"""Bounded W30 worker API, with exact mid-episode recovery and isolated evaluation."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .agent import Power30Agent
from .ledger import CallLedger
from .control import RuntimeControl
from .logging import EvidenceLogs, atomic_json, content_sha

DEVELOPMENT_NODES = (50000, 100000, 150000)
POLICY_NODES = (0,) + DEVELOPMENT_NODES
IO_HOOK = None


class RecoveryAccountingRequired(RuntimeError):
    pass


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if value["cuda"] is not None:
        # map_location=agent.device also remaps RNG byte tensors to CUDA.
        # Both CPU and CUDA generator setters require host ByteTensor state.
        torch.cuda.set_rng_state_all([state.cpu() for state in value["cuda"]])


@contextmanager
def isolated_rng(seed):
    before = rng_state()
    try:
        seed_all(seed)
        yield
    finally:
        restore_rng(before)


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    if IO_HOOK is not None:
        IO_HOOK(str(path),path.stat().st_size)


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_equal(a, b):
    """Exact checkpoint readback without an extra forward/update/PHY."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_tree_equal(a[k],b[k]) for k in a)
    if isinstance(a, (tuple,list)):
        return type(a) is type(b) and len(a)==len(b) and all(_tree_equal(x,y) for x,y in zip(a,b))
    return a == b


def _default_loader(record):
    from satellite_rl.t0_width.data import load_published_layout
    from satellite_rl.shared_interface.stages import UniformStage2Input
    from satellite_rl.shared_interface.spherical_gaussian import PHYSICS
    case, layout, authority = load_published_layout(record, role=record["role"])
    return UniformStage2Input(case, layout, authority, PHYSICS)


def _default_factory(stage_input, group):
    raise RuntimeError("W30 environment factory must bind menu and permanent call ledger")


def _case_identity(stage_input, record):
    if hasattr(stage_input, "layout"):
        layout = stage_input.layout
        from satellite_rl.shared_interface.spherical_gaussian import PHYSICS_SHA
        return dict(case_id=record.get("case_id"), layout_SHA=layout["content_sha256"],
                    aircraft_SHA=layout.get("parent_case_SHA"), physics_SHA=PHYSICS_SHA)
    # Injectable tiny fixtures are restricted to the correctness phase.
    return dict(input_reference=record)


def _terminal(env, identity, **fields):
    # The actual wire allocation is preserved once; service rows do not repeat its action arrays.
    beams = [{key: row[key] for key in ("beam_id", "demand_bps", "capacity_bps", "SI", "SGM")}
             for row in env.beam_results()]
    value = dict(identity=identity, metrics=env.metrics(), beams=beams, **fields)
    allocation = env.export_allocation()
    value["allocation_SHA"] = allocation.get("content_sha256")
    value["allocation_parent_SHA"] = allocation.get("parent_layout_SHA", identity.get("layout_SHA"))
    from satellite_rl.shared_interface.content import wire_bytes
    value["allocation_wire"] = json.loads(wire_bytes(allocation))
    return value


def evaluate_cases(agent, records, group, *, loader=None, env_factory=None, seed_base=274000,
                   repeats=3, node=150000, logs=None, phase="development", stop_requested=None,
                   start_cursor=0, wal_path=None, row_committed=None, wal_context=None,
                   mother_offset=0, repeat_offset=0, call_ledger=None):
    """Paired mother/repeat seed map; no sampling, replay or initialization leakage."""
    loader = loader or _default_loader
    env_factory = env_factory or _default_factory
    rows = []
    with isolated_rng(seed_base):
        for mother, record in enumerate(records):
            for repeat in range(repeats):
                cursor = mother * repeats + repeat
                if cursor < start_cursor:
                    continue
                if stop_requested is not None and stop_requested():
                    return rows, False
                actual_mother, actual_repeat = mother + mother_offset, repeat + repeat_offset
                action_seed = int(seed_base + actual_mother + actual_repeat * 1000000)
                seed_all(action_seed)
                if wal_path is not None and Path(wal_path).exists():
                    preflight_wal=torch.load(wal_path,map_location='cpu',weights_only=False)
                    if preflight_wal.get('stage')=='intent':
                        raise RecoveryAccountingRequired('Unknown evaluation intent; no new kernel or feature allowed before root accounting')
                started = time.perf_counter()
                stage_input = loader(record)
                identity = _case_identity(stage_input, record)
                env = env_factory(stage_input, group)
                state = env.reset()
                if hasattr(agent, "bind_environment"):
                    agent.bind_environment(env)
                timing = dict(input_and_reset_s=time.perf_counter() - started, mask_s=0.,
                              inference_s=0., environment_s=0.)
                done = False
                resumed_from_WAL = False
                prior_algorithm_s = 0.
                index = 0
                step_rows = []
                wal_identity = dict(node=node, group=group, phase=phase, cursor=cursor,
                                    case=identity, action_seed=action_seed, context=wal_context)
                if wal_path is not None and Path(wal_path).exists():
                    saved = torch.load(wal_path, map_location="cpu", weights_only=False)
                    if saved["identity"] == wal_identity:
                        if saved.get("stage") == "intent":
                            raise RecoveryAccountingRequired("Evaluation decision intent has unknown actual spend; preserve WAL")
                        env.load_state_dict(saved["environment"])
                        state = saved["observation"]
                        timing = saved["timing"]
                        resumed_from_WAL = True
                        prior_algorithm_s = sum(timing.values())
                        step_rows = saved["step_rows"]
                        index = len(step_rows)
                        done = saved["done"]
                        restore_rng(saved["rng"])
                    elif saved["identity"].get("node") == node and saved["identity"].get("cursor", 0) < cursor:
                        Path(wal_path).unlink()  # Already committed same-node predecessor only.
                    else:
                        raise ValueError("Evaluation WAL identity mismatch")
                if call_ledger is not None and not resumed_from_WAL:
                    call_ledger.charge("evaluation_episode")
                while not done:
                    if stop_requested is not None and stop_requested():
                        return rows, False  # Preserve committed per-step WAL for exact same repeat.
                    if wal_path is not None:
                        _atomic_torch(wal_path, dict(identity=wal_identity, stage="intent",
                                      environment=env.state_dict(), observation=state, timing=timing,
                                      step_rows=step_rows, done=done, rng=rng_state(),
                                      unknown_next_step_spend_not_zero=True))
                    t = time.perf_counter()
                    mask = env.action_mask()
                    timing["mask_s"] += time.perf_counter() - t
                    _sync(agent.device)
                    t = time.perf_counter()
                    action = agent.act(state, deterministic=False, action_mask=mask)
                    _sync(agent.device)
                    timing["inference_s"] += time.perf_counter() - t
                    t = time.perf_counter()
                    relation_audit = _relation_audit(env)
                    state, reward, done, info = env.step(action)
                    if hasattr(agent,"last_action_info"):
                        info["policy_diagnostics"] = deepcopy(agent.last_action_info)
                    if not np.isfinite(reward) or not np.isfinite(state).all():
                        raise FloatingPointError("Nonfinite evaluation transition")
                    if hasattr(agent, "last_preference"):
                        info["source_preference"] = agent.last_preference
                    timing["environment_s"] += time.perf_counter() - t
                    step_rows.append(_step_record(info, identity, phase=phase, node=node,
                                    mother=actual_mother, repeat=actual_repeat, action_seed=action_seed,
                                    beam_step=index, reward=reward, relation_audit=relation_audit))
                    index += 1
                    if wal_path is not None:
                        _atomic_torch(wal_path, dict(identity=wal_identity, stage="committed", environment=env.state_dict(),
                                      observation=state, timing=timing, step_rows=step_rows, done=done, rng=rng_state()))
                timing["total_s"] = sum(timing.values())
                terminal_started = time.perf_counter()
                row = _terminal(env, identity, phase=phase, node=node, mother=actual_mother,
                                repeat=actual_repeat, action_seed=action_seed, timing=timing)
                terminal_finished = time.perf_counter()
                # Preserve historical total_s; these metadata-only fields include
                # terminal metrics/export without changing actions or RNG state.
                timing["terminal_s"] = terminal_finished - terminal_started
                timing["algorithm_with_terminal_s"] = timing["total_s"] + timing["terminal_s"]
                timing["wall_to_terminal_s"] = terminal_finished - started
                timing["resumed_from_WAL"] = resumed_from_WAL
                timing["previous_algorithm_s"] = prior_algorithm_s
                timing["wall_to_terminal_scope"] = "current_invocation_including_restore"
                rows.append(row)
                if logs is not None:
                    for step_row in step_rows:
                        logs.append("steps", step_row)
                    logs.append("evaluations", row)
                if row_committed is not None:
                    row_committed(cursor + 1)
                if wal_path is not None and Path(wal_path).exists():
                    Path(wal_path).unlink()
    return rows, True


def _step_record(info, identity, **fields):
    # Never repeat terminal beam_results, layout, full arrays, or a policy tensor per step.
    keys = ("beam_id", "requested_action", "actual_action", "rate_bps", "required_rate_bps",
            "satisfaction_increment", "beam_service_step", "timing", "source_preference", "policy_diagnostics")
    return dict(identity=identity, **{key: info[key] for key in keys if key in info}, **fields)


def _relation_audit(env):
    audit = getattr(env, "relation_audit", None)
    if audit is None:
        return None
    keys = ("out_raw_w_float64", "future_in_raw_w_float64",
            "out_observation_float32", "future_in_observation_float32")
    return {key: np.asarray(audit[key]).copy() for key in keys}


def _validate_task(config, records, dev, loader, env_factory):
    phase = config.get("phase", "scientific")
    total = config["training"]["total_steps"]
    if phase == "scientific":
        if total != 150000 or len(records) != 576 or len(dev) != 20:
            raise ValueError("Scientific W30 requires 150000 steps,576 train,20 development mothers")
        if loader is not None or env_factory is not None:
            raise ValueError("Scientific tasks cannot inject fixture loaders/environments")
        if not config.get("source_sha") or not config.get("data_manifest_sha"):
            raise ValueError("Frozen source and qualified data manifest required")
        from .control import validate_data_admission
        validate_data_admission(config)
    elif phase == "engineering":
        if not 2000 <= total <= 2048:
            raise ValueError("Engineering must include real T0 update, at most2048 steps")
        if dev:
            raise ValueError("Engineering cannot consume science evaluation budget")
    elif phase == "correctness":
        if not 1 <= total <= 16 or dev:
            raise ValueError("Tiny correctness fixtures only")
    else:
        raise ValueError("Unknown task phase")
    if not records:
        raise ValueError("No training cases")
    if phase != "correctness":
        t = config["training"]
        required = dict(batch_size=256, warmup_steps=2000, replay_capacity=20000,
                        update_every=1, gradient_steps=1, gradient_clip=10)
        if any(t[k] != v for k, v in required.items()):
            raise ValueError("T0 training budget/update contract altered")


def run_task(config, run_dir, progress_callback=None, stop_requested=None, *, loader=None, env_factory=None):
    """Workbench worker entry. No background submission or remote operations here.

    config adds train_cases/dev_cases (data manifest case records), source_sha,
    data_manifest_sha and run_id to build_config(). Stop saves exact current state.
    """
    global IO_HOOK
    session_start = time.perf_counter()
    config = deepcopy(config)
    if config.get("task_kind") is not None:
        raise ValueError("W30 conditional confirmation requires separate root-admitted executor")
    records = config.get("train_cases", [])
    dev = config.get("dev_cases", [])
    _validate_task(config, records, dev, loader, env_factory)
    loader = loader or _default_loader
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    config_sha = content_sha(config)
    manifest = directory / "config.json"
    if manifest.exists() and content_sha(json.loads(manifest.read_text(encoding="utf8"))) != config_sha:
        raise ValueError("Existing run config/source/data identity differs")
    atomic_json(manifest, config)
    checkpoint_dir = Path(config.get("model_dir", directory / "models"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seed_all(config["training"]["seed"])
    control = RuntimeControl(config, directory)
    control.validate_owner()
    total_steps=config['training']['total_steps']
    phase=config['phase']
    interaction_kind='engineering_interaction' if phase=='engineering' else 'scientific_interaction'
    update_kind='engineering_update' if phase=='engineering' else 'scientific_update'
    expected_updates=max(0,total_steps-config['training']['warmup_steps']+1)
    ledger = CallLedger(config["ledger_root"], config["run_id"], phase,
        allowances={interaction_kind:total_steps,update_kind:expected_updates,
                    'update':expected_updates,'evaluation_episode':180 if phase=='scientific' else 0})
    IO_HOOK=ledger.record_io
    def call_hook(kind, **metadata):
        control.check_or_raise()
        ledger.charge(kind, **metadata)
    from .environment import Power30Env
    env_factory = env_factory or (lambda stage_input, group: Power30Env(stage_input, config["menu"], group, call_hook=call_hook))
    if config["phase"] == "engineering" and str(config["training"]["device"]).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    agent = Power30Agent(config, call_hook=call_hook)
    def parameter_hashes():
        return {name: hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in getattr(agent,name).parameters())).hexdigest()
                for name in ("actor", "critic1", "critic2")}
    initial_parameter_hashes = parameter_hashes()
    user_stop = stop_requested
    stop_requested = lambda: (user_stop is not None and user_stop()) or control.stop_requested()
    order_rng = random.Random(config["training"]["scene_order_seed"])
    progress = dict(env_step=0, episode=0, cycle=0, position=0, order=[], completed_nodes=[],
                    episode_return=0., episode_timing={}, wall_seconds=0., cumulative_update_seconds=0.)
    current = None
    env = None
    state = None
    resumed = False
    recoveries = []
    def needs_accounting(details):
        receipt = dict(stage="needs_root_recovery_accounting", phase=config.get("phase"),
                       group=config["group"], env_step=progress["env_step"],
                       config_sha=config_sha, reasons=details,
                       preserved_original_files=True, automatic_replay_allowed=False,
                       unknown_spend_not_zero=True)
        atomic_json(directory / "recovery_accounting_required.json", receipt)
        if progress_callback is not None:
            progress_callback(receipt)
        return receipt
    for name in ("current.pt", "previous.pt"):
        path = checkpoint_dir / name
        if not path.exists():
            continue
        try:
            saved = torch.load(path, map_location=agent.device, weights_only=False)
            if saved.get("config_sha") != config_sha:
                raise ValueError("Checkpoint configuration identity mismatch")
            preflight_wal_path=directory/'training_decision_wal.pt'
            if preflight_wal_path.exists():
                preflight_wal=torch.load(preflight_wal_path,map_location='cpu',weights_only=False)
                checkpoint_step=saved['progress']['env_step']
                if (preflight_wal.get('config_sha')!=config_sha
                        or preflight_wal.get('env_step',0)>checkpoint_step
                        or (preflight_wal.get('stage')=='intent' and preflight_wal.get('next_env_step',0)>checkpoint_step)):
                    raise RecoveryAccountingRequired('Post-checkpoint actual/unknown spend; refuse new kernel before root accounting')
            agent.load_state_dict(saved["agent"])
            if not _tree_equal(saved["agent"], agent.state_dict()):
                raise ValueError("Exact complete optimizer/replay/network/RNG restore differs")
            progress = saved["progress"]
            order_rng.setstate(saved["scene_rng"])
            current = saved["current_case"]
            if current is not None:
                stage_input = loader(records[current])
                env = env_factory(stage_input, config["group"])
                env.load_state_dict(saved["environment"])
                state = np.asarray(saved["observation"], dtype=np.float32)
                if not np.array_equal(state, env._observation()):
                    raise ValueError("Saved observation differs from recovered environment")
            # Reject invalid offsets before modifying logs.
            for log_name, wanted in saved["log_offsets"].items():
                path_log = directory / "logs" / (log_name + ".jsonl.gz")
                if wanted and (not path_log.exists() or path_log.stat().st_size < wanted):
                    raise ValueError("Missing checkpoint evidence prefix")
            evaluation_progress = directory / "evaluation_cursor.json"
            if evaluation_progress.exists():
                partial = json.loads(evaluation_progress.read_text(encoding="utf8"))
                if partial.get("config_sha") != config_sha:
                    raise ValueError("Evaluation continuation identity mismatch")
                if partial["node"] == progress["env_step"]:
                    progress["evaluation_cursor"] = partial["cursor"]
                    progress["evaluation_node"] = partial["node"]
                    progress["completed_nodes"] = partial["completed_nodes"]
                    saved["log_offsets"] = partial["log_offsets"]
            post_checkpoint = []
            decision_wal = directory / "training_decision_wal.pt"
            if decision_wal.exists():
                last_decision = torch.load(decision_wal, map_location="cpu", weights_only=False)
                if last_decision.get("config_sha") != config_sha:
                    post_checkpoint.append(dict(reason="decision WAL identity differs"))
                elif (last_decision.get("env_step", 0) > progress["env_step"] or
                      (last_decision.get("stage") == "intent" and
                       last_decision.get("next_env_step", 0) > progress["env_step"])):
                    post_checkpoint.append(dict(reason="actual or unknown post-checkpoint decision/update spend",
                                                checkpoint_step=progress["env_step"], decision=last_decision["receipt"]))
            for log_name, expected_offset in saved["log_offsets"].items():
                log_path = directory / "logs" / (log_name + ".jsonl.gz")
                actual_offset = log_path.stat().st_size if log_path.exists() else 0
                if actual_offset != expected_offset:
                    post_checkpoint.append(dict(reason="unique uncommitted evidence suffix preserved",
                        log=str(log_path), checkpoint_bytes=expected_offset, actual_bytes=actual_offset))
            if recoveries or post_checkpoint:
                return needs_accounting(recoveries + post_checkpoint)
            restore_rng(saved["rng"])
            logs = EvidenceLogs(directory / "logs", saved["log_offsets"])
            atomic_json(directory / "recovery_readback.json",dict(
                source_sha=config['source_sha'],config_sha=config_sha,
                checkpoint_sha=_sha_file(path),env_step=progress['env_step'],
                exact_optimizer_replay_network_rng_readback=True,
                extra_phy=0,extra_forward=0,extra_update=0))
            resumed = True
            break
        except Exception as exc:
            recoveries.append(dict(path=str(path), reason=repr(exc)))
    if not resumed:
        if recoveries or any((directory / "logs" / (x + ".jsonl.gz")).exists() for x in EvidenceLogs.NAMES):
            return needs_accounting(recoveries or [dict(reason="orphan unique evidence without trustworthy checkpoint")])
        logs = EvidenceLogs(directory / "logs")
    atomic_json(directory / "model_structure.json", dict(schema=agent.schema,
                parameters=agent.parameter_counts(), resumed=resumed,
                input_projection_initialization_rng=agent.extension_rng))
    base_wall = progress["wall_seconds"]
    total = config["training"]["total_steps"]
    phase = config["phase"]
    interval = config["training"]["checkpoint_interval"]
    status = "running"

    def notify(stage):
        ledger.flush()
        nonlocal status
        status = stage
        value = dict(stage=stage, env_step=progress["env_step"], update_step=agent.update_step,
                     episode=progress["episode"], budget=total, group=config["group"], phase=phase,
                     run_id=config.get("run_id"), wall_seconds=base_wall + time.perf_counter() - session_start,
                     calls=ledger.snapshot(),
                     disk_free_bytes=__import__("shutil").disk_usage(directory).free,
                     model_bytes=sum(p.stat().st_size for p in checkpoint_dir.glob("*.pt")))
        atomic_json(directory / "progress.json", value)
        if progress_callback is not None:
            progress_callback(value)
        return value

    def save():
        ledger.flush()
        progress["wall_seconds"] = base_wall + time.perf_counter() - session_start
        value = dict(ledger_snapshot=ledger.snapshot(), version="W30ResumeV1", config_sha=config_sha, agent=agent.state_dict(),
                     progress=deepcopy(progress), scene_rng=order_rng.getstate(), rng=rng_state(),
                     current_case=current, environment=env.state_dict() if env is not None else None,
                     observation=state, log_offsets=logs.barrier())
        staging = checkpoint_dir / "staging.pt"
        _atomic_torch(staging, value)
        # Before rotation verify that the new file can be read, not just that torch.save returned.
        verified = torch.load(staging, map_location="cpu", weights_only=False)
        if verified["config_sha"] != config_sha or verified["progress"]["env_step"] != progress["env_step"]:
            raise RuntimeError("Checkpoint readback mismatch")
        current_path = checkpoint_dir / "current.pt"
        if current_path.exists():
            os.replace(current_path, checkpoint_dir / "previous.pt")
        os.replace(staging, current_path)
        atomic_json(checkpoint_dir / "resume_index.json", dict(current_SHA=_sha_file(current_path),
                    env_step=progress["env_step"], retained=[x for x in ("current.pt", "previous.pt") if (checkpoint_dir / x).exists()]))

    def develop():
        node = progress["env_step"]
        if phase != "scientific" or node not in POLICY_NODES or node in progress["completed_nodes"]:
            return True
        policy_path = checkpoint_dir / ("policy_%06d.pt" % node)
        if policy_path.exists():
            old = torch.load(policy_path, map_location="cpu", weights_only=False)
            if old.get("config_sha") != config_sha or old.get("env_step") != node:
                raise ValueError("Evaluation policy identity mismatch")
        else:
            _atomic_torch(policy_path, dict(**agent.policy_state(), config_sha=config_sha, env_step=node,
                         source_sha=config.get("source_sha"), data_manifest_sha=config.get("data_manifest_sha")))
        if node == 0:
            progress["completed_nodes"].append(0)
            return True
        # Freeze the exact training state before evaluation; each episode then only writes a
        # small cursor receipt/WAL, avoiding thousands of redundant full-replay saves.
        if progress.get("evaluation_node") != node:
            progress["evaluation_node"] = node
            progress["evaluation_cursor"] = 0
            save()
        def row_committed(cursor):
            progress["evaluation_cursor"] = cursor
            atomic_json(directory / "evaluation_cursor.json", dict(config_sha=config_sha,
                        node=node, cursor=cursor, completed_nodes=progress["completed_nodes"],
                        log_offsets=logs.barrier()))
            notify("development")
        ledger.set_phase("evaluation")
        _, finished = evaluate_cases(agent, dev, config["group"], loader=loader, env_factory=env_factory,
                            node=node, logs=logs, stop_requested=stop_requested,
                            start_cursor=progress["evaluation_cursor"],
                            wal_path=directory / "evaluation_wal.pt", row_committed=row_committed, call_ledger=ledger)
        ledger.set_phase(config["phase"])
        if finished:
            wal = directory / "evaluation_wal.pt"
            if wal.exists():
                saved_wal = torch.load(wal, map_location="cpu", weights_only=False)
                if saved_wal["identity"]["node"] != node:
                    raise ValueError("Unexpected stale evaluation WAL")
                wal.unlink()
            progress["completed_nodes"].append(node)
            row_committed(len(dev) * 3)
            save()
        return finished

    try:
        if progress.get("completed"):
            ledger.close()
            return notify("completed")
        # Initial checkpoint makes interruption during initial validation recoverable.
        if not resumed:
            save()
        while progress["env_step"] < total:
            if (phase == "engineering" and progress["env_step"] == 2016
                    and not progress.get("engineering_recovery_probe_done")):
                # Exactly 17 updates before and 32 after one same-ID pause.
                # This does not allocate a second optimizer trajectory.
                if agent.update_step != 17:
                    raise RuntimeError("Engineering recovery probe expected17updates at2016")
                progress["engineering_recovery_probe_done"] = True
                save()
                ledger.close()
                return notify("stopped")
            if stop_requested is not None and stop_requested():
                save()
                ledger.close()
                return notify("stopped")
            # Continue the exact WAL mother/repeat and committed prefix. Unknown
            # intent or spend beyond a full checkpoint requires root accounting.
            if not develop():
                ledger.close()
                return notify("stopped_during_evaluation")
            if env is None:
                if progress["position"] == len(progress["order"]):
                    progress["order"] = list(range(len(records)))
                    order_rng.shuffle(progress["order"])
                    progress["position"] = 0
                    progress["cycle"] += 1
                current = progress["order"][progress["position"]]
                t = time.perf_counter()
                stage_input = loader(records[current])
                env = env_factory(stage_input, config["group"])
                state = env.reset()
                progress["episode_return"] = 0.
                progress["episode_timing"] = dict(input_and_reset_s=time.perf_counter() - t,
                                                   mask_s=0., inference_s=0., environment_s=0., update_s=0.)
            identity = _case_identity(env.stage_input, records[current])
            timing = progress["episode_timing"]
            t = time.perf_counter()
            mask = env.action_mask()
            timing["mask_s"] += time.perf_counter() - t
            decision_receipt = dict(episode=progress["episode"], next_env_step=progress["env_step"] + 1,
                                    update_step_before=agent.update_step, case=identity)
            _atomic_torch(directory / "training_decision_wal.pt", dict(config_sha=config_sha,
                          stage="intent", next_env_step=progress["env_step"] + 1,
                          env_step=progress["env_step"], receipt=decision_receipt,
                          rng=rng_state(), log_offsets=dict(logs.offsets)))
            t = time.perf_counter()
            action = agent.act(state, action_mask=mask, env_step=progress["env_step"])
            timing["inference_s"] += time.perf_counter() - t
            t = time.perf_counter()
            relation_audit = _relation_audit(env)
            ledger.charge("engineering_interaction" if phase == "engineering" else "scientific_interaction")
            next_state, reward, done, info = env.step(action)
            if hasattr(agent,"last_action_info"):
                info["policy_diagnostics"] = deepcopy(agent.last_action_info)
            if not np.isfinite(reward) or not np.isfinite(next_state).all():
                raise FloatingPointError("Nonfinite training transition; checkpoint preserved")
            timing["environment_s"] += time.perf_counter() - t
            t = time.perf_counter()
            next_mask = None if done else env.action_mask()
            timing["mask_s"] += time.perf_counter() - t
            agent.observe(state, action, reward, next_state, done, mask, next_mask)
            progress["env_step"] += 1
            progress["episode_return"] += reward
            t = time.perf_counter()
            if agent.ready(progress["env_step"], done):
                ledger.charge("engineering_update" if phase == "engineering" else "scientific_update")
                ledger.charge("update")
                update = agent.update()
            else:
                update = None
            update_elapsed = time.perf_counter() - t
            timing["update_s"] += update_elapsed
            progress["cumulative_update_seconds"] = progress.get("cumulative_update_seconds", 0.) + update_elapsed
            logs.append("steps", _step_record(info, identity, phase=phase,
                        env_step=progress["env_step"], episode=progress["episode"],
                        reward=reward, update=update, relation_audit=relation_audit))
            step_offsets = logs.barrier()
            decision_receipt.update(actual_action=info.get("actual_action"), requested_action=list(action),
                                    env_step=progress["env_step"], update_step_after=agent.update_step)
            _atomic_torch(directory / "training_decision_wal.pt", dict(config_sha=config_sha,
                          stage="committed", env_step=progress["env_step"], receipt=decision_receipt,
                          rng=rng_state(), log_offsets=step_offsets))
            state = next_state
            if done:
                logs.append("episodes", _terminal(env, identity, phase=phase,
                            episode=progress["episode"], env_step=progress["env_step"],
                            episode_return=progress["episode_return"], timing=timing))
                progress["episode"] += 1
                progress["position"] += 1
                env = current = state = None
            if progress["env_step"] % interval == 0 or progress["env_step"] in POLICY_NODES:
                save()
            if progress["env_step"] % config["training"]["log_interval"] == 0:
                notify("running")
        if not develop():
            ledger.close()
            return notify("stopped_during_evaluation")
        if env is not None:
            logs.append("episodes", _terminal(env, identity, phase=phase,
                        episode=progress["episode"], env_step=progress["env_step"],
                        budget_truncated=True, natural_terminal=False,
                        episode_return=progress["episode_return"], timing=progress["episode_timing"]))
        expected_updates = total - config["training"]["warmup_steps"] + 1
        if phase in ("engineering", "scientific") and agent.update_step != expected_updates:
            raise RuntimeError("Actual update count differs from frozen T0 inclusive warmup schedule")
        if phase == "engineering" and agent.update_step < 1:
            raise RuntimeError("Engineering qualification must execute a real update")
        if phase == "engineering":
            final_hashes = parameter_hashes()
            if any(final_hashes[k] == initial_parameter_hashes[k] for k in final_hashes):
                raise RuntimeError("Engineering actor/online Critic parameters did not all change")
            atomic_json(directory / "engineering_result.json", dict(
                initial_parameter_hashes=initial_parameter_hashes, final_parameter_hashes=final_hashes,
                actual_updates=agent.update_step, expected_updates=expected_updates,
                measured_update_seconds=progress["cumulative_update_seconds"],
                forecast_150k_update_seconds=progress["cumulative_update_seconds"] / agent.update_step * 148001,
                measured_complete_worker_wall_seconds=base_wall+time.perf_counter()-session_start,
                forecast_150k_complete_seconds=(base_wall+time.perf_counter()-session_start-progress["cumulative_update_seconds"])/total*150000+progress["cumulative_update_seconds"]/agent.update_step*148001,
                logical_atomic_io=ledger.snapshot().get('logical_atomic_io'),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(agent.device) if agent.device.type == "cuda" else None,
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(agent.device) if agent.device.type == "cuda" else None,
                model_bytes=sum(p.stat().st_size for p in checkpoint_dir.glob("*.pt")),
                log_bytes=sum(p.stat().st_size for p in (directory / "logs").glob("*")),
                calls=ledger.snapshot(), source_sha=config["source_sha"], root_acceptance_required=True))
        progress["completed"] = True
        save()
        ledger.close()
        return notify("completed")
    except RecoveryAccountingRequired as exc:
        ledger.flush()
        return needs_accounting([dict(reason=str(exc), evaluation_WAL_preserved=True)])
    except BaseException as exc:
        # Do not certify a failed/mutating update as a trustworthy continuation.
        atomic_json(directory / "failure.json", dict(reason=repr(exc), phase=phase,
                    env_step=progress["env_step"], update_step=agent.update_step))
        ledger.flush()
        notify("failed")
        raise
    finally:
        logs.close()


