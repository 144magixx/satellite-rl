"""Bounded W26 worker API, with exact mid-episode recovery and isolated evaluation."""
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

from .agent import WidthAgent, GROUPS, state_dimension
from .logging import EvidenceLogs, atomic_json, content_sha

DEVELOPMENT_NODES = (0, 2000, 5000) + tuple(range(10000, 150001, 10000))


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
        torch.cuda.set_rng_state_all(value["cuda"])


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


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_loader(record):
    from .data import load_published_layout
    from satellite_rl.shared_interface.stages import UniformStage2Input
    from satellite_rl.shared_interface.spherical_gaussian import PHYSICS
    case, layout, authority = load_published_layout(record, role=record["role"])
    return UniformStage2Input(case, layout, authority, PHYSICS)


def _default_factory(stage_input, group):
    from .environment import T0WidthEnv
    return T0WidthEnv(stage_input, group)


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
                   mother_offset=0, repeat_offset=0):
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
                        step_rows = saved["step_rows"]
                        index = len(step_rows)
                        done = saved["done"]
                        restore_rng(saved["rng"])
                    elif saved["identity"].get("node") == node and saved["identity"].get("cursor", 0) < cursor:
                        Path(wal_path).unlink()  # Already committed same-node predecessor only.
                    else:
                        raise ValueError("Evaluation WAL identity mismatch")
                while not done:
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
                row = _terminal(env, identity, phase=phase, node=node, mother=actual_mother,
                                repeat=actual_repeat, action_seed=action_seed, timing=timing)
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
            "satisfaction_increment", "beam_service_step", "timing", "source_preference")
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
            raise ValueError("Scientific W26 requires 150000 steps,576 train,20 development mothers")
        if loader is not None or env_factory is not None:
            raise ValueError("Scientific tasks cannot inject fixture loaders/environments")
        if not config.get("source_sha") or not config.get("data_manifest_sha"):
            raise ValueError("Frozen source and qualified data manifest required")
        from .data import validate_training_admission
        validate_training_admission(config)
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
    config = deepcopy(config)
    if config.get("task_kind") == "confirmation":
        if loader is not None or env_factory is not None:
            raise ValueError("Confirmation cannot use injected fixtures")
        return run_confirmation_task(config, run_dir, progress_callback, stop_requested)
    records = config.get("train_cases", [])
    dev = config.get("dev_cases", [])
    _validate_task(config, records, dev, loader, env_factory)
    loader = loader or _default_loader
    env_factory = env_factory or _default_factory
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
    agent = WidthAgent(config)
    order_rng = random.Random(config["training"]["scene_order_seed"])
    progress = dict(env_step=0, episode=0, cycle=0, position=0, order=[], completed_nodes=[],
                    episode_return=0., episode_timing={}, wall_seconds=0.)
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
            agent.load_state_dict(saved["agent"])
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
            resumed = True
            break
        except Exception as exc:
            recoveries.append(dict(path=str(path), reason=repr(exc)))
    if not resumed:
        if recoveries or any((directory / "logs" / (x + ".jsonl.gz")).exists() for x in EvidenceLogs.NAMES):
            return needs_accounting(recoveries or [dict(reason="orphan unique evidence without trustworthy checkpoint")])
        logs = EvidenceLogs(directory / "logs")
    atomic_json(directory / "model_structure.json", dict(schema=agent.schema,
                parameters=agent.parameter_counts(), resumed=resumed))
    session_start = time.perf_counter()
    base_wall = progress["wall_seconds"]
    total = config["training"]["total_steps"]
    phase = config["phase"]
    interval = config["training"]["checkpoint_interval"]
    status = "running"

    def notify(stage):
        nonlocal status
        status = stage
        value = dict(stage=stage, env_step=progress["env_step"], update_step=agent.update_step,
                     episode=progress["episode"], budget=total, group=config["group"], phase=phase,
                     run_id=config.get("run_id"), wall_seconds=base_wall + time.perf_counter() - session_start,
                     disk_free_bytes=__import__("shutil").disk_usage(directory).free,
                     model_bytes=sum(p.stat().st_size for p in checkpoint_dir.glob("*.pt")))
        atomic_json(directory / "progress.json", value)
        if progress_callback is not None:
            progress_callback(value)
        return value

    def save():
        progress["wall_seconds"] = base_wall + time.perf_counter() - session_start
        value = dict(version="W26ResumeV1", config_sha=config_sha, agent=agent.state_dict(),
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
        if phase != "scientific" or node not in DEVELOPMENT_NODES or node in progress["completed_nodes"]:
            return True
        policy_path = checkpoint_dir / ("policy_%06d.pt" % node)
        if policy_path.exists():
            old = torch.load(policy_path, map_location="cpu", weights_only=False)
            if old.get("config_sha") != config_sha or old.get("env_step") != node:
                raise ValueError("Evaluation policy identity mismatch")
        else:
            _atomic_torch(policy_path, dict(**agent.policy_state(), config_sha=config_sha, env_step=node,
                         source_sha=config.get("source_sha"), data_manifest_sha=config.get("data_manifest_sha")))
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
        _, finished = evaluate_cases(agent, dev, config["group"], loader=loader, env_factory=env_factory,
                            node=node, logs=logs, stop_requested=stop_requested,
                            start_cursor=progress["evaluation_cursor"],
                            wal_path=directory / "evaluation_wal.pt", row_committed=row_committed)
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
            return notify("completed")
        # Initial checkpoint makes interruption during initial validation recoverable.
        if not resumed:
            save()
        while progress["env_step"] < total:
            if stop_requested is not None and stop_requested():
                save()
                return notify("stopped")
            # No partial node logs are committed: interruption reruns that fixed node with isolated RNG.
            if not develop():
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
            next_state, reward, done, info = env.step(action)
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
            update = agent.update() if agent.ready(progress["env_step"], done) else None
            timing["update_s"] += time.perf_counter() - t
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
            if progress["env_step"] % interval == 0 or progress["env_step"] in DEVELOPMENT_NODES:
                save()
            if progress["env_step"] % config["training"]["log_interval"] == 0:
                notify("running")
        if not develop():
            return notify("stopped_during_evaluation")
        if env is not None:
            logs.append("episodes", _terminal(env, identity, phase=phase,
                        episode=progress["episode"], env_step=progress["env_step"],
                        budget_truncated=True, natural_terminal=False,
                        episode_return=progress["episode_return"], timing=progress["episode_timing"]))
        if phase == "engineering" and agent.update_step < 1:
            raise RuntimeError("Engineering qualification must execute a real update")
        progress["completed"] = True
        save()
        return notify("completed")
    except RecoveryAccountingRequired as exc:
        return needs_accounting([dict(reason=str(exc), evaluation_WAL_preserved=True)])
    except BaseException as exc:
        # Do not certify a failed/mutating update as a trustworthy continuation.
        atomic_json(directory / "failure.json", dict(reason=repr(exc), phase=phase,
                    env_step=progress["env_step"], update_step=agent.update_step))
        notify("failed")
        raise
    finally:
        logs.close()


class BackgroundPolicy:
    """Historical preference then the identical W26 feasible domain; no lookahead PHY."""
    def __init__(self, name, device):
        if name not in ("random", "fixed", "greedy"):
            raise ValueError("Unknown background policy")
        self.name = name
        self.device = torch.device(device)
        self.last_preference = None

    def bind_environment(self, env):
        self.env = env

    def act(self, state, deterministic=False, action_mask=None):
        pairs = np.argwhere(action_mask["frequency_slots"])
        powers = np.flatnonzero(action_mask["power"])
        if self.name == "random":
            f, l = pairs[np.random.randint(len(pairs))]
            p = powers[np.random.randint(len(powers))]
            self.last_preference = dict(rule="random", uniform="legal_FL_and_legal_P")
            return int(f), int(l), int(p)
        if self.name == "fixed":
            desired_start, desired_count, desired_power = self.env.beam_idx // 8 * 8, 8, 30
        else:
            # This is the original code's all-eight-group load preference, including
            # its historical 10-minus-column-count convention, not a new greedy search.
            column_load = self.env.occupied.sum(axis=0)
            desired_start = int(np.argmin(column_load))
            desired_count = max(1, min(5, int(10 - column_load[desired_start])))
            current = self.env.indices[self.env.beam_idx]
            demand = self.env.demands[current]
            desired_power = 30 if demand >= 800e6 else 25 if demand >= 400e6 else 20
        clamped_start = min(99, max(0, desired_start))
        f, l = min(((int(f), int(l)) for f, l in pairs),
                   key=lambda pair: (abs(pair[1] + 1 - desired_count),
                                     abs(pair[0] - clamped_start), pair[0], pair[1]))
        p = min((int(p) for p in powers), key=lambda p: (abs((p + 1) * 5 - desired_power), (p + 1) * 5, p))
        self.last_preference = dict(rule=self.name, start=desired_start, count=desired_count,
                                    power_w=desired_power, mapped_indices=[f, l, p],
                                    mapping_changed=(f != desired_start or l + 1 != desired_count or (p + 1) * 5 != desired_power))
        return f, l, p


def run_confirmation_task(config, run_dir, progress_callback=None, stop_requested=None):
    """One locked, interleaved 680-episode Direct matrix; no optimizer or replay loading."""
    from .agent import build_config
    from .data import read_json_ref, load_published_layout
    from satellite_rl.shared_interface.stages import UniformStage2Input
    from satellite_rl.shared_interface.spherical_gaussian import PHYSICS
    manifest = read_json_ref(config["confirmation_manifest_ref"])
    rows = manifest["rows"]
    if (manifest.get("schema") != "W26PublishedDatasetV1" or manifest.get("root_approved") is not True
            or manifest.get("role") != "test" or len(rows) != 40 or len({r["case_id"] for r in rows}) != 40):
        raise ValueError("Complete root-pinned confirmation partition required")
    unlock = deepcopy(config["confirmation_unlock"])
    if unlock.get("manifest_SHA") != config["confirmation_manifest_ref"]["sha256"]:
        raise ValueError("Confirmation unlock belongs to another partition")
    unlock["manifest_ref"] = config["confirmation_manifest_ref"]
    receipt = read_json_ref(unlock["receipt_ref"])
    if (receipt.get("schema") != "W26ConfirmationUnlockV1" or receipt.get("root_approved") is not True
            or receipt.get("manifest_SHA") != unlock.get("manifest_SHA")):
        raise ValueError("Confirmation terminal-lock receipt differs")
    refs = config["terminal_policies"]
    if set(refs) != set(GROUPS) or receipt.get("terminal_model_SHAs") != {g: refs[g]["sha256"] for g in GROUPS}:
        raise ValueError("All four immutable terminal policies must match root lock")
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    task_sha = content_sha(config)
    status_path = directory / "confirmation_cursor.json"
    progress = dict(task_sha=task_sha, cursor=0, log_offsets={name: 0 for name in EvidenceLogs.NAMES})
    if status_path.exists():
        progress = json.loads(status_path.read_text(encoding="utf8"))
        if progress.get("task_sha") != task_sha:
            raise ValueError("Confirmation task identity differs")
    for name, expected in progress["log_offsets"].items():
        log_path = directory / "logs" / (name + ".jsonl.gz")
        actual = log_path.stat().st_size if log_path.exists() else 0
        if actual != expected:
            result = dict(stage="needs_root_recovery_accounting", task_sha=task_sha,
                          completed_episodes=progress["cursor"], preserved_original_files=True,
                          unknown_spend_not_zero=True, log=str(log_path), expected_bytes=expected, actual_bytes=actual)
            atomic_json(directory / "recovery_accounting_required.json", result)
            return result
    logs = EvidenceLogs(directory / "logs", progress["log_offsets"])
    device = config.get("device", "cuda")
    seed_all(51001)
    agents = {}
    for group, ref in refs.items():
        if _sha_file(ref["path"]) != ref["sha256"]:
            raise ValueError("Terminal policy file SHA differs")
        policy = torch.load(ref["path"], map_location=device, weights_only=False)
        cfg = build_config(group, device)
        agent = WidthAgent(cfg, device)
        if (policy.get("schema") != agent.schema or policy.get("env_step") != 150000
                or policy.get("source_sha") != config["source_sha"]):
            raise ValueError("Terminal policy provenance differs")
        agent.actor.load_state_dict(policy["actor"], strict=True)
        agents[group] = agent
    for name in ("random", "fixed", "greedy"):
        agents[name] = BackgroundPolicy(name, device)
    # 17 episodes/mother: four learned*three + random*three + two rules.
    jobs = [(g, r) for g in GROUPS for r in range(3)] + [("random", r) for r in range(3)] + [("fixed",0),("greedy",0)]
    def load_confirm(row):
        case, layout, authority = load_published_layout(row, role="test", confirmation_unlock=unlock)
        return UniformStage2Input(case, layout, authority, PHYSICS)
    try:
        for mother, row in enumerate(rows):
            rotated = jobs[mother % len(jobs):] + jobs[:mother % len(jobs)]
            for slot, (method, repeat) in enumerate(rotated):
                cursor = mother * 17 + slot
                if cursor < progress["cursor"]:
                    continue
                if stop_requested is not None and stop_requested():
                    return dict(stage="stopped", completed_episodes=progress["cursor"], budget=680)
                group = method if method in GROUPS else "S00"
                def commit(_):
                    progress["cursor"] = cursor + 1
                    progress["log_offsets"] = logs.barrier()
                    atomic_json(status_path, progress)
                    if progress_callback is not None:
                        progress_callback(dict(stage="confirmation", completed_episodes=progress["cursor"],
                                              budget=680, method=method, mother=mother))
                wal = directory / "confirmation_wal.pt"
                if wal.exists():
                    previous = torch.load(wal, map_location="cpu", weights_only=False)["identity"].get("context", {})
                    if previous.get("task_sha") == task_sha and previous.get("matrix_cursor", cursor) < cursor:
                        wal.unlink()
                evaluate_cases(agents[method], [row], group, loader=load_confirm,
                               seed_base=275000, repeats=1, mother_offset=mother, repeat_offset=repeat,
                               node=150000, logs=logs, phase="confirmation_" + method,
                               wal_path=wal, row_committed=commit,
                               wal_context=dict(task_sha=task_sha, matrix_cursor=cursor))
        return dict(stage="completed", completed_episodes=progress["cursor"], budget=680)
    except RecoveryAccountingRequired as exc:
        result = dict(stage="needs_root_recovery_accounting", task_sha=task_sha,
                      completed_episodes=progress["cursor"], preserved_original_files=True,
                      unknown_spend_not_zero=True, reason=str(exc))
        atomic_json(directory / "recovery_accounting_required.json", result)
        return result
    finally:
        logs.close()
