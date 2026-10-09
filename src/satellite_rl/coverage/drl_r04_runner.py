"""Workbench-only S1-D12A learning, demonstration, and evaluation jobs.

The runner owns persistence and sampling.  Geometry and reward live in
``drl_r04_core``; the network has no dependency on either module.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .drl_r04_config import DrlR04Config
from .drl_r04_core import (DrlR04Env, h1_teacher_label,
                           run_h1, run_h2, run_i0)
from .drl_r04_policy import DrlR04Policy, Transition, collate, ppo_update
from .runner import (_event, _jsonable, _jsonl, _jsonl_many, _load_torch, _now, _read, _restore_rng,
                     _rng_state, _save_torch, _seed, _sha256, _status,
                     _stop_requested, _write_json)
from .variable_dataset import load_variable_dataset


POLICY_FORMAT = "drl_r04_set_ppo_v1"
CHECKPOINT_FORMAT = "s1_d12a_checkpoint_v1"
MULTIPLIERS = (0.75, 1.0, 1.25)
_RESIDENT_STREAMS = None
_RESIDENT_DATASET = None
_RESIDENT_CORE = None


def _config_hash(config):
    """Canonical schema-6 hash; legacy config_hash validates only older schemas."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _learner_hash(config):
    learning_keys = ("seed", "num_envs", "rollout_steps", "minibatch_size", "epochs",
                     "lr", "gamma", "gae_lambda", "clip_ratio", "entropy_coef",
                     "value_coef", "max_grad_norm")
    return _config_hash({"policy_format": POLICY_FORMAT,
                         "observation_shape": {"beam": 12, "candidate": 16, "global": 14},
                         "network": config["network"],
                         "training": {key: config["training"][key] for key in learning_keys},
                         "core": config["coverage"]["drl_r04"]})


def _write_observation_shard(directory, step, observations):
    """Keep every dynamic network input once, compressed and content-addressed."""
    folder = directory / "observations"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"rollout_{step:06d}.jsonl.gz"
    if path.exists():
        orphan = path.with_name(path.name + ".interrupted." + _sha256(path)[:16])
        if orphan.exists():
            raise FileExistsError("D12-A interrupted observation shard already exists")
        path.replace(orphan)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temp.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                               compresslevel=3, mtime=0) as packed:
                for item in observations:
                    packed.write((json.dumps(_jsonable(item), ensure_ascii=False,
                                             allow_nan=False, separators=(",", ":")) + "\n").encode())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return {"file": str(path.relative_to(directory)),
            "sha256": _sha256(path), "rows": len(observations)}


def _dataset(config, role):
    record = config["datasets"][role]
    path = Path(record["path"]).resolve()
    if not path.is_file() or _sha256(path) != record["sha256"]:
        raise ValueError(f"D12-A dataset SHA mismatch: {role}")
    dataset = load_variable_dataset(path)
    if not dataset.scenes:
        raise ValueError(f"D12-A empty dataset: {role}")
    return dataset


def _scaled(scene, multiplier):
    return replace(scene, demands=np.asarray(scene.demands, dtype=np.float64) * multiplier,
                   metadata={**scene.metadata, "demand_multiplier": float(multiplier)})


def _core_config(config):
    values = config.get("coverage", {}).get("drl_r04", {})
    return DrlR04Config(**{key: value for key, value in values.items()
                           if key in DrlR04Config.__dataclass_fields__})


def _reachability_contract(job, config):
    frozen = job.get("reachability_contract")
    if not isinstance(frozen, dict):
        raise ValueError("D12-A formal PPO requires pretraining H1/H2 reachability proof")
    path = Path(frozen.get("path", ""))
    indices = frozen.get("reachable_mother_indices")
    if (not path.is_file() or _sha256(path) != frozen.get("sha256") or
            frozen.get("validation_dataset_sha256") != config["datasets"]["validation"]["sha256"] or
            not isinstance(indices, list) or not indices or
            len(indices) != len(set(indices)) or
            any(type(index) is not int or not 0 <= index < 20 for index in indices)):
        raise ValueError("D12-A frozen reachability proof differs")
    source = _read(path, {})
    if (source.get("reachable_mother_indices") != indices or
            source.get("validation_dataset_sha256") != frozen["validation_dataset_sha256"]):
        raise ValueError("D12-A reachability content differs from frozen job")
    return indices


def _validate_training(config, job, preflight):
    settings = config["training"]
    formal = job.get("experiment_role", "formal") == "formal"
    total = 262144 if formal else 2048
    interval = 16384 if formal else 2048
    expected = {"seed": 51001, "num_envs": 32, "rollout_steps": 64,
                "gamma": 1.0, "gae_lambda": .95, "lr": .0003,
                "epochs": 4, "minibatch_size": 64,
                "entropy_coef": .01, "clip_ratio": .2,
                "value_coef": .5, "max_grad_norm": .5,
                "checkpoint_interval": 2048, "validation_interval": interval,
                "pilot_step": 65536, "cpu_workers": 16,
                "torch_threads": 8}
    for name, value in expected.items():
        if settings.get(name) != value:
            raise ValueError(f"D12-A frozen training {name} differs")
    if settings.get("total_steps") != total:
        raise ValueError("D12-A decision budget differs")
    if settings.get("device") != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D12-A formal learning requires DQN/CUDA")
    if preflight is None:
        raise RuntimeError("D12-A formal job requires workbench CUDA preflight")
    if settings["total_steps"] % 2048 or settings["validation_interval"] % 2048:
        raise ValueError("D12-A checkpoint boundary differs")
    return settings


def _action_digest(env):
    menu = env.menu
    records = [item[0].to_record() for item in menu]
    return hashlib.sha256(json.dumps(records, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _worker_reset(packet):
    scene, core = packet
    env = DrlR04Env(scene, core)
    obs, info = env.reset()
    return env, obs, info


def _worker_step(packet):
    env, choice = packet
    menu_hash = _action_digest(env)
    requested = env.menu[choice][0].to_record()
    before_current, before_best = float(env.current_j), float(env.best_j)
    began = time.monotonic()
    observation, reward, terminated, truncated, info = env.step(choice)
    elapsed = time.monotonic() - began
    physical = _terminal_physics(env) if terminated or truncated else None
    return env, observation, float(reward), bool(terminated), bool(truncated), info, {
        "candidate_menu_sha256": menu_hash, "requested_action": requested,
        "current_j_before": before_current, "best_j_before": before_best,
        "current_j_after": float(env.current_j), "best_j_after": float(env.best_j),
        "execution_seconds": elapsed, "terminal_physics": physical}


def _terminal_physics(env):
    """Measure the raw current and published best terminal layouts exactly once each."""
    from .drl_r04_observation import layout_sha256
    from .variable_downstream import evaluate_layout

    if env.initial_j is None:
        return {"status": "initialization_failed", "current": None, "best": None,
                "physical_allocation_calls": 0}
    layouts = {"initial": env.initial_layout, "current": env.current_layout,
               "best": env.best_layout}
    cache = {}
    aliases = {}
    for name, layout in layouts.items():
        digest = layout_sha256(layout)
        if digest not in cache:
            measured = evaluate_layout(env.scene, layout, {}, route="R04")
            cache[digest] = {"layout_sha256": digest, "beams": measured["beams"],
                             "metrics": measured["metrics"],
                             "timing": measured["timing"], "calls": measured["calls"],
                             "physics_version": measured["physics_version"],
                             "aircraft_service": {key: value for key, value in
                                                  measured["aircraft_service"].items()
                                                  if key != "physical_scene"}}
        aliases[name] = digest
    return {"status": "evaluated_all_actual_beams_and_users",
            "layouts_by_sha256": cache,
            "physical_allocation_calls": len(cache),
            "layout_aliases": aliases}


def _worker_teacher(packet):
    scene, core, remaining_labels, remaining_trials = packet
    env, observation, reset_info = _worker_reset((scene, core))
    labels = []
    began = time.monotonic()
    while not env.done and len(labels) < remaining_labels and env.trials < remaining_trials:
        menu_sha = _action_digest(env)
        before_trials, before_geometry = env.trials, env.geometry_calls
        choice = h1_teacher_label(env)
        previous_observation = observation
        selected_action = env.menu[choice][0].to_record()
        teacher_trials = env.trials - before_trials
        teacher_geometry = env.geometry_calls - before_geometry
        observation, reward, terminated, truncated, step_info = env.step(choice)
        labels.append({"observation": previous_observation, "label": choice,
                      "menu_sha256": menu_sha,
                      "teacher_trials": teacher_trials,
                      "teacher_geometry_calls": teacher_geometry,
                      "selected_action": selected_action,
                      "actual_action": selected_action if step_info["accepted"] else None,
                      "step_info": step_info, "reward": reward,
                      "terminated": terminated, "truncated": truncated})
    return labels, {"result": env.result(), "reset_info": reset_info,
                    "elapsed_seconds": time.monotonic() - began,
                    "complete": bool(env.done)}


def _resident_init(streams, dataset_record, core):
    """One process owns fixed stream IDs for its entire rollout and resume."""
    global _RESIDENT_STREAMS, _RESIDENT_DATASET, _RESIDENT_CORE
    torch.set_num_threads(1)
    _RESIDENT_STREAMS = {stream_id: stream for stream_id, stream in streams}
    _RESIDENT_DATASET = _dataset({"datasets": {"train": dataset_record}}, "train")
    _RESIDENT_CORE = core


def _resident_step(packet):
    """Only compact choices/results cross IPC; live environments stay in worker."""
    global_step, choices = packet
    results = []
    for stream_id, choice in choices:
        stream = _RESIDENT_STREAMS[stream_id]
        env, observation, reward, terminated, truncated, info, audit = _worker_step(
            (stream["env"], choice))
        physical = audit.pop("terminal_physics")
        stream["env"], stream["observation"] = env, observation
        stream["episode_return"] += reward
        stream["action_count"] += 1
        scene = _scene_for(stream, _RESIDENT_DATASET)
        event = {"env_step": global_step + stream_id + 1,
                 "stream_id": stream_id,
                 "stream_episode_serial": stream["episode_serial"],
                 **_scene_reference(scene, MULTIPLIERS[stream["level"]]),
                 "candidate_index": int(choice),
                 "actual_action": info.get("action") if info.get("accepted") else None,
                 "reward": reward, "terminated": terminated, "truncated": truncated,
                 "info": info, **audit}
        terminal = None
        if terminated or truncated:
            terminal = _terminal(stream, env_step=global_step + stream_id + 1,
                                 stream_id=stream_id,
                                 reason=info.get("reason", "terminated" if terminated else "budget"),
                                 complete=True)
            terminal["terminal_physics"] = physical
            stream["scene_index"] = int(stream["sampler"].integers(len(_RESIDENT_DATASET.scenes)))
            stream["level"] = int(stream["sampler"].integers(len(MULTIPLIERS)))
            stream["episode_serial"] += 1
            fresh_env, fresh_obs, reset_info = _worker_reset((
                _scene_for(stream, _RESIDENT_DATASET), _RESIDENT_CORE))
            if fresh_env.done:
                raise ValueError(f"D12-A resident initializer failed: {reset_info}")
            stream.update(env=fresh_env, observation=fresh_obs, reset_info=reset_info,
                          episode_return=0.0, action_count=0,
                          episode_start_step=global_step + stream_id + 33)
            observation = fresh_obs
        results.append((stream_id, observation, reward, terminated or truncated,
                        event, terminal))
    return results


def _resident_state(_):
    return list(_RESIDENT_STREAMS.items())


def _resident_tail(last_step):
    rows = []
    for stream_id, stream in _RESIDENT_STREAMS.items():
        if stream["action_count"] == 0:
            env = stream["env"]
            row = env.result()
            row.pop("ledger", None)
            row.update(_scene_reference(env.scene, MULTIPLIERS[stream["level"]]))
            row.update({"stream_id": stream_id,
                        "stream_episode_serial": stream["episode_serial"],
                        "purpose": "budget_tail_snapshot", "episode": None,
                        "completed_episode": False,
                        "stop_reason": "no_action_since_reset",
                        "terminal_env_step": None,
                        "action_trace": {"file": "actions.jsonl", "stream_id": stream_id,
                                         "stream_episode_serial": stream["episode_serial"],
                                         "first_env_step": None, "last_env_step": None,
                                         "count": 0},
                        "terminal_physics": {
                            "status": "not_run_no_action_since_reset",
                            "missing_reason": "no_action_since_reset",
                            "physical_allocation_calls": 0}})
            rows.append(row)
            continue
        last_action_step = last_step - 32 + stream_id + 1
        row = _terminal(stream, env_step=last_action_step, stream_id=stream_id,
                        reason="budget_truncated", complete=False)
        row["terminal_physics"] = _terminal_physics(stream["env"])
        row["episode"] = None
        row["purpose"] = "budget_tail_snapshot"
        rows.append(row)
    return rows


def _scene_for(stream, dataset):
    return _scaled(dataset.scenes[stream["scene_index"]], MULTIPLIERS[stream["level"]])


def _fresh_stream(dataset, core, seed):
    sampler = np.random.default_rng(seed)
    index = int(sampler.integers(len(dataset.scenes)))
    level = int(sampler.integers(len(MULTIPLIERS)))
    scene = _scaled(dataset.scenes[index], MULTIPLIERS[level])
    env, obs, info = _worker_reset((scene, core))
    if env.done:
        raise ValueError(f"D12-A training initializer failed: {scene.scene_id}: {info}")
    return {"sampler": sampler, "scene_index": index, "level": level,
            "env": env, "observation": obs, "episode_serial": 0,
            "episode_return": 0.0, "episode_start_step": 1,
            "action_count": 0, "reset_info": info}


def _scene_reference(scene, multiplier):
    return {"scene_id": scene.scene_id, "scene_sha256": scene.source_hash,
            "demand_multiplier": multiplier}


def _terminal(stream, *, env_step, stream_id, reason, complete):
    env = stream["env"]
    record = env.result()
    # The complete action ledger is in actions.jsonl; keep a precise pointer
    # instead of repeating every step in each terminal row.
    record.pop("ledger", None)
    record.update(_scene_reference(env.scene, MULTIPLIERS[stream["level"]]))
    record.update({"terminal_env_step": env_step, "stream_id": stream_id,
                   "stream_episode_serial": stream["episode_serial"],
                   "episode_return": stream["episode_return"],
                   "stop_reason": reason, "completed_episode": complete,
                   "action_trace": {"file": "actions.jsonl", "stream_id": stream_id,
                                    "stream_episode_serial": stream["episode_serial"],
                                    "first_env_step": stream["episode_start_step"],
                                    "last_env_step": env_step,
                                    "count": stream["action_count"]}})
    return record


def _save_checkpoint(directory, job, config, policy, optimizer, *, streams,
                     env_step, update_step, episode, last_validation, ledgers):
    path = directory / "checkpoint.pt"
    payload = {"format": CHECKPOINT_FORMAT, "run_id": job["run_id"],
               "source_commit": job["git_commit"], "config_hash": _config_hash(config),
               "learner_hyperparameter_hash": _learner_hash(config),
               "policy_format": POLICY_FORMAT, "policy": policy.state_dict(),
               "optimizer": optimizer.state_dict(), "rng": _rng_state(),
               "cpu_workers": config["training"].get("cpu_workers"),
               "cpu_worker_backend": "spawn_resident",
               "streams": streams, "env_step": env_step, "update_step": update_step,
               "episode": episode, "last_validation": last_validation,
               "ledger_counts": dict(ledgers)}
    _save_torch(path, payload)
    return _sha256(path)


def _load_checkpoint(directory, job, config, policy, optimizer):
    payload = _load_torch(directory / "checkpoint.pt", "cpu")
    for key, value in (("format", CHECKPOINT_FORMAT), ("run_id", job["run_id"]),
                       ("source_commit", job["git_commit"]),
                       ("config_hash", _config_hash(config)),
                       ("learner_hyperparameter_hash", _learner_hash(config)),
                       ("policy_format", POLICY_FORMAT),
                       ("cpu_workers", config.get("training", {}).get("cpu_workers"))):
        if payload.get(key) != value:
            raise ValueError(f"D12-A checkpoint {key} differs")
    policy.load_state_dict(payload["policy"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    _restore_rng(payload["rng"])
    for name, count in payload["ledger_counts"].items():
        path = directory / name
        rows = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        if len(rows) < count:
            raise ValueError(f"D12-A ledger shorter than checkpoint: {name}")
        if len(rows) > count:
            path.write_text("\n".join(rows[:count]) + "\n", encoding="utf-8")
    updates_path = directory / "updates.jsonl"
    for line in (updates_path.read_text(encoding="utf-8").splitlines()
                 if updates_path.exists() else []):
        shard = json.loads(line).get("observation_shard", {})
        path = (directory / shard.get("file", "")).resolve()
        if (not path.is_relative_to(directory.resolve()) or not path.is_file() or
                _sha256(path) != shard.get("sha256") or shard.get("rows") != 2048):
            raise ValueError("D12-A committed observation shard differs")
    return payload


def _model_metadata(path, job, config, *, step, selection, preflight):
    return {"policy_format": POLICY_FORMAT, "research_stage": "coverage",
            "run_id": job["run_id"], "route": "D12A", "network": config["network"],
            "train_mode": job.get("train_mode"),
            "bc_source": job.get("bc_source"),
            "training_seed": 51001, "env_step": step, "selection": selection,
            "cpu_workers": config["training"].get("cpu_workers"),
            "cpu_worker_backend": "spawn_resident" if selection != "bc_endpoint" else None,
            "source_commit": job["git_commit"], "config_hash": _config_hash(config),
            "learner_hyperparameter_hash": _learner_hash(config),
            "policy_sha256": _sha256(path), "gpu": (preflight or {}).get("gpu"),
            "torch_version": torch.__version__, "numpy_version": np.__version__,
            "created_at": _now()}


def _save_model(directory, job, config, policy, *, step, selection, preflight):
    folder = {"fixed_budget_endpoint": "final"}.get(selection, selection)
    target = Path(job["model_dir"]) / folder
    target.mkdir(parents=True, exist_ok=True)
    _save_torch(target / "policy.pt", policy.state_dict())
    metadata = _model_metadata(target / "policy.pt", job, config, step=step,
                               selection=selection, preflight=preflight)
    metadata["parameter_count"] = sum(p.numel() for p in policy.parameters())
    _write_json(target / "model.json", metadata)
    _write_json(directory / "artifacts.json", {
        "checkpoint": str(directory / "checkpoint.pt"),
        "final_policy": str(target), "model": str(target / "policy.pt"),
        "model_metadata": str(target / "model.json"),
        "actions": str(directory / "actions.jsonl"),
        "episodes": str(directory / "episodes.jsonl"),
        "budget_tails": str(directory / "budget_tails.jsonl"),
        "updates": str(directory / "updates.jsonl"),
        "evaluations": str(directory / "evaluations.jsonl")})
    return target


def _load_model(job, config, device):
    source = Path(job.get("model_path", job["model_dir"]))
    metadata = _read(source / "model.json", {})
    if (metadata.get("policy_format") != POLICY_FORMAT or
            metadata.get("route") != "D12A" or
            metadata.get("network") != config["network"] or
            metadata.get("learner_hyperparameter_hash") != _learner_hash(config) or
            metadata.get("policy_sha256") != _sha256(source / "policy.pt")):
        raise ValueError("D12-A model provenance differs; old R08 is prohibited")
    policy = DrlR04Policy().to(device)
    policy.load_state_dict(_load_torch(source / "policy.pt", device), strict=True)
    policy.eval()
    return policy, metadata


def _initialize_p1_from_bc(job, config, policy):
    teacher = Path(job["bc_model_dir"])
    metadata = _read(teacher / "model.json", {})
    if (metadata.get("policy_format") != POLICY_FORMAT or
            metadata.get("route") != "D12A" or
            metadata.get("network") != config["network"] or
            metadata.get("learner_hyperparameter_hash") != _learner_hash(config) or
            metadata.get("selection") != "bc_endpoint" or
            metadata.get("rl_env_step") != 0 or
            metadata.get("policy_sha256") != _sha256(teacher / "policy.pt")):
        raise ValueError("P1 requires frozen D12-A BC endpoint")
    policy.load_state_dict(_load_torch(teacher / "policy.pt",
                                       next(policy.parameters()).device), strict=True)
    return metadata


def _evaluate_policy(policy, dataset, core, *, device, step, output,
                     indices=None, baseline=None, completed=0, on_row=None):
    rows = []
    indices = list(range(len(dataset.scenes))) if indices is None else list(indices)
    if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(dataset.scenes)
                                                for i in indices):
        raise ValueError("D12-A scene indices differ")
    ordinal = 0
    for index in indices:
        scene = dataset.scenes[index]
        for multiplier in MULTIPLIERS:
            if ordinal < completed:
                ordinal += 1
                continue
            scaled = _scaled(scene, multiplier)
            env = DrlR04Env(scaled, core)
            began = time.monotonic()
            if baseline is not None:
                {"I0": run_i0, "H1": run_h1, "H2": run_h2}[baseline](env)
                reset_info = None
            else:
                observation, reset_info = env.reset()
                while not env.done:
                    batch = collate([observation], device)
                    with torch.no_grad():
                        choice = int(policy.act(batch, deterministic=True)[0].item())
                    observation, reward, terminated, truncated, info = env.step(choice)
                    if terminated or truncated:
                        break
            algorithm_seconds = time.monotonic() - began
            row = env.result()
            row["terminal_physics"] = _terminal_physics(env)
            row.update(_scene_reference(scaled, multiplier))
            row.update({"dataset_index": index, "checkpoint_step": step,
                        "reset_info": reset_info,
                        "algorithm_seconds": algorithm_seconds,
                        "total_seconds": algorithm_seconds + sum(
                            item["timing"]["total_seconds"] for item in
                            row["terminal_physics"].get("layouts_by_sha256", {}).values())})
            _jsonl(output, row)
            rows.append(row)
            ordinal += 1
            if on_row is not None and not on_row(ordinal):
                return rows
    return rows


def _teacher(directory, job, config, resume, preflight):
    if preflight is None:
        raise RuntimeError("D12-A teacher requires workbench preflight")
    if job.get("preparation_mode") != "teacher":
        raise ValueError("D12-A prepare mode must be teacher")
    dataset = _dataset(config, "train")
    if len(dataset.scenes) != 576:
        raise ValueError("D12-A teacher requires 576 train mother scenes")
    core = _core_config(config)
    target = directory / "demonstrations"
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "teacher_manifest.json"
    if manifest_path.exists():
        manifest = _read(manifest_path, {})
        if (not resume or manifest.get("run_id") != job["run_id"] or
                manifest.get("config_hash") != _config_hash(config)):
            raise ValueError("D12-A teacher manifest differs")
    else:
        if resume or any(target.iterdir()):
            raise ValueError("D12-A teacher has incomplete provenance")
        manifest = {"format": "d12a_teacher_v1", "run_id": job["run_id"],
                    "source_commit": job["git_commit"],
                    "config_hash": _config_hash(config),
                    "dataset_sha256": config["datasets"]["train"]["sha256"],
                    "sampling": "all_train_mothers_in_input_order_then_multipliers_0.75_1_1.25",
                    "sampling_seed": None,
                    "max_labels": 20000, "max_trials": 442368,
                    "next_scene": 0, "labels": 0, "trials": 0,
                    "geometry_calls": 0, "unverified_trials": 0,
                    "label_operations": {"stop": 0, "split": 0, "merge": 0},
                    "interrupted_batches": [], "inflight": [], "scenes": []}
        _write_json(manifest_path, manifest)
    if manifest.get("inflight"):
        # A killed process may have spent every reserved trial.  Charge the
        # maximum conservatively; never reuse unknown physical work for free.
        abandoned = manifest.pop("inflight")
        manifest["unverified_trials"] = manifest.get("unverified_trials", 0) + sum(
            item["reserved_trials"] for item in abandoned)
        manifest.setdefault("interrupted_batches", []).append({
            "scenes": abandoned, "reason": "uncommitted_worker_result",
            "recovered_at": _now()})
        _write_json(manifest_path, manifest)
    with ProcessPoolExecutor(max_workers=8,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        while manifest["next_scene"] < len(dataset.scenes) * 3:
            remaining_labels = 20000 - manifest["labels"]
            remaining_trials = 442368 - manifest["trials"] - manifest.get("unverified_trials", 0)
            if remaining_labels < core.max_commits or remaining_trials < core.max_trials:
                break
            full_parallel = remaining_labels >= 8 * core.max_commits and remaining_trials >= 8 * core.max_trials
            width = 8 if full_parallel else 1
            width = min(width, len(dataset.scenes) * 3 - manifest["next_scene"])
            trial_reservation = core.max_trials
            label_reservation = core.max_commits
            entries, packets = [], []
            for ordinal in range(manifest["next_scene"], manifest["next_scene"] + width):
                scene_index, level = divmod(ordinal, 3)
                scene = _scaled(dataset.scenes[scene_index], MULTIPLIERS[level])
                entries.append({"ordinal": ordinal, "reserved_trials": trial_reservation,
                                "reserved_labels": label_reservation})
                packets.append((scene, core, label_reservation, trial_reservation))
            manifest["inflight"] = entries
            _write_json(manifest_path, manifest)
            for entry, packet, (labels, audit) in zip(entries, packets,
                                                      pool.map(_worker_teacher, packets)):
                ordinal = entry["ordinal"]
                scene_index, level = divmod(ordinal, 3)
                scene = packet[0]
                file = target / f"scene_{ordinal:04d}.pt"
                if file.exists():
                    orphan = file.with_name(file.stem + ".interrupted." +
                                            _sha256(file)[:16] + file.suffix)
                    if orphan.exists():
                        raise FileExistsError("D12-A teacher orphan evidence already exists")
                    file.replace(orphan)
                _save_torch(file, labels)
                result = audit["result"]
                if len(labels) > entry["reserved_labels"] or result["trials"] > entry["reserved_trials"]:
                    raise ValueError("D12-A teacher worker exceeded reserved budget")
                row = {"ordinal": ordinal, "scene_index": scene_index,
                       "level": level, **_scene_reference(scene, MULTIPLIERS[level]),
                       "path": str(file), "sha256": _sha256(file),
                       "labels": len(labels), "teacher_trials": result["trials"],
                       "geometry_calls": result["geometry_calls"],
                       "matrix_calls": result["matrix_calls"],
                       "ray_pair_checks": result["ray_pair_checks"],
                       "complete": audit["complete"],
                       "elapsed_seconds": audit["elapsed_seconds"],
                       "initial_sha256": result["initial_sha256"],
                       "initial_j": result["initial_j"],
                       "best_j": result["best_j"]}
                manifest["scenes"].append(row)
                manifest["labels"] += len(labels)
                for item in labels:
                    operation = item["selected_action"]["operation"]
                    manifest["label_operations"][operation] += 1
                manifest["trials"] += result["trials"]
                manifest["geometry_calls"] += result["geometry_calls"]
                manifest["next_scene"] = ordinal + 1
                manifest["inflight"] = [item for item in manifest["inflight"]
                                        if item["ordinal"] != ordinal]
                _write_json(manifest_path, manifest)
                _status(directory, job, stage="preparing", scene_count=ordinal + 1,
                        labels=manifest["labels"], candidate_trials=manifest["trials"],
                        unverified_candidate_trial_reservation=manifest["unverified_trials"])
            if _stop_requested(directory):
                _status(directory, job, stage="stopped", stop_reason="requested",
                        labels=manifest["labels"], candidate_trials=manifest["trials"],
                        unverified_candidate_trial_reservation=manifest["unverified_trials"])
                return
    manifest["completed"] = True
    _write_json(manifest_path, manifest)
    _write_json(directory / "artifacts.json", {
        "teacher_manifest": str(manifest_path), "demonstrations": str(target)})
    _status(directory, job, stage="completed", labels=manifest["labels"],
            candidate_trials=manifest["trials"],
            unverified_candidate_trial_reservation=manifest["unverified_trials"],
            scene_count=manifest["next_scene"])


def _bc(directory, job, config, resume, preflight):
    if preflight is None or config["training"]["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D12-A BC requires workbench DQN/CUDA preflight")
    for key, value in (("bc_epochs", 20), ("bc_batch_size", 256),
                       ("bc_max_updates", 1600), ("seed", 51001)):
        if config["training"].get(key) != value:
            raise ValueError(f"D12-A BC frozen {key} differs")
    torch.set_num_threads(config["training"]["torch_threads"])
    demonstrations_dir = Path(job["demonstrations_dir"])
    teacher_source = job.get("teacher_source", {})
    manifest_path = Path(teacher_source.get("manifest_path",
                                            demonstrations_dir.parent / "teacher_manifest.json"))
    manifest = _read(manifest_path, {})
    if (manifest.get("format") != "d12a_teacher_v1" or
            not manifest.get("completed") or
            manifest.get("dataset_sha256") != config["datasets"]["train"]["sha256"] or
            manifest.get("labels", 0) > 20000 or
            sum(manifest.get("label_operations", {}).values()) != manifest.get("labels") or
            manifest.get("trials", 0) + manifest.get("unverified_trials", 0) > 442368):
        raise ValueError("D12-A BC requires completed matching teacher artifact")
    demonstrations = []
    for row in manifest["scenes"]:
        file = Path(row["path"])
        if not file.resolve().is_relative_to(demonstrations_dir.resolve()):
            raise ValueError("D12-A teacher shard escaped demonstration directory")
        if _sha256(file) != row["sha256"]:
            raise ValueError("D12-A demonstration shard SHA differs")
        demonstrations.extend(_load_torch(file, "cpu"))
    if len(demonstrations) != manifest["labels"] or not demonstrations:
        raise ValueError("D12-A teacher label count differs")
    _seed(51001)
    policy = DrlR04Policy().to("cuda")
    optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
    step, epoch = 0, 0
    checkpoint = directory / "bc_checkpoint.pt"
    if resume:
        saved = _load_torch(checkpoint, "cpu")
        if (saved.get("format") != "d12a_bc_v1" or
                saved.get("run_id") != job["run_id"] or
                saved.get("teacher_manifest_sha256") != _sha256(manifest_path)):
            raise ValueError("D12-A BC checkpoint provenance differs")
        policy.load_state_dict(saved["policy"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        _restore_rng(saved["rng"])
        step, epoch = saved["update_step"], saved["epoch"]
        updates_path = directory / "bc_updates.jsonl"
        lines = updates_path.read_text(encoding="utf-8").splitlines() if updates_path.exists() else []
        if len(lines) < epoch:
            raise ValueError("D12-A BC update ledger shorter than checkpoint")
        if len(lines) > epoch:
            updates_path.write_text("\n".join(lines[:epoch]) + "\n", encoding="utf-8")
    elif checkpoint.exists() or (directory / "bc_updates.jsonl").exists():
        raise FileExistsError("D12-A BC requires same-ID resume")
    epoch_index = epoch - 1
    for epoch_index in range(epoch, 20):
        permutation = torch.randperm(len(demonstrations)).tolist()
        losses = []
        for offset in range(0, len(permutation), 256):
            if step >= 1600:
                break
            rows = [demonstrations[i] for i in permutation[offset:offset + 256]]
            batch = collate([row["observation"] for row in rows], "cuda")
            labels = torch.as_tensor([row["label"] for row in rows],
                                     dtype=torch.int64, device="cuda")
            logits, _ = policy(batch)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
            step += 1
            losses.append(float(loss.detach()))
        _jsonl(directory / "bc_updates.jsonl", {"epoch": epoch_index + 1,
                "update_step": step, "labels": len(demonstrations),
                "mean_cross_entropy": float(np.mean(losses)) if losses else None})
        _save_torch(checkpoint, {"format": "d12a_bc_v1", "run_id": job["run_id"],
                    "teacher_manifest_sha256": _sha256(manifest_path),
                    "policy": policy.state_dict(), "optimizer": optimizer.state_dict(),
                    "rng": _rng_state(), "update_step": step, "epoch": epoch_index + 1})
        _status(directory, job, stage="training", update_step=step,
                epoch=epoch_index + 1, label_count=len(demonstrations))
        if _stop_requested(directory):
            _status(directory, job, stage="stopped", stop_reason="requested",
                    update_step=step, epoch=epoch_index + 1)
            return
        if step >= 1600:
            break
    final = _save_model(directory, job, config, policy, step=0,
                        selection="bc_endpoint", preflight=preflight)
    metadata_file = final / "model.json"
    metadata = _read(metadata_file, {})
    metadata.update({"teacher_manifest_sha256": _sha256(manifest_path),
                     "teacher_run_id": manifest["run_id"], "bc_updates": step,
                     "bc_epochs": epoch_index + 1, "bc_labels": len(demonstrations),
                     "rl_env_step": 0})
    _write_json(metadata_file, metadata)
    _status(directory, job, stage="completed", update_step=step,
            epoch=epoch_index + 1, final_policy=str(final))


def _validate_if_due(directory, job, config, policy, core, dataset, *, step,
                     last_validation, ledgers):
    if step == 0 or step % config["training"]["validation_interval"] or step <= last_validation:
        return last_validation
    # Deterministic argmax evaluation does not consume the training RNG.
    output = directory / "validation" / f"step_{step:06d}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        interrupted = output.with_name(output.stem + ".interrupted." +
                                       _sha256(output)[:16] + output.suffix)
        if interrupted.exists():
            raise FileExistsError("D12-A interrupted validation evidence already exists")
        output.replace(interrupted)
        _event(directory, job, "d12a_validation_restarted_after_uncommitted_node",
               env_step=step, previous_sha256=_sha256(interrupted),
               previous_path=str(interrupted))
    rows = _evaluate_policy(policy, dataset, core, device="cuda", step=step, output=output)
    if len(rows) != 60:
        raise ValueError("D12-A validation node must contain 20 mothers × three levels")
    def refreshed(row):
        return any(item["action"]["operation"] != "stop" and
                   item["best_j"] < row["initial_j"] - 1e-10
                   for item in row.get("ledger", []))

    by_mother = [rows[start:start + 3] for start in range(0, 60, 3)]
    reachable = (_reachability_contract(job, config)
                 if job.get("experiment_role", "formal") == "formal" else [])
    improved_mothers = sum(
        all(row["initial_j"] is not None and row["best_j"] is not None and
            np.isfinite(row["initial_j"]) and np.isfinite(row["best_j"])
            for row in group) and
        float(np.mean([row["initial_j"] - row["best_j"] for row in group])) > 1e-8 and
        any(refreshed(row) for row in group)
        for group in by_mother)
    feasible_rows = sum(
        bool(row.get("initialization", {}).get("metrics", {}).get("coverage_100")) and
        bool(row.get("initialization", {}).get("metrics", {}).get("demand_conserved")) and
        row["best_j"] is not None and np.isfinite(row["best_j"]) and
        row["geometry_calls"] <= core.max_geometry_calls and
        row["trials"] <= core.max_trials and row["commits"] <= core.max_commits
        for row in rows)
    summary = {"env_step": step, "rows": len(rows), "path": str(output),
               "sha256": _sha256(output),
               "mean_best_j": (float(np.mean([row["best_j"] for row in rows
                                               if row["best_j"] is not None]))
                               if any(row["best_j"] is not None for row in rows) else None),
               "first_step_stop_count": sum(bool(row.get("ledger") and
                                            row["ledger"][0]["action"]["operation"] == "stop")
                                            for row in rows),
               "best_j_improved_count": sum(row["best_j"] is not None and
                                            row["best_j"] < row["initial_j"] - 1e-10
                                            for row in rows),
               "three_level_improved_mother_count": int(improved_mothers),
               "reachable_mother_indices": reachable,
               "reachable_mother_count": len(reachable),
               "reachable_subset_first_stop_zero_progress": bool(reachable) and all(
                    row.get("ledger") and row["ledger"][0]["action"]["operation"] == "stop" and
                    row["best_j"] is not None and row["initial_j"] is not None and
                    abs(row["best_j"] - row["initial_j"]) <= 1e-10
                    for index in reachable for row in by_mother[index]),
               "nonstop_best_refresh_rows": sum(refreshed(row) for row in rows),
               "feasible_row_count": feasible_rows,
               "all_60_first_step_stop_zero_progress": all(
                    row.get("ledger") and row["ledger"][0]["action"]["operation"] == "stop" and
                    row["best_j"] is not None and row["initial_j"] is not None and
                    abs(row["best_j"] - row["initial_j"]) <= 1e-10
                    for row in rows)}
    _jsonl(directory / "evaluations.jsonl", summary)
    ledgers["evaluations.jsonl"] += 1
    _status(directory, job, stage="training", env_step=step,
            last_validation_step=step, latest_validation=summary)
    return step


def _scientific_stop_if_due(directory, job, config, policy, *, step, preflight,
                            checkpoint_sha256):
    if job.get("experiment_role", "formal") != "formal":
        return False
    _reachability_contract(job, config)
    path = directory / "evaluations.jsonl"
    if not path.is_file():
        return False
    nodes = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(nodes) < 2 or [row["env_step"] for row in nodes[-2:]] != [
            step - config["training"]["validation_interval"], step]:
        return False
    for row in nodes[-2:]:
        result_path = Path(row["path"])
        if (row.get("rows") != 60 or not result_path.is_file() or
                _sha256(result_path) != row.get("sha256")):
            raise ValueError("D12-A scientific-stop validation evidence differs")
    if not all(row.get("all_60_first_step_stop_zero_progress") and
               row.get("reachable_subset_first_stop_zero_progress") and
               row.get("best_j_improved_count") == 0 and
               row.get("feasible_row_count") == 60 for row in nodes[-2:]):
        return False
    selection = ("pilot_endpoint" if step == config["training"]["pilot_step"]
                 else "scientific_stop_endpoint")
    endpoint = _save_model(directory, job, config, policy, step=step,
                           selection=selection, preflight=preflight)
    _status(directory, job, stage="stopped", env_step=step,
            stop_reason="consecutive_validation_all_stop_zero_progress",
            scientific_stop=True,
            checkpoint_sha256=checkpoint_sha256, final_policy=str(endpoint),
            scientific_stop_validation_nodes=[row["env_step"] for row in nodes[-2:]])
    _event(directory, job, "d12a_scientific_stop", env_step=step,
           endpoint=str(endpoint))
    return True


def _train_resident(directory, job, config, preflight, settings, core,
                    validation_set, policy, optimizer, streams, step,
                    updates, episode, last_validation, ledgers):
    count, workers = 32, settings["cpu_workers"]
    observations = [stream["observation"] for stream in streams]
    with ExitStack() as stack:
        pools = [stack.enter_context(ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn"),
            initializer=_resident_init,
            initargs=([(i, streams[i]) for i in range(worker, count, workers)],
                      config["datasets"]["train"], core)))
                 for worker in range(workers)]

        def snapshot():
            returned = [pool.submit(_resident_state, None) for pool in pools]
            ordered = [None] * count
            for task in returned:
                for stream_id, stream in task.result():
                    ordered[stream_id] = stream
            if any(stream is None for stream in ordered):
                raise ValueError("D12-A resident checkpoint lacks a stream")
            return ordered

        if step and step % settings["validation_interval"] == 0 and step > last_validation:
            # Resume after a worker/host interruption during an uncommitted
            # validation node, before taking any further training action.
            last_validation = _validate_if_due(directory, job, config, policy, core,
                                               validation_set, step=step,
                                               last_validation=last_validation,
                                               ledgers=ledgers)
            sha = _save_checkpoint(directory, job, config, policy, optimizer,
                             streams=streams, env_step=step, update_step=updates,
                             episode=episode, last_validation=last_validation,
                             ledgers=ledgers)
            if _scientific_stop_if_due(directory, job, config, policy, step=step,
                                       preflight=preflight, checkpoint_sha256=sha):
                return
            if step == settings["pilot_step"] and job.get("experiment_role", "formal") == "formal":
                pilot = _save_model(directory, job, config, policy, step=step,
                                    selection="pilot_endpoint", preflight=preflight)
                _status(directory, job, stage="pilot_pending", env_step=step,
                        stop_reason="pilot_gate", checkpoint_sha256=sha,
                        pilot_policy=str(pilot))
                return

        while step < settings["total_steps"]:
            transitions, action_rows, episode_rows = [], [], []
            began = time.monotonic()
            for _ in range(64):
                with torch.no_grad():
                    choice, logp, value = policy.act(collate(observations, "cuda"))
                indices, logps, values = choice.tolist(), logp.tolist(), value.tolist()
                offset = step + len(transitions)
                tasks = [pool.submit(_resident_step, (offset,
                          [(i, indices[i]) for i in range(worker, count, workers)]))
                         for worker, pool in enumerate(pools)]
                results = sorted((result for task in tasks for result in task.result()),
                                 key=lambda row: row[0])
                if len(results) != count:
                    raise ValueError("D12-A resident workers returned incomplete step")
                next_observations = [None] * count
                for i, next_obs, reward, done, event, terminal in results:
                    transitions.append(Transition(observations[i], indices[i],
                                                  logps[i], values[i], reward, done))
                    event.update(candidate_mask=observations[i]["candidate_mask"].tolist(),
                                 choice_log_prob=logps[i], value=values[i])
                    action_rows.append(event)
                    next_observations[i] = next_obs
                    if terminal is not None:
                        episode += 1
                        terminal["episode"] = episode
                        episode_rows.append(terminal)
                observations = next_observations
            with torch.no_grad():
                bootstrap = policy(collate(observations, "cuda"))[1].cpu().numpy()
            summary = ppo_update(policy, optimizer, transitions, bootstrap, device="cuda",
                                 gamma=1.0, gae_lambda=.95, clip_ratio=.2,
                                 entropy_coef=.01, value_coef=.5, max_grad_norm=.5,
                                 epochs=4, minibatch_size=64, num_envs=count)
            step += len(transitions)
            updates += summary["minibatch_updates"]
            shard = _write_observation_shard(directory, step,
                                             [item.observation for item in transitions])
            for row_index, row in enumerate(action_rows):
                row["observation_ref"] = {**shard, "row": row_index}
            _jsonl_many(directory / "actions.jsonl", action_rows)
            _jsonl_many(directory / "episodes.jsonl", episode_rows)
            _jsonl(directory / "updates.jsonl", {"env_step": step,
                    "update_step": updates, "episode": episode,
                    "rollout_seconds": time.monotonic() - began,
                    "cpu_worker_backend": "spawn_resident", "cpu_workers": workers,
                    "observation_shard": shard,
                    **summary})
            ledgers["actions.jsonl"] += len(action_rows)
            ledgers["episodes.jsonl"] += len(episode_rows)
            ledgers["updates.jsonl"] += 1
            streams = snapshot()
            sha = _save_checkpoint(directory, job, config, policy, optimizer,
                                   streams=streams, env_step=step, update_step=updates,
                                   episode=episode, last_validation=last_validation,
                                   ledgers=ledgers)
            _status(directory, job, stage="training", env_step=step,
                    budget_env_steps=settings["total_steps"], update_step=updates,
                    episode=episode, checkpoint_sha256=sha,
                    cpu_worker_backend="spawn_resident", cpu_workers=workers,
                    effective_steps_per_second=len(transitions) /
                    max(time.monotonic() - began, 1e-9))
            if step % settings["validation_interval"] == 0:
                last_validation = _validate_if_due(directory, job, config, policy, core,
                                                   validation_set, step=step,
                                                   last_validation=last_validation,
                                                   ledgers=ledgers)
                sha = _save_checkpoint(directory, job, config, policy, optimizer,
                                       streams=streams, env_step=step, update_step=updates,
                                       episode=episode, last_validation=last_validation,
                                       ledgers=ledgers)
                if _scientific_stop_if_due(directory, job, config, policy, step=step,
                                           preflight=preflight, checkpoint_sha256=sha):
                    return
            if _stop_requested(directory):
                _status(directory, job, stage="stopped", env_step=step,
                        stop_reason="requested", checkpoint_sha256=sha)
                return
            if step == settings["pilot_step"] and job.get("experiment_role", "formal") == "formal":
                pilot = _save_model(directory, job, config, policy, step=step,
                                    selection="pilot_endpoint", preflight=preflight)
                _status(directory, job, stage="pilot_pending", env_step=step,
                        stop_reason="pilot_gate", checkpoint_sha256=sha,
                        pilot_policy=str(pilot))
                return
        if step == settings["total_steps"] and ledgers.get("budget_tails.jsonl", 0) == 0:
            tasks = [pool.submit(_resident_tail, step) for pool in pools]
            tails = sorted((row for task in tasks for row in task.result()),
                           key=lambda row: row["stream_id"])
            _jsonl_many(directory / "budget_tails.jsonl", tails)
            ledgers["budget_tails.jsonl"] = len(tails)
            streams = snapshot()
            _save_checkpoint(directory, job, config, policy, optimizer,
                             streams=streams, env_step=step, update_step=updates,
                             episode=episode, last_validation=last_validation,
                             ledgers=ledgers)
    if step != settings["total_steps"]:
        raise ValueError("D12-A training ended short of frozen budget")
    selection = ("fixed_budget_endpoint" if job.get("experiment_role", "formal") == "formal"
                 else "verification_endpoint")
    final = _save_model(directory, job, config, policy, step=step,
                        selection=selection, preflight=preflight)
    _status(directory, job, stage="completed", env_step=step,
            budget_env_steps=settings["total_steps"], update_step=updates,
            episode=episode, final_policy=str(final))
    _event(directory, job, "d12a_training_completed", env_step=step)


def _train(directory, job, config, resume, preflight):
    settings = _validate_training(config, job, preflight)
    torch.set_num_threads(settings["torch_threads"])
    core = _core_config(config)
    train_set, validation_set = _dataset(config, "train"), _dataset(config, "validation")
    if len(validation_set.scenes) != 20:
        raise ValueError("D12-A validation must have 20 mother scenes")
    if job.get("experiment_role", "formal") == "formal":
        _reachability_contract(job, config)
    _seed(settings["seed"])
    policy = DrlR04Policy().to("cuda")
    optimizer = torch.optim.Adam(policy.parameters(), lr=settings["lr"])
    ledger_names = ("actions.jsonl", "episodes.jsonl", "updates.jsonl",
                    "evaluations.jsonl", "budget_tails.jsonl")
    ledgers = {name: 0 for name in ledger_names}
    if resume:
        saved = _load_checkpoint(directory, job, config, policy, optimizer)
        streams = saved["streams"]
        step, updates, episode = saved["env_step"], saved["update_step"], saved["episode"]
        last_validation, ledgers = saved["last_validation"], saved["ledger_counts"]
        if job.get("experiment_role", "formal") == "formal" and step >= settings["pilot_step"]:
            release = _read(directory / "pilot_gate.json", {})
            passed = (release.get("passed") is True and
                      release.get("run_id") == job["run_id"] and
                      release.get("env_step") == settings["pilot_step"])
            if step == settings["pilot_step"] and last_validation < step:
                # An interrupted validation may be repaired, but it must stop
                # at the pilot gate before consuming decision 65537.
                passed = False
            elif step == settings["pilot_step"]:
                passed = passed and release.get("checkpoint_sha256") == _sha256(
                    directory / "checkpoint.pt")
            if not passed and not (step == settings["pilot_step"] and last_validation < step):
                raise ValueError("D12-A pilot gate release missing or differs")
    else:
        if (directory / "checkpoint.pt").exists() or any((directory / name).exists()
                                                      for name in ledger_names):
            raise FileExistsError("D12-A run directory requires same-ID resume")
        if job.get("train_mode") == "P1":
            _initialize_p1_from_bc(job, config, policy)
        elif job.get("train_mode") != "P0":
            raise ValueError("D12-A train_mode must be P0 or P1")
        streams = [_fresh_stream(train_set, core, settings["seed"] + 101 + i * 1009)
                   for i in range(32)]
        for i, stream in enumerate(streams):
            stream["episode_start_step"] = i + 1
        step = updates = episode = last_validation = 0
    _status(directory, job, stage="training", env_step=step,
            budget_env_steps=settings["total_steps"], update_step=updates,
            episode=episode, num_envs=32)
    return _train_resident(directory, job, config, preflight, settings, core,
                           validation_set, policy, optimizer, streams, step,
                           updates, episode, last_validation, ledgers)


def _evaluate(directory, job, config, resume, preflight):
    baseline = job.get("variant") if job.get("variant") in ("I0", "H1", "H2") else None
    if baseline is None and (config["training"]["device"] != "cuda" or
                             not torch.cuda.is_available()):
        raise RuntimeError("D12-A policy evaluation requires CUDA")
    if preflight is None:
        raise RuntimeError("D12-A evaluation requires workbench preflight")
    policy, metadata = (None, {"env_step": 0}) if baseline else _load_model(job, config, "cuda")
    role = next(iter(config["datasets"]))
    dataset = _dataset(config, role)
    indices = job.get("scene_indices")
    if role == "test" and (indices != list(range(job.get("shard_id", -1), 100, 4)) or
                           len(indices) != 25):
        raise ValueError("D12-A test shard identity differs")
    output = directory / "results.partial.jsonl"
    completed = 0
    if output.exists():
        if not resume:
            raise FileExistsError("Existing D12-A evaluation requires same-ID resume")
        lines = output.read_text(encoding="utf-8").splitlines()
        completed = len(lines)
        if completed > (len(indices) if indices is not None else len(dataset.scenes)) * 3:
            raise ValueError("D12-A evaluation ledger exceeds expected rows")
        for ordinal, line in enumerate(lines):
            row = json.loads(line)
            expected_index = (indices if indices is not None else range(len(dataset.scenes)))[ordinal // 3]
            if row.get("dataset_index") != expected_index or row.get("demand_multiplier") != MULTIPLIERS[ordinal % 3]:
                raise ValueError("D12-A evaluation row prefix differs")
    expected = (len(indices) if indices is not None else len(dataset.scenes)) * 3
    _status(directory, job, stage="evaluating", scene_count=completed,
            expected_scene_count=expected, results_partial=str(output))
    def progress(count):
        _status(directory, job, stage="evaluating", scene_count=count,
                expected_scene_count=expected, results_partial=str(output))
        return not _stop_requested(directory)

    rows = _evaluate_policy(policy, dataset, _core_config(config), device="cuda",
                            step=metadata["env_step"], output=output,
                            indices=indices, baseline=baseline, completed=completed,
                            on_row=progress)
    if len(rows) + completed < expected and _stop_requested(directory):
        _status(directory, job, stage="stopped", stop_reason="requested",
                scene_count=len(rows) + completed, results_partial=str(output))
        return
    if len(rows) + completed != expected:
        raise ValueError("D12-A evaluation row count differs")
    output.replace(directory / "results.jsonl")
    output = directory / "results.jsonl"
    all_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    _write_json(directory / "summary.json", {"scene_count": len(all_rows),
                "result_sha256": _sha256(output), "model": metadata,
                "method": job.get("variant"),
                "mean_best_j": (float(np.mean([x["best_j"] for x in all_rows
                                               if x["best_j"] is not None]))
                                if any(x["best_j"] is not None for x in all_rows)
                                else None)})
    _status(directory, job, stage="completed", scene_count=len(all_rows),
            results=str(output))


def run_drl_r04_job(directory, job, config, *, resume=False, preflight=None):
    """Dispatch a frozen D12-A task through the normal workbench worker."""
    directory = Path(directory)
    if config.get("schema_version") != 6 or config.get("research_stage") != "coverage":
        raise ValueError("D12-A requires coverage schema 6")
    directory.mkdir(parents=True, exist_ok=True)
    _status(directory, job, stage="initializing", preflight=preflight)
    if job["kind"] == "train":
        if job.get("train_mode") == "BC":
            _bc(directory, job, config, resume, preflight)
        else:
            _train(directory, job, config, resume, preflight)
    elif job["kind"] == "evaluate":
        _evaluate(directory, job, config, resume, preflight)
    elif job["kind"] == "prepare":
        _teacher(directory, job, config, resume, preflight)
    else:
        raise ValueError("D12-A job kind must be train, prepare or evaluate")
