"""Persistent workbench worker for first-stage coverage PPO and evaluation."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback

import numpy as np
import torch

from .ppo import CoveragePolicy, Rollout, ppo_update


REMOTE_DQN_PREFIX = Path("/home/u2024110224/first_stage_satellite_rl/envs/DQN")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _layout_hash(centers):
    return hashlib.sha256(np.asarray(centers, dtype="<f8").tobytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _write_json(path, value):
    from satellite_rl.workbench.runner import _atomic_json
    _atomic_json(path, _jsonable(value))


def _jsonl(path, value):
    path = Path(path)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def _jsonl_many(path, values):
    if not values:
        return
    path = Path(path)
    lines = [json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False) + "\n"
             for value in values]
    with path.open("a", encoding="utf-8") as stream:
        stream.writelines(lines)
        stream.flush()


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _status(directory, job, **fields):
    path = directory / "status.json"
    state = _read(path, {"run_id": job["run_id"]})
    state.update(fields, heartbeat_at=_now(), pid=os.getpid())
    _write_json(path, state)
    return state


def _event(directory, job, message, **fields):
    _jsonl(directory / "events.jsonl", {"time": _now(), "run_id": job["run_id"],
                                      "message": message, **fields})


def _stop_requested(directory):
    return (directory / "stop.request").exists() or (directory / "stop_request.json").exists()


def _execution_target(preflight):
    """Enforce the approved remote interpreter and GPU for real workbench jobs.

    Kept separate from the imported preflight so workbench tests can replace
    both interfaces without requiring a GPU on their test host.
    """
    prefix = Path(sys.prefix)
    if prefix != REMOTE_DQN_PREFIX or "4090 D" not in str(preflight.get("gpu", "")):
        raise RuntimeError(f"Coverage workbench requires {REMOTE_DQN_PREFIX} and RTX 4090 D; "
                           f"got prefix={prefix}, gpu={preflight.get('gpu')}")
    return {**preflight, "hostname": socket.gethostname(),
            "execution_target": "remote_rtx4090d_dqn",
            "python_prefix": str(prefix)}


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.load(map_location="cuda") also moves the *CPU* RNG ByteTensor.
    # PyTorch's RNG setters require CPU ByteTensors even for CUDA generators.
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([generator_state.cpu()
                                  for generator_state in state["cuda"]])


@contextmanager
def _isolated_rng(seed):
    before = _rng_state()
    try:
        _seed(seed)
        yield
    finally:
        _restore_rng(before)


def _dataset(config, role):
    from .scenes import load_dataset
    record = config["datasets"][role]
    path = Path(record["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen {role} dataset missing: {path}")
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"Frozen {role} dataset hash mismatch")
    return load_dataset(path, config["coverage"]["initialization"], config["coverage"])


def _policy(config, obs_dim, action_dim, device):
    network = config["network"]
    if network["kind"] != "coverage_candidate_mlp" or float(network.get("dropout", 0)) != 0:
        raise ValueError("Coverage PPO requires coverage_candidate_mlp with dropout=0")
    return CoveragePolicy(obs_dim, action_dim=action_dim,
                          hidden_sizes=network["hidden_sizes"], candidate_dim=4).to(device)


def _save_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _load_torch(path, device):
    # weights_only=False is required for NumPy/Python RNG and the environment
    # state; inputs are private workbench checkpoint files, never user uploads.
    return torch.load(path, map_location=device, weights_only=False)


def _model_metadata(job, config, policy, dataset_hashes, step):
    from .config import config_hash
    return {"format_version": 2, "policy_format": "coverage_candidate_v2",
            "algorithm": "ppo", "research_stage": "coverage",
            "experiment_role": job.get("experiment_role", "formal"),
            "network": "coverage_candidate_mlp", "obs_dim": policy.obs_dim,
            "action_dim": policy.action_dim, "candidate_dim": policy.candidate_dim,
            "hidden_sizes": list(policy.hidden_sizes),
            "dropout": 0, "training_seed": config["training"]["seed"],
            "learning_reward_scale": config["training"]["reward_scale"],
            "env_step": step, "selection": "fixed_budget_endpoint",
            "git_commit": job["git_commit"], "source_root": job["source_root"],
            "config_hash": config_hash(config), "datasets": dataset_hashes,
            "coverage": config["coverage"], "created_at": _now(),
            "hostname": socket.gethostname(),
            "execution_target": "remote_rtx4090d_dqn",
            "python_prefix": sys.prefix, "gpu": torch.cuda.get_device_name(0)}


def _export_policy(job, config, policy, dataset_hashes, step):
    directory = Path(job["model_dir"]) / "final"
    directory.mkdir(parents=True, exist_ok=True)
    _save_torch(directory / "policy.pt", policy.state_dict())
    _write_json(directory / "model.json", _model_metadata(job, config, policy, dataset_hashes, step))
    return directory


def _checkpoint(directory, job, config, policy, optimizer, env, obs, mask, candidates, step, update,
                episode, episode_return, last_validation_step, elapsed, dataset_hashes):
    from .config import config_hash
    payload = {"format_version": 2, "policy_format": "coverage_candidate_v2",
               "config_hash": config_hash(config),
               "source_commit": job["git_commit"], "source_root": job["source_root"],
               "dataset_hashes": dataset_hashes, "model": policy.state_dict(),
               "optimizer": optimizer.state_dict(), "env": env.state_dict(),
               "obs": obs, "mask": mask, "candidate_features": candidates,
               "step": step, "update": update,
               "episode": episode, "episode_return": episode_return,
               "last_validation_step": last_validation_step,
               "elapsed_seconds": elapsed, "rng": _rng_state()}
    path = directory / "checkpoint.pt"
    _save_torch(path, payload)
    _write_json(directory / "artifacts.json", {"run_id": job["run_id"],
                "config": str(directory / "config.json"), "checkpoint": str(path),
                "source_root": job["source_root"], "datasets": config["datasets"]})
    return _sha256(path)


def _restore(directory, job, config, policy, optimizer, env, dataset_hashes, device):
    from .config import config_hash
    payload = _load_torch(directory / "checkpoint.pt", device)
    if (payload.get("format_version") != 2 or
            payload.get("policy_format") != "coverage_candidate_v2" or
            payload["config_hash"] != config_hash(config) or
            payload["source_commit"] != job["git_commit"] or
            Path(payload["source_root"]).resolve() != Path(job["source_root"]).resolve() or
            payload["dataset_hashes"] != dataset_hashes):
        raise ValueError("Resume checkpoint source, configuration or datasets differ")
    policy.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    env.load_state_dict(payload["env"])
    if not torch.equal(payload["candidate_features"], env.candidate_features()):
        raise ValueError("Resume candidate features differ from restored environment")
    _restore_rng(payload["rng"])
    return payload


def _metrics_row(metrics, index):
    return {key: float(value[index].item()) for key, value in metrics.items()
            if key not in ("done", "scene_index")}


def _summary(rows):
    if not rows:
        return {"scene_count": 0, "valid_scene_count": 0,
                "valid_metric_scene_count": 0, "failed_scene_count": 0,
                "pooled_demand_coverage": 0.0}
    valid_metrics = [row for row in rows if row.get("metrics") is not None]
    failed = [row.get("metrics") is None or
              row.get("input", {}).get("metadata", {}).get("initialization_valid") is False
              for row in rows]
    result = {"scene_count": len(rows), "valid_scene_count": len(rows) - sum(failed),
              "valid_metric_scene_count": len(valid_metrics),
              "failed_scene_count": sum(failed)}
    movement_metrics = [row for row in valid_metrics
                        if bool(row["metrics"].get("movement_applicable", True))]
    result["movement_metric_scene_count"] = len(movement_metrics)
    requested = [float(np.asarray(row["input"]["demands_bps"], dtype=np.float64).sum())
                 for row in rows]
    users = [len(row["input"]["demands_bps"]) for row in rows]
    coverage_demand = [float(np.clip(row["metrics"]["coverage_demand"], 0, 1))
                       if not is_failed else 0.0 for row, is_failed in zip(rows, failed)]
    coverage_users = [float(np.clip(row["metrics"]["coverage_users"], 0, 1))
                      if not is_failed else 0.0 for row, is_failed in zip(rows, failed)]
    uncovered_demand = [max(0.0, float(row["metrics"]["uncovered_demand_bps"]))
                        if not is_failed else total
                        for row, total, is_failed in zip(rows, requested, failed)]
    uncovered_users = [float(row["metrics"]["uncovered_users"])
                       if not is_failed else float(count)
                       for row, count, is_failed in zip(rows, users, failed)]
    result.update(coverage_demand=float(np.mean(coverage_demand)),
                  coverage_users=float(np.mean(coverage_users)),
                  uncovered_demand_bps=float(np.mean(uncovered_demand)),
                  uncovered_users=float(np.mean(uncovered_users)),
                  feasible=float(np.mean([float(row["metrics"]["feasible"])
                                          if not is_failed else 0.0
                                          for row, is_failed in zip(rows, failed)])),
                  pooled_demand_coverage=(float(np.dot(coverage_demand, requested) /
                                                sum(requested)) if sum(requested) > 0 else 0.0),
                  requested_demand_bps=float(sum(requested)),
                  total_uncovered_demand_bps=float(sum(uncovered_demand)),
                  total_uncovered_users=float(sum(uncovered_users)))
    if valid_metrics:
        result.update({key: float(np.mean([row["metrics"][key] for row in valid_metrics]))
                       for key in valid_metrics[0]["metrics"]
                       if key not in ("coverage_demand", "coverage_users",
                                      "uncovered_demand_bps", "uncovered_users", "feasible",
                                      "movement_mean_deg", "movement_max_deg",
                                      "movement_applicable")})
    if movement_metrics:
        for key in ("movement_mean_deg", "movement_max_deg"):
            if key in movement_metrics[0]["metrics"]:
                result[key] = float(np.mean([row["metrics"][key]
                                             for row in movement_metrics]))
    return result


def _summary_scopes():
    return {"coverage_demand": "all evaluations; initializer failure=0 even if a partial layout exists",
            "coverage_users": "all evaluations; initializer failure=0 even if a partial layout exists",
            "feasible": "all evaluations; initializer failure=false",
            "uncovered_demand_bps": "all evaluations; initializer failure=all input demand",
            "uncovered_users": "all evaluations; initializer failure=all input aircraft",
            "pooled_demand_coverage": "sum covered requested demand / sum all input requested demand",
            "load_cv/load_std_bps/max_load_bps/score":
                "representable layouts only; denominator=valid_metric_scene_count; partial failed initializers remain raw",
            "movement_mean_deg/movement_max_deg":
                "movement_applicable layouts only; absent for empty-start P0; denominator=movement_metric_scene_count",
            "learning_reward":
                "environment score differences multiplied by shared training.reward_scale; raw terminal score remains separate"}


def _evaluate_policy(config, policy, dataset, *, device, job=None, directory=None,
                     group="validation", step=0):
    from .environment import CoverageBatchEnv

    def failure_row(index, repeat, action_seed):
        scene = dataset.scenes[index]
        return {"scene_index": index, "scene_id": scene.scene_id,
                "scene_sha256": scene.source_hash, "repeat": repeat,
                "action_seed": action_seed, "status": "initialization_failed",
                "metadata": scene.metadata, "metrics": None,
                "input": {"aircraft_ids": scene.aircraft_ids,
                          "positions": scene.positions, "demands_bps": scene.demands}}

    def terminal_row(index, local, env, metrics, layouts, assignments, aircraft_mask,
                     repeat, action_seed, elapsed, batch_size):
        scene = dataset.scenes[index]
        associated = assignments[local][aircraft_mask[local]]
        return {"scene_index": index, "scene_id": scene.scene_id,
                "scene_sha256": scene.source_hash, "repeat": repeat,
                "action_seed": action_seed, "status": "completed",
                "initial_layout_hash": _layout_hash(scene.centers),
                "final_layout_hash": _layout_hash(layouts[local].cpu().numpy()),
                "input": {"aircraft_ids": scene.aircraft_ids,
                          "positions": scene.positions, "demands_bps": scene.demands,
                          "initial_centers": scene.centers, "metadata": scene.metadata},
                "metrics": _metrics_row(metrics, local), "final_centers": layouts[local],
                "associations": associated, "beam_loads_bps": env.loads[local],
                "active_beam_mask": env.active_masks()[local],
                "uncovered_aircraft_ids": [scene.aircraft_ids[i] for i in
                    (associated < 0).nonzero(as_tuple=False).flatten().tolist()],
                # All scenes in a batch share the same latency; the allocated
                # GPU time is separate for throughput accounting.
                "elapsed_seconds": elapsed, "allocated_gpu_seconds": elapsed / batch_size,
                "batch_elapsed_seconds": elapsed, "batch_size": batch_size}

    def publish(row):
        rows.append(row)
        if directory is not None:
            _jsonl(directory / "evaluations.jsonl", {"step": step, "group": group,
                   "scene_index": row["scene_index"], "scene_id": row["scene_id"],
                   "scene_sha256": row["scene_sha256"], "repeat": row["repeat"],
                   "action_seed": row["action_seed"], "status": row["status"],
                   "metrics": row["metrics"]})
            _status(directory, job, stage="evaluating" if group == "test" else "validating",
                    evaluated_scenes=len(rows), evaluation_total=len(dataset) * repeats,
                    evaluation_execution=("batched_deterministic" if deterministic
                                          else "serial_stochastic_reference"))

    rows = []
    deterministic = config["evaluation"]["action_mode"] == "deterministic"
    repeats = int(config["evaluation"]["repeats"])
    base_seed = int(config["training"]["seed"]) + 100003
    batch_limit = min(int(config["training"]["num_envs"]), len(dataset))
    previous_mode = policy.training
    policy.eval()
    # A dedicated environment and isolated RNG leave training's stochastic
    # action stream and scene-order stream exactly where they were.
    try:
        for repeat in range(repeats):
            action_seed = (base_seed + repeat) % (2 ** 32)
            with _isolated_rng(action_seed):
                width = batch_limit if deterministic else 1
                for start in range(0, len(dataset), width):
                    if directory is not None and _stop_requested(directory):
                        break
                    indices = list(range(start, min(start + width, len(dataset))))
                    valid_indices = [i for i in indices
                                     if (np.isfinite(dataset.scenes[i].centers).all() and
                                         dataset.scenes[i].metadata.get("initialization_valid") is not False)]
                    terminal_rows = {}
                    if valid_indices:
                        env = CoverageBatchEnv(dataset, config["coverage"],
                                               num_envs=len(valid_indices), device=device,
                                               seed=action_seed, auto_reset=False)
                        if env.obs_dim != policy.obs_dim or env.action_dim != policy.action_dim:
                            raise ValueError("Evaluation environment and policy dimensions differ")
                        obs, mask = env.reset(scene_indices=valid_indices)
                        candidates = env.candidate_features()
                        started = time.monotonic()
                        last_heartbeat = started
                        for action_step in range(env.horizon):
                            action, _, _ = policy.act(obs, mask, candidates,
                                                      deterministic=deterministic)
                            obs, _, done, info = env.step(action)
                            mask = info.get("mask") if "mask" in info else env.action_mask()
                            candidates = env.candidate_features()
                            if directory is not None and time.monotonic() - last_heartbeat >= 20:
                                _status(directory, job,
                                        stage="evaluating" if group == "test" else "validating",
                                        evaluated_scenes=len(rows),
                                        evaluation_total=len(dataset) * repeats,
                                        evaluation_batch_size=len(valid_indices),
                                        evaluation_action_step=action_step + 1)
                                last_heartbeat = time.monotonic()
                        if not bool(done.all().item()):
                            raise RuntimeError("Coverage evaluation did not reach the fixed horizon")
                        elapsed = time.monotonic() - started
                        metrics = env.metrics()
                        layouts = env.layouts()
                        assignments, aircraft_mask = env.associations()
                        for local, index in enumerate(valid_indices):
                            terminal_rows[index] = terminal_row(
                                index, local, env, metrics, layouts, assignments,
                                aircraft_mask, repeat, action_seed, elapsed,
                                len(valid_indices))
                    for index in indices:
                        publish(terminal_rows[index] if index in terminal_rows else
                                failure_row(index, repeat, action_seed))
                if directory is not None and _stop_requested(directory):
                    break
    finally:
        policy.train(previous_mode)
    summary = _summary(rows)
    summary.update(unique_scene_count=len(dataset), repeat_count=repeats)
    return rows, summary


def _train(directory, job, config, resume, preflight):
    from .environment import CoverageBatchEnv
    training = config["training"]
    if training["gamma"] != 1.0:
        raise ValueError("Finite-horizon telescoping coverage reward requires gamma=1")
    reward_scale = float(training["reward_scale"])
    if not 0 < reward_scale <= 1:
        raise ValueError("Coverage learning reward scale must be in (0,1]")
    count = int(training["num_envs"])
    total = int(training["total_steps"])
    if total % count:
        raise ValueError("Training budget must be divisible by num_envs")
    device = torch.device("cuda")
    dataset_hashes = {role: record["sha256"] for role, record in config["datasets"].items()}
    train_data = _dataset(config, "train")
    validation_data = _dataset(config, "validation")
    invalid_train = [scene.scene_id for scene in train_data.scenes
                     if (not np.isfinite(scene.centers).all() or
                         scene.metadata.get("initialization_valid") is False)]
    if invalid_train:
        raise ValueError(f"Training scenes have invalid initial layouts: {invalid_train[:10]}")
    _seed(int(training["seed"]))
    env = CoverageBatchEnv(train_data, config["coverage"], num_envs=count, device=device,
                           seed=int(training["seed"]))
    policy = _policy(config, env.obs_dim, env.action_dim, device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(training["lr"]))
    if resume:
        payload = _restore(directory, job, config, policy, optimizer, env, dataset_hashes, device)
        obs, mask = payload["obs"], payload["mask"]
        candidates = payload["candidate_features"]
        step, update, episode = payload["step"], payload["update"], payload["episode"]
        episode_return = payload["episode_return"]
        last_validation_step = payload["last_validation_step"]
        elapsed_base = payload["elapsed_seconds"]
    else:
        obs, mask = env.reset()
        candidates = env.candidate_features()
        step = update = episode = 0
        episode_return = torch.zeros(count, device=device)
        last_validation_step = -1
        elapsed_base = 0.0
    started = time.monotonic()
    last_heartbeat = started
    _event(directory, job, "training_resumed" if resume else "training_started")
    _status(directory, job, stage="training", env_step=step, update_step=update,
            episode=episode, budget_env_steps=total, preflight=preflight,
            dataset_hashes=dataset_hashes, learning_reward_scale=reward_scale)
    while step < total:
        if _stop_requested(directory):
            sha = _checkpoint(directory, job, config, policy, optimizer, env, obs, mask,
                              candidates, step,
                              update, episode, episode_return, last_validation_step,
                              elapsed_base + time.monotonic() - started, dataset_hashes)
            _status(directory, job, stage="stopped", env_step=step, update_step=update,
                    episode=episode, checkpoint_step=step, checkpoint_sha256=sha)
            _event(directory, job, "training_stopped", env_step=step)
            return
        length = min(int(training["rollout_steps"]), (total - step) // count)
        observations, masks, candidate_batches, actions, old_logp, old_values, rewards, dones = (
            [] for _ in range(8))
        completed = []
        for offset in range(length):
            with torch.no_grad():
                action, logp, value = policy.act(obs, mask, candidates)
            next_obs, raw_reward, done, info = env.step(action)
            reward = raw_reward * reward_scale
            observations.append(obs)
            masks.append(mask)
            candidate_batches.append(candidates)
            actions.append(action)
            old_logp.append(logp)
            old_values.append(value)
            rewards.append(reward)
            dones.append(done)
            episode_return += reward
            next_mask = info.get("mask") if "mask" in info else env.action_mask()
            if info.get("episodes_finished", False):
                terminal = info["episode_metrics"]
                metric_names = [name for name in terminal
                                if name not in ("done", "scene_index")]
                # One device-to-host transfer for the complete vector batch;
                # preserve the original per-environment row order below.
                packed = torch.stack(
                    [terminal[name].to(torch.float64) for name in metric_names] +
                    [terminal["scene_index"].to(torch.float64),
                     episode_return.to(torch.float64)], dim=1).cpu().numpy()
                for i in range(count):
                    scene_index = int(packed[i, -2])
                    completed.append({**{name: float(packed[i, j])
                                          for j, name in enumerate(metric_names)},
                                      "episode_return": float(packed[i, -1]),
                                      "scene_index": scene_index,
                                      "scene_sha256": train_data.scenes[scene_index].source_hash,
                                      "completion_env_step": step + (offset + 1) * count,
                                      "collected_update_step": update})
                episode_return.zero_()
            obs, mask = next_obs, next_mask
            candidates = env.candidate_features()
            if time.monotonic() - last_heartbeat >= 20:
                _status(directory, job, stage="stopping" if _stop_requested(directory)
                        else "collecting", env_step=step + (offset + 1) * count,
                        update_step=update, episode=episode + len(completed),
                        budget_env_steps=total)
                last_heartbeat = time.monotonic()
        with torch.no_grad():
            bootstrap_value = policy.value(obs)
        rollout = Rollout(*(torch.stack(items) for items in
                            (observations, masks, candidate_batches, actions, old_logp,
                             old_values, rewards, dones)),
                          bootstrap_value)
        diagnostics = ppo_update(policy, optimizer, rollout,
            gamma=float(training["gamma"]), gae_lambda=float(training["gae_lambda"]),
            clip_ratio=float(training["clip_ratio"]), entropy_coef=float(training["entropy_coef"]),
            value_coef=float(training["value_coef"]), max_grad_norm=float(training["max_grad_norm"]),
            epochs=int(training["epochs"]), minibatch_size=int(training["minibatch_size"]))
        step += length * count
        update += diagnostics.pop("updates")
        first_episode = episode
        episode += len(completed)
        elapsed = elapsed_base + time.monotonic() - started
        rate = step / max(elapsed, 1e-9)
        _jsonl(directory / "updates.jsonl", {"env_step": step, "update_step": update,
               "rollout_steps": length * count, "episode": episode,
               "learning_reward_scale": reward_scale, **diagnostics})
        _jsonl(directory / "metrics.jsonl", {"step": step, "group": "ppo_update",
               "values": diagnostics, "wall_time": _now()})
        _jsonl_many(directory / "episodes.jsonl", [
            {"env_step": metrics["completion_env_step"],
             "update_step": metrics["collected_update_step"],
             "episode": first_episode + offset,
             "seed": int(training["seed"]),
             "training_seed": int(training["seed"]),
             "learning_reward_scale": reward_scale,
             "scene_index": metrics["scene_index"],
             "scene_sha256": metrics["scene_sha256"],
             "metrics": {key: value for key, value in metrics.items()
                         if key not in ("completion_env_step", "collected_update_step")}}
            for offset, metrics in enumerate(completed, start=1)])
        if completed:
            movement_available = any(m.get("movement_applicable", 1.0) > 0
                                     for m in completed)
            terminal_mean = {name: float(np.mean([m[name] for m in completed]))
                             for name in completed[0]
                             if name not in ("scene_index", "scene_sha256",
                                             "completion_env_step", "collected_update_step")
                             and (movement_available or name not in
                                  ("movement_mean_deg", "movement_max_deg"))}
            _jsonl(directory / "metrics.jsonl", {"step": step, "group": "terminal",
                   "values": terminal_mean, "wall_time": _now()})
        if step - last_validation_step >= int(training["validation_interval"]) or step == total:
            _, summary = _evaluate_policy(config, policy, validation_data, device=device,
                                            job=job, directory=directory,
                                            group="validation", step=step)
            if _stop_requested(directory):
                sha = _checkpoint(directory, job, config, policy, optimizer, env, obs, mask,
                                  candidates,
                                  step, update, episode, episode_return,
                                  last_validation_step, elapsed,
                                  dataset_hashes)
                _status(directory, job, stage="stopped", env_step=step, update_step=update,
                        episode=episode, checkpoint_step=step, checkpoint_sha256=sha)
                _event(directory, job, "training_stopped", env_step=step)
                return
            _jsonl(directory / "metrics.jsonl", {"step": step, "group": "validation",
                   "values": summary, "metric_scopes": _summary_scopes(),
                   "wall_time": _now()})
            last_validation_step = step
        elapsed = elapsed_base + time.monotonic() - started
        rate = step / max(elapsed, 1e-9)
        if step % int(training["checkpoint_interval"]) < length * count or step == total or _stop_requested(directory):
            sha = _checkpoint(directory, job, config, policy, optimizer, env, obs, mask,
                              candidates, step,
                              update, episode, episode_return, last_validation_step,
                              elapsed, dataset_hashes)
            checkpoint_fields = {"checkpoint_step": step, "checkpoint_sha256": sha}
        else:
            checkpoint_fields = {}
        _status(directory, job, stage="training", env_step=step, update_step=update,
                episode=episode, budget_env_steps=total, steps_per_second=rate,
                eta_seconds=(total - step) / max(rate, 1e-9), wall_seconds=elapsed,
                gpu_memory_mb=torch.cuda.memory_allocated() / 1048576,
                latest_update=diagnostics, **checkpoint_fields)
    if _stop_requested(directory):
        sha = _checkpoint(directory, job, config, policy, optimizer, env, obs, mask,
                          candidates, step,
                          update, episode, episode_return, last_validation_step,
                          elapsed_base + time.monotonic() - started, dataset_hashes)
        _status(directory, job, stage="stopped", env_step=step, update_step=update,
                episode=episode, checkpoint_step=step, checkpoint_sha256=sha)
        _event(directory, job, "training_stopped", env_step=step)
        return
    final = _export_policy(job, config, policy, dataset_hashes, step)
    artifacts = _read(directory / "artifacts.json", {})
    artifacts.update(final_policy=str(final), model=str(final / "policy.pt"),
                     model_metadata=str(final / "model.json"))
    _write_json(directory / "artifacts.json", artifacts)
    _status(directory, job, stage="completed", env_step=step, update_step=update,
            episode=episode, final_policy=str(final))
    _event(directory, job, "training_completed", env_step=step)


def _evaluation(directory, job, config, preflight):
    from .baselines import run_baseline
    dataset = _dataset(config, "test")
    coverage = config["coverage"]
    method = coverage.get("method", "ppo")
    if method not in ("ppo", "none", "local_search"):
        raise ValueError("Unknown coverage evaluation method")
    _status(directory, job, stage="evaluating", evaluated_scenes=0,
            evaluation_total=len(dataset) * (int(config["evaluation"]["repeats"])
                                            if method == "ppo" else 1),
            preflight=preflight)
    if method == "ppo":
        model_dir = Path(job["model_path"])
        if model_dir.is_file():
            model_dir = model_dir.parent
        metadata = _read(model_dir / "model.json")
        if (metadata is None or metadata.get("research_stage") != "coverage" or
                metadata.get("format_version") != 2 or
                metadata.get("policy_format") != "coverage_candidate_v2"):
            raise ValueError("Coverage model metadata missing")
        expected_action_dim = (int(coverage["candidate_capacity"])
                               if coverage["initialization"] == "empty" else 9)
        if (metadata.get("hidden_sizes") != list(config["network"]["hidden_sizes"]) or
                metadata.get("dropout") != 0 or
                metadata.get("network") != "coverage_candidate_mlp" or
                metadata.get("obs_dim") != 964 or
                metadata.get("candidate_dim") != 4 or
                metadata.get("action_dim") != expected_action_dim):
            raise ValueError("Evaluation policy architecture differs from frozen model")
        model_coverage = dict(metadata.get("coverage", {}))
        eval_coverage = dict(coverage)
        model_coverage.pop("method", None)
        eval_coverage.pop("method", None)
        if model_coverage != eval_coverage:
            raise ValueError("Evaluation coverage physics/initializer differs from frozen model")
        policy = _policy(config, int(metadata["obs_dim"]), expected_action_dim, "cuda")
        policy.load_state_dict(_load_torch(model_dir / "policy.pt", "cuda"))
        rows, summary = _evaluate_policy(config, policy, dataset, device="cuda",
                                         job=job, directory=directory, group="test")
    else:
        rows = []
        for index, scene in enumerate(dataset.scenes):
            if _stop_requested(directory):
                break
            result = run_baseline(scene, config, method=method)
            rows.append({"scene_index": index, "scene_sha256": scene.source_hash,
                         "input": {"aircraft_ids": scene.aircraft_ids,
                                   "positions": scene.positions, "demands_bps": scene.demands,
                                   "initial_centers": scene.centers if
                                   np.isfinite(scene.centers).all() else None,
                                   "metadata": scene.metadata}, **result})
            _jsonl(directory / "evaluations.jsonl", {"group": "test", "scene_index": index,
                   "scene_id": scene.scene_id, "scene_sha256": scene.source_hash,
                   "method": method, "status": result["status"],
                   "metrics": result.get("metrics")})
            _status(directory, job, stage="evaluating", evaluated_scenes=len(rows),
                    evaluation_total=len(dataset))
        summary = _summary(rows)
    if _stop_requested(directory):
        _status(directory, job, stage="stopped", evaluated_scenes=len(rows))
        _event(directory, job, "evaluation_stopped", evaluated_scenes=len(rows))
        return
    result_path = directory / "results.json"
    _write_json(result_path, {"summary": summary, "scenes": rows,
                "metadata": {"research_stage": "coverage", "method": method,
                             "policy_format": "coverage_candidate_v2" if method == "ppo" else None,
                             "git_commit": job["git_commit"], "datasets": config["datasets"],
                             "model_path": job.get("model_path"), "config": config,
                             "learning_reward_scale": config["training"]["reward_scale"],
                             "preflight": preflight,
                             "metric_scopes": _summary_scopes(),
                             "evaluation_execution": ("batched_deterministic" if
                                 method == "ppo" and config["evaluation"]["action_mode"] == "deterministic"
                                 else "serial_stochastic_reference" if method == "ppo"
                                 else "baseline"),
                             "action_seed_rule": "(training.seed + 100003 + repeat) mod 2^32" if
                                 method == "ppo" else None}})
    _write_json(directory / "artifacts.json", {"run_id": job["run_id"],
                "config": str(directory / "config.json"), "results": str(result_path),
                "source_root": job["source_root"], "datasets": config["datasets"]})
    _status(directory, job, stage="completed", evaluated_scenes=len(rows), metrics=summary,
            results=str(result_path))
    _event(directory, job, "evaluation_completed", scene_count=len(rows))


def run_coverage_job(job_dir, resume=False):
    """Run a frozen job dispatched by the existing workbench scheduler."""
    from .config import validate_config
    from satellite_rl.workbench.runner import strict_cuda_preflight

    directory = Path(job_dir).resolve()
    job = _read(directory / "job.json")
    if job is None:
        raise FileNotFoundError(directory / "job.json")
    try:
        config = validate_config(_read(directory / "config.json"))
        if config["research_stage"] != "coverage" or config["algorithm"] != "ppo":
            raise ValueError("Job is not a coverage PPO task")
        if config["training"]["device"] != "cuda":
            raise ValueError("Persistent coverage jobs require CUDA")
        preflight = _execution_target(strict_cuda_preflight(job))
        torch.set_num_threads(int(config["training"]["torch_threads"]))
        preflight["torch_threads"] = torch.get_num_threads()
        for role, record in config["datasets"].items():
            if _sha256(record["path"]) != record["sha256"]:
                raise ValueError(f"Frozen {role} dataset hash mismatch")
        if config["schema_version"] == 3:
            from .balance_runner import run_balance_job
            run_balance_job(directory, job, config, resume=resume, preflight=preflight)
        elif job["kind"] == "train":
            _train(directory, job, config, resume, preflight)
        elif job["kind"] == "evaluate":
            if resume:
                raise ValueError("Evaluation restart requires a new workbench job")
            _evaluation(directory, job, config, preflight)
        else:
            raise ValueError("Unknown workbench task kind")
    except BaseException as error:
        failure = {"run_id": job.get("run_id"), "time": _now(), "error": str(error),
                   "traceback": traceback.format_exc(),
                   "checkpoint": str(directory / "checkpoint.pt") if
                   (directory / "checkpoint.pt").exists() else None}
        _write_json(directory / "failure.json", failure)
        _status(directory, job, stage="failed", error=str(error),
                traceback=failure["traceback"], failure_path=str(directory / "failure.json"))
        _event(directory, job, "worker_failed", error=str(error))
        raise


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run_coverage_job(args.job_dir, resume=args.resume)


if __name__ == "__main__":
    main()
