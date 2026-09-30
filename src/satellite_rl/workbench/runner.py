"""Persistent CUDA worker for workbench training and scene-based evaluation."""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch

from satellite_rl.agents.workbench_algorithms import build_agent, ensure_finite_modules
from satellite_rl.training.run_monitor import replace_with_retry
from satellite_rl.workbench.checkpoints import (config_hash, export_policy, load_checkpoint,
                                                 load_policy, save_checkpoint)
from satellite_rl.workbench.learning_snapshots import FORMAT as LEARNING_SNAPSHOT_FORMAT, save_learning_snapshot
from satellite_rl.workbench.scene_order import SceneOrder


# Only asynchronous jobs register here. The historical synchronous path remains
# byte-for-byte identical in its per-call persistence behavior.
_TELEMETRY_SINKS = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    try:
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(_jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _jsonl(path, value):
    if _TELEMETRY_SINKS:
        # These paths are constructed from the already resolved job directory;
        # avoid a filesystem resolve/open on every parameter update.
        sink = _TELEMETRY_SINKS.get(Path(path).absolute().parent)
        if sink is not None:
            sink.append_jsonl(path, value)
            return
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def _tail_jsonl(path, count=10):
    if not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8") as stream:
        tail = deque((line for line in stream if line.strip()), maxlen=count)
    rows = []
    for line in tail:
        try:
            rows.append(json.loads(line))
        except ValueError:
            rows.append({"unparseable_line": line[:500]})
    return rows


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_cuda_preflight(job):
    """Verify the frozen source, DQN interpreter and working CUDA before work."""
    import satellite_rl

    root = Path(job["source_root"]).resolve()
    manifest_path = root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["git_commit"] != job["git_commit"]:
        raise RuntimeError("Job commit and frozen source manifest disagree")
    files = manifest["files"]
    if not files:
        raise RuntimeError("Frozen source manifest has no files")
    for relative, expected in files.items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file() or _sha256(target) != expected:
            raise RuntimeError(f"Frozen source checksum mismatch: {relative}")
    package = Path(satellite_rl.__file__).resolve()
    if not package.is_relative_to(root / "src"):
        raise RuntimeError(f"satellite_rl import is {package}; expected frozen {root / 'src'}")
    # Linux virtual environments may symlink their interpreter outside the
    # prefix. Verify the active environment, retaining both paths as evidence.
    interpreter = Path(sys.executable).absolute()
    environment_prefix = Path(sys.prefix).resolve()
    parts = [part.lower() for part in environment_prefix.parts]
    if "envs" not in parts or "dqn" not in parts:
        raise RuntimeError(f"Training/evaluation requires DQN Python, got {interpreter} ({environment_prefix})")
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError(f"DQN CUDA unavailable: torch={torch.__version__}, cuda={torch.version.cuda}")
    torch.set_num_threads(1)
    return {"python": str(interpreter), "python_resolved": str(interpreter.resolve()),
            "environment_prefix": str(environment_prefix), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "numpy": np.__version__,
            "source_root": str(root), "package_file": str(package), "git_commit": job["git_commit"],
            "manifest_sha256": _sha256(manifest_path)}


def seed_training(seed):
    """Set every initialization stream before constructing networks and optimizers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def isolated_rng(seed):
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
              torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(before[0])
        np.random.set_state(before[1])
        torch.set_rng_state(before[2])
        if before[3] is not None:
            torch.cuda.set_rng_state_all(before[3])


class Ledger:
    def __init__(self, directory, job, config, preflight, *, resume=False):
        self.directory = Path(directory)
        self.job = job
        self.config = config
        self.started = time.monotonic()
        self.status_path = self.directory / "status.json"
        self.status = (json.loads(self.status_path.read_text(encoding="utf-8")) if resume
                       and self.status_path.exists() else {
                           "run_id": job["run_id"], "stage": "initializing", "env_step": 0,
                           "update_step": 0, "episode": 0,
                           "budget_env_steps": config["training"]["total_steps"],
                           "created_at": now()})
        self.status.update({"pid": os.getpid(), "heartbeat_at": now(), "preflight": preflight,
                            "action_mask": config["environment"].get("action_mask", "none")})
        self.writer = None
        self.sink = None
        self.closed = False
        self.telemetry = config.get("telemetry", {"mode": "sync", "flush_interval_seconds": 2.0,
                                                  "queue_capacity": 2048})
        if self.telemetry["mode"] == "async":
            from satellite_rl.workbench.telemetry import AsyncTelemetrySink
            key = self.directory.resolve()
            if key in _TELEMETRY_SINKS:
                raise RuntimeError("A telemetry sink is already active for this job")
            self.sink = AsyncTelemetrySink(max_queue=self.telemetry["queue_capacity"],
                                           flush_interval=self.telemetry["flush_interval_seconds"])
            _TELEMETRY_SINKS[key] = self.sink
        if not resume:
            self.open_tensorboard()
        self.save_status()

    def open_tensorboard(self, *, purge_step=None):
        """On resume, use the loaded checkpoint boundary, never the stale status step."""
        if self.writer is not None:
            raise RuntimeError("TensorBoard writer already open")
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.directory / "tensorboard"), purge_step=purge_step)
            if self.sink is not None:
                self.sink.attach_tensorboard(self.writer)
        except ImportError:
            self.event("TensorBoard is unavailable; JSONL metrics remain canonical")

    def event(self, message, **fields):
        _jsonl(self.directory / "events.jsonl", {"time": now(), "run_id": self.job["run_id"],
                                                     "message": message, **fields})

    def record(self, group, step, values):
        clean = {}
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, (int, float, np.number, bool)):
                number = float(value)
                if not np.isfinite(number):
                    raise FloatingPointError(f"Nonfinite metric: {group}/{key}")
                clean[key] = number
        row = {"step": int(step), "wall_time": now(), "group": group, "values": clean}
        _jsonl(self.directory / "metrics.jsonl", row)
        if self.writer:
            if self.sink is not None:
                self.sink.record_tensorboard(group, step, clean)
            else:
                for key, value in clean.items():
                    self.writer.add_scalar(f"{group}/{key}", value, step)
                self.writer.flush()

    def save_status(self, **fields):
        self.status.update(fields)
        self.status["heartbeat_at"] = now()
        if self.sink is not None:
            self.status["telemetry"] = {**self.telemetry, **self.sink.health}
            self.sink.write_status(self.status_path, self.status)
        else:
            _atomic_json(self.status_path, self.status)

    def flush(self):
        if self.sink is not None:
            self.sink.flush()

    def close(self):
        if self.closed:
            return
        try:
            if self.sink is not None:
                self.sink.close()
            elif self.writer:
                self.writer.close()
        finally:
            self.closed = True
            if self.sink is not None:
                _TELEMETRY_SINKS.pop(self.directory.resolve(), None)


def _save_scene(path, scene):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(value) for key, value in scene.items()}
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise RuntimeError(f"Existing scene differs: {path}")
    else:
        temp = path.with_name(path.name + ".tmp." + str(os.getpid()))
        try:
            with temp.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            replace_with_retry(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    return {"path": str(path), "sha256": _sha256(path)}


def _load_scene(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: np.array(data[key], copy=True) for key in data.files}


def _scene_files(path):
    path = Path(path)
    files = sorted(path.glob("*.npz")) if path.is_dir() else [path]
    if not files or any(not file.is_file() for file in files):
        raise ValueError(f"No valid scene files: {path}")
    return files


def _reconcile(directory, checkpoint_step):
    """Discard log rows after the last complete checkpoint on resume."""
    for name in ("metrics", "episodes", "updates", "evaluations", "traces"):
        path = Path(directory) / f"{name}.jsonl"
        if not path.exists():
            continue
        temp = path.with_name(path.name + ".tmp")
        try:
            with path.open(encoding="utf-8") as source, temp.open("w", encoding="utf-8") as stream:
                for line in source:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        # Abrupt process exit can leave the final append partial.
                        # Never hide corruption of a newline-terminated record.
                        if not line.endswith("\n"):
                            break
                        raise
                    if int(row.get("env_step", row.get("step", 0))) <= checkpoint_step:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            replace_with_retry(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def _restore_logging(ledger, directory, checkpoint_step):
    ledger.flush()
    _reconcile(directory, checkpoint_step)
    # Metrics at the checkpoint's own step were written before it and are
    # canonical; only later steps belong to the rolled-back tail.
    ledger.open_tensorboard(purge_step=checkpoint_step + 1)


def _ensure_finite_agent(agent):
    names = ("actor", "critic1", "critic2", "target1", "target2", "value", "online", "target")
    try:
        ensure_finite_modules(((name, getattr(agent, name)) for name in names
                               if getattr(agent, name, None) is not None),
                              getattr(agent, "log_alpha", None))
    except FloatingPointError as error:
        raise FloatingPointError(f"{error}; last checkpoint preserved") from error


def _evaluation_episode(env, policy, scene, scene_id, seed, action_mode, trace_path=None,
                        env_step=0, policy_name="model"):
    state = env.reset(seed=seed, scene=scene)
    reward_total = 0.0
    trace = []
    done = False
    step = 0
    while not done:
        mask = env.action_mask()
        policy_options = {"action_mask": mask} if mask is not None else {}
        action = policy.act(state, deterministic=action_mode == "deterministic", **policy_options)
        next_state, reward, done, info = env.step(action)
        reward_total += float(reward)
        row = {"scene_id": scene_id, "step": step, "env_step": env_step, "policy": policy_name,
               "requested_action": info.get("requested_action", list(action)),
               "actual_action": info.get("actual_action"), "reward": reward,
               "reward_terms": info.get("reward_terms"), "done": done,
               "beam_id": info.get("beam_id"),
               "action_mask": info.get("action_mask", "none"),
               "legal_action_count": info.get("legal_action_count"),
               "policy_action_info": getattr(policy, "last_action_info", {})}
        trace.append(_jsonable(row))
        if trace_path:
            _jsonl(trace_path, row)
        state = next_state
        step += 1
        if step > 1000:
            raise RuntimeError("Evaluation environment did not terminate")
    return {"scene_id": scene_id, "metrics": {**env.metrics(), "return": reward_total},
            "beams": env.beam_results(), "trace": trace}


def _summarize_scenes(results):
    keys = set.intersection(*(set(row["metrics"]) for row in results)) if results else set()
    return {key: float(np.mean([row["metrics"][key] for row in results]))
            for key in keys if all(isinstance(row["metrics"][key], (int, float)) for row in results)}


def _validation_due(evaluation, progress, env_step):
    """Keep legacy cadence; explicit early nodes do not shift the periodic grid.

    Evaluation happens at episode boundaries. A requested off-boundary node is
    evaluated at the first following boundary and logged at that actual step.
    The last completed node is part of the full checkpoint, avoiding duplicate
    evaluation at a restored boundary.
    """
    last = progress.get("last_validation_step", -1)
    if env_step <= last:
        return False
    interval = evaluation["interval_steps"]
    extra = evaluation.get("extra_steps", [])
    if not extra:
        return env_step > 0 and interval > 0 and env_step - last >= interval
    extra_due = any(last < step <= env_step for step in extra)
    periodic_due = (interval > 0 and env_step > 0
                    and env_step // interval > max(last, 0) // interval)
    return extra_due or periodic_due


def _fixed_evaluate(config, agent, directory, group, count, seed_base, env_step, ledger, scene_set=None):
    from satellite_rl.workbench.environment import WorkbenchEnv
    files = None
    if scene_set is not None:
        from satellite_rl.workbench.datasets import assert_scene_profile, scene_set_files
        assert_scene_profile(scene_set, config["environment"].get("profile", "china_legacy"))
        files = scene_set_files(scene_set)
        if len(files) != count:
            raise ValueError("冻结评价场景数量与配置不一致")

    results = []
    for index in range(count):
        seed = seed_base + index
        scene_path = Path(directory) / "scenes" / group / f"scene_{seed}.npz"
        with isolated_rng(seed):
            env = WorkbenchEnv(config)
            env.reset(seed=seed, scene=_load_scene(files[index][0]) if files is not None else None)
            scene = env.scene()
            scene_ref = _save_scene(scene_path, scene)
            if files is not None:
                scene_ref.update(source_path=str(files[index][0]), source_sha256=files[index][1],
                                 source_scene_id=scene_set["id"])
        for repeat in range(config["evaluation"]["repeats"]):
            action_seed = seed + repeat * 1000000
            with isolated_rng(action_seed):
                agent_rng = agent.rng.getstate()
                agent.rng.seed(action_seed + 7321)
                try:
                    env = WorkbenchEnv(config)
                    result = _evaluation_episode(env, agent, scene, f"{group}_{seed}_r{repeat}",
                                                 action_seed, config["evaluation"]["action_mode"],
                                                 Path(directory) / "traces.jsonl" if group == "test" else None,
                                                 env_step)
                finally:
                    agent.rng.setstate(agent_rng)
            result["scene"] = scene_ref
            result["action_seed"] = action_seed
            _jsonl(Path(directory) / "evaluations.jsonl", {"env_step": env_step, "group": group,
                                                            "scene_id": result["scene_id"],
                                                            "action_seed": action_seed,
                                                            "metrics": result["metrics"],
                                                            "scene_sha256": scene_ref["sha256"]})
            results.append(result)
    summary = _summarize_scenes(results)
    ledger.record(group, env_step, summary)
    return results, summary


def _artifacts(directory, job, **extras):
    path = Path(directory) / "artifacts.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.update({"run_id": job["run_id"], "config": str(Path(directory) / "config.json"),
                 "source_root": job["source_root"], **extras})
    _atomic_json(path, data)


def _job_provenance(job, config):
    from satellite_rl.envs.regions import profile_metadata
    metadata = profile_metadata(config["environment"].get("profile", "china_legacy"))
    return {"run_id": job["run_id"], "git_commit": job["git_commit"],
            "profile": metadata["id"], "environment_version": metadata["environment_version"],
            "environment_profile": metadata, "scene_sets": job.get("scene_sets", {}),
            **({key: job.get(key) for key in ("model_files", "model_source_path", "model_provenance")}
               if job.get("model_files") else {})}


def _training(directory, job, config, ledger, resume):
    from satellite_rl.workbench.config import config_summary
    from satellite_rl.workbench.environment import WorkbenchEnv

    training = config["training"]
    summary = config_summary(config)
    seed_training(training["seed"])
    agent = build_agent(config, summary["state_dim"], summary["action_dims"], "cuda")
    checkpoint = Path(directory) / "checkpoint.pt"
    if resume:
        if not checkpoint.exists():
            raise RuntimeError("No recoverable checkpoint")
        progress, restored_environment = load_checkpoint(checkpoint, agent, config)
        restored_sets = restored_environment.get("scene_sets") if restored_environment else None
        if restored_sets is not None and restored_sets != job.get("scene_sets", {}):
            raise ValueError("恢复任务的冻结场景与断点不一致")
    else:
        progress = {"env_step": 0, "episode": 0, "best_validation": None,
                    "last_validation_step": -1, "last_checkpoint_step": 0,
                    "wall_seconds": 0.0}
    env_step = progress["env_step"]
    episode = progress["episode"]
    session_started = time.monotonic()
    session_base_steps = env_step
    session_base_updates = agent.update_step
    base_wall_seconds = float(progress.get("wall_seconds", 0.0))

    def save_progress(stage=None, **fields):
        elapsed = max(time.monotonic() - session_started, 1e-9)
        step_rate = max(0, env_step - session_base_steps) / elapsed
        update_rate = max(0, agent.update_step - session_base_updates) / elapsed
        current_stage = stage or ("warmup" if config["algorithm"] != "ppo" and
                                  env_step < training["warmup_steps"] else "training")
        ledger.save_status(stage=current_stage, env_step=env_step, episode=episode,
                           update_step=agent.update_step, budget_env_steps=training["total_steps"],
                           steps_per_second=step_rate, updates_per_second=update_rate,
                           eta_seconds=((training["total_steps"] - env_step) / step_rate
                                        if step_rate > 0 else None),
                           wall_seconds=base_wall_seconds + elapsed,
                           gpu=ledger.status["preflight"]["gpu"],
                           gpu_memory_mb=torch.cuda.memory_allocated() / 1048576,
                           **fields)

    model_dir = Path(job["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    scene_sets = job.get("scene_sets", {})
    if scene_sets:
        from satellite_rl.workbench.datasets import assert_scene_profile, scene_set_files
        for record in scene_sets.values():
            assert_scene_profile(record, config["environment"].get("profile", "china_legacy"))
            scene_set_files(record)
    supplied_scenes = (scene_set_files(scene_sets["train"]) if "train" in scene_sets else
                       [(path, _sha256(path)) for path in _scene_files(job["training_scene_path"])]
                       if job.get("training_scene_path") else [])
    scene_order = SceneOrder(training.get("scene_order", "sequential"),
                             training.get("scene_order_seed", 51001),
                             [digest for _, digest in supplied_scenes])
    if resume:
        scene_order.verify_checkpoint((restored_environment or {}).get("scene_order"), episode)
    order_provenance = {"mode": scene_order.mode, "seed": scene_order.seed,
                        "source_count": len(supplied_scenes),
                        "source_set_sha256": scene_order.source_set_sha256}
    provenance = {**_job_provenance(job, config), "training_scene_order": order_provenance}
    source_manifest = Path(job["source_root"]) / "source_manifest.json"
    snapshot_provenance = {**provenance, "source_root": job["source_root"],
                           "source_manifest_sha256": _sha256(source_manifest) if source_manifest.is_file() else None,
                           "training_scene_path": job.get("training_scene_path")}

    def snapshot_artifact(step):
        path = save_learning_snapshot(directory, agent, config, snapshot_provenance, step,
                                      source_history=job.get("source_history"))
        if path is None:
            return None
        recovery_records = sorted(str(item) for item in Path(path).parent.glob(
            f"{Path(path).stem}_recovery_*.json"))
        return {"directory": str(Path(directory) / "learning_snapshots"), "latest": path,
                "format": LEARNING_SNAPSHOT_FORMAT,
                "source_recovery_records": recovery_records,
                "description": "SAC网络参数及最多64条终止、64条非终止转移；用于离线学习诊断，不可用于恢复训练"}

    def run_validation():
        save_progress("validating")
        ledger.flush()
        _, validation = _fixed_evaluate(config, agent, directory, "validation",
                                         config["evaluation"]["validation_count"],
                                         config["evaluation"]["validation_seed"], env_step, ledger,
                                         scene_set=scene_sets.get("validation"))
        progress["last_validation_step"] = env_step
        selection = validation.get("final_si_mean")
        if selection is not None and (progress["best_validation"] is None or selection > progress["best_validation"]):
            progress["best_validation"] = selection
            export_policy(model_dir / "best_validation", agent, config,
                          {**provenance,
                           "env_step": env_step, "selection": "validation_si_mean"})
        save_progress()

    if resume:
        snapshot_record = snapshot_artifact(env_step)
        _restore_logging(ledger, directory, env_step)
        if snapshot_record:
            _artifacts(directory, job, learning_snapshots=snapshot_record)
    save_progress(scene_order=scene_order.checkpoint_state(episode))
    ledger.event("training_started" if not resume else "training_resumed", env_step=env_step,
                 config_hash=config_hash(config), training_scene_order=order_provenance)
    last_saved_step = progress.get("last_checkpoint_step", env_step)
    if not resume:
        if _validation_due(config["evaluation"], progress, env_step):
            run_validation()
        save_checkpoint(checkpoint, agent, progress, config,
                        {**provenance, "boundary": "next_episode",
                         "scene_order": scene_order.checkpoint_state(episode)})
        ledger.status["checkpoint_step"] = env_step
        snapshot_record = snapshot_artifact(env_step)
        _artifacts(directory, job, checkpoint=str(checkpoint),
                   **({"learning_snapshots": snapshot_record} if snapshot_record else {}))
    while env_step < training["total_steps"]:
        env = WorkbenchEnv(config)
        scene_seed = training["seed"] + 100000 + episode
        selection = scene_order.selection(episode)
        source_index = selection["source_index"]
        source_scene = supplied_scenes[source_index][0] if source_index is not None else None
        if source_scene is not None and _sha256(source_scene) != supplied_scenes[source_index][1]:
            raise ValueError("训练场景文件SHA-256与冻结场景来源不一致")
        with isolated_rng(scene_seed):
            state = env.reset(seed=scene_seed, scene=_load_scene(source_scene) if source_scene else None)
            scene = env.scene()
        scene_ref = _save_scene(Path(directory) / "scenes" / "train" / f"scene_{episode:07d}.npz", scene)
        done = False
        episode_return = 0.0
        episode_start = time.monotonic()
        length = 0
        episode_terms = {}
        while not done and env_step < training["total_steps"]:
            mask = env.action_mask()
            policy_options = {"action_mask": mask} if mask is not None else {}
            action = agent.act(state, env_step=env_step, **policy_options)
            next_state, reward, done, info = env.step(action)
            if not np.isfinite(reward) or not np.isfinite(next_state).all():
                raise FloatingPointError("Nonfinite environment transition; last checkpoint preserved")
            replay_options = ({"action_mask": mask,
                               "next_action_mask": None if done else env.action_mask()}
                              if mask is not None else {})
            agent.observe(state, action, reward, next_state, done, **replay_options)
            env_step += 1
            length += 1
            episode_return += float(reward)
            for name, values in info.get("reward_terms", {}).items():
                if isinstance(values, dict):
                    for component in ("raw", "scaled", "weighted"):
                        key = f"{name}/{component}"
                        episode_terms[key] = episode_terms.get(key, 0.0) + float(values.get(component, 0.0))
            if agent.ready(env_step, done):
                updates = training["gradient_steps"] if config["algorithm"] != "ppo" else 1
                for _ in range(updates):
                    diagnostics = agent.update()
                    _ensure_finite_agent(agent)
                    _jsonl(Path(directory) / "updates.jsonl", {"env_step": env_step,
                                                                "update_step": agent.update_step,
                                                                "values": diagnostics})
                    if agent.update_step % training["log_interval"] == 0:
                        ledger.record("learning", env_step, diagnostics)
            if env_step % training["log_interval"] == 0:
                elapsed = max(time.monotonic() - session_started, 1e-6)
                ledger.record("runtime", env_step, {"env_steps_per_second": (env_step - session_base_steps) / elapsed,
                                                     "updates_per_second": (agent.update_step - session_base_updates) / elapsed,
                                                     "wall_seconds": base_wall_seconds + elapsed,
                                                     "replay_size": len(agent.replay.rows) if hasattr(agent, "replay") else len(agent.rollout),
                                                     "cuda_allocated_mb": torch.cuda.memory_allocated() / 1048576,
                                                     "cuda_reserved_mb": torch.cuda.memory_reserved() / 1048576,
                                                     **{k: v for k, v in agent.last_action_info.items() if isinstance(v, (int, float))}})
                requested = info.get("requested_action") or {}
                actual = info.get("actual_action") or {}
                ledger.record("action", env_step, {
                    **{f"requested_{key}": requested.get(key) for key in ("freq", "slots", "power_w")},
                    **{f"actual_{key}": actual.get(key) for key in ("freq", "slots", "power_w", "charged_power_w")},
                    "clipped_slots": (requested.get("slots", 0) - actual.get("slots", 0)),
                })
                save_progress()
            if length <= 5 or env_step % max(1, training["log_interval"]) == 0 or done:
                _jsonl(Path(directory) / "traces.jsonl", {"env_step": env_step, "episode": episode,
                                                              "step": length - 1, "scene_sha256": scene_ref["sha256"],
                                                              "requested_action": info.get("requested_action", list(action)),
                                                              "actual_action": info.get("actual_action"),
                                                              "reward": reward, "reward_terms": info.get("reward_terms"),
                                                              "done": done})
            state = next_state
        if config["algorithm"] == "ppo" and agent.rollout:
            diagnostics = agent.update()
            _ensure_finite_agent(agent)
            _jsonl(Path(directory) / "updates.jsonl", {"env_step": env_step,
                                                        "update_step": agent.update_step, "values": diagnostics})
            ledger.record("learning", env_step, diagnostics)
        terminal_metrics = env.metrics()
        episode_row = {"env_step": env_step, "episode": episode, "length": length,
                       "return": episode_return, "duration_seconds": time.monotonic() - episode_start,
                       "scene_seed": scene_seed, "scene": scene_ref, "terminal": bool(done),
                       "source_scene": ({"path": str(source_scene), "sha256": supplied_scenes[source_index][1],
                                         "index": source_index,
                                         "scene_id": scene_sets["train"]["files"][source_index].get("scene_id")
                                         if "train" in scene_sets else source_scene.stem,
                                         "set_id": scene_sets["train"]["id"] if "train" in scene_sets else None}
                                        if source_scene else None),
                       "scene_order": selection,
                       "metrics": terminal_metrics, "reward_terms": episode_terms}
        _jsonl(Path(directory) / "episodes.jsonl", episode_row)
        ledger.record("training", env_step, {"return": episode_return, "episode_length": length,
                                             **{k: v for k, v in terminal_metrics.items() if isinstance(v, (int, float))}})
        ledger.record("reward", env_step, episode_terms)
        episode += 1
        save_progress(scene_order=scene_order.checkpoint_state(episode))
        if _validation_due(config["evaluation"], progress, env_step):
            run_validation()
        stopped = (Path(directory) / "stop.request").exists()
        if stopped or env_step - last_saved_step >= training["checkpoint_interval"] or env_step >= training["total_steps"]:
            _ensure_finite_agent(agent)
            ledger.flush()
            progress.update({"env_step": env_step, "episode": episode, "last_checkpoint_step": env_step,
                             "wall_seconds": base_wall_seconds + time.monotonic() - session_started})
            save_checkpoint(checkpoint, agent, progress, config,
                            {**provenance, "boundary": "next_episode",
                             "scene_order": scene_order.checkpoint_state(episode)})
            ledger.status["checkpoint_step"] = env_step
            snapshot_record = snapshot_artifact(env_step)
            last_saved_step = env_step
            export_policy(model_dir / "latest", agent, config,
                          {**provenance, "env_step": env_step})
            _artifacts(directory, job, checkpoint=str(checkpoint), latest_policy=str(model_dir / "latest"),
                       best_validation_policy=str(model_dir / "best_validation") if (model_dir / "best_validation").exists() else None,
                       **({"learning_snapshots": snapshot_record} if snapshot_record else {}))
            save_progress(checkpoint_step=env_step, best_validation=progress["best_validation"])
            ledger.flush()
        if stopped:
            ledger.event("training_stopped", env_step=env_step)
            save_progress("stopped")
            return
    # Final test is never used to select a model.
    save_progress("evaluating")
    test_rows, test_summary = _fixed_evaluate(config, agent, directory, "test",
                                               config["evaluation"]["test_count"],
                                               config["evaluation"]["test_seed"], env_step, ledger,
                                               scene_set=scene_sets.get("test"))
    export_policy(model_dir / "final", agent, config,
                  {**provenance, "env_step": env_step})
    _atomic_json(Path(directory) / "results.json", {"summary": test_summary, "scenes": test_rows,
                                                    "metadata": {**provenance, "group": "test", "run_id": job["run_id"],
                                                                 "git_commit": job["git_commit"],
                                                                 "config_hash": config_hash(config),
                                                                 "action_mode": config["evaluation"]["action_mode"],
                                                                 "repeats": config["evaluation"]["repeats"],
                                                                 "actual_scene_count": config["evaluation"]["test_count"],
                                                                 "repeated_actions_are_not_independent_scenes": True,
                                                                 "model_selection": "fixed_budget_final"}})
    _artifacts(directory, job, final_policy=str(model_dir / "final"), results=str(Path(directory) / "results.json"))
    ledger.event("training_completed", env_step=env_step)
    ledger.flush()
    save_progress("completed", metrics=test_summary)


def _baseline_environment_config(config):
    """Original rule baselines share physics, not a learned SAC action head.

    Removing the model's feasible-action mask is intentional for the historical
    rule baseline. Its environment must also drop SAC-only structure selectors,
    which require that mask but are never used to build a baseline policy.
    """
    baseline = deepcopy(config)
    baseline["environment"]["action_mask"] = "none"
    baseline["network"]["sac_policy_structure"] = "factorized"
    baseline["network"]["sac_critic_structure"] = "additive"
    return baseline


def _evaluate_job(directory, job, config, ledger, *, device="cuda"):
    from satellite_rl.workbench.environment import WorkbenchEnv
    for reference in job.get("model_files", []):
        if _sha256(reference["path"]) != reference["sha256"]:
            raise ValueError("冻结评价模型校验失败")

    if job.get("legacy_type"):
        if config["environment"].get("profile", "china_legacy") != "china_legacy":
            raise ValueError("中国历史权重不支持欧洲环境身份")
        from satellite_rl.workbench.legacy import load_legacy_policy
        policy = load_legacy_policy(job["model_path"], job["legacy_type"], device=device)
        environment_config = config
    else:
        policy = load_policy(job["model_path"], config=job.get("model_config"), device=device)
        environment_config = policy.config
    scene_set = job.get("scene_sets", {}).get("test")
    if scene_set is not None:
        from satellite_rl.workbench.datasets import assert_scene_profile, scene_set_files
        assert_scene_profile(scene_set, environment_config["environment"].get("profile", "china_legacy"))
        files = [path for path, _ in scene_set_files(scene_set)]
    else:
        # Historical evaluation jobs predate region metadata and freezing.
        if environment_config["environment"].get("profile", "china_legacy") != "china_legacy":
            raise ValueError("欧洲评价任务缺少冻结场景与环境来历")
        files = _scene_files(job["scene_path"])
    results = []
    repeat_count = int(job.get("repeats", 1))
    action_mode = job.get("action_mode", "deterministic")
    baseline_names = list(job.get("baselines", []))
    if len(set(baseline_names)) != len(baseline_names) or any(name not in ("fixed", "greedy") for name in baseline_names):
        raise ValueError("Unknown or repeated baseline")
    budget = len(files) * (repeat_count + len(baseline_names))
    started = time.monotonic()

    def eval_progress(processed, stage="evaluating", **fields):
        elapsed = max(time.monotonic() - started, 1e-9)
        rate = processed / elapsed
        ledger.save_status(stage=stage, env_step=processed, budget_env_steps=budget,
                           steps_per_second=rate, updates_per_second=0,
                           eta_seconds=(budget - processed) / rate if rate > 0 else None,
                           wall_seconds=elapsed,
                           gpu=getattr(ledger, "status", {}).get("preflight", {}).get("gpu"),
                           gpu_memory_mb=torch.cuda.memory_allocated() / 1048576 if torch.cuda.is_available() else 0,
                           **fields)

    eval_progress(0)
    for index, file in enumerate(files):
        scene = _load_scene(file)
        scene_ref = _save_scene(Path(directory) / "scenes" / "test" / f"scene_{index:05d}.npz", scene)
        for repeat in range(repeat_count):
            if (Path(directory) / "stop.request").exists():
                eval_progress(len(results), "stopped")
                return
            seed = config["evaluation"]["test_seed"] + repeat
            with isolated_rng(seed):
                if hasattr(policy, "agent") and hasattr(policy.agent, "rng"):
                    policy.agent.rng.seed(seed + 7321)
                env = WorkbenchEnv(environment_config)
                result = _evaluation_episode(env, policy, scene, f"{file.stem}_r{repeat}", seed,
                                             action_mode, Path(directory) / "traces.jsonl", len(results))
            result["scene"] = {**scene_ref, "source_path": str(file), "source_sha256": _sha256(file)}
            results.append(result)
            _jsonl(Path(directory) / "evaluations.jsonl", {"env_step": len(results),
                                                            "scene_id": result["scene_id"],
                                                            "metrics": result["metrics"],
                                                            "scene_sha256": result["scene"]["sha256"]})
            eval_progress(len(results))
    summary = _summarize_scenes(results)
    ledger.record("test", len(results), summary)
    baselines = {}
    completed_evaluations = len(results)
    if baseline_names:
        from satellite_rl.workbench.baselines import BaselinePolicy
        # Baseline names denote the original rules, not silently masked variants.
        baseline_config = _baseline_environment_config(environment_config)
        for name in baseline_names:
            baseline_scenes = []
            mappings = {}
            for index, file in enumerate(files):
                if (Path(directory) / "stop.request").exists():
                    eval_progress(completed_evaluations, "stopped")
                    return
                scene_ref = _save_scene(Path(directory) / "scenes" / "test" / f"scene_{index:05d}.npz",
                                        _load_scene(file))
                seed = config["evaluation"]["test_seed"]
                with isolated_rng(seed):
                    env = WorkbenchEnv(baseline_config)
                    baseline_policy = BaselinePolicy(name, env)
                    baseline = _evaluation_episode(env, baseline_policy, _load_scene(scene_ref["path"]),
                                                   f"{file.stem}_{name}", seed, "deterministic",
                                                   Path(directory) / "traces.jsonl", completed_evaluations,
                                                   policy_name=name)
                baseline["scene"] = {**scene_ref, "source_path": str(file), "source_sha256": _sha256(file)}
                baseline_scenes.append(baseline)
                completed_evaluations += 1
                for row in baseline_policy.metadata()["power_mappings"]:
                    key = (row["source_power_w"], row["mapped_power_w"])
                    mappings[key] = mappings.get(key, 0) + row["count"]
                _jsonl(Path(directory) / "evaluations.jsonl", {"env_step": completed_evaluations,
                                                                "group": f"baseline_{name}",
                                                                "scene_id": baseline["scene_id"],
                                                                "metrics": baseline["metrics"],
                                                                "scene_sha256": scene_ref["sha256"]})
                eval_progress(completed_evaluations)
            baseline_summary = _summarize_scenes(baseline_scenes)
            mapping_rows = [{"source_power_w": source, "mapped_power_w": mapped, "count": count}
                            for (source, mapped), count in sorted(mappings.items())]
            baselines[name] = {"name": name, "summary": baseline_summary, "scenes": baseline_scenes,
                               "metadata": {"variant": f"workbench_adapted_{name}_v1", "repeats": 1,
                                            "action_mask": "none",
                                            "power_mappings": mapping_rows,
                                            "exact_power_match": all(np.isclose(row["source_power_w"], row["mapped_power_w"])
                                                                     for row in mapping_rows),
                                            "note": "原规则按本次功率档位映射后，在相同实际场景各运行一次。"}}
            ledger.record(f"baseline_{name}", completed_evaluations, baseline_summary)
    result_path = Path(directory) / "results.json"
    _atomic_json(result_path, {"summary": summary, "scenes": results, "baselines": baselines,
                               "metadata": {**_job_provenance(job, environment_config), "model_path": job["model_path"],
                                            "git_commit": job["git_commit"], "action_mode": action_mode,
                                            "repeats": repeat_count}})
    _artifacts(directory, job, results=str(result_path))
    eval_progress(completed_evaluations, "completed", metrics=summary)


def run_job(job_dir, *, resume=False):
    from satellite_rl.workbench.config import validate_config

    directory = Path(job_dir).resolve()
    if _read_json(directory / "config.json", {}).get("research_stage") == "coverage":
        from satellite_rl.coverage.runner import run_coverage_job
        return run_coverage_job(directory, resume=resume)
    job = {}
    config = None
    ledger = None
    try:
        job = json.loads((directory / "job.json").read_text(encoding="utf-8"))
        raw_config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        config = validate_config(raw_config)
        if config["training"]["device"] != "cuda":
            raise ValueError("Persistent workbench jobs require training.device=cuda")
        preflight = strict_cuda_preflight(job)
        ledger = Ledger(directory, job, config, preflight, resume=resume)
        if job["kind"] == "train":
            _training(directory, job, config, ledger, resume)
        elif job["kind"] == "evaluate":
            if resume:
                raise ValueError("Evaluation restart must create a fresh job")
            _evaluate_job(directory, job, config, ledger)
        else:
            raise ValueError(f"Unsupported job kind: {job['kind']}")
        # Surface background writer/flush/close failures inside the normal
        # failure-record path, before run_job can return success.
        ledger.close()
    except BaseException as error:
        failure_traceback = traceback.format_exc()
        if ledger is not None:
            try:
                ledger.close()
            except BaseException:
                pass  # Preserve the original failure; use synchronous reporting below.
        checkpoint_path = directory / "checkpoint.pt"
        prior_status = _read_json(directory / "status.json", {})
        # The checkpoint may already be at a newer boundary when an artifact
        # write fails before status.json is persisted. The in-memory boundary
        # advances only after save_checkpoint returns successfully.
        checkpoint_step = (ledger.status.get("checkpoint_step", prior_status.get("checkpoint_step"))
                           if ledger is not None else prior_status.get("checkpoint_step"))
        failure = {"run_id": job.get("run_id"), "time": now(), "error": str(error),
                   "traceback": failure_traceback, "config_hash": config_hash(config) if config else None,
                   "last_normal_checkpoint": {"path": str(checkpoint_path),
                                              "sha256": _sha256(checkpoint_path),
                                              "env_step": checkpoint_step}
                   if checkpoint_path.exists() else None,
                   "recent_trace": _tail_jsonl(directory / "traces.jsonl", 10),
                   "last_status": prior_status,
                   "recent_scenes": [str(p) for p in sorted((directory / "scenes").rglob("*.npz"))[-3:]]
                   if (directory / "scenes").exists() else []}
        _atomic_json(directory / "failure.json", failure)
        if ledger is None:
            status = {"run_id": job.get("run_id"), "stage": "failed", "env_step": 0,
                      "budget_env_steps": config["training"]["total_steps"] if config else 0,
                      "heartbeat_at": now(),
                      "pid": os.getpid(), "error": str(error), "traceback": failure_traceback,
                      "failure_path": str(directory / "failure.json")}
            _atomic_json(directory / "status.json", status)
        else:
            _jsonl(directory / "events.jsonl", {"time": now(), "run_id": job["run_id"],
                                                "message": "worker_failed", "error": str(error),
                                                "traceback": failure_traceback})
            ledger.status.update(stage="failed", error=str(error), traceback=failure_traceback,
                                 failure_path=str(directory / "failure.json"), heartbeat_at=now())
            _atomic_json(ledger.status_path, ledger.status)
        raise
    finally:
        if ledger:
            ledger.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run_job(args.job_dir, resume=args.resume)


if __name__ == "__main__":
    main()
