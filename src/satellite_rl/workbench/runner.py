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
from satellite_rl.workbench.experiments import is_variable_beam_experiment
from satellite_rl.workbench.learning_snapshots import FORMAT as LEARNING_SNAPSHOT_FORMAT, save_learning_snapshot
from satellite_rl.workbench.scene_order import SceneOrder, VariableBeamSceneOrder


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


def s2p19_pilot_gate_passed(directory, run_id, checkpoint, pilot_stop_steps):
    """Fail closed on a missing, stale or cross-task workbench gate receipt."""
    gate = _read_json(Path(directory) / "pilot_gate.json", {})
    return (gate.get("pass") is True and gate.get("run_id") == run_id and
            gate.get("checkpoint_step") == pilot_stop_steps and
            gate.get("checkpoint_sha256") == _sha256(checkpoint))


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


def _reconcile(directory, checkpoint_step, last_validation_step=None):
    """Discard log rows after the last complete checkpoint on resume."""
    for name in ("metrics", "episodes", "updates", "evaluations", "traces", "count_metrics"):
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
                    row_step = int(row.get("env_step", row.get("step", 0)))
                    group = row.get("group")
                    uncommitted_evaluation = (last_validation_step is not None and
                        ((name in ("traces", "evaluations", "metrics", "count_metrics") and
                          group == "validation" and row_step > last_validation_step) or
                         (name in ("traces", "evaluations") and
                          group not in (None, "training", "validation"))))
                    if row_step <= checkpoint_step and not uncommitted_evaluation:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            replace_with_retry(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def _restore_logging(ledger, directory, checkpoint_step, last_validation_step=None):
    ledger.flush()
    _reconcile(directory, checkpoint_step, last_validation_step)
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
                        env_step=0, policy_name="model", scene_sha256=None, group="test",
                        scene_metadata=None):
    policy_device = getattr(getattr(policy, "agent", policy), "device", None)
    timed_cuda = (torch.cuda.is_available() and
                  isinstance(policy_device, torch.device) and policy_device.type == "cuda")
    if timed_cuda:
        torch.cuda.synchronize()
    episode_started = time.perf_counter()
    state = env.reset(seed=seed, scene=scene)
    reset_history = getattr(policy, "reset_episode", None)
    if reset_history is not None:
        reset_history()
    preparation_seconds = time.perf_counter() - episode_started
    reward_total = 0.0
    trace = []
    done = False
    step = 0
    inference_seconds = 0.0
    while not done:
        mask = env.action_mask()
        policy_options = {"action_mask": mask} if mask is not None else {}
        if getattr(policy, "variable_width", False):
            # 逐束实体档位：观测按束数上限补零，必须把本场景真实束数传给编码器，
            # 否则补零行会参与注意力池化与 N 条件调制。
            policy_options["beam_entity_count"] = int(env.raw_env.beam_num)
        if timed_cuda:
            torch.cuda.synchronize()
        action_started = time.perf_counter()
        action = policy.act(state, deterministic=action_mode == "deterministic", **policy_options)
        if timed_cuda:
            torch.cuda.synchronize()
        action_seconds = time.perf_counter() - action_started
        inference_seconds += action_seconds
        next_state, reward, done, info = env.step(action)
        reward_total += float(reward)
        row = {"scene_id": scene_id, "scene_sha256": scene_sha256, "group": group,
               "step": step, "env_step": env_step, "policy": policy_name,
               "beam_count": len(scene["rate"]),
               "mother_scene_id": (scene_metadata or {}).get("mother_scene_id"),
               "load_multiplier": (scene_metadata or {}).get("load_multiplier"),
               "requested_action": info.get("requested_action", list(action)),
               "actual_action": info.get("actual_action"), "reward": reward,
               "reward_terms": info.get("reward_terms"), "done": done,
               "beam_id": info.get("beam_id"),
               "beam_service_step": info.get("beam_service_step"),
               "service_timing": "after_action_before_later_interference",
               "action_mask": info.get("action_mask", "none"),
               "legal_action_count": info.get("legal_action_count"),
               "policy_action_info": getattr(policy, "last_action_info", {})}
        row["inference_seconds"] = action_seconds
        trace.append(_jsonable(row))
        if trace_path:
            _jsonl(trace_path, row)
        state = next_state
        step += 1
        if step > 1000:
            raise RuntimeError("Evaluation environment did not terminate")
    verification_started = time.perf_counter()
    final_beams = env.beam_results()
    metrics = {**env.metrics(beams=final_beams), "return": reward_total}
    verification_seconds = time.perf_counter() - verification_started
    total_seconds = time.perf_counter() - episode_started
    return {"scene_id": scene_id, "metrics": metrics,
            "beams": final_beams,
            "beam_count": len(final_beams), "scene_metadata": scene_metadata,
            "inference_seconds": inference_seconds,
            "initial_preparation_seconds": preparation_seconds,
            "initial_verification_seconds": verification_seconds,
            "initial_allocation_seconds": total_seconds,
            "service_timing": "after_last_recorded_action_including_later_interference",
            "trace": trace}


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


def _zero_validation_case(path, contract, result=None):
    """Immutable variable-beam validation@0 allocation, bound to the initial checkpoint."""
    path = Path(path)
    if path.exists():
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            body = envelope["body"]
            actual = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
            if (envelope.get("sha256") != actual or body.get("contract") != contract or
                    not isinstance(body.get("result"), dict)):
                raise ValueError("可变束数零步验证缓存SHA或冻结合同不一致")
            saved = body["result"]
            scene = saved.get("scene") or {}
            if (saved.get("action_seed") != contract["action_seed"] or
                    scene.get("source_sha256") != contract["source_scene_sha256"] or
                    _sha256(scene["path"]) != scene.get("sha256") or
                    not isinstance(saved.get("trace"), list) or not saved["trace"]):
                raise ValueError("可变束数零步验证缓存案例身份或归档场景损坏")
            return saved
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"可变束数零步验证缓存损坏，拒绝重算：{path}") from error
    if result is None:
        return None
    body = {"contract": contract, "result": _jsonable(result)}
    digest = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    _atomic_json(path, {"body": body, "sha256": digest})
    return body["result"]


def _s2p19_validation_case(path, contract, result=None):
    """Freeze one complete inline allocation for later SCA without inference."""
    path = Path(path)
    if path.exists():
        envelope = _read_json(path, {})
        body = envelope.get("body") or {}
        saved = body.get("result") or {}
        payload = json.dumps(body, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        if (envelope.get("sha256") != hashlib.sha256(payload).hexdigest() or
                body.get("contract") != contract or
                saved.get("action_seed") != contract["action_seed"] or
                (saved.get("scene") or {}).get("source_sha256") != contract["source_scene_sha256"] or
                len(saved.get("beams") or []) != saved.get("beam_count") or
                len(saved.get("trace") or []) != saved.get("beam_count") or
                saved.get("initial_allocation_seconds") is None):
            raise ValueError("S2-P19 inline验证初解case损坏或来源不符；拒绝重算覆盖")
        return saved
    if result is None:
        return None
    body = {"contract": contract, "result": _jsonable(result)}
    payload = json.dumps(body, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    _atomic_json(path, {"body": body, "sha256": hashlib.sha256(payload).hexdigest()})
    return body["result"]


def _policy_parameters_sha256(agent):
    """Deterministic actor snapshot digest; copies no RNG or optimizer state."""
    digest = hashlib.sha256()
    for name, tensor in sorted(agent.policy_network().state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _verify_c1_demonstration_provenance(job, config, train_record, scene_order, agent):
    """Bind C1 to one completed workbench teacher, not arbitrary manifest bytes."""
    from .s2p19_service import select_teacher_scenes
    from .storage import validate_id

    source = job.get("demonstration_source") or {}
    required = ("run_id", "job_path", "job_sha256", "result_path", "result_sha256",
                "manifest_path", "manifest_sha256")
    if set(source) != set(required):
        raise ValueError("C1任务缺完整教师工作台来源合同")
    source_id = validate_id(source["run_id"])
    archive = Path(job["archive_root"]).resolve(strict=True)
    job_path = Path(source["job_path"]).resolve(strict=True)
    teacher_dir = job_path.parent
    if (teacher_dir.name != source_id or teacher_dir.parent != archive / "outputs" / "workbench" / "jobs" or
            job_path.name != "job.json" or _sha256(job_path) != source["job_sha256"]):
        raise ValueError("C1教师job身份、归属或SHA不符")
    teacher_job = _read_json(job_path, {})
    teacher_status = _read_json(teacher_dir / "status.json", {})
    if (teacher_job.get("run_id") != source_id or teacher_job.get("kind") != "prepare" or
            teacher_job.get("experiment_id") != "S2-P19" or
            (teacher_job.get("scene_sets") or {}).get("train", {}).get("id") != train_record["id"] or
            teacher_status.get("stage") != "completed"):
        raise ValueError("C1教师工作台任务未完成或训练场景不一致")
    result_path = Path(source["result_path"]).resolve(strict=True)
    manifest_path = Path(source["manifest_path"]).resolve(strict=True)
    if (result_path.parent != teacher_dir or result_path.name not in ("preparation_result.json", "results.json") or
            manifest_path != teacher_dir / "teacher" / "manifest.json" or
            _sha256(result_path) != source["result_sha256"] or
            _sha256(manifest_path) != source["manifest_sha256"]):
        raise ValueError("C1教师结果或manifest路径/SHA不符")
    result = _read_json(result_path, {})
    summary = result.get("summary", result)
    demo = config["training"]["demonstrations"]
    if (summary.get("manifest_path") != str(manifest_path) or
            summary.get("manifest_sha256") != source["manifest_sha256"] or
            summary.get("source_set_sha256") != scene_order.source_set_sha256 or
            demo["manifest_path"] != str(manifest_path) or
            demo["manifest_sha256"] != source["manifest_sha256"] or
            demo["source_set_sha256"] != scene_order.source_set_sha256):
        raise ValueError("C1配置/教师结果/冻结训练集来源不一致")
    selected = select_teacher_scenes(train_record)
    teacher_config = (_read_json(teacher_dir / "config.json", {}).get("preparation") or {})
    if (teacher_config.get("preparation_type") != "demonstrations" or
            teacher_config.get("selected_sha256") != selected or
            [row["scene_sha256"] for row in agent.demonstrations.records] != selected):
        raise ValueError("C1教师清单不是冻结训练集逐母预选的576条有序轨迹")


def _fixed_evaluate(config, agent, directory, group, count, seed_base, env_step, ledger,
                    scene_set=None, zero_cache=None):
    from satellite_rl.workbench.environment import WorkbenchEnv
    files = None
    if scene_set is not None:
        from satellite_rl.workbench.datasets import assert_scene_profile, scene_set_files
        assert_scene_profile(scene_set, config["environment"].get("profile", "china_legacy"))
        files = scene_set_files(scene_set)
        if len(files) != count:
            raise ValueError("冻结评价场景数量与配置不一致")

    results = []
    p19_policy_sha = (_policy_parameters_sha256(agent)
                      if config.get("experiment", {}).get("id") == "S2-P19" and group == "validation"
                      else None)
    if zero_cache is not None and (group != "validation" or env_step != 0 or
                                   config["training"].get("beam_count_mode", "none") == "none" or
                                   config["evaluation"]["repeats"] != 1):
        raise ValueError("零步案例缓存仅支持可变束数单动作验证")
    for index in range(count):
        seed = seed_base + index
        scene_path = Path(directory) / "scenes" / group / f"scene_{seed}.npz"
        cache_contract = ({"checkpoint_sha256": zero_cache["checkpoint_sha256"],
                           "config_hash": zero_cache["config_hash"],
                           "scene_index": index, "source_scene_sha256": files[index][1],
                           "action_seed": seed, "group": "validation", "env_step": 0}
                          if zero_cache is not None else None)
        cache_path = (Path(directory) / "validation_zero_cases" / f"scene_{index:05d}.json"
                      if zero_cache is not None else None)
        cached = _zero_validation_case(cache_path, cache_contract) if cache_path else None
        if cached is not None:
            result = cached
            scene_ref = result["scene"]
            for trace in result["trace"]:
                _jsonl(Path(directory) / "traces.jsonl", trace)
            _jsonl(Path(directory) / "evaluations.jsonl", {"env_step": 0, "group": group,
                "scene_id": result["scene_id"], "action_seed": seed,
                "metrics": result["metrics"], "beam_count": result["beam_count"],
                "scene_metadata": result["scene_metadata"],
                "inference_seconds": result["inference_seconds"],
                "initial_preparation_seconds": result["initial_preparation_seconds"],
                "initial_verification_seconds": result["initial_verification_seconds"],
                "initial_allocation_seconds": result["initial_allocation_seconds"],
                "beams_final": result["beams"], "service_timing": result["service_timing"],
                "scene_sha256": scene_ref["sha256"]})
            results.append(result)
            continue
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
            p19_case = (Path(directory) / "validation_cases" / f"step_{env_step:09d}" /
                        f"scene_{index:05d}_r{repeat}.json"
                        if config.get("experiment", {}).get("id") == "S2-P19" and group == "validation"
                        else None)
            p19_contract = ({"config_hash": config_hash(config), "env_step": env_step,
                             "scene_index": index, "repeat_index": repeat,
                             "source_scene_sha256": files[index][1], "action_seed": action_seed,
                             "policy_sha256": p19_policy_sha}
                            if p19_case is not None else None)
            result = (_s2p19_validation_case(p19_case, p19_contract)
                      if p19_case is not None else None)
            if result is None:
                with isolated_rng(action_seed):
                    agent_rng = agent.rng.getstate()
                    agent.rng.seed(action_seed + 7321)
                    try:
                        env = WorkbenchEnv(config)
                        result = _evaluation_episode(env, agent, scene, f"{group}_{seed}_r{repeat}",
                                                     action_seed, config["evaluation"]["action_mode"],
                                                     None if zero_cache is not None or p19_case is not None
                                                     else Path(directory) / "traces.jsonl",
                                                     env_step, scene_sha256=scene_ref["sha256"], group=group,
                                                     scene_metadata=(scene_set["files"][index] if files is not None else None))
                    finally:
                        agent.rng.setstate(agent_rng)
            result["scene"] = scene_ref
            result["action_seed"] = action_seed
            result.setdefault("beam_count", len(scene["rate"]))
            result.setdefault("scene_metadata", scene_set["files"][index] if files is not None else None)
            result.setdefault("inference_seconds", None)
            result.setdefault("initial_preparation_seconds", None)
            result.setdefault("initial_verification_seconds", None)
            result.setdefault("initial_allocation_seconds", None)
            if zero_cache is not None:
                result = _zero_validation_case(cache_path, cache_contract, result)
                for trace in result["trace"]:
                    _jsonl(Path(directory) / "traces.jsonl", trace)
            if p19_case is not None:
                result = _s2p19_validation_case(p19_case, p19_contract, result)
                for trace in result["trace"]:
                    _jsonl(Path(directory) / "traces.jsonl", trace)
            _jsonl(Path(directory) / "evaluations.jsonl", {"env_step": env_step, "group": group,
                                                            "scene_id": result["scene_id"],
                                                            "action_seed": action_seed,
                                                            "metrics": result["metrics"],
                                                            "beam_count": result["beam_count"],
                                                            "scene_metadata": result["scene_metadata"],
                                                            "inference_seconds": result["inference_seconds"],
                                                            "initial_preparation_seconds": result["initial_preparation_seconds"],
                                                            "initial_verification_seconds": result["initial_verification_seconds"],
                                                            "initial_allocation_seconds": result["initial_allocation_seconds"],
                                                            "beams_final": result["beams"],
                                                            "service_timing": result["service_timing"],
                                                            "scene_sha256": scene_ref["sha256"]})
            results.append(result)
    summary = _summarize_scenes(results)
    ledger.record(group, env_step, summary)
    if any(row.get("scene_metadata", {}).get("beam_count") for row in results if row.get("scene_metadata")):
        by_count = {}
        for row in results:
            n = row["beam_count"]
            by_count.setdefault(n, []).append(row)
        _jsonl(Path(directory) / "count_metrics.jsonl", {
            "group": group, "env_step": env_step,
            "by_beam_count": {str(n): {"scene_actions": len(rows),
                                       "metrics": _summarize_scenes(rows)}
                              for n, rows in sorted(by_count.items())}})
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
    pilot_stop = (training["pilot_stop_steps"] if config["experiment"].get("id") == "S2-P19"
                  else None)
    if pilot_stop is not None and resume and pilot_stop <= env_step < training["total_steps"]:
        if not s2p19_pilot_gate_passed(directory, job["run_id"], checkpoint, pilot_stop):
            ledger.event("s2p19_pilot_gate_pending", env_step=env_step)
            ledger.save_status(stage="stopped", env_step=env_step,
                               stop_reason="s2p19_pilot_gate_pending",
                               checkpoint_step=env_step)
            return
        ledger.status.pop("stop_reason", None)
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
        if training.get("beam_count_mode", "none") != "none":
            fields.setdefault("beam_count_steps", dict(count_steps))
            fields.setdefault("beam_count_episodes", dict(count_episodes))
            fields.setdefault("beam_bucket_steps", dict(bucket_steps))
            fields.setdefault("beam_bucket_episodes", dict(bucket_episodes))
            fields.setdefault("replay_bucket_sizes", {key: len(value) for key, value in
                                                      agent.replay.bucket_indices.items()})
            fields.setdefault("replay_sampled_by_bucket", dict(agent.replay.sampled_by_bucket))
            if agent.replay.segment_indices is not None:
                # S2-P18 手段E：每桶、每进度段的配额与有放回抽样次数都必须可回读。
                fields.setdefault("progress_strata", agent.replay.progress_strata)
                fields.setdefault("replay_sampled_by_stratum", dict(agent.replay.sampled_by_stratum))
                fields.setdefault("replay_replacement_draws_by_stratum",
                                  dict(agent.replay.replacement_draws_by_stratum))
                fields.setdefault("replay_stratum_sizes", {
                    key: [len(cell) for cell in bucket]
                    for key, bucket in agent.replay.segment_indices.items()})
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
    variable_mode = training.get("beam_count_mode", "none")
    if variable_mode != "none":
        if "train" not in scene_sets or training.get("scene_order") != "variable_beams":
            raise ValueError("Variable beam training requires a frozen train set and variable_beams order")
        scene_order = VariableBeamSceneOrder(
            variable_mode, training.get("scene_order_seed", 51001),
            [digest for _, digest in supplied_scenes], scene_sets["train"]["files"])
        if config["experiment"].get("id") == "S2-P19" and config["experiment"].get("arm") == "C1":
            demonstration = training["demonstrations"]
            if demonstration["source_set_sha256"] != scene_order.source_set_sha256:
                raise ValueError("C1示范来源与冻结训练场景集SHA不一致")
            source_hashes = {digest for _, digest in supplied_scenes}
            teacher_hashes = {row["scene_sha256"] for row in agent.demonstrations.records}
            if not teacher_hashes.issubset(source_hashes):
                raise ValueError("C1示范含不在冻结训练集中的场景")
            if len(agent.demonstrations.records) != 576 or len(teacher_hashes) != 576:
                raise ValueError("C1示范须为576个不同训练母场景各一条完整轨迹")
            source_metadata = {digest: metadata for (_, digest), metadata in zip(
                supplied_scenes, scene_sets["train"]["files"])}
            mothers = [source_metadata[digest].get("mother_scene_id") for digest in teacher_hashes]
            if None in mothers or len(set(mothers)) != 576:
                raise ValueError("C1示范的576条轨迹未覆盖576个不同训练母场景")
            _verify_c1_demonstration_provenance(job, config, scene_sets["train"], scene_order, agent)
    else:
        scene_order = SceneOrder(training.get("scene_order", "sequential"),
                                 training.get("scene_order_seed", 51001),
                                 [digest for _, digest in supplied_scenes])
    if resume:
        scene_order.verify_checkpoint((restored_environment or {}).get("scene_order"), episode)
    partial_episode = (restored_environment or {}).get("partial_episode") if resume else None
    if partial_episode is not None and variable_mode == "none":
        raise ValueError("Legacy training cannot resume a variable beam partial episode")
    order_provenance = {"mode": scene_order.mode, "seed": scene_order.seed,
                        "source_count": len(supplied_scenes),
                        "source_set_sha256": scene_order.source_set_sha256}
    if variable_mode != "none":
        order_provenance["beam_count_mode"] = variable_mode
    bucket_steps = dict(progress.get("bucket_steps", {}))
    bucket_episodes = dict(progress.get("bucket_episodes", {}))
    count_steps = dict(progress.get("beam_count_steps", {}))
    count_episodes = dict(progress.get("beam_count_episodes", {}))
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
        zero_cache = ({"checkpoint_sha256": _sha256(checkpoint),
                       "config_hash": config_hash(config)}
                      if variable_mode != "none" and env_step == 0 else None)
        _, validation = _fixed_evaluate(config, agent, directory, "validation",
                                         config["evaluation"]["validation_count"],
                                         config["evaluation"]["validation_seed"], env_step, ledger,
                                         scene_set=scene_sets.get("validation"),
                                         zero_cache=zero_cache)
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
        _restore_logging(ledger, directory, env_step, progress.get("last_validation_step", -1))
        if snapshot_record:
            _artifacts(directory, job, learning_snapshots=snapshot_record)
    save_progress(scene_order=scene_order.checkpoint_state(episode))
    ledger.event("training_started" if not resume else "training_resumed", env_step=env_step,
                 config_hash=config_hash(config), training_scene_order=order_provenance)
    last_saved_step = progress.get("last_checkpoint_step", env_step)
    if not resume:
        if variable_mode != "none" and env_step == 0:
            # A recoverable initial state must exist before the 144-scene node.
            save_checkpoint(checkpoint, agent, progress, config,
                            {**provenance, "boundary": "next_episode",
                             "scene_order": scene_order.checkpoint_state(episode)})
            ledger.status["checkpoint_step"] = 0
        if _validation_due(config["evaluation"], progress, env_step):
            run_validation()
        save_checkpoint(checkpoint, agent, progress, config,
                        {**provenance, "boundary": "next_episode",
                         "scene_order": scene_order.checkpoint_state(episode)})
        ledger.status["checkpoint_step"] = env_step
        snapshot_record = snapshot_artifact(env_step)
        _artifacts(directory, job, checkpoint=str(checkpoint),
                   **({"learning_snapshots": snapshot_record} if snapshot_record else {}))
    elif variable_mode != "none" and env_step == 0 and \
            _validation_due(config["evaluation"], progress, env_step):
        # The loaded checkpoint is the pre-validation state; finish node zero
        # before entering the training loop, then commit its completed marker.
        run_validation()
        save_checkpoint(checkpoint, agent, progress, config,
                        {**provenance, "boundary": "next_episode",
                         "scene_order": scene_order.checkpoint_state(episode)})
        ledger.status["checkpoint_step"] = 0
        snapshot_record = snapshot_artifact(0)
        _artifacts(directory, job, checkpoint=str(checkpoint),
                   **({"learning_snapshots": snapshot_record} if snapshot_record else {}))
    while env_step < training["total_steps"]:
        scene_seed = training["seed"] + 100000 + episode
        selection = scene_order.selection(episode)
        replay_bucket = selection.get("bucket") if variable_mode != "none" else None
        source_index = selection["source_index"]
        source_scene = supplied_scenes[source_index][0] if source_index is not None else None
        if source_scene is not None and _sha256(source_scene) != supplied_scenes[source_index][1]:
            raise ValueError("训练场景文件SHA-256与冻结场景来源不一致")
        if partial_episode is None:
            env = WorkbenchEnv(config)
            with isolated_rng(scene_seed):
                state = env.reset(seed=scene_seed, scene=_load_scene(source_scene) if source_scene else None)
                scene = env.scene()
            if variable_mode != "none" and len(scene["rate"]) != selection["beam_count"]:
                raise ValueError("Frozen scene actual beam count differs from training metadata")
            # 逐束实体编码档位：观测宽度随 N 变化，权重按束数上界建立，必须与
            # 配置的 beam_count_max 一致（否则策略输入会在训练中途改变形状）。
            if config["network"].get("context_beam_count", "fixed_100") == "actual":
                from satellite_rl.workbench.config import expected_state_dim
                if state.shape[0] != expected_state_dim(config):
                    raise ValueError(
                        "逐束实体编码要求场景束数等于配置的 beam_count_max；"
                        f"当前场景 {len(scene['rate'])} 束、配置上限 "
                        f"{config['environment']['beam_count_max']}")
            reset_history = getattr(agent, "reset_episode", None)
            if reset_history is not None:
                reset_history()
            scene_ref = _save_scene(Path(directory) / "scenes" / "train" / f"scene_{episode:07d}.npz", scene)
            episode_return, length, episode_terms = 0.0, 0, {}
            episode_start = time.monotonic()
        else:
            if (partial_episode.get("episode") != episode or
                    partial_episode.get("selection") != selection or
                    partial_episode.get("scene_sha256") != supplied_scenes[source_index][1]):
                raise ValueError("Partial episode source differs from frozen checkpoint")
            env = partial_episode["environment"]
            if not isinstance(env, WorkbenchEnv):
                raise ValueError("Partial episode environment type mismatch")
            state = partial_episode["state"]
            scene_ref = partial_episode["scene_ref"]
            if _sha256(scene_ref["path"]) != scene_ref["sha256"]:
                raise ValueError("Partial episode archived scene differs from checkpoint")
            episode_return = partial_episode["return"]
            length = partial_episode["length"]
            episode_terms = partial_episode["reward_terms"]
            episode_start = time.monotonic() - partial_episode["elapsed_seconds"]
            partial_episode = None
        # S2-P18 手段D：按本回合实际束数解析折扣，并写进环境供奖励整形使用。
        beam_count_for_gamma = (selection.get("beam_count")
                                or (len(scene["rate"]) if "rate" in scene else None)
                                or getattr(getattr(env, "raw_env", None), "beam_num", None))
        gamma_lookup = getattr(agent, "gamma_for", None)
        effective_gamma = (gamma_lookup(beam_count_for_gamma) if gamma_lookup is not None
                           else float(training["gamma"]))
        env._effective_gamma = effective_gamma
        done = False
        while not done and env_step < training["total_steps"]:
            mask = env.action_mask()
            progress_segment = None
            replay = getattr(agent, "replay", None)
            if getattr(replay, "segment_indices", None) is not None:
                # 回合内进度分位：用掩码前可见的 beam_idx/beam_num 分段。
                progress_segment = min(replay.progress_strata - 1,
                                       int(env.raw_env.beam_idx / env.raw_env.beam_num
                                           * replay.progress_strata))
            policy_options = {"action_mask": mask} if mask is not None else {}
            if getattr(agent, "variable_width", False):
                # 逐束实体档位：观测按束数上限补零，必须把真实束数传给编码器，
                # 否则补零行会参与注意力池化与 N 条件调制。
                if beam_count_for_gamma is None:
                    raise RuntimeError("逐束实体档位缺少真实束数")
                policy_options["beam_entity_count"] = int(beam_count_for_gamma)
            action = agent.act(state, env_step=env_step, **policy_options)
            next_state, reward, done, info = env.step(action)
            if not np.isfinite(reward) or not np.isfinite(next_state).all():
                raise FloatingPointError("Nonfinite environment transition; last checkpoint preserved")
            replay_options = ({"action_mask": mask,
                               "next_action_mask": None if done else env.action_mask()}
                              if mask is not None else {})
            if getattr(agent, "temporal_enabled", False):
                replay_options["actual_action"] = info["actual_action"]
            if replay_bucket is not None:
                replay_options["replay_bucket"] = replay_bucket
            if progress_segment is not None:
                replay_options["progress_segment"] = progress_segment
            if getattr(agent, "variable_width", False):
                # 逐束实体档位：真实束数随转移写入回放（自然布局观测无法还原束数）。
                replay_options["entity_count"] = int(beam_count_for_gamma)
            agent.observe(state, action, reward, next_state, done, **replay_options)
            env_step += 1
            if replay_bucket is not None:
                bucket_steps[replay_bucket] = bucket_steps.get(replay_bucket, 0) + 1
                count_key = str(selection["beam_count"])
                count_steps[count_key] = count_steps.get(count_key, 0) + 1
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
            if variable_mode != "none" and _validation_due(config["evaluation"], progress, env_step):
                run_validation()
            if env_step % training["log_interval"] == 0:
                elapsed = max(time.monotonic() - session_started, 1e-6)
                ledger.record("runtime", env_step, {"env_steps_per_second": (env_step - session_base_steps) / elapsed,
                                                     "updates_per_second": (agent.update_step - session_base_updates) / elapsed,
                                                     "wall_seconds": base_wall_seconds + elapsed,
                                                     "replay_size": len(agent.replay) if hasattr(agent, "replay") else len(agent.rollout),
                                                     **({"replay_bucket_sizes": {key: len(value) for key, value in agent.replay.bucket_indices.items()},
                                                         "replay_sampled_by_bucket": dict(agent.replay.sampled_by_bucket),
                                                         "replay_replacement_draws": dict(agent.replay.replacement_draws)}
                                                        if variable_mode != "none" else {}),
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
            _jsonl(Path(directory) / "traces.jsonl", {"env_step": env_step, "group": "training",
                                                          "episode": episode,
                                                          "step": length - 1, "scene_sha256": scene_ref["sha256"],
                                                          "beam_count": selection.get("beam_count"),
                                                          "replay_bucket": replay_bucket,
                                                          "mother_scene_id": selection.get("mother_scene_id"),
                                                          "load_multiplier": selection.get("load_multiplier"),
                                                          "beam_id": info.get("beam_id"),
                                                          "beam_service_step": info.get("beam_service_step"),
                                                          "service_timing": "after_action_before_later_interference",
                                                          "requested_action": info.get("requested_action", list(action)),
                                                          "actual_action": info.get("actual_action"),
                                                          "reward": reward, "reward_terms": info.get("reward_terms"),
                                                          "done": done})
            state = next_state
            if (variable_mode != "none" and not done and
                    env_step < training["total_steps"] and
                    env_step % training["checkpoint_interval"] == 0):
                _ensure_finite_agent(agent)
                ledger.flush()
                progress.update({"env_step": env_step, "episode": episode,
                                 "last_checkpoint_step": env_step,
                                 "wall_seconds": base_wall_seconds + time.monotonic() - session_started,
                                 "bucket_steps": bucket_steps,
                                 "beam_count_steps": count_steps})
                partial = {"episode": episode, "selection": selection,
                           "scene_sha256": supplied_scenes[source_index][1],
                           "environment": env, "state": state,
                           "scene_ref": scene_ref, "return": episode_return,
                           "length": length, "reward_terms": episode_terms,
                           "elapsed_seconds": time.monotonic() - episode_start}
                save_checkpoint(checkpoint, agent, progress, config,
                                {**provenance, "boundary": "within_episode",
                                 "scene_order": scene_order.checkpoint_state(episode),
                                 "partial_episode": partial})
                ledger.status["checkpoint_step"] = env_step
                last_saved_step = env_step
                snapshot_record = snapshot_artifact(env_step)
                _artifacts(directory, job, checkpoint=str(checkpoint),
                           **({"learning_snapshots": snapshot_record} if snapshot_record else {}))
                save_progress(checkpoint_step=env_step)
                ledger.flush()
                if pilot_stop is not None and env_step == pilot_stop:
                    export_policy(model_dir / "latest", agent, config,
                                  {**provenance, "env_step": env_step,
                                   "selection": "s2p19_pilot_gate_pending"})
                    _artifacts(directory, job, checkpoint=str(checkpoint),
                               latest_policy=str(model_dir / "latest"))
                    ledger.event("s2p19_pilot_gate_pending", env_step=env_step,
                                 checkpoint_sha256=_sha256(checkpoint))
                    save_progress("stopped", stop_reason="s2p19_pilot_gate_pending",
                                  checkpoint_step=env_step)
                    ledger.flush()
                    return
        if config["algorithm"] == "ppo" and agent.rollout:
            diagnostics = agent.update()
            _ensure_finite_agent(agent)
            _jsonl(Path(directory) / "updates.jsonl", {"env_step": env_step,
                                                        "update_step": agent.update_step, "values": diagnostics})
            ledger.record("learning", env_step, diagnostics)
        final_beams = env.beam_results() if hasattr(env, "beam_results") else []
        terminal_metrics = env.metrics(beams=final_beams) if final_beams else env.metrics()
        episode_row = {"env_step": env_step, "episode": episode, "length": length,
                       "return": episode_return, "duration_seconds": time.monotonic() - episode_start,
                       "scene_seed": scene_seed, "scene": scene_ref, "terminal": bool(done),
                       "episode_end_reason": (terminal_metrics.get("termination_reason", "complete")
                                              if done else "training_budget_exhausted"),
                       "source_scene": ({"path": str(source_scene), "sha256": supplied_scenes[source_index][1],
                                         "index": source_index,
                                         "scene_id": scene_sets["train"]["files"][source_index].get("scene_id")
                                         if "train" in scene_sets else source_scene.stem,
                                         "set_id": scene_sets["train"]["id"] if "train" in scene_sets else None}
                                        if source_scene else None),
                       "scene_order": selection,
                       "beam_count": len(final_beams), "replay_bucket": replay_bucket,
                       "scene_metadata": (scene_sets["train"]["files"][source_index]
                                          if variable_mode != "none" else None),
                       "metrics": terminal_metrics,
                       "beams_final": final_beams,
                       "service_timing": "after_last_recorded_action_including_later_interference",
                       "reward_terms": episode_terms}
        _jsonl(Path(directory) / "episodes.jsonl", episode_row)
        ledger.record("training", env_step, {"return": episode_return, "episode_length": length,
                                             **{k: v for k, v in terminal_metrics.items() if isinstance(v, (int, float))}})
        ledger.record("reward", env_step, episode_terms)
        episode += 1
        if replay_bucket is not None:
            bucket_episodes[replay_bucket] = bucket_episodes.get(replay_bucket, 0) + 1
            count_key = str(selection["beam_count"])
            count_episodes[count_key] = count_episodes.get(count_key, 0) + 1
            progress.update(bucket_steps=bucket_steps, bucket_episodes=bucket_episodes,
                            beam_count_steps=count_steps, beam_count_episodes=count_episodes)
        save_progress(scene_order=scene_order.checkpoint_state(episode))
        if variable_mode == "none" and _validation_due(config["evaluation"], progress, env_step):
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
        if pilot_stop is not None and env_step == pilot_stop:
            ledger.event("s2p19_pilot_gate_pending", env_step=env_step,
                         checkpoint_sha256=_sha256(checkpoint))
            save_progress("stopped", stop_reason="s2p19_pilot_gate_pending",
                          checkpoint_step=env_step)
            ledger.flush()
            return
        if stopped:
            ledger.event("training_stopped", env_step=env_step)
            save_progress("stopped")
            return
    # The approved V0/V1 assessment is a separate frozen 440-scene job. A
    # default workbench test_count would otherwise add unapproved allocations.
    if variable_mode != "none":
        export_policy(model_dir / "final", agent, config,
                      {**provenance, "env_step": env_step,
                       "selection": "fixed_budget_final"})
        result_path = Path(directory) / "results.json"
        _atomic_json(result_path, {"summary": None, "scenes": [],
                                   "metadata": {**provenance, "group": "training_endpoint",
                                                "run_id": job["run_id"],
                                                "git_commit": job["git_commit"],
                                                "config_hash": config_hash(config),
                                                "model_selection": "fixed_budget_final",
                                                "independent_assessment_pending": True}})
        _artifacts(directory, job, final_policy=str(model_dir / "final"), results=str(result_path))
        ledger.event("training_completed_assessment_pending", env_step=env_step)
        ledger.flush()
        save_progress("completed", metrics=None, independent_assessment_pending=True)
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

    Historical 100-beam rules retain their unmasked environment. The explicit
    200-beam adaptation projects their requested actions into the same feasible
    resource set as the learned policies.
    """
    baseline = deepcopy(config)
    if baseline["environment"].get("beam_count_max", 100) == 200:
        baseline["environment"]["action_mask"] = "joint_feasible"
    else:
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
        overrides = job.get("environment_overrides")
        if overrides is not None:
            if (not isinstance(overrides, dict) or overrides != {"beam_count_max": 200} or
                    environment_config["environment"].get("profile") != "europe_v1" or
                    policy.metadata.get("state_dim") != 207 or
                    policy.metadata.get("provenance", {}).get("run_id") !=
                    "train_sac_20261001_162126_33e89e95"):
                raise ValueError("Unsupported model evaluation environment override")
            environment_config = deepcopy(environment_config)
            environment_config["environment"]["beam_count_max"] = 200
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
    action_seed_base = int(job.get("action_seed", config["evaluation"]["test_seed"]))
    if action_seed_base < 0 or action_seed_base + repeat_count - 1 > 2 ** 32 - 1:
        raise ValueError("Evaluation action seed is outside the supported range")
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
        source_metadata = scene_set["files"][index] if scene_set is not None else None
        scene_ref = _save_scene(Path(directory) / "scenes" / "test" / f"scene_{index:05d}.npz", scene)
        for repeat in range(repeat_count):
            if (Path(directory) / "stop.request").exists():
                eval_progress(len(results), "stopped")
                return
            seed = action_seed_base + repeat
            with isolated_rng(seed):
                if hasattr(policy, "agent") and hasattr(policy.agent, "rng"):
                    policy.agent.rng.seed(seed + 7321)
                env = WorkbenchEnv(environment_config)
                result = _evaluation_episode(env, policy, scene, f"{file.stem}_r{repeat}", seed,
                                             action_mode, Path(directory) / "traces.jsonl", len(results),
                                             scene_sha256=scene_ref["sha256"],
                                             scene_metadata=source_metadata)
            result["scene"] = {**scene_ref, "source_path": str(file), "source_sha256": _sha256(file)}
            result.setdefault("beam_count", len(scene["rate"]))
            result.setdefault("scene_metadata", source_metadata)
            result.setdefault("inference_seconds", None)
            result.setdefault("initial_preparation_seconds", None)
            result.setdefault("initial_verification_seconds", None)
            result.setdefault("initial_allocation_seconds", None)
            result["repeat_index"] = repeat
            result["action_seed"] = seed
            results.append(result)
            _jsonl(Path(directory) / "evaluations.jsonl", {"env_step": len(results),
                                                            "group": "test",
                                                            "scene_id": result["scene_id"],
                                                            "repeat_index": repeat,
                                                            "action_seed": seed,
                                                            "metrics": result["metrics"],
                                                            "beam_count": result["beam_count"],
                                                            "scene_metadata": result["scene_metadata"],
                                                            "inference_seconds": result["inference_seconds"],
                                                            "initial_preparation_seconds": result["initial_preparation_seconds"],
                                                            "initial_verification_seconds": result["initial_verification_seconds"],
                                                            "initial_allocation_seconds": result["initial_allocation_seconds"],
                                                            "beams_final": result["beams"],
                                                            "service_timing": result["service_timing"],
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
                baseline_scene = _load_scene(file)
                scene_ref = _save_scene(Path(directory) / "scenes" / "test" / f"scene_{index:05d}.npz",
                                        baseline_scene)
                source_metadata = scene_set["files"][index] if scene_set is not None else None
                seed = action_seed_base
                with isolated_rng(seed):
                    env = WorkbenchEnv(baseline_config)
                    baseline_policy = BaselinePolicy(name, env)
                    baseline = _evaluation_episode(env, baseline_policy, baseline_scene,
                                                   f"{file.stem}_{name}", seed, "deterministic",
                                                   Path(directory) / "traces.jsonl", completed_evaluations,
                                                   policy_name=name, scene_sha256=scene_ref["sha256"],
                                                   group=f"baseline_{name}", scene_metadata=source_metadata)
                baseline["scene"] = {**scene_ref, "source_path": str(file), "source_sha256": _sha256(file)}
                baseline.setdefault("beam_count", len(baseline_scene["rate"]))
                baseline.setdefault("scene_metadata", source_metadata)
                baseline.setdefault("inference_seconds", None)
                baseline.setdefault("initial_preparation_seconds", None)
                baseline.setdefault("initial_verification_seconds", None)
                baseline.setdefault("initial_allocation_seconds", None)
                baseline["repeat_index"] = 0
                baseline["action_seed"] = seed
                baseline_scenes.append(baseline)
                completed_evaluations += 1
                for row in baseline_policy.metadata()["power_mappings"]:
                    key = (row["source_power_w"], row["mapped_power_w"])
                    mappings[key] = mappings.get(key, 0) + row["count"]
                _jsonl(Path(directory) / "evaluations.jsonl", {"env_step": completed_evaluations,
                                                                "group": f"baseline_{name}",
                                                                "scene_id": baseline["scene_id"],
                                                                "repeat_index": 0,
                                                                "action_seed": seed,
                                                                "metrics": baseline["metrics"],
                                                                "beam_count": baseline["beam_count"],
                                                                "scene_metadata": baseline["scene_metadata"],
                                                                "inference_seconds": baseline["inference_seconds"],
                                                                "initial_preparation_seconds": baseline["initial_preparation_seconds"],
                                                                "initial_verification_seconds": baseline["initial_verification_seconds"],
                                                                "initial_allocation_seconds": baseline["initial_allocation_seconds"],
                                                                "beams_final": baseline["beams"],
                                                                "service_timing": baseline["service_timing"],
                                                                "scene_sha256": scene_ref["sha256"]})
                eval_progress(completed_evaluations)
            baseline_summary = _summarize_scenes(baseline_scenes)
            mapping_rows = [{"source_power_w": source, "mapped_power_w": mapped, "count": count}
                            for (source, mapped), count in sorted(mappings.items())]
            feasible_baseline = baseline_config["environment"]["action_mask"] == "joint_feasible"
            baselines[name] = {"name": name, "summary": baseline_summary, "scenes": baseline_scenes,
                               "metadata": {"variant": (f"workbench_feasible_{name}_v2" if feasible_baseline
                                                        else f"workbench_adapted_{name}_v1"), "repeats": 1,
                                            "action_mask": baseline_config["environment"]["action_mask"],
                                            "power_mappings": mapping_rows,
                                            "exact_power_match": all(np.isclose(row["source_power_w"], row["mapped_power_w"])
                                                                     for row in mapping_rows),
                                            "note": ("原规则请求动作投影至同等可行资源；保存请求与实际动作。"
                                                     if feasible_baseline else
                                                     "原规则按本次功率档位映射后，在相同实际场景各运行一次。")}}
            ledger.record(f"baseline_{name}", completed_evaluations, baseline_summary)
    result_path = Path(directory) / "results.json"
    _atomic_json(result_path, {"summary": summary, "scenes": results, "baselines": baselines,
                               "metadata": {**_job_provenance(job, environment_config), "model_path": job["model_path"],
                                            "git_commit": job["git_commit"], "action_mode": action_mode,
                                            "repeats": repeat_count,
                                            "action_seed_base": action_seed_base,
                                            "environment_overrides": job.get("environment_overrides")}})
    _artifacts(directory, job, results=str(result_path))
    eval_progress(completed_evaluations, "completed", metrics=summary)


def _validate_variable_assessment_scene_set(experiment_id, scene_set, files):
    """Keep P17/P18 440 and require the exact new P19 40×11×2 test grid."""
    records = scene_set.get("files") or []
    expected_count = 880 if experiment_id == "S2-P19" else 440
    if len(files) != expected_count or len(records) != expected_count:
        raise ValueError(f"{experiment_id} independent assessment requires all {expected_count} frozen scenarios")
    counts = (50, 60, 75, 90, 100, 125, 140, 150, 175, 180, 200)
    for index, (_, digest) in enumerate(files):
        record = records[index]
        if (record.get("beam_count") not in counts or
                record.get("partition_role") != "test" or
                record.get("sha256") != digest):
            raise ValueError(f"{experiment_id} frozen scenario count/role/SHA differs")
    if experiment_id != "S2-P19":
        return
    source = scene_set.get("source") or {}
    dates = ("2024-02-01", "2024-02-02")
    if (source.get("type") != "aviation_variable_beams_s2p19" or
            source.get("experiment_id") != "S2-P19" or
            source.get("partition_role") != "test" or
            source.get("mother_scene_count") != 40 or
            source.get("source_dates") != list(dates) or
            source.get("snapshot_dates") != list(dates) or
            source.get("load_multipliers") != [0.75, 1.25]):
        raise ValueError("S2-P19考核来源不是冻结的Feb01/02双日期40母test分区")
    mothers = source.get("mother_scene_inputs") or []
    if len(mothers) != 40:
        raise ValueError("S2-P19考核缺40个冻结母场景来源")
    expected = {}
    date_counts = {day: 0 for day in dates}
    for index, mother in enumerate(mothers):
        stamp = mother.get("selected_time_utc")
        digest = mother.get("sha256")
        day = stamp[:10] if isinstance(stamp, str) else None
        if (day not in date_counts or not isinstance(stamp, str) or not stamp.endswith("Z") or
                not isinstance(digest, str) or len(digest) != 64 or
                any(char not in "0123456789abcdef" for char in digest) or
                not isinstance(mother.get("path"), str) or not mother["path"]):
            raise ValueError("S2-P19母场景日期、源路径或SHA不完整")
        try:
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("S2-P19母场景UTC时间戳格式无效") from error
        if parsed.tzinfo != timezone.utc:
            raise ValueError("S2-P19母场景时间戳须为UTC")
        date_counts[day] += 1
        mother_id = f"final_{index:05d}_{hashlib.sha256(stamp.encode()).hexdigest()[:12]}"
        if mother_id in expected:
            raise ValueError("S2-P19母场景身份重复")
        expected[mother_id] = (index, stamp, digest)
    timestamps = [row[1] for row in expected.values()]
    if (date_counts != {day: 20 for day in dates} or len(set(timestamps)) != 40 or
            timestamps != sorted(timestamps)):
        raise ValueError("S2-P19考核母时刻须两日期各20个且全局不重复")
    combinations = {mother_id: set() for mother_id in expected}
    for record in records:
        mother_id = record.get("mother_scene_id")
        if mother_id not in expected:
            raise ValueError("S2-P19衍生场景未绑定40个冻结母场景")
        _, stamp, mother_sha = expected[mother_id]
        n, load = record["beam_count"], record.get("load_multiplier")
        if type(n) is not int or type(load) not in (int, float) or load not in (0.75, 1.25):
            raise ValueError("S2-P19场景束数或负载档位无效")
        # import_scenes and freeze_scene_set deliberately rename scene_id;
        # scientific identity is the mother UTC/source SHA/N/load tuple plus
        # the verified bytes of each imported/frozen NPZ.
        if (record.get("selected_time_utc") != stamp or
                record.get("source_sha256") != mother_sha):
            raise ValueError("S2-P19衍生场景与母时刻、源SHA或身份不一致")
        key = (n, load)
        if key in combinations[mother_id]:
            raise ValueError("S2-P19同母场景有重复N/负载组合")
        combinations[mother_id].add(key)
    required = {(n, load) for n in counts for load in (0.75, 1.25)}
    if any(rows != required for rows in combinations.values()):
        raise ValueError("S2-P19每个母场景必须完整覆盖11束数×2负载")


def _evaluate_variable_job(directory, job, config, ledger, *, device="cuda"):
    """Variable-beam (S2-P17/S2-P18/S2-P19) cases survive interruption unchanged."""
    from satellite_rl.workbench.baselines import BaselinePolicy
    from satellite_rl.workbench.datasets import assert_scene_profile, scene_set_files
    from satellite_rl.workbench.environment import WorkbenchEnv
    from satellite_rl.workbench.experiments import VARIABLE_BEAM_EXPERIMENTS
    from satellite_rl.workbench.variable_evaluation import VariableEvaluationCases

    experiment_id = job.get("experiment_id")
    if job.get("legacy_type") or not is_variable_beam_experiment(experiment_id):
        raise ValueError("可变束数评价需要一个工作台策略：" + "/".join(VARIABLE_BEAM_EXPERIMENTS))
    for reference in job.get("model_files", []):
        if _sha256(reference["path"]) != reference["sha256"]:
            raise ValueError(f"Frozen {experiment_id} model file differs")
    policy = load_policy(job["model_path"], config=job.get("model_config"), device=device)
    environment_config = policy.config
    overrides = job.get("environment_overrides")
    if overrides is not None:
        # 该覆盖只为把固定 100 束布局的冻结 T0（H0）抬到 200 束上限，其身份必须精确匹配；
        # S2-P18 的逐束自然布局不需要覆盖，其配置本身已声明上限，走下面的常规校验。
        if (overrides != {"beam_count_max": 200} or
                environment_config["environment"].get("profile") != "europe_v1" or
                policy.metadata.get("state_dim") != 207 or
                policy.metadata.get("provenance", {}).get("run_id") !=
                "train_sac_20261001_162126_33e89e95"):
            raise ValueError(f"{experiment_id} H0 environment override provenance mismatch")
        environment_config = deepcopy(environment_config)
        environment_config["environment"]["beam_count_max"] = 200
    if environment_config["environment"].get("profile") != "europe_v1" or \
            environment_config["environment"].get("beam_count_max") != 200:
        raise ValueError(f"{experiment_id} model environment must support 200 European beams")
    scene_set = job.get("scene_sets", {}).get("test")
    if not scene_set:
        raise ValueError(f"{experiment_id} evaluation requires a frozen test set")
    assert_scene_profile(scene_set, "europe_v1")
    files = scene_set_files(scene_set)
    _validate_variable_assessment_scene_set(experiment_id, scene_set, files)
    action_seed_base = int(job.get("action_seed", -1))
    repeats = int(job.get("repeats", 0))
    action_mode = job.get("action_mode")
    baseline_names = list(job.get("baselines", []))
    if (action_seed_base != 53001 or repeats != 3 or action_mode != "stochastic" or
            len(set(baseline_names)) != len(baseline_names) or
            any(name not in ("fixed", "greedy") for name in baseline_names)):
        raise ValueError(f"{experiment_id} evaluation action budget or baseline list differs")
    model_root = Path(job["model_path"])
    if model_root.is_file():
        model_root = model_root.parent
    contract = {"experiment_id": experiment_id, "run_id": job["run_id"],
                "git_commit": job["git_commit"], "job_config_hash": config_hash(config),
                "policy_config_hash": config_hash(policy.config),
                "model_path": str(model_root),
                "model_policy_sha256": _sha256(model_root / "policy.pt"),
                "model_metadata_sha256": _sha256(model_root / "model.json"),
                "model_files": job.get("model_files", []),
                "scene_set": scene_set,
                "action_seed_base": action_seed_base, "repeats": repeats,
                "action_mode": action_mode, "baselines": baseline_names,
                "environment_overrides": overrides}
    store = VariableEvaluationCases(directory, _jsonable(contract))
    model_cases = []
    baseline_cases = {name: [] for name in baseline_names}
    for index, (_, digest) in enumerate(files):
        record = scene_set["files"][index]
        for repeat in range(repeats):
            expected = {"kind": "model", "scene_index": index,
                        "scene_sha256": digest, "beam_count": record["beam_count"],
                        "scene_metadata": record,
                        "repeat_index": repeat, "action_seed": action_seed_base + repeat,
                        "action_mode": action_mode}
            model_cases.append((f"model_{index:05d}_{repeat:02d}", expected))
        for name in baseline_names:
            expected = {"kind": name, "scene_index": index,
                        "scene_sha256": digest, "beam_count": record["beam_count"],
                        "scene_metadata": record,
                        "repeat_index": 0, "action_seed": action_seed_base,
                        "action_mode": "deterministic"}
            baseline_cases[name].append((f"{name}_{index:05d}_00", expected))
    plan = model_cases + [case for name in baseline_names for case in baseline_cases[name]]
    budget = len(plan)
    started = time.monotonic()
    new_completed = 0
    reused_completed = 0

    def progress(processed, stage="evaluating", metrics=None):
        elapsed = max(time.monotonic() - started, 1e-9)
        rate = new_completed / elapsed
        ledger.save_status(stage=stage, env_step=processed, budget_env_steps=budget,
                           completed_cases=processed, case_directory=str(store.directory),
                           newly_completed_cases=new_completed,
                           reused_completed_cases=reused_completed,
                           **({"metrics": metrics} if metrics is not None else {}),
                           steps_per_second=rate, updates_per_second=0,
                           eta_seconds=(budget - processed) / rate if rate > 0 else None,
                           wall_seconds=elapsed,
                           gpu=ledger.status.get("preflight", {}).get("gpu"),
                           gpu_memory_mb=torch.cuda.memory_allocated() / 1048576
                           if torch.cuda.is_available() else 0)

    progress(0)
    completed = 0
    for case_id, expected in plan:
        if (Path(directory) / "stop.request").exists():
            progress(completed, "stopped")
            return
        previous = store.load(case_id, expected)
        if previous is None:
            index = expected["scene_index"]
            file, digest = files[index]
            scene = _load_scene(file)
            if len(scene["rate"]) != expected["beam_count"]:
                raise ValueError(f"{experiment_id} frozen scene arrays differ from metadata")
            scene_ref = _save_scene(Path(directory) / "scenes" / "test" /
                                    f"scene_{index:05d}.npz", scene)
            seed = expected["action_seed"]
            with isolated_rng(seed):
                if expected["kind"] == "model":
                    policy.agent.rng.seed(seed + 7321)
                    env = WorkbenchEnv(environment_config)
                    result = _evaluation_episode(
                        env, policy, scene, f"{file.stem}_r{expected['repeat_index']}",
                        seed, action_mode, None, completed,
                        scene_sha256=scene_ref["sha256"],
                        scene_metadata=scene_set["files"][index])
                else:
                    baseline_config = _baseline_environment_config(environment_config)
                    env = WorkbenchEnv(baseline_config)
                    baseline_policy = BaselinePolicy(expected["kind"], env)
                    result = _evaluation_episode(
                        env, baseline_policy, scene, f"{file.stem}_{expected['kind']}",
                        seed, "deterministic", None, completed,
                        policy_name=expected["kind"], scene_sha256=scene_ref["sha256"],
                        group=f"baseline_{expected['kind']}",
                        scene_metadata=scene_set["files"][index])
                    result["baseline_power_mappings"] = baseline_policy.metadata()["power_mappings"]
            result["scene"] = {**scene_ref, "source_path": str(file), "source_sha256": digest}
            result["repeat_index"] = expected["repeat_index"]
            result["action_seed"] = seed
            result["case_id"] = case_id
            result["case_completed_at"] = now()
            store.save(case_id, expected, _jsonable(result))
            new_completed += 1
        else:
            archived = previous["scene"]
            if _sha256(archived["path"]) != archived["sha256"]:
                raise ValueError(f"{experiment_id} completed case archived scene is missing or damaged: {case_id}")
            reused_completed += 1
        completed += 1
        progress(completed)
        if completed % 20 == 0:
            ledger.event("evaluation_cases_progress", completed_cases=completed, total_cases=budget)

    # Build the public result and raw logs from immutable cases in planned order.
    for _ in store.require_all(plan):
        pass
    model_metrics = [{"metrics": store.load(case_id, expected)["metrics"]}
                     for case_id, expected in model_cases]
    model_summary = _summarize_scenes(model_metrics)
    baseline_summaries = {}
    baseline_mappings = {}
    for name in baseline_names:
        rows = []
        mappings = {}
        for case_id, expected in baseline_cases[name]:
            result = store.load(case_id, expected)
            rows.append({"metrics": result["metrics"]})
            for mapping in result.get("baseline_power_mappings", []):
                key = (mapping["source_power_w"], mapping["mapped_power_w"])
                mappings[key] = mappings.get(key, 0) + mapping["count"]
        baseline_summaries[name] = _summarize_scenes(rows)
        baseline_mappings[name] = [dict(source_power_w=a, mapped_power_w=b, count=count)
                                   for (a, b), count in sorted(mappings.items())]
    result_path = Path(directory) / "results.json"
    temp_result = result_path.with_name(result_path.name + ".pending")
    trace_path = Path(directory) / "traces.jsonl"
    eval_path = Path(directory) / "evaluations.jsonl"
    metric_path = Path(directory) / "metrics.jsonl"
    temp_trace = trace_path.with_name(trace_path.name + ".pending")
    temp_eval = eval_path.with_name(eval_path.name + ".pending")
    temp_metric = metric_path.with_name(metric_path.name + ".pending")
    try:
        with temp_result.open("w", encoding="utf-8") as result_stream, \
                temp_trace.open("w", encoding="utf-8") as trace_stream, \
                temp_eval.open("w", encoding="utf-8") as eval_stream, \
                temp_metric.open("w", encoding="utf-8") as metric_stream:
            result_stream.write('{"summary":' + json.dumps(model_summary, ensure_ascii=False, allow_nan=False) + ',"scenes":[')
            for ordinal, (case_id, expected) in enumerate(model_cases):
                result = store.load(case_id, expected)
                row = {key: value for key, value in result.items() if key != "trace"}
                row["case_path"] = str(store.path(case_id))
                result_stream.write(("," if ordinal else "") + json.dumps(row, ensure_ascii=False, allow_nan=False))
            result_stream.write('],"baselines":{')
            for baseline_index, name in enumerate(baseline_names):
                mappings = baseline_mappings[name]
                metadata = {"variant": f"workbench_feasible_{name}_v2", "repeats": 1,
                            "action_mask": "joint_feasible", "power_mappings": mappings,
                            "exact_power_match": all(np.isclose(row["source_power_w"], row["mapped_power_w"])
                                                     for row in mappings),
                            "note": "原规则请求动作投影至同等可行资源；保存请求与实际动作。"}
                prefix = {"name": name, "summary": baseline_summaries[name], "metadata": metadata}
                result_stream.write(("," if baseline_index else "") + json.dumps(name) + ":" +
                                    json.dumps(prefix, ensure_ascii=False, allow_nan=False)[:-1] + ',"scenes":[')
                for ordinal, (case_id, expected) in enumerate(baseline_cases[name]):
                    result = store.load(case_id, expected)
                    row = {key: value for key, value in result.items() if key != "trace"}
                    row["case_path"] = str(store.path(case_id))
                    result_stream.write(("," if ordinal else "") + json.dumps(row, ensure_ascii=False, allow_nan=False))
                result_stream.write("]}")
            result_stream.write('},"metadata":' + json.dumps({**_job_provenance(job, environment_config),
                "experiment_id": experiment_id, "model_path": job["model_path"],
                "git_commit": job["git_commit"], "action_mode": action_mode,
                "repeats": repeats, "action_seed_base": action_seed_base,
                "case_contract_sha256": store.contract_sha256,
                "completed_cases": budget}, ensure_ascii=False, allow_nan=False) + "}")
            for ordinal, (case_id, expected) in enumerate(plan, 1):
                result = store.load(case_id, expected)
                group = "test" if expected["kind"] == "model" else f"baseline_{expected['kind']}"
                eval_row = {"env_step": ordinal, "group": group, "case_id": case_id,
                            "scene_id": result["scene_id"],
                            "repeat_index": result["repeat_index"],
                            "action_seed": result["action_seed"],
                            "metrics": result["metrics"], "beam_count": result["beam_count"],
                            "scene_metadata": result["scene_metadata"],
                            "inference_seconds": result["inference_seconds"],
                            "initial_preparation_seconds": result.get("initial_preparation_seconds"),
                            "initial_verification_seconds": result.get("initial_verification_seconds"),
                            "initial_allocation_seconds": result["initial_allocation_seconds"],
                            "beams_final": result["beams"],
                            "service_timing": result["service_timing"],
                            "scene_sha256": result["scene"]["sha256"]}
                eval_stream.write(json.dumps(eval_row, ensure_ascii=False, allow_nan=False) + "\n")
                for trace in result["trace"]:
                    trace_stream.write(json.dumps({**trace, "case_id": case_id,
                        "repeat_index": result["repeat_index"],
                        "action_seed": result["action_seed"]}, ensure_ascii=False,
                        allow_nan=False) + "\n")
            for group, values in [("test", model_summary)] + [
                    (f"baseline_{name}", baseline_summaries[name]) for name in baseline_names]:
                metric_stream.write(json.dumps({"step": budget, "wall_time": now(),
                    "group": group, "values": values}, ensure_ascii=False,
                    allow_nan=False) + "\n")
            for stream in (result_stream, trace_stream, eval_stream, metric_stream):
                stream.flush()
                os.fsync(stream.fileno())
        replace_with_retry(temp_result, result_path)
        replace_with_retry(temp_trace, trace_path)
        replace_with_retry(temp_eval, eval_path)
        replace_with_retry(temp_metric, metric_path)
    finally:
        for path in (temp_result, temp_trace, temp_eval, temp_metric):
            path.unlink(missing_ok=True)
    _artifacts(directory, job, results=str(result_path), case_directory=str(store.directory))
    progress(budget, "completed", model_summary)


def run_job(job_dir, *, resume=False):
    from satellite_rl.workbench.config import validate_config

    directory = Path(job_dir).resolve()
    research_stage = _read_json(directory / "config.json", {}).get("research_stage")
    if research_stage == "fair20":
        from .fair20_runner import run_job as run_fair20_job
        return run_fair20_job(directory, resume=resume)
    if research_stage == "sac_improvement":
        from satellite_rl.sac_improvement import run_job as run_sac_improvement_job
        return run_sac_improvement_job(directory, resume=resume)
    if research_stage == "s2p19_preparation":
        from satellite_rl.workbench.s2p19_preparation import run_job as run_s2p19_preparation_job
        return run_s2p19_preparation_job(directory, resume=resume)
    if research_stage == "power_refinement":
        from satellite_rl.power_refinement.runner import run_refinement_job
        return run_refinement_job(directory, resume=resume)
    if research_stage == "coverage":
        if _read_json(directory / "config.json", {}).get("schema_version") == 8:
            from .geoload_runner import run_job as run_geoload_job
            return run_geoload_job(directory, resume=resume)
        if _read_json(directory / "config.json", {}).get("schema_version") == 7:
            from satellite_rl.coverage.nondrl.runner import run_nondrl_job
            job = _read_json(directory / "job.json", {})
            config = _read_json(directory / "config.json", {})
            return run_nondrl_job(directory, job, config, resume=resume,
                                  preflight=strict_cuda_preflight(job))
        if _read_json(directory / "config.json", {}).get("schema_version") == 6:
            job = _read_json(directory / "job.json", {})
            config = _read_json(directory / "config.json", {})
            if job.get("kind") == "prepare":
                from satellite_rl.workbench.drl_r04_preparation import run_drl_r04_preparation_job
                return run_drl_r04_preparation_job(directory, job, config, resume=resume,
                                                   preflight=strict_cuda_preflight(job))
            from satellite_rl.coverage.drl_r04_runner import run_drl_r04_job
            return run_drl_r04_job(directory, job, config, resume=resume,
                                   preflight=strict_cuda_preflight(job))
        if _read_json(directory / "config.json", {}).get("schema_version") == 5:
            job = _read_json(directory / "job.json", {})
            config = _read_json(directory / "config.json", {})
            if job.get("kind") == "prepare":
                from satellite_rl.workbench.variable_preparation import run_variable_preparation_job
                return run_variable_preparation_job(directory, job, config,
                                                    resume=resume,
                                                    preflight=strict_cuda_preflight(job))
            from satellite_rl.coverage.variable_runner import run_variable_job
            return run_variable_job(directory, job, config, resume=resume,
                                    preflight=strict_cuda_preflight(job))
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
            if is_variable_beam_experiment(job.get("experiment_id")):
                _evaluate_variable_job(directory, job, config, ledger)
            elif resume:
                raise ValueError("Evaluation restart must create a fresh job")
            else:
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
