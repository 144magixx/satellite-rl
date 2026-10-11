"""Durable S1-ND12 worker with complete-row commits and compressed raw evidence."""
from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from contextlib import contextmanager

import numpy as np

from satellite_rl.coverage.variable_dataset import load_variable_dataset
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.storage import clean_json, digest, now, read_json


def jsonable(value):
    if hasattr(value, "to_record"):
        return jsonable(value.to_record())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return clean_json(value)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


class RowStore:
    """One gzip scene shard per durable commit; manifest is the transaction boundary."""
    def __init__(self, root, provenance, *, resume=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "manifest.json"
        self.provenance = jsonable(provenance)
        old = read_json(self.path)
        if old is not None and not resume:
            raise FileExistsError("existing evidence requires same-ID resume")
        self.manifest = old or {"format": "s1nd12_scene_shards_v1", "provenance": self.provenance, "rows": []}
        if self.manifest.get("provenance") != self.provenance:
            raise ValueError("resume evidence provenance differs")
        for row in self.manifest["rows"]:
            path = self.root / row["file"]
            if digest(path) != row["sha256"]:
                raise ValueError("committed scene evidence SHA differs")
        self.keys = {r["key"] for r in self.manifest["rows"]}
        if len(self.keys) != len(self.manifest["rows"]):
            raise ValueError("duplicate committed scene key")

    def append(self, key, value):
        if key in self.keys:
            raise ValueError("refusing duplicate scene evidence")
        name = f"scene_{len(self.manifest['rows']):06d}.json.gz"
        path, temporary = self.root / name, self.root / (name + ".partial")
        payload = json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        with temporary.open("wb") as stream:
            with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
                compressed.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        self.manifest["rows"].append({"key": key, "file": name, "sha256": digest(path)})
        atomic_json(self.path, self.manifest)
        self.keys.add(key)

    def load(self, row):
        with gzip.open(self.root / row["file"], "rt", encoding="utf-8") as stream:
            return json.load(stream)


def scaled_scene(scene, multiplier):
    value = deepcopy(scene)
    value.positions = np.asarray(scene.positions).copy()
    value.demands = np.asarray(scene.demands).copy() * multiplier
    value.metadata = {**scene.metadata, "demand_multiplier": multiplier, "mother_scene_id": scene.scene_id}
    return value


def _initials(job, expected_sha, key="initial_source"):
    source = job.get(key)
    if not source:
        return {}
    if digest(source["path"]) != source["sha256"]:
        raise ValueError("initial manifest SHA differs")
    manifest = read_json(source["path"])
    if manifest.get("dataset_sha256") != expected_sha or manifest.get("completed") is not True:
        raise ValueError("common initial dataset/completion differs")
    raw = Path(manifest["evidence_path"])
    evidence = RowStore(raw, manifest["provenance"], resume=True)
    return {row["key"]: evidence.load(row) for row in evidence.manifest["rows"]}


def fixed_validation(directory, job, config, provenance, progress, stopped):
    """Return a callback; retain every validation layout and fixed-input hash."""
    from .core import solve
    from .learning import LearningPolicy
    source = config["datasets"]["validation"]
    if digest(source["path"]) != source["sha256"]:
        raise ValueError("fixed validation dataset SHA differs")
    dataset = load_variable_dataset(source["path"])
    initials = _initials(job, source["sha256"], "validation_initial_source")
    route = config["coverage"]["nondrl"]["route"]
    def callback(context):
        model_sha = hashlib.sha256()
        for name, value in context["model"].state_dict().items():
            model_sha.update(name.encode())
            model_sha.update(value.detach().cpu().contiguous().numpy().tobytes())
        node = {**provenance, "updates": context["updates"], "model_sha256": model_sha.hexdigest()}
        store = RowStore(Path(directory) / "validation" / f"update_{context['updates']:06d}", node,
                         resume=(Path(directory) / "validation" / f"update_{context['updates']:06d}" / "manifest.json").exists())
        policy = LearningPolicy(route, context["model"], context["contract"], context["device"], "main")
        total = len(dataset.scenes) * len(config["evaluation"]["demand_multipliers"])
        for index, original in enumerate(dataset.scenes):
            for multiplier in config["evaluation"]["demand_multipliers"]:
                key = f"{index}:{multiplier:g}"
                if key in store.keys:
                    continue
                if stopped():
                    return {"stopped": True, "completed_scenes": len(store.keys), "expected_scenes": total}
                initial = initials.get(key)
                if initial is None:
                    raise ValueError("fixed validation initial is incomplete")
                result = solve(scaled_scene(original, multiplier), route,
                               config["coverage"]["nondrl"]["limits"],
                               initial=initial["result"]["initial_layout"], policy=policy)
                if result["initial_sha256"] != initial["result"]["initial_sha256"]:
                    raise ValueError("validation changed shared initial")
                store.append(key, {"key": key, "dataset_index": index, "mother_scene_id": original.scene_id,
                                  "demand_multiplier": multiplier, "result": result})
                progress(stage="validating", validation_updates=context["updates"],
                         validation_completed_scenes=len(store.keys), validation_total_scenes=total)
        rows = [store.load(row) for row in store.manifest["rows"]]
        mothers = []
        for index in range(len(dataset.scenes)):
            subset = [r for r in rows if r["dataset_index"] == index]
            cvs = [r["result"]["metrics_all"]["load_cv_all"] for r in subset]
            valid = [cv for cv in cvs if cv is not None]
            mothers.append({"dataset_index": index, "cv_mean": float(np.mean(valid)) if valid else None,
                            "all_zero_demand": not valid})
        valid = [m["cv_mean"] for m in mothers if m["cv_mean"] is not None]
        return {"completed_scenes": len(rows), "expected_scenes": total, "mother_count": len(mothers),
                "mother_equal_cv_mean": float(np.mean(valid)) if valid else None,
                "zero_demand_mothers": sum(m["all_zero_demand"] for m in mothers),
                "mother_metrics": mothers, "raw_manifest_path": str(store.path), "raw_manifest_sha256": digest(store.path)}
    return callback


def evaluate(directory, job, config, provenance, *, resume, progress, stopped):
    from .core import solve
    route = config["coverage"]["nondrl"]["route"]
    method = config["coverage"]["nondrl"]["method"]
    mode = job.get("preparation_mode")
    role = job["dataset_role"]
    source = config["datasets"][role]
    if digest(source["path"]) != source["sha256"]:
        raise ValueError("frozen evaluation data SHA differs")
    dataset = load_variable_dataset(source["path"])
    indices = job.get("scene_indices", list(range(len(dataset.scenes))))
    multipliers = config["evaluation"]["demand_multipliers"]
    total = len(indices) * len(multipliers)
    store = RowStore(directory / "raw_scenes", provenance, resume=resume)
    initials = _initials(job, source["sha256"])
    policy = None
    if job.get("model_path"):
        from .learning import load_policy
        for item in job["model_files"]:
            if digest(item["path"]) != item["sha256"]:
                raise ValueError("frozen evaluation model SHA differs")
        policy = load_policy(Path(job["model_path"]) / "policy.pt", expected_provenance={"route": route, "schema_version": 7},
                             device=job.get("execution_device", "cpu"), mode=method)
    for index in indices:
        original = dataset.scenes[index]
        for multiplier in multipliers:
            key = f"{index}:{multiplier:g}"
            if key in store.keys:
                continue
            if stopped():
                return {"stopped": True, "completed_scenes": len(store.keys)}
            scene = scaled_scene(original, multiplier)
            start = time.perf_counter()
            shared = initials.get(key)
            if mode != "initial" and shared is None:
                raise ValueError("common I0 evidence lacks requested scene")
            initial = shared["result"]["initial_layout"] if shared else None
            effective_route = "I0" if mode == "initial" or method == "I0" else "BH" if method == "BH" else route
            result = solve(scene, effective_route, config["coverage"]["nondrl"]["limits"], initial=initial, policy=policy,
                           method="main" if method in ("I0", "BH") or mode == "initial" else method)
            if shared and result["initial_sha256"] != shared["result"]["initial_sha256"]:
                raise ValueError("route changed common I0 layout")
            row = {"key": key, "run_id": job["run_id"], "route": route, "method": method,
                   "dataset_index": index, "mother_scene_id": original.scene_id,
                   "source_hash": original.source_hash, "demand_multiplier": multiplier,
                   "aircraft_background": {"dataset_sha256": source["sha256"], "index": index},
                   "result": result, "measured_task_seconds": time.perf_counter() - start,
                   "shared_initial_record": shared.get("key") if shared else None}
            if shared:
                initial_time = shared["result"]["timing"].get("total_seconds", shared["measured_task_seconds"])
                row["shared_initial_seconds"] = initial_time
                row["end_to_end_segment_sum_seconds"] = initial_time + row["measured_task_seconds"]
            store.append(key, row)
            progress(stage="preparing" if mode == "initial" else "evaluating", completed_scenes=len(store.keys), total_scenes=total,
                     latest_metrics=jsonable(result.get("metrics_all", {})), latest_timing=jsonable(result.get("timing", {})))
    if mode == "initial":
        atomic_json(directory / "initial_manifest.json", {"completed": True, "dataset_sha256": source["sha256"],
                    "evidence_path": str(store.root), "provenance": provenance, "rows": len(store.keys)})
    return {"summary": {"rows": len(store.keys), "mother_count": len(indices), "route": route,
                        "dataset_sha256": source["sha256"], "raw_manifest": str(store.path)}}


@contextmanager
def worker_lock(directory):
    """OS-held job lock releases on process death, with no stale PID guessing."""
    path = Path(directory) / "worker.lock"
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_nondrl_job(directory, job, config, *, resume=False, preflight=None):
    with worker_lock(directory):
        return _run_nondrl_job(directory, job, config, resume=resume, preflight=preflight)


def _run_nondrl_job(directory, job, config, *, resume=False, preflight=None):
    from satellite_rl.workbench.coverage_nondrl import validate_nondrl_config
    directory = Path(directory)
    status = read_json(directory / "status.json", {})
    started = time.perf_counter()
    def progress(**fields):
        status.update(fields, heartbeat_at=now(), elapsed_seconds=time.perf_counter() - started)
        atomic_json(directory / "status.json", jsonable(status))
    def stopped():
        return (directory / "stop.request").exists()
    try:
        config = validate_nondrl_config(config)
        provenance = {"schema_version": 7, "experiment_id": "S1-ND12", "run_id": job["run_id"],
                      "git_commit": job["git_commit"], "source_sha256": preflight.get("manifest_sha256") if preflight else None,
                      "config_sha256": canonical_sha(config), "dataset_sha256": job.get("dataset_sha256", {})}
        progress(stage="initializing", pid=os.getpid(), preflight=preflight, execution_device=job.get("execution_device"),
                 budget_env_steps=0, budget_updates=config["training"]["updates"] if job["kind"] == "train" else 0)
        route = config["coverage"]["nondrl"]["route"]
        if job["kind"] == "prepare" and job.get("preparation_mode") == "dataset":
            from satellite_rl.workbench.nondrl_preparation import run_dataset
            result = run_dataset(directory, job, config, resume=resume, progress=progress, stopped=stopped)
        elif job["kind"] == "prepare" and job.get("preparation_mode") == "teacher":
            from .core import Problem
            from .config import NondrlConfig
            from .teachers import generate_teacher
            role = "development" if route == "05" else "train"
            source = config["datasets"][role]
            if digest(source["path"]) != source["sha256"]:
                raise ValueError("teacher data SHA differs")
            dataset = load_variable_dataset(source["path"])
            limits = {**config["coverage"]["nondrl"]["limits"], "online_seconds": {"01": 1., "02": 10., "03": 4., "04": 6.4, "05": 1.2}[route]}
            def problems():
                for scene in dataset.scenes:
                    for multiplier in config["evaluation"]["demand_multipliers"]:
                        if stopped():
                            return
                        yield Problem(scaled_scene(scene, multiplier), NondrlConfig.from_dict(limits))
            teacher_progress = lambda values: progress(**{**values, "stage": "preparing"})
            result = generate_teacher(problems(), route, directory / "teacher", config["teacher"],
                                      stop_callback=stopped, progress_callback=teacher_progress, provenance=provenance,
                                      resume=resume)
            atomic_json(directory / "teacher_manifest.json", {**jsonable(result),
                        "completed": result.get("status") == "completed", "dataset_sha256": source["sha256"],
                        "evidence_path": result["path"]})
        elif job["kind"] == "train":
            from .learning import train
            source = job["teacher_source"]
            if digest(source["path"]) != source["sha256"]:
                raise ValueError("teacher manifest SHA differs")
            evidence = read_json(source["path"])["evidence_path"]
            gate = job["reachability_source"]
            if digest(gate["path"]) != gate["sha256"]:
                raise ValueError("frozen development gate SHA differs")
            provenance["reachability_sha"] = gate["sha256"]
            result = train(evidence, route, config["training"], job["model_dir"],
                           resume=resume, stop_callback=stopped,
                           validation_callback=fixed_validation(directory, job, config, provenance, progress, stopped),
                           progress_callback=lambda values: progress(**{**values, "stage": "training"}), provenance=provenance)
        else:
            result = evaluate(directory, job, config, provenance, resume=resume, progress=progress, stopped=stopped)
        atomic_json(directory / "results.json", jsonable(result))
        scientific_stop = result.get("status") == "scientific_stop" or result.get("scientific_stop") is True
        is_stopped = result.get("stopped") is True or result.get("status") == "stopped" or stopped() or scientific_stop
        progress(stage="stopped" if is_stopped else "completed", ended_at=now(), scientific_stop=scientific_stop,
                 stop_reason=result.get("stop_reason", "user_stop" if is_stopped else None))
        return result
    except BaseException as error:
        progress(stage="failed", ended_at=now(), error=repr(error))
        atomic_json(directory / "failure.json", {"error": repr(error), "traceback": traceback.format_exc(), "at": now()})
        raise
