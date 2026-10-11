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
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
import multiprocessing

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


def _initial_worker_setup():
    import torch
    torch.set_num_threads(1)


def _initial_worker(payload):
    """Single CPU scene; receipt remains usable if its coordinator dies."""
    from .core import solve
    target = Path(payload["receipt"])
    target.parent.mkdir(parents=True, exist_ok=True)
    with worker_lock(target.parent / target.stem):
        existing = read_json(target)
        if existing:
            if existing.get("provenance_sha256") != payload["provenance_sha256"]:
                raise ValueError("initial worker receipt provenance differs")
            return str(target)
        begin = time.perf_counter()
        scene = payload["scene"]
        result = solve(scaled_scene(scene, payload["multiplier"]), "I0", payload["limits"])
        row = {"key": payload["key"], "run_id": payload["run_id"], "route": payload["route"], "method": "I0",
               "dataset_index": payload["index"], "mother_scene_id": scene.scene_id, "source_hash": scene.source_hash,
               "demand_multiplier": payload["multiplier"], "aircraft_background": {"dataset_sha256": payload["dataset_sha256"], "index": payload["index"]},
               "result": jsonable(result), "measured_task_seconds": time.perf_counter() - begin,
               "worker_pid": os.getpid(), "worker_device": "cpu", "shared_initial_record": None}
        atomic_json(target, {"provenance_sha256": payload["provenance_sha256"], "row": row})
    return str(target)


def parallel_initial(directory, job, config, provenance, dataset, source, indices, store, progress, stopped):
    """Bounded spawn pool, fixed order commits, completed orphan receipts retained."""
    workers = min(config["preparation"]["initial_workers"], len(indices) * 3)
    todo = [(i, m) for i in indices for m in config["evaluation"]["demand_multipliers"]
            if f"{i}:{m:g}" not in store.keys]
    provenance_sha = canonical_sha(provenance)
    total = len(indices) * len(config["evaluation"]["demand_multipliers"])
    pending, cursor = [], 0
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_initial_worker_setup) as pool:
        while cursor < len(todo) or pending:
            while cursor < len(todo) and len(pending) < workers * 2 and not stopped():
                index, multiplier = todo[cursor]
                key = f"{index}:{multiplier:g}"
                receipt = directory / "initial_receipts" / (canonical_sha(key)[:24] + ".json")
                payload = {"scene": dataset.scenes[index], "multiplier": multiplier, "index": index,
                           "key": key, "receipt": str(receipt), "provenance_sha256": provenance_sha,
                           "dataset_sha256": source["sha256"], "limits": config["coverage"]["nondrl"]["limits"],
                           "run_id": job["run_id"], "route": config["coverage"]["nondrl"]["route"]}
                # Reuse a complete orphan receipt without repeating its calculation.
                old = read_json(receipt)
                if old is not None:
                    if old.get("provenance_sha256") != provenance_sha:
                        raise ValueError("initial receipt belongs to another frozen task")
                    pending.append((key, receipt, None))
                else:
                    pending.append((key, receipt, pool.submit(_initial_worker, payload)))
                cursor += 1
            if not pending:
                break
            key, receipt, future = pending[0]
            if future is not None:
                try:
                    future.result(timeout=5)
                except FutureTimeout:
                    progress(stage="preparing", completed_scenes=len(store.keys), total_scenes=total,
                             initial_workers=workers, initial_inflight=len(pending))
                    continue
            value = read_json(receipt)
            if value is None or value.get("provenance_sha256") != provenance_sha or value["row"]["key"] != key:
                raise ValueError("initial worker returned an invalid complete receipt")
            store.append(key, value["row"])
            # Commit first, then remove the temporary complete receipt. Native
            # crashes between these operations are safe because RowStore wins.
            receipt.unlink(missing_ok=True)
            pending.pop(0)
            progress(stage="preparing", completed_scenes=len(store.keys), total_scenes=total,
                     initial_workers=workers, initial_inflight=len(pending))
    return {"stopped": stopped(), "completed_scenes": len(store.keys)}


def _initials(job, expected_sha, key="initial_source"):
    source = job.get(key)
    if not source:
        return {}
    if isinstance(source, list):
        combined = {}
        for value in source:
            rows = _initials({key: value}, expected_sha, key)
            if set(combined).intersection(rows):
                raise ValueError("common initial shard sources overlap")
            combined.update(rows)
        return combined
    if digest(source["path"]) != source["sha256"]:
        raise ValueError("initial manifest SHA differs")
    manifest = read_json(source["path"])
    if manifest.get("dataset_sha256") != expected_sha or manifest.get("completed") is not True:
        raise ValueError("common initial dataset/completion differs")
    raw = Path(manifest["evidence_path"])
    if digest(raw / "manifest.json") != manifest.get("raw_manifest_sha256"):
        raise ValueError("common initial raw manifest SHA differs")
    evidence = RowStore(raw, manifest["provenance"], resume=True)
    return {row["key"]: evidence.load(row) for row in evidence.manifest["rows"]}


def fixed_validation(directory, job, config, provenance, progress, stopped):
    """Return a callback; retain every validation layout and fixed-input hash."""
    from .core import solve, metrics_all
    from .learning import LearningPolicy, audit_ranker
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
                extra = {}
                if route == "03":
                    raw = result.get("raw_layout")
                    try:
                        raw_metrics = metrics_all(scaled_scene(original, multiplier), raw,
                                                  config["coverage"]["nondrl"]["limits"]) if raw is not None else None
                        extra["raw_scene_feasible"] = (raw_metrics is not None and
                            raw_metrics["beam_count"] == result["initial_metrics_all"]["beam_count"] and
                            all(raw_metrics.get(k) is True for k in ("geometry_feasible", "unique_assignment", "demand_conserved")))
                        extra["raw_metrics_all"] = raw_metrics
                    except (ValueError, TypeError, AttributeError) as error:
                        extra.update(raw_scene_feasible=False, raw_error=str(error))
                    if context["updates"] == 2000:
                        extra["HR_result"] = solve(scaled_scene(original, multiplier), "03",
                            config["coverage"]["nondrl"]["limits"], initial=initial["result"]["initial_layout"], method="HR")
                store.append(key, {"key": key, "dataset_index": index, "mother_scene_id": original.scene_id,
                                  "demand_multiplier": multiplier, "result": result, **extra})
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
        summary = {"completed_scenes": len(rows), "expected_scenes": total, "mother_count": len(mothers),
                "mother_equal_cv_mean": float(np.mean(valid)) if valid else None,
                "zero_demand_mothers": sum(m["all_zero_demand"] for m in mothers),
                "mother_metrics": mothers, "raw_manifest_path": str(store.path), "raw_manifest_sha256": digest(store.path)}
        if context["updates"] == 2000 and route == "03":
            if len(rows) != total or any("raw_scene_feasible" not in row or "HR_result" not in row for row in rows):
                return {**summary, "attention_required": True, "reason": "route03 raw/HR pilot evidence incomplete"}
            raw_rate = sum(row["raw_scene_feasible"] for row in rows) / len(rows)
            improved = []
            for index in range(len(dataset.scenes)):
                subset = [row for row in rows if row["dataset_index"] == index]
                model = [row["result"]["metrics_all"]["load_cv_all"] for row in subset]
                baseline = [row["HR_result"]["metrics_all"]["load_cv_all"] for row in subset]
                if all(x is not None for x in model + baseline):
                    base_mean, model_mean = float(np.mean(baseline)), float(np.mean(model))
                    if base_mean > 1e-12 and (base_mean - model_mean) / base_mean > .01:
                        improved.append(index)
            summary.update(pilot={"raw_scene_feasible_fraction": raw_rate, "improved_over_HR_mothers": improved,
                                  "passed": raw_rate >= .8 and len(improved) >= 6})
            if not summary["pilot"]["passed"]:
                summary.update(scientific_stop=True, stop_reason="measured_route03_2000_update_gate_failed")
        if context["updates"] == 2000 and route == "04":
            source = job.get("ranking_audit_source")
            if not source or digest(source["path"]) != source["sha256"]:
                return {**summary, "attention_required": True, "reason": "fixed route04 candidate evidence source missing"}
            wrapper = read_json(source["path"])
            path = Path(wrapper["evidence_path"])
            if digest(path) != wrapper["sha256"]:
                return {**summary, "attention_required": True, "reason": "fixed route04 candidate manifest SHA differs"}
            manifest = read_json(path)
            audits = []
            for shard in manifest["shards"]:
                file = path.parent / shard["path"]
                if digest(file) != shard["sha256"]:
                    return {**summary, "attention_required": True, "reason": "fixed route04 candidate shard SHA differs"}
                with gzip.open(file, "rt", encoding="utf-8") as stream:
                    for line in stream:
                        sample = json.loads(line)
                        audit = audit_ranker(policy, sample, topk=8)
                        audits.append({"mother_id": sample["mother_id"], **audit})
            atomic_json(store.root / "ranking_audit.json", audits)
            if len(audits) != len(dataset.scenes) or any(a.get("complete") is not True for a in audits):
                return {**summary, "attention_required": True, "reason": "route04 fixed labelled pools incomplete or omitted mothers",
                        "ranking_audit_path": str(store.root / "ranking_audit.json")}
            relevant = [a for a in audits if a.get("model_topk_recall") is not None]
            recall_gain = float(np.mean([a["model_topk_recall"] - a["rule_topk_recall"] for a in relevant])) if relevant else None
            best_gain = float(np.mean([a["model_completed_best_improvement"] - a["rule_completed_best_improvement"] for a in audits]))
            passed = recall_gain is not None and recall_gain >= .05 and best_gain >= -1e-12
            summary.update(pilot={"passed": passed, "top8_recall_gain_over_rule": recall_gain,
                                  "mean_true_best_gain_delta": best_gain, "states_without_improvement": len(audits) - len(relevant),
                                  "ranking_audit_path": str(store.root / "ranking_audit.json"),
                                  "ranking_audit_sha256": digest(store.root / "ranking_audit.json")})
            if not passed:
                summary.update(scientific_stop=True, stop_reason="measured_route04_2000_update_gate_failed")
        return summary
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
    if mode == "initial" and config["preparation"]["initial_workers"] > 1 and len(store.keys) < total:
        outcome = parallel_initial(directory, job, config, provenance, dataset, source, indices, store, progress, stopped)
        if outcome["stopped"]:
            return outcome
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
            result = solve(scene, effective_route, job.get("resolved_limits", config["coverage"]["nondrl"]["limits"]), initial=initial, policy=policy,
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
                    "evidence_path": str(store.root), "raw_manifest_sha256": digest(store.path),
                    "provenance": provenance, "rows": len(store.keys)})
    return {"summary": {"rows": len(store.keys), "mother_count": len(indices), "route": route,
                        "dataset_sha256": source["sha256"], "raw_manifest": str(store.path)}}


@contextmanager
def worker_lock(directory):
    """OS-held job lock releases on process death, with no stale PID guessing."""
    Path(directory).mkdir(parents=True, exist_ok=True)
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
        elif job["kind"] == "prepare" and job.get("preparation_mode") in ("teacher", "ranking_audit", "teacher_probe"):
            from .core import Problem
            from .config import NondrlConfig
            from .teachers import generate_teacher
            audit_mode = job.get("preparation_mode") == "ranking_audit"
            role = "validation" if audit_mode else "development" if route == "05" or job.get("preparation_mode") == "teacher_probe" else "train"
            source = config["datasets"][role]
            if digest(source["path"]) != source["sha256"]:
                raise ValueError("teacher data SHA differs")
            dataset = load_variable_dataset(source["path"])
            limits = {**config["coverage"]["nondrl"]["limits"], "online_seconds": {"01": 1., "02": 10., "03": 4., "04": 6.4, "05": 4.}[route]}
            if audit_mode:
                limits["max_menu"] = 64
            initials = _initials(job, source["sha256"])
            def problems():
                for index, scene in enumerate(dataset.scenes):
                    for multiplier in ([1.] if audit_mode else config["evaluation"]["demand_multipliers"]):
                        if stopped():
                            return
                        initial = initials.get(f"{index}:{multiplier:g}")
                        if initial is None:
                            raise ValueError("teacher shared initial evidence lacks scene")
                        yield Problem(scaled_scene(scene, multiplier), NondrlConfig.from_dict(limits),
                                      initial=initial["result"]["initial_layout"])
            teacher_progress = lambda values: progress(**{**values, "stage": "preparing"})
            result = generate_teacher(problems(), route, directory / "teacher", job.get("teacher_limits", config["teacher"]),
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
            teacher_manifest = read_json(source["path"])
            evidence = teacher_manifest["evidence_path"]
            if digest(evidence) != teacher_manifest.get("sha256"):
                raise ValueError("frozen teacher evidence manifest SHA differs")
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
        from .learning import ScientificStop
        if isinstance(error, ScientificStop):
            result = {"status": "scientific_stop", "scientific_stop": True, "stop_reason": str(error)}
            atomic_json(directory / "scientific_stop.json", {**result, "at": now(), "traceback": traceback.format_exc()})
            atomic_json(directory / "results.json", result)
            progress(stage="stopped", ended_at=now(), scientific_stop=True, stop_reason=str(error))
            return result
        progress(stage="failed", ended_at=now(), error=repr(error))
        atomic_json(directory / "failure.json", {"error": repr(error), "traceback": traceback.format_exc(), "at": now()})
        raise
