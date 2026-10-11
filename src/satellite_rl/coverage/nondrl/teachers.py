"""Bounded, auditable supervision made from common executable operations.

Only workbench runners may invoke formal generation. The module itself does
not launch jobs or connect to hosts. Every completed physical trial is saved;
unfinished trials are unknown and never negative training labels.
"""
from __future__ import annotations

from collections import OrderedDict
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .models import canonical_route, center_rays

EVIDENCE_FORMAT = "s1_nondrl_teacher_v1"


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items() if k != "layout"} | (
            {"layout": (value["layout"].to_record() if hasattr(value["layout"], "to_record")
                        else jsonable(value["layout"]))} if value.get("layout") is not None else {})
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False,
                               indent=2), encoding="utf-8")
    temp.replace(path)


def scene_features(scene):
    """Only immutable scene inputs, no current or teacher layout features."""
    positions = np.asarray(scene.positions, dtype=np.float64)
    demands = np.asarray(scene.demands, dtype=np.float64)
    rays = center_rays(torch.as_tensor(positions, dtype=torch.float64)).numpy()
    total = max(float(demands.sum()), 1.)
    return np.column_stack((positions / [90., 180.], rays,
                            demands / total, np.log1p(demands), np.ones(len(demands))))


def beam_features(scene, centers, visible):
    centers = np.asarray(centers, dtype=np.float64)
    rays = center_rays(torch.as_tensor(centers, dtype=torch.float64)).numpy()
    visible = np.asarray(visible, dtype=np.int64)
    known = visible >= 0
    loads = np.bincount(visible[known], weights=np.asarray(scene.demands)[known], minlength=len(centers))
    counts = np.bincount(visible[known], minlength=len(centers))
    # Visible loads/counts, NEVER complete hidden loads or teacher activation.
    return np.column_stack((centers / [90., 180.], rays,
                            loads / max(float(np.sum(scene.demands)), 1.),
                            counts / max(len(visible), 1), np.full(len(centers), .5)))


def feature16(values):
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(result) > 16 or not np.isfinite(result).all():
        raise ValueError("candidate features exceed frozen schema")
    return np.pad(result, (0, 16 - len(result)))


def state_sample(problem, objects=None):
    current = problem.current
    aircraft = scene_features(problem.scene)
    # State routes may see current ownership, which is never used in MLM/03.
    owner_centers = current.centers[current.assignment]
    loads = np.bincount(current.assignment, weights=problem.scene.demands, minlength=len(current.centers))
    aircraft[:, 2:5] = np.column_stack((owner_centers / [90., 180.],
                                       loads[current.assignment] / max(float(np.sum(problem.scene.demands)), 1.)))
    sample = {"mother_id": str(problem.scene.scene_id),
              "source_hash": str(problem.scene.source_hash),
              "demand_multiplier": float(problem.scene.metadata.get("demand_multiplier", 1.)),
              "aircraft": aircraft.tolist(),
              "beams": beam_features(problem.scene, current.centers, current.assignment).tolist(),
              "state": current.to_record(),
              # Audit-only payload; collate never supplies these to a network.
              "scene_demands_bps": np.asarray(problem.scene.demands).tolist(),
              "scene_positions": np.asarray(problem.scene.positions).tolist(),
              "initial_layout_sha256": problem.initial_sha256,
              "initial_carry": problem.carry(problem.initial_physical),
              "initial_metrics": problem.initial_metrics}
    if objects is not None:
        sample["actions"] = [feature16(obj["features"]).tolist() for obj in objects]
        sample["objects"] = jsonable(objects)
    return sample


def cv_value(problem, layout=None):
    layout = problem.current if layout is None else layout
    loads = np.bincount(layout.assignment, weights=problem.scene.demands,
                        minlength=len(layout.centers))
    mean = float(loads.mean())
    return float(loads.std() / mean) if mean else 0.


def trial_label(problem, trial, before_cv):
    reason = str(trial.get("reason", ""))
    complete = (reason not in ("timeout", "budget", "budget_exhausted", "deadline", "unknown") and
                "unknown" not in reason)
    if not complete:
        return {"known": False, "utility": 0., "status": "unknown", "reason": reason}
    accepted = bool(trial.get("accepted", False))
    valid_metric = trial.get("metrics_all") and trial["metrics_all"].get("load_cv_all") is not None
    gain = ((before_cv - float(trial["metrics_all"]["load_cv_all"])) / max(before_cv, 1e-12)) if valid_metric else 0.
    # Only fully accepted improvements are positive; completed failures are negative.
    utility = max(-1., min(gain, 1.)) if accepted or reason in ("no_exact_CV_improvement", "I0_CV_protection") else -1.
    return {"known": True, "utility": utility,
            "status": "accepted" if accepted else "rejected", "reason": reason}


def rebuild_trial(problem, region, source, **kwargs):
    """A solver timeout without incumbent is unknown, never a negative label."""
    trace_start = len(problem.trace)
    proposal = problem.rebuild(region, **kwargs)
    if proposal is not None:
        return problem.consider(proposal, source)
    records = problem.trace[trace_start:]
    certificate = next((record for record in reversed(records) if record.get("status") == "infeasible_pool"), None)
    return {"accepted": False, "layout": None,
            "reason": "certified_infeasible_pool" if certificate else "rebuild_no_incumbent_unknown",
            "solver_records": records}


class EvidenceWriter:
    def __init__(self, folder, route, provenance=None, shard_rows=128, resume=False):
        self.folder, self.route = Path(folder), canonical_route(route)
        self.folder.mkdir(parents=True, exist_ok=True)
        if (self.folder / "manifest.json").exists() and not resume:
            raise FileExistsError("teacher evidence exists; resume must reuse audited shards")
        self.rows, self.pending = 0, []
        self.shard_rows = shard_rows
        self.manifest = {"format": EVIDENCE_FORMAT, "route": self.route,
                         "provenance": provenance or {}, "shards": [], "rows": 0,
                         "status": "generating", "trials": 0, "elapsed_seconds": 0.}
        if resume:
            self.manifest = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
            if (self.manifest["format"] != EVIDENCE_FORMAT or self.manifest["route"] != self.route or
                    self.manifest["provenance"] != (provenance or {})):
                raise ValueError("teacher resume provenance/schema differs")
            self.rows = self.manifest["rows"]
            for record in self.manifest["shards"]:
                if digest(self.folder / record["path"]) != record["sha256"]:
                    raise ValueError("teacher resume shard differs")
        self.flush()

    def append(self, sample):
        sample = {**sample, "route": self.route, "format": EVIDENCE_FORMAT}
        self.pending.append(jsonable(sample))
        self.rows += 1
        if len(self.pending) >= self.shard_rows:
            self.flush()

    def flush(self):
        if self.pending:
            path = self.folder / f"samples_{len(self.manifest['shards']):05d}.jsonl.gz"
            temp = path.with_suffix(path.suffix + ".tmp")
            with temp.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                    for sample in self.pending:
                        stream.write((json.dumps(sample, ensure_ascii=False, allow_nan=False,
                                                 separators=(",", ":")) + "\n").encode())
            temp.replace(path)
            self.manifest["shards"].append({"path": path.name, "sha256": digest(path),
                                               "rows": len(self.pending)})
            self.pending = []
        self.manifest["rows"] = self.rows
        atomic_json(self.folder / "manifest.json", self.manifest)


def mlm_sample(problem, reference, rng, ratio=.5, frozen=None, region=None, candidate_pool=None, legal_matrix=None):
    """A coherent whole reference; mask augmentation never mixes solutions."""
    context = problem.current if frozen is None else frozen
    beams = list(region["beams"]) if region is not None else list(range(len(context.centers)))
    members = np.flatnonzero(np.isin(context.assignment, beams))
    centers = np.asarray(candidate_pool if candidate_pool is not None else context.centers[beams], dtype=float)
    if not np.array_equal(centers[:len(beams)], context.centers[beams]):
        raise ValueError("MLM pool must retain original region centers first")
    original = np.asarray([beams.index(int(b)) for b in context.assignment[members]], dtype=np.int64)
    mapping = {}
    for b in beams:
        ids = np.flatnonzero(np.all(centers == reference.centers[b], axis=1))
        if not len(ids):
            raise ValueError("MLM teacher center absent from pre-frozen pool")
        mapping[b] = int(ids[0])
    if not np.isin(reference.assignment[members], beams).all():
        raise ValueError("MLM teacher changed external region membership")
    labels = np.asarray([mapping[int(b)] for b in reference.assignment[members]], dtype=np.int64)
    changed = original != labels
    hidden = rng.random(len(labels)) < ratio
    hidden |= changed  # Any changed label is hidden; all visible labels match reference.
    if not hidden.any():
        hidden[rng.integers(len(hidden))] = True
    visible = np.where(hidden, -1, original)
    aircraft = scene_features(problem.scene)[members]
    legal = (problem.legal(centers=centers, positions=problem.scene.positions[members])
             if legal_matrix is None else np.asarray(legal_matrix, dtype=bool))
    if not legal[np.arange(len(labels)), labels].all():
        raise ValueError("MLM teacher label lies outside legal frozen edge")
    # Only this region's visible demand/count is encoded, with a shared scene denominator.
    candidates = beam_features(problem.scene, centers, np.full(len(problem.scene.demands), -1))
    known = visible >= 0
    candidates[:, 5] = np.bincount(visible[known], weights=aircraft[known, 5], minlength=len(centers))
    candidates[:, 6] = np.bincount(visible[known], minlength=len(centers)) / len(members)
    candidates[:, 7] = len(beams) / 200.
    return {"mother_id": str(problem.scene.scene_id), "source_hash": str(problem.scene.source_hash),
            "aircraft": aircraft.tolist(), "candidates": candidates.tolist(), "legal": legal.tolist(),
            "visible": visible.tolist(), "hidden": hidden.tolist(), "labels": labels.tolist(),
            "original_assignment": original.tolist(), "centers": centers.tolist(),
            "target_n": len(beams), "pool_hash": hashlib.sha256(centers.tobytes() + legal.tobytes()).hexdigest(),
            "members": members.tolist(), "region_beams": beams, "external_context": context.to_record(),
            "mask_ratio": ratio, "input_pool": "original_and_scene_geometry_before_teacher"}


def generate_teacher(problems, route, output, limits=None, stop_callback=None,
                     progress_callback=None, provenance=None, resume=False):
    """Consume caller-owned Problems with strict local AND aggregate budgets.

    Route 01 freezes scene-generated region pools BEFORE solving, route 02 saves
    actual executed monotone traces, route 03 stores complete continuous-center
    references, 04 labels candidate trials, 05 labels shared-rebuilder regions.
    Offline operation/geometry/physical ledgers remain owned by each Problem.
    """
    route, limits = canonical_route(route), limits or {}
    if int(limits.get("workers", 1)) > 1:
        return _parallel_teacher(problems, route, output, limits, stop_callback,
                                 progress_callback, provenance, resume)
    caps = {"01": (20000, 20000.), "02": (1000000, 50000.),
            "03": (1280000, 40000.), "04": (128000, 12800.), "05": (11520, 7200.)}
    max_trials = int(limits.get("max_trials", caps[route][0]))
    max_seconds = float(limits.get("max_seconds", caps[route][1]))
    if not 0 < max_trials <= caps[route][0] or not 0 < max_seconds <= caps[route][1]:
        raise ValueError("teacher budget exceeds approved route bound")
    problem_caps = {"01": 20000, "02": 5000, "03": 10000, "04": 2000, "05": 960}
    max_problems = min(int(limits.get("max_problems", problem_caps[route])), problem_caps[route])
    max_steps = min(int(limits.get("max_steps", 4 if route == "03" else 20)), 4 if route == "03" else 20)
    problem_seconds_cap = 1. if route == "01" else 10. if route == "02" else 4.
    per_problem = min(float(limits.get("per_problem_seconds", problem_seconds_cap)), problem_seconds_cap)
    writer = EvidenceWriter(output, route, provenance, shard_rows=100000000, resume=resume)
    if resume and writer.manifest.get("limits", limits) != limits:
        raise ValueError("teacher same-ID recovery budget/config differs")
    writer.manifest["limits"] = dict(limits)
    rng = np.random.default_rng(51001)
    if resume and "rng" in writer.manifest:
        rng.bit_generator.state = writer.manifest["rng"]
    previous = float(writer.manifest.get("elapsed_seconds", 0.))
    started, trials, completed = time.perf_counter() - previous, int(writer.manifest.get("trials", 0)), int(writer.manifest.get("problems", 0))
    active = writer.manifest.get("active")
    stop_reason = "input_exhausted"

    def persist(problem=None, state=None):
        writer.manifest.update(trials=trials, problems=completed, rng=rng.bit_generator.state,
                               elapsed_seconds=time.perf_counter() - started,
                               budget={"max_trials": max_trials, "max_seconds": max_seconds},
                               active=state)
        if state is not None and problem is not None:
            state.update(current=problem.current.to_record(), current_metrics=problem.current_metrics,
                         current_physical=problem.current_physical, calls=problem.calls.copy(),
                         physical_cache=problem._physical_cache, times=problem.times.copy(), trace=problem.trace)
        writer.flush()

    def can_start(local_start):
        return (trials < max_trials and time.perf_counter() - started < max_seconds and
                time.perf_counter() - local_start < per_problem and
                not (stop_callback and stop_callback()))

    from .core import BudgetStop
    from .classical import region_pool
    for problem_index, problem in enumerate(problems):
        if problem_index < completed:
            continue
        if completed >= max_problems:
            stop_reason = "problem_limit"
            break
        if not can_start(time.perf_counter()) and not active:
            if stop_callback and stop_callback():
                stop_reason = "stopped"
                break
            # An approved input still receives a legal reference. No candidate,
            # solver, model label or PHY work is started after the search cap.
            _problem_receipt(writer, completed, problem, 0., "global_budget_exhausted_I0_reference_only")
            completed += 1
            writer.manifest["reference_only_problems"] = writer.manifest.get("reference_only_problems",0) + 1
            persist()
            continue
        local_start = time.perf_counter()
        if active:
            if active["mother_id"] != str(problem.scene.scene_id) or active["source_hash"] != str(problem.scene.source_hash):
                raise ValueError("teacher resume scene identity differs")
            problem.current = type(problem.current).from_record(active["current"])
            problem.current_metrics, problem.current_physical = active["current_metrics"], active["current_physical"]
            problem.calls.update(active["calls"])
            problem.times.update(active.get("times", {}))
            problem._physical_cache.update(active.get("physical_cache", {}))
            problem.trace = list(active.get("trace", []))
            frozen = type(problem.current).from_record(active["frozen"])
            references = active["references"]
            local_start -= active["local_elapsed"]
        else:
            frozen = type(problem.current).from_record(problem.current.to_record())
            references = [problem.current.to_record()]
        active = active or {"mother_id": str(problem.scene.scene_id), "source_hash": str(problem.scene.source_hash),
                            "frozen": frozen.to_record(), "references": references, "step": 0,
                            "local_elapsed": 0., "pending": None}
        problem.deadline = min(problem.deadline, time.perf_counter() + max(0., per_problem - active["local_elapsed"]))
        interrupted = False
        problem_completion_reason = "menu_or_step_finished"
        try:
            for step in range(active["step"], max_steps):
                if route in ("04", "05") and writer.rows >= problem_caps[route]:
                    stop_reason = "state_budget_exhausted"
                    break
                if not can_start(local_start):
                    interrupted = bool(stop_callback and stop_callback())
                    problem_completion_reason = "user_stop" if interrupted else "search_budget_exhausted_unknown"
                    break
                before = cv_value(problem)
                if route == "01":
                    # Each local solve is ONE teacher problem; mask augmentation is free of solves.
                    pending = active.get("pending")
                    regions = pending["regions"] if pending else problem.regions()
                    if not regions:
                        break
                    for region_index in range(pending.get("region_index", 0) if pending else 0, len(regions)):
                        region = regions[region_index]
                        if not can_start(local_start):
                            interrupted = bool(stop_callback and stop_callback())
                            problem_completion_reason = "user_stop" if interrupted else "search_budget_exhausted_unknown"
                            active["pending"] = {"regions": regions, "region_index": region_index}
                            break
                        context = type(problem.current).from_record(problem.current.to_record())
                        pool = region_pool(problem, region)  # must precede teacher call
                        members = np.flatnonzero(np.isin(context.assignment, region["beams"]))
                        legal = problem.legal(centers=pool, positions=problem.scene.positions[members])
                        persist(problem, {**active, "pending": {"regions": regions, "region_index": region_index,
                                                                "pool": pool.tolist()}, "local_elapsed": time.perf_counter() - local_start})
                        try:
                            trial = rebuild_trial(problem, region, "01_teacher", candidate_pool=pool)
                        except BudgetStop:
                            candidate, trial = None, {"accepted": False, "reason": "unknown"}
                            problem_completion_reason = "trial_budget_exhausted_unknown"
                        trials += 1
                        reference = trial["layout"] if trial.get("accepted") else context
                        for ratio in (.25, .5, .75, 1.):
                            record = mlm_sample(problem, reference, rng, ratio, context, region, pool, legal)
                            record["teacher_trial"] = jsonable(trial)
                            writer.append(record)
                        problem.commit(trial)
                        active["pending"] = {"regions": regions, "region_index": region_index + 1}
                        active["local_elapsed"] = time.perf_counter() - local_start
                        persist(problem, active)
                    break
                pending = active.get("pending")
                objects = pending["objects"] if pending else (problem.regions() if route == "05" else problem.menu())
                if not objects:
                    if route == "02":
                        sample = state_sample(problem, [{"kind": "padding", "features": [0.]}])
                        sample.update(positive_actions=[], stop_positive=True, stop_reason="verified_empty_menu",
                                      trajectory_step=step, trials=[])
                        writer.append(sample)
                    break
                objects = objects[:12 if route == "05" else 32 if route in ("01", "02", "03") else 64]
                sample = pending["sample"] if pending else state_sample(problem, objects)
                results, labels = [], []
                if pending:
                    labels = pending["labels"]
                    for stored in pending["results"]:
                        if stored and stored.get("layout") is not None:
                            stored = {**stored, "layout": type(problem.current).from_record(stored["layout"])}
                        results.append(stored)
                for object_index in range(len(labels), len(objects)):
                    obj = objects[object_index]
                    if not can_start(local_start):
                        if stop_callback and stop_callback():
                            interrupted = True
                            break
                        labels.append({"known": False, "utility": 0., "status": "unknown", "reason": "budget"})
                        results.append(None)
                        continue
                    trial_start = time.perf_counter()
                    try:
                        if route == "05":
                            trial = rebuild_trial(problem, obj, "teacher_05")
                        else:
                            trial = problem.preview(obj)
                    except BudgetStop:
                        trial = {"reason": "unknown"}
                        problem_completion_reason = "trial_budget_exhausted_unknown"
                    trials += 1
                    label = trial_label(problem, trial, before)
                    label["elapsed_seconds"] = time.perf_counter() - trial_start
                    labels.append(label)
                    results.append(trial)
                    active["pending"] = {"objects": objects, "sample": sample, "labels": labels,
                                         "results": [jsonable(result) if result else None for result in results]}
                    active["local_elapsed"] = time.perf_counter() - local_start
                    persist(problem, active)
                if interrupted:
                    break
                sample["trials"] = [{"object": obj, "label": label, "result": jsonable(trial) if trial else None}
                                    for obj, label, trial in zip(objects, labels, results)]
                best = max((i for i, label in enumerate(labels) if label["known"] and label["utility"] > 0),
                           key=lambda i: labels[i]["utility"], default=None)
                if route == "02":
                    # A STOP label requires a COMPLETELY checked menu. Truncation is not optimal STOP.
                    if best is not None:
                        positives = [i for i, label in enumerate(labels) if label["known"] and
                                     abs(label["utility"] - labels[best]["utility"]) <= 1e-6]
                        sample.update(positive_actions=positives, stop_positive=False, trajectory_step=step)
                        sample["executed_after"] = results[best]["layout"].to_record()
                        writer.append(sample)
                    elif all(label["known"] for label in labels):
                        sample.update(positive_actions=[], stop_positive=True,
                                      stop_reason="verified_no_improvement", trajectory_step=step)
                        writer.append(sample)
                elif route in ("04", "05"):
                    sample.update(known=[label["known"] for label in labels],
                                  utility=[label["utility"] for label in labels], trajectory_step=step)
                    writer.append(sample)
                if best is None:
                    break
                if not problem.commit(results[best]):
                    raise ValueError("teacher best action failed common commit")
                references.append(problem.current.to_record())
                active.update(references=references, step=step + 1, pending=None,
                              local_elapsed=time.perf_counter() - local_start)
                persist(problem, active)
                if route == "05":
                    break  # State labels are joint regions; caller may supply more rule-generated states.
        except BudgetStop as error:
            problem_completion_reason = "local_budget_exhausted_unknown: " + str(error)
            active["local_elapsed"] = time.perf_counter() - local_start
            persist(problem, active)
        if interrupted:
            active["local_elapsed"] = time.perf_counter() - local_start
            persist(problem, active)
            stop_reason = "stopped"
            break
        if route == "03":
            # Multi-reference loss selects a whole solution, never per-aircraft mixed labels.
            writer.append({"mother_id": str(problem.scene.scene_id), "source_hash": str(problem.scene.source_hash),
                           "aircraft": scene_features(problem.scene).tolist(), "positions": problem.scene.positions.tolist(),
                           "target_n": len(frozen.centers),
                           "references": [{"centers": ref["centers"], "assignment": ref["assignment"]}
                                          for ref in references[-4:]], "teacher_trials": trials,
                           "teacher_elapsed_seconds": time.perf_counter() - local_start})
        _problem_receipt(writer, completed, problem, time.perf_counter() - local_start, problem_completion_reason)
        completed += 1
        active = None
        persist()
        if progress_callback:
            progress_callback({"stage": "teacher", "problems": completed, "samples": writer.rows,
                               "trials": trials, "elapsed_seconds": time.perf_counter() - started})
    writer.manifest.update(status="stopped" if stop_reason == "stopped" else "completed",
                           stop_reason=stop_reason, trials=trials, problems=completed,
                           elapsed_seconds=time.perf_counter() - started)
    if not writer.rows and stop_reason != "stopped":
        writer.manifest.update(status="scientific_stop", stop_reason="no_valid_supervision")
    writer.flush()
    return {**writer.manifest, "path": str(writer.folder / "manifest.json"),
            "sha256": digest(writer.folder / "manifest.json")}


def _problem_receipt(writer, index, problem, elapsed, completion_reason="menu_or_step_finished"):
    """Complete actual problem facts, stored once for independent audit."""
    folder = writer.folder / "problem_receipts"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"problem_{index:06d}.json.gz"
    record = {"format": "s1_nondrl_teacher_problem_v1", "route": writer.route,
              "provenance": writer.manifest["provenance"], "target_n": problem.n,
              "mother_id": str(problem.scene.scene_id), "source_hash": str(problem.scene.source_hash),
              "demand_multiplier": float(problem.scene.metadata.get("demand_multiplier", 1.)),
              "scene_positions": np.asarray(problem.scene.positions).tolist(),
              "scene_demands_bps": np.asarray(problem.scene.demands).tolist(),
              "aircraft_ids": list(problem.scene.aircraft_ids),
              "initial_layout": problem.initial.to_record(), "final_layout": problem.current.to_record(),
              "initial_metrics_all": problem.initial_metrics, "final_metrics_all": problem.current_metrics,
              "initial_physical": problem.initial_physical, "final_physical": problem.current_physical,
              "initial_sha256": problem.initial_sha256, "config": problem.config.to_dict(),
              "initial_provenance": getattr(problem,"initial_provenance",{}),
              "trace": problem.trace, "calls": problem.calls, "timing": problem.times,
              "teacher_elapsed_seconds": elapsed, "completion_reason": completion_reason,
              "reference_scope": "last_complete_certified_layout; unknown search is not a negative label"}
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as packed:
            packed.write(json.dumps(jsonable(record), separators=(",", ":"), allow_nan=False).encode())
    temp.replace(path)
    writer.manifest.setdefault("problem_receipts", []).append({"path": str(path.relative_to(writer.folder)),
                "sha256": digest(path), "mother_id": record["mother_id"], "demand_multiplier": record["demand_multiplier"]})


def _reference_only(writer,index,problem,reason):
    """Finalize identity/valid I0 reference, never extend a search reservation."""
    started=time.perf_counter()
    _problem_receipt(writer,index,problem,0.,reason)
    writer.manifest["reference_only_problems"]=writer.manifest.get("reference_only_problems",0)+1
    writer.manifest["reference_finalize_seconds"]=writer.manifest.get("reference_finalize_seconds",0.)+time.perf_counter()-started


def _teacher_worker(problem, route, folder, limits, provenance):
    """CPU-only bounded process; the parent owns deterministic result ordering."""
    torch.set_num_threads(int(limits.pop("worker_threads", 1)))
    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    owner = target / "owner.json"
    if owner.exists():
        old = json.loads(owner.read_text(encoding="utf-8"))
        if old.get("running"):
            try:
                os.kill(int(old["pid"]), 0)
            except (OSError, ProcessLookupError):
                pass
            else:
                raise RuntimeError("a previous teacher worker still owns this problem")
    existing = target / "manifest.json"
    if existing.exists():
        record = json.loads(existing.read_text(encoding="utf-8"))
        if record.get("route") != route or record.get("provenance") != (provenance or {}) or record.get("limits") != limits:
            raise ValueError("teacher worker receipt provenance/budget differs")
        if record.get("status") in ("completed", "scientific_stop"):
            return {**record, "path": str(existing), "sha256": digest(existing)}
    atomic_json(owner, {"pid": os.getpid(), "running": True, "started_unix": time.time()})
    # Problem preparation happened before submission and is already accounted by
    # the caller. Queue/import time is separate, never deducted as solver work.
    problem.online_started = time.perf_counter()
    problem.deadline = problem.online_started + problem.config.online_seconds
    try:
        result = generate_teacher([problem], route, target, limits, provenance=provenance,
                                  resume=existing.exists())
        atomic_json(owner, {"pid": os.getpid(), "running": False, "finished_unix": time.time()})
        return result
    except Exception:
        atomic_json(owner, {"pid": os.getpid(), "running": False, "failed_unix": time.time()})
        raise


def _parallel_teacher(problems, route, output, limits, stop_callback, progress_callback, provenance, resume):
    """Reserve worst-case calls/time before dispatch, then refund unused budget.

    CPU workers have independent bounded Problems. Finished shards are kept once
    in their original worker folders and folded in fixed input order. Recovery
    reuses completed receipts and unfinished workers' own cursors; a live old
    worker fails closed. No change to mother split, labels or per-task seed.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    workers = int(limits.get("workers", 8))
    if not 2 <= workers <= 8:
        raise ValueError("teacher CPU worker count must be 2..8")
    caps = {"01": (20000, 20000.), "02": (1000000, 50000.),
            "03": (1280000, 40000.), "04": (128000, 12800.), "05": (11520, 7200.)}
    max_trials = int(limits.get("max_trials", caps[route][0]))
    max_seconds = float(limits.get("max_seconds", caps[route][1]))
    if not 0 < max_trials <= caps[route][0] or not 0 < max_seconds <= caps[route][1]:
        raise ValueError("parallel teacher budget exceeds approved route bound")
    problem_caps = {"01": 20000, "02": 5000, "03": 10000, "04": 2000, "05": 960}
    max_problems = min(int(limits.get("max_problems", problem_caps[route])), problem_caps[route])
    max_steps = min(int(limits.get("max_steps", 4 if route == "03" else 20)), 4 if route == "03" else 20)
    if route == "04":
        # At most 2000 labelled states, including multistep worker trajectories.
        max_problems = min(max_problems, problem_caps[route] // max_steps)
    problem_seconds_cap = 1. if route == "01" else 10. if route == "02" else 4.
    per_seconds = min(float(limits.get("per_problem_seconds", problem_seconds_cap)), problem_seconds_cap)
    per_calls = 12 if route in ("01", "05") else 32 * max_steps
    writer = EvidenceWriter(output, route, provenance, resume=resume)
    manifest = writer.manifest
    if resume and manifest.get("limits", limits) != limits:
        raise ValueError("parallel teacher same-ID recovery budget/config differs")
    manifest["limits"] = dict(limits)
    folded = int(manifest.get("problems", 0))
    trials = int(manifest.get("trials", 0))
    compute_seconds = float(manifest.get("teacher_compute_seconds", 0.))
    pending_receipts = dict(manifest.get("reservations", {}))
    previous_wall = float(manifest.get("coordinator_wall_seconds", 0.))
    start = time.perf_counter() - previous_wall
    iterator, index, exhausted = iter(problems), 0, False
    for _ in range(folded):
        next(iterator)
        index += 1
    active = OrderedDict()
    stopped = False

    def persist():
        manifest.update(problems=folded, trials=trials, teacher_compute_seconds=compute_seconds,
                        elapsed_seconds=compute_seconds, coordinator_wall_seconds=time.perf_counter() - start,
                        reservations=pending_receipts, workers=workers,
                        worker_threads=int(limits.get("worker_threads", 1)),
                        cpu_parallel_scheme="spawn_fixed_order_with_worst_case_reservations",
                        budget={"max_trials": max_trials, "max_seconds": max_seconds})
        writer.flush()

    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        while active or not exhausted:
            while len(active) < workers and not exhausted:
                if stop_callback and stop_callback():
                    stopped, exhausted = True, True
                    break
                if index >= max_problems:
                    exhausted = True
                    break
                reserved_calls = sum(record["max_trials"] for record in pending_receipts.values())
                reserved_seconds = sum(record["max_seconds"] for record in pending_receipts.values())
                remaining_calls = max_trials - trials - reserved_calls
                remaining_seconds = max_seconds - compute_seconds - reserved_seconds
                existing = pending_receipts.get(str(index))
                if existing:
                    reservation = existing
                elif remaining_calls > 0 and remaining_seconds > 0:
                    reservation = {"max_trials": min(per_calls, remaining_calls),
                                   "max_seconds": min(per_seconds, remaining_seconds)}
                else:
                    # Waiting jobs may refund unused reservations. Only final
                    # exhaustion ends the iterator; it does not lose scenes.
                    if not active:
                        # Preserve all remaining approved mother/tier identities
                        # without running additional teacher trials or native
                        # solves. Shared I0 physics is cached by the runner.
                        for problem in iterator:
                            if index>=max_problems:break
                            if stop_callback and stop_callback():
                                stopped=True
                                break
                            _reference_only(writer,index,problem,"global_budget_exhausted_I0_reference_only")
                            index+=1;folded+=1;persist()
                        exhausted = True
                    break
                try:
                    problem = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                key = str(index)
                pending_receipts[key] = reservation
                persist()
                settings = {**limits, **reservation, "workers": 1,
                            "max_problems": 1, "per_problem_seconds": per_seconds}
                folder = writer.folder / "workers" / f"problem_{index:06d}"
                active[index] = pool.submit(_teacher_worker, problem, route, str(folder), settings, provenance)
                index += 1
            if not active:
                break
            task_index, future = next(iter(active.items()))
            result = future.result()  # bounded per-problem work; stable fold order
            source_manifest = Path(result["path"])
            if result.get("problems",0)!=1 or len(result.get("problem_receipts",[]))!=1:
                raise ValueError("teacher worker lacks complete approved problem reference; never fold as completed")
            for shard in result["shards"]:
                source = source_manifest.parent / shard["path"]
                writer.manifest["shards"].append({**shard, "path": str(source.relative_to(writer.folder))})
            for receipt in result.get("problem_receipts", []):
                source = source_manifest.parent / receipt["path"]
                writer.manifest.setdefault("problem_receipts", []).append({**receipt, "path": str(source.relative_to(writer.folder))})
            writer.rows += result["rows"]
            trials += result["trials"]
            compute_seconds += result["elapsed_seconds"]
            pending_receipts.pop(str(task_index))
            active.pop(task_index)
            folded += 1
            persist()
            if progress_callback:
                progress_callback({"stage": "teacher", "problems": folded, "samples": writer.rows,
                                   "trials": trials, "elapsed_seconds": compute_seconds,
                                   "workers": workers, "coordinator_wall_seconds": time.perf_counter() - start})
            if stop_callback and stop_callback():
                stopped, exhausted = True, True
    manifest.update(status="stopped" if stopped else "completed",
                    stop_reason="stopped" if stopped else "search_budget_exhausted_references_retained" if manifest.get("reference_only_problems") else "input_or_budget_exhausted",
                    completion_scope="complete input references; supervision may be limited/unknown")
    if not writer.rows and not stopped:
        manifest.update(status="scientific_stop", stop_reason="no_valid_supervision")
    persist()
    return {**manifest, "path": str(writer.folder / "manifest.json"),
            "sha256": digest(writer.folder / "manifest.json")}
