"""Independently recalculate every saved stage 3 beam and fixed-slot capacity.

Reads the SHA archived cases, initial allocations, scenes and frozen physical
source. This performs no policy inference or new evaluation action sampling.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
import sys

import numpy as np


ROOT = Path("D:/paper/outputs/experiments/stage3_si_priority_20261003")
ARCHIVE = ROOT / "remote_archive"
OUT = ROOT / "completion"
PORTS = (20418, 20424)
METHODS = ("sca", "sca_si_lp", "sca_si_lb")
TIMING_KEYS = ("initialization_seconds", "physics_setup_seconds", "optimizer_wall_seconds",
               "optimizer_callback_io_seconds", "optimizer_excluding_callback_seconds",
               "verification_seconds", "total_pipeline_seconds")
METRIC_KEYS = ("final_si_mean", "final_sgm_mean", "basic_met_count", "all_basic_met",
               "power_w", "power_budget_w", "throughput_bps", "effective_carried_bps",
               "zero_capacity_count", "allocated_slots")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value) -> str:
    content = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def beam_sgm(capacity: float, demand: float, beta: float) -> float:
    if capacity <= 0:
        return 0.0
    ratio = capacity / demand
    real = ratio - 1.0 if ratio >= 1.0 else 1.0 - 1.0 / ratio
    imaginary = (capacity - demand) / beta
    return 1.0 - (1.0 - math.exp(-math.hypot(real, imaginary))) ** 3


def close(actual, expected, *, rtol=1e-8, atol=1e-9) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol)


def assert_close(actual, expected, label, *, rtol=1e-8, atol=1e-9) -> None:
    if not close(actual, expected, rtol=rtol, atol=atol):
        raise RuntimeError(f"{label}: {actual!r} differs from {expected!r}")


def bool_csv(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError("Saved CSV boolean is not explicit True/False")


def check_priority_floors(capacity, demand, original, reference) -> None:
    """Keep T0's floor and A's strict basic-service beam count."""
    ratio = np.asarray(capacity) / demand
    reference_ratio = np.asarray(reference) / demand
    t0_floor = np.minimum(original, 0.1 * demand)
    if np.any(ratio < t0_floor / demand - 1e-8):
        raise RuntimeError("B/C cumulatively violated T0 basic-service floor")
    if np.any((reference_ratio >= 0.1) & (ratio < 0.1)):
        raise RuntimeError("B/C newly crossed A's strict basic-service threshold")
    if np.any((reference_ratio < 0.1) & (ratio < reference_ratio - 1e-8)):
        raise RuntimeError("B/C lowered a beam already short in A beyond tolerance")


def self_check() -> None:
    assert beam_sgm(0.0, 100.0, 100.0) == 0.0
    assert_close(beam_sgm(100.0, 100.0, 100.0), 1.0, "exact-demand SGM")
    assert_close(min(150.0 / 100.0, 1.0), 1.0, "SI cap")
    assert bool_csv("True") and not bool_csv("False")
    try:
        bool_csv("1")
    except ValueError:
        pass
    else:
        raise AssertionError("Ambiguous boolean accepted")
    demand = np.array([100.0, 100.0])
    check_priority_floors(np.array([10.0, 8.0]), demand,
                          np.array([10.0, 7.0]), np.array([10.0, 8.0]))
    for bad in (np.array([9.999, 8.0]), np.array([10.0, 7.999])):
        try:
            check_priority_floors(bad, demand, np.array([10.0, 7.0]),
                                  np.array([10.0, 8.0]))
        except RuntimeError:
            pass
        else:
            raise AssertionError("A strict or already-short beam floor was not enforced")
    try:
        check_priority_floors(np.array([10.0 - 1.5e-6]), np.array([100.0]),
                              np.array([10.0]), np.array([10.0 - 5e-7]))
    except RuntimeError:
        pass
    else:
        raise AssertionError("T0 tolerance was incorrectly added to A tolerance")


def archived_file(manifest: dict, label: str, relative: str) -> Path:
    items = [item for item in manifest["files"]
             if item["label"] == label and item["relative"] == relative]
    if len(items) != 1:
        raise RuntimeError(f"Archive manifest missing unique {label}/{relative}")
    item = items[0]
    path = Path(item["local_path"])
    if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
        raise RuntimeError(f"Archived file SHA differs: {label}/{relative}")
    return path


def check_priority_record(case: dict, a_case: dict, a_file_sha: str) -> None:
    """Audit one B/C record against its exact common A and true physics."""
    row, csv_row = case["row"], case["csv"]
    a_row = a_case["row"]
    problem = case["problem"]
    demand = case["demand"]
    capacity = case["capacity"]
    original = np.asarray([beam["rate_bps"] for beam in case["initial"]["beams"]])
    reference = a_case["capacity"]
    a_hash = fingerprint(a_row["beams"])
    if (row.get("reference_allocation_sha256") != a_hash or
            csv_row.get("reference_allocation_sha256") != a_hash or
            row.get("reference_case_sha256") != a_file_sha or
            csv_row.get("reference_case_sha256") != a_file_sha or
            row.get("reference_method") != "sca"):
        raise RuntimeError("B/C did not use the exact archived common A allocation")
    if row["initial_allocation_sha256"] != a_row["initial_allocation_sha256"]:
        raise RuntimeError("B/C and A use different T0 allocations")
    if not np.allclose(row["solver"]["initial_metrics"]["capacity_bps"], reference,
                       rtol=1e-8, atol=1e-3):
        raise RuntimeError("B/C solver did not start from A physical capacity")
    if not np.allclose(row["solver"]["initial_metrics"]["final_si_mean"],
                       a_row["metrics"]["final_si_mean"], rtol=1e-10, atol=1e-10):
        raise RuntimeError("B/C solver did not start from A SI")
    for name in ("final_si_mean", "final_sgm_mean"):
        assert_close(row["reference_metrics"][name], a_row["metrics"][name],
                     "B/C reference " + name)
    check_priority_floors(capacity, demand, original, reference)
    reference_ratio = reference / demand
    t0_floor = np.minimum(original, 0.1 * demand)
    solver = row["solver"]
    phase1 = float(solver["phase1_best_si_mean"])
    best = float(solver["best_feasible_si_mean"])
    locked = solver.get("phase2_si_floor")
    final_si = float(case["row"]["metrics"]["final_si_mean"])
    a_si = float(a_row["metrics"]["final_si_mean"])
    if (not np.isfinite([phase1, best]).all() or
            phase1 < a_si - 1e-10 or best < phase1 - 1e-10 or
            final_si < a_si - 1e-8 or final_si < phase1 - 1e-8 or
            final_si < best - 1e-8):
        raise RuntimeError("B/C SI priority or best feasible SI ledger differs")
    if locked is not None:
        assert_close(float(locked), best - 1e-8, "locked SI floor", rtol=0, atol=1e-12)
        if final_si < float(locked):
            raise RuntimeError("B/C final SI fell below locked floor")
    for key, expected in (("phase1_best_si_mean", phase1),
                          ("best_feasible_si_mean", best),
                          ("locked_si_floor", locked)):
        actual = row.get(key)
        saved = csv_row.get(key)
        if expected is None:
            if actual is not None or saved not in (None, ""):
                raise RuntimeError("B/C phase value unexpectedly populated: " + key)
        elif actual is None or saved in (None, ""):
            raise RuntimeError("B/C phase value missing: " + key)
        else:
            assert_close(actual, expected, key, rtol=0, atol=1e-12)
            assert_close(float(saved), expected, key + " CSV", rtol=0, atol=1e-12)
    highest = np.asarray(solver["highest_si_power_w"], dtype=float)
    if (highest.shape != case["power"].shape or not np.isfinite(highest).all() or
            np.any(highest < 0) or np.any(highest[~problem.active] != 0) or
            highest.sum() > float(case["row"]["metrics"]["power_budget_w"]) + 1e-7):
        raise RuntimeError("B/C highest-SI candidate violates power or empty-slot rules")
    highest_ev = problem.evaluate(highest)
    highest_ratio = highest_ev.capacity_bps / demand
    if (np.any(highest_ratio < t0_floor / demand - 1e-8) or
            np.any((reference_ratio >= 0.1) & (highest_ratio < 0.1)) or
            np.any((reference_ratio < 0.1) & (highest_ratio < reference_ratio - 1e-8))):
        raise RuntimeError("B/C highest-SI candidate violates basic-service protection")
    assert_close(highest_ev.si, best, "recomputed best feasible SI", rtol=0, atol=1e-10)
    assert_close(solver["highest_si_metrics"]["final_si_mean"], highest_ev.si,
                 "saved highest-SI metric", rtol=0, atol=1e-10)
    assert_close(solver["highest_si_metrics"]["final_sgm_mean"], highest_ev.sgm,
                 "saved highest-SI SGM", rtol=0, atol=1e-10)
    if not np.allclose(solver["highest_si_metrics"]["capacity_bps"],
                       highest_ev.capacity_bps, rtol=1e-8, atol=1e-3):
        raise RuntimeError("Saved highest-SI beam capacities differ from physical recomputation")
    own_calls = int(solver["evaluations"])
    a_calls = int(a_row["solver"]["evaluations"])
    if (own_calls > 2000 or a_calls > 2000 or a_calls + own_calls > 4000 or
            int(row["reference_solver_evaluations"]) != a_calls or
            int(row["complete_route_solver_evaluations"]) != a_calls + own_calls or
            int(row["complete_route_capacity_budget"]) != 4000 or
            float(row["complete_route_optimizer_time_budget_seconds"]) != 60.0 or
            int(csv_row["reference_solver_evaluations"]) != a_calls or
            int(csv_row["complete_route_solver_evaluations"]) != a_calls + own_calls or
            int(csv_row["complete_route_capacity_budget"]) != 4000 or
            float(csv_row["complete_route_optimizer_time_budget_seconds"]) != 60.0):
        raise RuntimeError("B/C route capacity-call accounting differs")
    timing, a_timing = row["timing"], a_row["timing"]
    for field, source in (("reference_optimizer_wall_seconds", "optimizer_wall_seconds"),
                          ("reference_verification_seconds", "verification_seconds")):
        assert_close(timing[field], a_timing[source], field, rtol=1e-8, atol=1e-6)
    assert_close(timing["total_pipeline_seconds"],
                 timing["initialization_seconds"] + timing["physics_setup_seconds"] +
                 a_timing["optimizer_wall_seconds"] + a_timing["verification_seconds"] +
                 timing["optimizer_wall_seconds"] + timing["verification_seconds"],
                 "B/C complete route time", rtol=1e-8, atol=1e-6)


def audit_one(port: int, physics_class) -> dict:
    base = ARCHIVE / str(port)
    manifest = json.loads((base / "archive_manifest.json").read_text(encoding="utf-8"))
    job = json.loads(archived_file(manifest, "job", "job.json").read_text(encoding="utf-8"))
    config = json.loads(archived_file(manifest, "job", "config.json").read_text(encoding="utf-8"))
    if job.get("kind") != "refine" or config.get("research_stage") != "power_refinement":
        raise RuntimeError("Archived job is not a stage3 refinement")
    for reference in job["model_files"]:
        local = archived_file(manifest, "job", "model_input/" + Path(reference["path"]).name)
        if sha(local) != reference["sha256"]:
            raise RuntimeError("Frozen T0 model input SHA differs from job record")
    policy_shas = [item["sha256"] for item in job["model_files"]
                   if Path(item["path"]).name == "policy.pt"]
    if len(policy_shas) != 1 or manifest["run_id"] != job["run_id"]:
        raise RuntimeError("Archived job has no unique frozen T0 policy or task identity")
    options = config["power_refinement"]
    expected_mode = "total" if port == 20418 else "initial"
    if (options["power_budget_mode"] != expected_mode or options["repeats"] != 3
            or options["methods"] != list(METHODS)
            or options["action_seed"] != 53001 or options["optimizer_seed"] != 63001
            or options["protect_basic_service"] is not True):
        raise RuntimeError("Archived scientific comparison configuration differs")
    frozen = {"si_tolerance": 1e-8, "sgm_tolerance": 1e-8,
              "maxiter": 300, "maxeval": 2000, "max_time_seconds": 30.0,
              "si_stall_window": 200, "si_stall_tolerance": 1e-6,
              "si_min_evaluations": 200, "si_max_fraction": 0.85,
              "service_margin_ratio": 5e-9}
    if any(options.get(name) != value for name, value in frozen.items()):
        raise RuntimeError("Archived SI-priority phase rules or budgets differ")
    scenes = {}
    for item in job["scene_sets"]["test"]["files"]:
        local = archived_file(manifest, "job", "scene_inputs/test/" + Path(item["path"]).name)
        if sha(local) != item["sha256"]:
            raise RuntimeError("Frozen scene SHA differs from job.json")
        scenes[item["sha256"]] = local
    if len(scenes) != 100:
        raise RuntimeError("Archived job does not have 100 unique frozen scenes")
    scene_positions = {digest: index for index, digest in enumerate(scenes)}
    case_path = archived_file(manifest, "job", "power_cases.jsonl")
    csv_path = archived_file(manifest, "job", "power_results.csv")
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        csv_list = list(csv.DictReader(stream))
    csv_rows = {(row["source_scene_sha256"], int(row["repeat"]), row["method"]): row
                for row in csv_list}
    if len(csv_list) != 900 or len(csv_rows) != 900:
        raise RuntimeError("Archived power_results.csv is incomplete")
    timing_path = archived_file(manifest, "job", "power_scene_times.csv")
    with timing_path.open(encoding="utf-8-sig", newline="") as stream:
        timing_table = list(csv.DictReader(stream))
    timing_rows = {(row["source_scene_sha256"], row["method"]): row for row in timing_table}
    if len(timing_table) != 300 or len(timing_rows) != 300:
        raise RuntimeError("Archived power_scene_times.csv is incomplete")
    seen = set()
    initial_cache = {}
    initial_hashes = {}
    timing_groups = {}
    audited_cases = {}
    cases = beams_checked = capacity_checks = highest_checks = 0
    for line in case_path.open(encoding="utf-8"):
        row = json.loads(line)
        scene_sha = row["source_scene_sha256"]
        repeat = int(row["repeat"])
        method = row["method"]
        key = (scene_sha, repeat, method)
        if (scene_sha not in scenes or repeat not in range(3) or method not in METHODS
                or row["power_budget_mode"] != expected_mode or key in seen):
            raise RuntimeError("Case grid, source or mode differs")
        archived_case = archived_file(manifest, "job",
            f"power_cases/scene_{scene_positions[scene_sha]:05d}_r{repeat}_{method}.json")
        if fingerprint(row) != fingerprint(json.loads(archived_case.read_text(encoding="utf-8"))):
            raise RuntimeError("Canonical case differs from independently archived case file")
        if (row["git_commit"] != job["git_commit"] or row["run_id"] != job["run_id"] or
                row["source_model_sha256"] != policy_shas[0] or
                csv_rows[key]["git_commit"] != job["git_commit"] or
                csv_rows[key]["run_id"] != job["run_id"]):
            raise RuntimeError("Case source commit or task identity differs from archived job")
        expected_seed = None
        if row.get("optimizer_seed") != expected_seed:
            raise RuntimeError("Optimizer seed differs from frozen scene/repeat stream")
        seen.add(key)
        initial_key = (scene_sha, repeat)
        if initial_key not in initial_cache:
            source_initial = Path(row["initial_path"])
            initial_path = archived_file(manifest, "job", "initial_allocations/" + source_initial.name)
            initial = json.loads(initial_path.read_text(encoding="utf-8"))
            if len(initial["beams"]) != 100 or initial["source_scene_sha256"] != scene_sha:
                raise RuntimeError("Frozen initial allocation identity differs")
            scene_path = scenes[scene_sha]
            with np.load(scene_path, allow_pickle=False) as arrays:
                scene = {name: np.asarray(arrays[name]) for name in arrays.files}
            problem = physics_class.from_scene(scene, job["model_config"], initial["beams"])
            initial_power = np.asarray([beam["power_w"] for beam in initial["beams"]], dtype=float)
            calculated = problem.evaluate(initial_power).capacity_bps
            recorded = np.asarray([beam["rate_bps"] for beam in initial["beams"]], dtype=float)
            if not np.allclose(calculated, recorded, rtol=1e-8, atol=1e-3):
                raise RuntimeError("Frozen physical source did not reproduce initial capacity")
            demand_initial = np.asarray([beam["required_bps"] for beam in initial["beams"]], dtype=float)
            initial_si = float(np.minimum(recorded / demand_initial, 1.0).mean())
            initial_sgm = fmean(beam_sgm(float(c), float(d), float(demand_initial.mean()))
                                for c, d in zip(recorded, demand_initial))
            assert_close(initial["metrics"]["final_si_mean"], initial_si, "initial SI")
            assert_close(initial["metrics"]["final_sgm_mean"], initial_sgm, "initial SGM")
            identity_sha = fingerprint(initial["beams"])
            initial_cache[initial_key] = (initial, problem, identity_sha, initial_power)
            initial_hashes[initial_key] = identity_sha
        initial, problem, identity_sha, initial_power = initial_cache[initial_key]
        if row["initial_allocation_sha256"] != identity_sha:
            raise RuntimeError("Saved case initial-allocation SHA differs")
        for metric in ("final_si_mean", "final_sgm_mean"):
            assert_close(row["initial_metrics"][metric], initial["metrics"][metric],
                         "Case initial " + metric)
        if len(row["beams"]) != 100:
            raise RuntimeError("Case missing target beams")
        cap = min(float(initial_power.sum()), 3000.0) if expected_mode == "initial" else 3000.0
        demand = np.asarray([beam["required_bps"] for beam in initial["beams"]], dtype=float)
        power = np.asarray([beam["power_w"] for beam in row["beams"]], dtype=float)
        capacity = np.asarray([beam["rate_bps"] for beam in row["beams"]], dtype=float)
        if (not np.isfinite(power).all() or not np.isfinite(capacity).all()
                or np.any(power < -1e-8) or np.any(capacity < -1e-6)
                or power.sum() > cap + 1e-6):
            raise RuntimeError("Nonfinite, negative or over-budget power/capacity")
        calculated = problem.evaluate(power).capacity_bps
        if not np.allclose(calculated, capacity, rtol=1e-8, atol=1e-3):
            raise RuntimeError("Saved beam capacities differ from frozen physical calculation")
        capacity_checks += 100
        beta = float(demand.mean())
        sgm_values = []
        for index, (old, new) in enumerate(zip(initial["beams"], row["beams"])):
            for field in ("beam_id", "original_index", "group", "freq", "slots", "required_bps"):
                if old[field] != new[field]:
                    raise RuntimeError("Power refinement changed frozen beam, demand or frequency slot")
            c, d = float(capacity[index]), float(demand[index])
            si = min(c / d, 1.0)
            sgm = beam_sgm(c, d, beta)
            sgm_values.append(sgm)
            for field, expected in (("capacity_bps", c), ("effective_served_bps", min(c, d)),
                                    ("capacity_demand_ratio", c / d), ("si", si), ("sgm", sgm),
                                    ("initial_capacity_bps", old["rate_bps"]),
                                    ("initial_power_w", old["power_w"])):
                assert_close(new[field], expected, field,
                             rtol=1e-8, atol=1e-3 if "bps" in field else 1e-9)
            if bool(new["basic_met"]) != (c >= 0.1 * d):
                raise RuntimeError("Saved per-beam basic service differs")
            if not old["slots"] and (abs(power[index]) > 1e-8 or abs(c) > 1e-6):
                raise RuntimeError("Beam without slots has power or capacity")
            service_floor = min(float(old["rate_bps"]), 0.1 * d)
            if c < service_floor - max(1e-3, d * 1e-8):
                raise RuntimeError("Saved case violated initial basic-service protection")
            beams_checked += 1
        metrics = {
            "final_si_mean": float(np.minimum(capacity / demand, 1.0).mean()),
            "final_sgm_mean": fmean(sgm_values),
            "basic_met_count": int(np.sum(capacity >= 0.1 * demand)),
            "all_basic_met": bool(np.all(capacity >= 0.1 * demand)),
            "power_w": float(power.sum()), "power_budget_w": cap,
            "throughput_bps": float(capacity.sum()),
            "effective_carried_bps": float(np.minimum(capacity, demand).sum()),
            "zero_capacity_count": int(np.sum(capacity <= 0)),
            "allocated_slots": sum(int(beam["slots"]) for beam in row["beams"]),
        }
        if metrics["final_si_mean"] < initial["metrics"]["final_si_mean"] - options["si_tolerance"] - 1e-10:
            raise RuntimeError("Saved case reduced SI below protected initial result")
        # The frozen A solver retains its original SGM guard.  B/C explicitly
        # prioritize SI and may return lower SGM than either T0 or A.
        if (method == "sca" and metrics["final_sgm_mean"] <
                initial["metrics"]["final_sgm_mean"] - options["sgm_tolerance"] - 1e-10):
            raise RuntimeError("Frozen A case reduced SGM below protected T0 result")
        csv_row = csv_rows[key]
        if csv_row.get("optimizer_seed") != ("" if expected_seed is None else str(expected_seed)):
            raise RuntimeError("Optimizer seed CSV differs")
        timing = row.get("timing")
        if not isinstance(timing, dict) or any(name not in timing for name in TIMING_KEYS):
            raise RuntimeError("Case timing is missing")
        scope = ("sum_of_measured_stages_shared_initializer_charged_once_per_method"
                 if method == "sca" else
                 "sum_of_measured_stages_shared_reference_charged_once_per_route")
        if timing.get("timing_scope") != scope or csv_row.get("timing_scope") != scope:
            raise RuntimeError("Case timing scope differs from independent deployment accounting")
        assert_close(timing["initialization_seconds"], initial["inference_seconds"],
                     "shared T0 initialization timing", rtol=1e-8, atol=1e-6)
        assert_close(timing["physics_setup_seconds"], initial["physics_setup_seconds"],
                     "shared fixed-slot physics setup timing", rtol=1e-8, atol=1e-6)
        if any(not math.isfinite(float(timing[name])) or float(timing[name]) < 0 for name in TIMING_KEYS):
            raise RuntimeError("Case timing is negative or nonfinite")
        assert_close(timing["optimizer_callback_io_seconds"] + timing["optimizer_excluding_callback_seconds"],
                     timing["optimizer_wall_seconds"], "optimizer timing decomposition", rtol=1e-8, atol=1e-6)
        timing_fields = TIMING_KEYS
        if method != "sca":
            timing_fields += ("reference_optimizer_wall_seconds",
                              "reference_verification_seconds", "reference_total_seconds")
            assert_close(timing["reference_total_seconds"],
                         timing["reference_optimizer_wall_seconds"] +
                         timing["reference_verification_seconds"], "reference stage sum")
        elif any(name in timing for name in ("reference_total_seconds",
                                              "reference_optimizer_wall_seconds")):
            raise RuntimeError("A must not charge a reference stage")
        assert_close(sum(timing[name] for name in ("initialization_seconds", "physics_setup_seconds",
                                                    "optimizer_wall_seconds", "verification_seconds"))
                     + (timing["reference_total_seconds"] if method != "sca" else 0),
                     timing["total_pipeline_seconds"], "total pipeline timing", rtol=1e-8, atol=1e-6)
        for name in timing_fields:
            assert_close(float(csv_row[name]), timing[name], "case timing CSV " + name,
                         rtol=1e-8, atol=1e-6)
        timing_groups.setdefault((scene_sha, method), []).append(timing)
        if csv_row["initial_allocation_sha256"] != identity_sha:
            raise RuntimeError("Complete CSV and case JSONL initial SHA differ")
        for metric in METRIC_KEYS:
            expected = metrics[metric]
            recorded = row["metrics"][metric]
            if isinstance(expected, bool):
                if type(recorded) is not bool or recorded != expected or bool_csv(csv_row[metric]) != expected:
                    raise RuntimeError("Boolean aggregate metric differs")
            else:
                atol = 1e-3 if metric.endswith("bps") else 1e-9
                assert_close(recorded, expected, metric, rtol=1e-8, atol=atol)
                assert_close(float(csv_row[metric]), expected, metric + " CSV", rtol=1e-8, atol=atol)
        solver = row["solver"]
        solver_calls = int(solver["evaluations"])
        if solver_calls < 1 or solver_calls > 2000 or int(csv_row["solver_evaluations"]) != solver_calls:
            raise RuntimeError("Saved solver exceeded its 2000-call budget")
        assert_close(float(csv_row["solver_seconds"]), solver["elapsed_seconds"],
                     "solver seconds")
        audited_cases[key] = {"row": row, "csv": csv_row, "capacity": capacity,
                              "power": power, "demand": demand,
                              "initial": initial, "problem": problem}
        cases += 1
    expected_keys = {(scene, repeat, method) for scene in scenes
                     for repeat in range(3) for method in METHODS}
    if seen != expected_keys or cases != 900 or beams_checked != 90000 or capacity_checks != 90000:
        raise RuntimeError("Case, beam or physical recomputation coverage incomplete")
    for scene, repeat in {(scene, repeat) for scene, repeat, _ in expected_keys}:
        a_case = audited_cases[scene, repeat, "sca"]
        a_name = f"power_cases/scene_{scene_positions[scene]:05d}_r{repeat}_sca.json"
        a_file = archived_file(manifest, "job", a_name)
        a_sha = sha(a_file)
        if fingerprint(a_case["row"]["beams"]) != fingerprint(
                json.loads(a_file.read_text(encoding="utf-8"))["beams"]):
            raise RuntimeError("Archived A case differs from canonical case stream")
        for method in METHODS[1:]:
            if Path(audited_cases[scene, repeat, method]["row"].get(
                    "reference_case_path", "")).name != Path(a_name).name:
                raise RuntimeError("B/C names another common A case file")
            check_priority_record(audited_cases[scene, repeat, method], a_case,
                                  a_sha)
            highest_checks += 100
    if set(timing_groups) != set(timing_rows) or any(len(group) != 3 for group in timing_groups.values()):
        raise RuntimeError("Scene timing grid does not have exactly three actions per method")
    for key, group in timing_groups.items():
        aggregate = timing_rows[key]
        if int(aggregate["repeats"]) != 3:
            raise RuntimeError("Scene timing CSV repeats differ")
        timing_fields = TIMING_KEYS + (("reference_optimizer_wall_seconds",
                                        "reference_verification_seconds",
                                        "reference_total_seconds") if key[1] != "sca" else ())
        for name in timing_fields:
            total = sum(float(item[name]) for item in group)
            assert_close(float(aggregate[name + "_sum"]), total,
                         "scene timing sum " + name, rtol=1e-8, atol=1e-6)
            assert_close(float(aggregate[name + "_mean_per_action"]), total / 3,
                         "scene timing mean " + name, rtol=1e-8, atol=1e-6)
    return {"port": port, "mode": expected_mode, "cases_checked": cases,
            "beams_checked": beams_checked, "physical_capacity_recomputed": capacity_checks,
            "highest_si_candidate_beams_recomputed": highest_checks,
            "independent_initial_allocations": len(initial_hashes),
            "case_jsonl_sha256": sha(case_path), "case_csv_sha256": sha(csv_path),
            "scene_timing_csv_sha256": sha(timing_path), "timed_scene_method_rows": len(timing_rows),
            "frozen_source_commit": job["git_commit"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("stage3_beam_audit_self_check_passed")
        return
    source_root = ARCHIVE / "20418/source/src"
    # Import frozen scientific source without rewriting archived .pyc files.
    # Their original bytes are included in the transfer SHA manifest.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source_root))
    import satellite_rl  # noqa: E402
    from satellite_rl.power_refinement.physics import FixedAllocationPhysics  # noqa: E402
    if Path(satellite_rl.__file__).resolve() != source_root / "satellite_rl/__init__.py":
        raise RuntimeError("Beam audit did not load archived frozen physical source")
    for relative in ("satellite_rl/power_refinement/physics.py",
                     "satellite_rl/power_refinement/model_based.py",
                     "satellite_rl/power_refinement/priority.py",
                     "satellite_rl/power_refinement/config.py",
                     "satellite_rl/power_refinement/runner.py",
                     "satellite_rl/power_refinement/timing.py",
                     "satellite_rl/workbench/environment.py"):
        if sha(ARCHIVE / "20418/source/src" / relative) != sha(ARCHIVE / "20424/source/src" / relative):
            raise RuntimeError("Two frozen task sources differ: " + relative)
    ports = {str(port): audit_one(port, FixedAllocationPhysics) for port in PORTS}
    if len({item["frozen_source_commit"] for item in ports.values()}) != 1:
        raise RuntimeError("Two archived budgets used different frozen source commits")
    result = {"status": "independent_all_1800_cases_180000_beams_and_timing_verified",
              "checked_utc": datetime.now(timezone.utc).isoformat(),
              "source_root": str(source_root),
              "scope": "saved cases only; no policy inference or new evaluation",
              "ports": ports, "total_cases": sum(x["cases_checked"] for x in ports.values()),
              "total_beams": sum(x["beams_checked"] for x in ports.values())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "beam_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "total_cases": result["total_cases"],
                      "total_beams": result["total_beams"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
