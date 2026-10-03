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


ROOT = Path("D:/paper/outputs/experiments/stage3_all_routes_20261003")
ARCHIVE = ROOT / "remote_archive"
OUT = ROOT / "completion"
PORTS = (20418, 20424)
METHODS = {"none", "discrete", "slsqp", "sca", "de", "pso", "wmmse"}
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
    options = config["power_refinement"]
    expected_mode = "total" if port == 20418 else "initial"
    if (options["power_budget_mode"] != expected_mode or options["repeats"] != 3
            or options["methods"] != ["none", "discrete", "slsqp", "sca", "de", "pso", "wmmse"]
            or options["action_seed"] != 53001 or options["optimizer_seed"] != 63001
            or options["protect_basic_service"] is not True):
        raise RuntimeError("Archived scientific comparison configuration differs")
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
    if len(csv_list) != 2100 or len(csv_rows) != 2100:
        raise RuntimeError("Archived power_results.csv is incomplete")
    timing_path = archived_file(manifest, "job", "power_scene_times.csv")
    with timing_path.open(encoding="utf-8-sig", newline="") as stream:
        timing_table = list(csv.DictReader(stream))
    timing_rows = {(row["source_scene_sha256"], row["method"]): row for row in timing_table}
    if len(timing_table) != 700 or len(timing_rows) != 700:
        raise RuntimeError("Archived power_scene_times.csv is incomplete")
    seen = set()
    initial_cache = {}
    initial_hashes = {}
    timing_groups = {}
    cases = beams_checked = capacity_checks = 0
    for line in case_path.open(encoding="utf-8"):
        row = json.loads(line)
        scene_sha = row["source_scene_sha256"]
        repeat = int(row["repeat"])
        method = row["method"]
        key = (scene_sha, repeat, method)
        if (scene_sha not in scenes or repeat not in range(3) or method not in METHODS
                or row["power_budget_mode"] != expected_mode or key in seen):
            raise RuntimeError("Case grid, source or mode differs")
        expected_seed = ((63001 + 1000 * scene_positions[scene_sha] + 10 * repeat
                          + (1 if method == "de" else 2)) % 2**32
                         if method in ("de", "pso") else None)
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
        if (metrics["final_si_mean"] < initial["metrics"]["final_si_mean"] - options["si_tolerance"] - 1e-10
                or metrics["final_sgm_mean"] < initial["metrics"]["final_sgm_mean"] - options["sgm_tolerance"] - 1e-10):
            raise RuntimeError("Saved case reduced SI/SGM below protected initial result")
        csv_row = csv_rows[key]
        if csv_row.get("optimizer_seed") != ("" if expected_seed is None else str(expected_seed)):
            raise RuntimeError("Optimizer seed CSV differs")
        timing = row.get("timing")
        if not isinstance(timing, dict) or any(name not in timing for name in TIMING_KEYS):
            raise RuntimeError("Case timing is missing")
        scope = "sum_of_measured_stages_shared_initializer_charged_once_per_method"
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
        assert_close(sum(timing[name] for name in ("initialization_seconds", "physics_setup_seconds",
                                                    "optimizer_wall_seconds", "verification_seconds")),
                     timing["total_pipeline_seconds"], "total pipeline timing", rtol=1e-8, atol=1e-6)
        for name in TIMING_KEYS:
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
        cases += 1
    expected_keys = {(scene, repeat, method) for scene in scenes
                     for repeat in range(3) for method in METHODS}
    if seen != expected_keys or cases != 2100 or beams_checked != 210000 or capacity_checks != 210000:
        raise RuntimeError("Case, beam or physical recomputation coverage incomplete")
    if set(timing_groups) != set(timing_rows) or any(len(group) != 3 for group in timing_groups.values()):
        raise RuntimeError("Scene timing grid does not have exactly three actions per method")
    for key, group in timing_groups.items():
        aggregate = timing_rows[key]
        if int(aggregate["repeats"]) != 3:
            raise RuntimeError("Scene timing CSV repeats differ")
        for name in TIMING_KEYS:
            total = sum(float(item[name]) for item in group)
            assert_close(float(aggregate[name + "_sum"]), total,
                         "scene timing sum " + name, rtol=1e-8, atol=1e-6)
            assert_close(float(aggregate[name + "_mean_per_action"]), total / 3,
                         "scene timing mean " + name, rtol=1e-8, atol=1e-6)
    return {"port": port, "mode": expected_mode, "cases_checked": cases,
            "beams_checked": beams_checked, "physical_capacity_recomputed": capacity_checks,
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
                     "satellite_rl/power_refinement/runner.py",
                     "satellite_rl/workbench/environment.py"):
        if sha(ARCHIVE / "20418/source/src" / relative) != sha(ARCHIVE / "20424/source/src" / relative):
            raise RuntimeError("Two frozen task sources differ: " + relative)
    ports = {str(port): audit_one(port, FixedAllocationPhysics) for port in PORTS}
    result = {"status": "independent_all_4200_cases_420000_beams_and_timing_verified",
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
