"""Bounded Jan12 A/B/C numerical check from two saved T0 allocations.

Runs in an isolated development snapshot. It performs no policy inference and
does not submit a workbench evaluation task.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback

import cvxpy
import clarabel
import numpy as np
import scipy
import torch
import satellite_rl
from satellite_rl.power_refinement.model_based import optimize_model_based
from satellite_rl.power_refinement.physics import FixedAllocationPhysics
from satellite_rl.power_refinement.priority import optimize_priority
from satellite_rl.workbench.runner import _load_scene


ROOT = Path(os.environ["S3P03_JAN12_ROOT"])
STATUS = ROOT / "status.json"
RECEIPT = ROOT / "receipt.json"
OPTIONS = {"si_tolerance": 1e-8, "sgm_tolerance": 1e-8,
           "maxiter": 300, "maxeval": 2000, "max_time_seconds": 30.0,
           "protect_basic_service": True}
PRIORITY = dict(OPTIONS, si_stall_window=200, si_stall_tolerance=1e-6,
                si_min_evaluations=200, si_max_fraction=0.85,
                service_margin_ratio=5e-9)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def validate(problem, p0, p_ref, candidate, cap, initial_ev, ref_ev, result):
    power = np.asarray(candidate, dtype=float)
    ev = problem.evaluate(power)
    baseline = initial_ev.capacity_bps / problem.demand_bps
    ref_ratio = ref_ev.capacity_bps / problem.demand_bps
    ratio = ev.capacity_bps / problem.demand_bps
    ref_met = ref_ratio >= 0.1
    floor = np.maximum(ref_ratio, np.minimum(baseline, 0.1)) - 1e-8
    strict_cross = int(np.count_nonzero(ref_met & (ratio < 0.1)))
    below_floor = float(np.max(np.maximum(floor[~ref_met] - ratio[~ref_met], 0))) if np.any(~ref_met) else 0.0
    checks = {
        "power_budget_excess_w": max(0.0, float(power.sum() - cap)),
        "negative_power_w": max(0.0, float(-power.min())),
        "empty_slot_power_w": float(np.max(np.abs(power[~problem.active]))) if np.any(~problem.active) else 0.0,
        "strict_new_basic_cross_down": strict_cross,
        "originally_short_floor_gap": below_floor,
        "si_below_A": max(0.0, float(ref_ev.si - ev.si)),
        "phase1_best_si_mean": result["phase1_best_si_mean"],
        "best_feasible_si_mean": result["best_feasible_si_mean"],
        "phase2_si_floor": result["phase2_si_floor"],
    }
    if (checks["power_budget_excess_w"] > 1e-7 or checks["negative_power_w"] > 0 or
            checks["empty_slot_power_w"] > 0 or checks["strict_new_basic_cross_down"] or
            checks["originally_short_floor_gap"] > 1e-12 or
            checks["si_below_A"] > 1e-8 + 1e-12 or
            result["evaluations"] > 2000 or
            ev.si + 1e-8 + 1e-12 < result["phase1_best_si_mean"] or
            ev.si + 1e-8 + 1e-12 < result["best_feasible_si_mean"] or
            not np.allclose(power, result["power_w"], rtol=0, atol=0)):
        raise RuntimeError("Jan12 numerical guard failed: " + json.dumps(checks))
    return ev, checks


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = ROOT / item["name"]
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise RuntimeError("Frozen Jan12 input differs: " + item["name"])
    metadata = json.loads((ROOT / "model.json").read_text(encoding="utf-8"))
    if (metadata["provenance"]["env_step"] != 150000 or
            metadata["provenance"]["run_id"] != "train_sac_20261001_162126_33e89e95" or
            sha(ROOT / "policy.pt") != "ae095e8246d22b2716ba154315cee18584917d215580821cb53f9aeba78eb824"):
        raise RuntimeError("T0 endpoint or policy SHA differs")
    config = json.loads((ROOT / "T0.json").read_text(encoding="utf-8"))
    package = Path(satellite_rl.__file__).resolve()
    if "dev_snapshots" not in str(package) or not torch.cuda.is_available():
        raise RuntimeError("Development snapshot or DQN/CUDA is unavailable")
    receipt = {
        "status": "running_development_only", "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": sha(ROOT / "manifest.json"),
        "training_interactions": 0, "policy_inference_calls": 0,
        "date_utc": "2024-01-12", "scene_count": 2, "budget_modes": ["total", "initial"],
        "methods": ["sca", "sca_si_lp", "sca_si_lb"],
        "runtime": {"python": sys.executable, "package": str(package),
                    "torch": torch.__version__, "cuda": torch.cuda.is_available(),
                    "numpy": np.__version__, "scipy": scipy.__version__,
                    "cvxpy": cvxpy.__version__, "clarabel": clarabel.__version__},
        "options_A": OPTIONS, "options_BC": PRIORITY, "cases": [],
    }
    write(STATUS, {"status": "running", "started_utc": receipt["started_utc"], "completed_cases": 0, "total_cases": 12})
    write(RECEIPT, receipt)
    for index in (0, 1):
        scene_path = ROOT / f"scene_{index:05d}.npz"
        initial_path = ROOT / f"initial_scene_{index:05d}.json"
        initial = json.loads(initial_path.read_text(encoding="utf-8"))
        scene_sha = sha(scene_path)
        initial_sha = sha(initial_path)
        problem = FixedAllocationPhysics.from_scene(_load_scene(scene_path), config, initial["beams"])
        p0 = np.asarray([beam["power_w"] for beam in initial["beams"]], dtype=float)
        initial_ev = problem.evaluate(p0)
        np.testing.assert_allclose(initial_ev.capacity_bps,
                                   [beam["rate_bps"] for beam in initial["beams"]],
                                   rtol=1e-8, atol=1e-3)
        baseline_floor = np.minimum(initial_ev.capacity_bps, 0.1 * problem.demand_bps)
        for mode, cap in (("total", 3000.0), ("initial", float(p0.sum()))):
            a_start = perf_counter()
            a = optimize_model_based(problem, p0, "sca", dict(OPTIONS, power_cap_w=cap))
            a_seconds = perf_counter() - a_start
            p_ref = np.asarray(a["power_w"], dtype=float)
            ref_ev = problem.evaluate(p_ref)
            if (a["evaluations"] > 2000 or p_ref.sum() > cap + 1e-7 or
                    np.any(p_ref[~problem.active] != 0) or
                    ref_ev.si < initial_ev.si - 1e-8 or
                    ref_ev.sgm < initial_ev.sgm - 1e-8):
                raise RuntimeError("A/reference Jan12 guard failed")
            reference_sha = hashlib.sha256(np.asarray(p_ref, dtype="<f8").tobytes()).hexdigest()
            common = {"scene_index": index, "scene_sha256": scene_sha,
                      "initial_sha256": initial_sha, "budget_mode": mode,
                      "power_cap_w": cap, "initial_si": initial_ev.si,
                      "initial_sgm": initial_ev.sgm, "reference_sha256": reference_sha,
                      "reference_si": ref_ev.si, "reference_sgm": ref_ev.sgm,
                      "reference_evaluations": a["evaluations"],
                      "reference_optimizer_seconds": a_seconds}
            receipt["cases"].append(dict(common, method="sca", final_si=ref_ev.si,
                                         final_sgm=ref_ev.sgm, evaluations=a["evaluations"],
                                         complete_route_evaluations=a["evaluations"],
                                         optimizer_seconds=a_seconds,
                                         complete_route_optimizer_seconds=a_seconds,
                                         termination_reason=a["termination_reason"]))
            write(RECEIPT, receipt)
            for method in ("sca_si_lp", "sca_si_lb"):
                t0 = perf_counter()
                result = optimize_priority(problem, p_ref, method,
                                           dict(PRIORITY, power_cap_w=cap,
                                                baseline_service_floor_bps=baseline_floor))
                elapsed = perf_counter() - t0
                ev, checks = validate(problem, p0, p_ref, result["power_w"], cap,
                                      initial_ev, ref_ev, result)
                receipt["cases"].append(dict(common, method=method,
                                             final_si=ev.si, final_sgm=ev.sgm,
                                             evaluations=result["evaluations"],
                                             complete_route_evaluations=a["evaluations"] + result["evaluations"],
                                             optimizer_seconds=elapsed,
                                             complete_route_optimizer_seconds=a_seconds + elapsed,
                                             termination_reason=result["termination_reason"],
                                             solver_status=result["solver_status"],
                                             guard=checks))
                write(RECEIPT, receipt)
            write(STATUS, {"status": "running", "last_scene_index": index,
                           "last_mode": mode, "completed_cases": len(receipt["cases"]), "total_cases": 12,
                           "checked_utc": datetime.now(timezone.utc).isoformat()})
            if sha(initial_path) != initial_sha or sha(scene_path) != scene_sha:
                raise RuntimeError("Jan12 saved input changed during development check")
    receipt["status"] = "completed_development_only"
    receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write(RECEIPT, receipt)
    write(STATUS, {"status": "completed", "completed_cases": 12, "total_cases": 12,
                   "checked_utc": receipt["completed_utc"]})


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write(STATUS, {"status": "failed", "error": type(exc).__name__ + ": " + str(exc),
                       "traceback": traceback.format_exc(),
                       "checked_utc": datetime.now(timezone.utc).isoformat()})
        raise
