"""Independent NumPy audit of S1-02 archived aircraft/beam results.

Never imports the planning environment, agents or metric implementation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

EARTH_KM, ORBIT_KM, SAT_LON = 6371.0, 42164.0, 13.0
BEAMS = 100
BPS_ATOL, OTHER_ATOL, RTOL = 1e-5, 1e-7, 1e-9
# Frozen CPU and CUDA action legality both use a 1e-9 degree allowance.
# This is one fixed allowance relative to the original centers, not per step.
MOVEMENT_ATOL_DEG = 1e-9


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_bundle(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = data.get("scenes") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"{path}: requires scene rows, not only an aggregate summary")
    config_file = path.parent / "config.json"
    config = (json.loads(config_file.read_text(encoding="utf-8-sig")) if config_file.is_file()
              else data.get("config", data.get("metadata", {}).get("config", {})))
    return rows, config


def parse_results(arguments):
    result = {}
    for item in arguments:
        label, filename = item.split("=", 1)
        if not label or label in result:
            raise ValueError("Each result label must be unique and nonempty")
        result[label] = Path(filename).resolve()
    return result


def rays(locations):
    locations = np.asarray(locations, dtype=np.float64)
    latitude, longitude = np.deg2rad(locations[:, 0]), np.deg2rad(locations[:, 1])
    ground = EARTH_KM * np.column_stack((np.cos(latitude) * np.cos(longitude),
                                        np.cos(latitude) * np.sin(longitude), np.sin(latitude)))
    angle = math.radians(SAT_LON)
    result = ground - ORBIT_KM * np.array([math.cos(angle), math.sin(angle), 0.0])
    return result / np.linalg.norm(result, axis=1)[:, None]


def visible(locations):
    p = np.asarray(locations, dtype=np.float64)
    cosine = np.cos(np.deg2rad(p[:, 0])) * np.cos(np.deg2rad(p[:, 1] - SAT_LON))
    elevation = np.rad2deg(np.arctan2(cosine - EARTH_KM / ORBIT_KM,
                                     np.sqrt(np.maximum(0.0, 1 - cosine ** 2))))
    return ((p[:, 0] >= 25) & (p[:, 0] <= 80) & (p[:, 1] >= -40) &
            (p[:, 1] <= 50) & (elevation >= 10))


def chord_limit(angle):
    return 4 * math.sin(math.radians(angle) / 2) ** 2


def _arrays(row):
    source = row["input"]
    ids = source["aircraft_ids"]
    positions = np.asarray(source["positions"], dtype=np.float64)
    demands = np.asarray(source["demands_bps"], dtype=np.float64)
    centers = np.asarray(row.get("final_positions", row.get("final_centers")), dtype=np.float64)
    assignment_raw = np.asarray(row.get("final_associations", row.get("associations")))
    if (len(ids) == 0 or len(set(ids)) != len(ids) or positions.shape != (len(ids), 2) or
            demands.shape != (len(ids),) or centers.shape != (BEAMS, 2) or
            assignment_raw.shape != (len(ids),)):
        raise ValueError("Invalid aircraft/beam shapes or duplicate aircraft IDs")
    if (not np.isfinite(positions).all() or not np.isfinite(demands).all() or
            not np.isfinite(centers).all() or not np.isfinite(assignment_raw).all() or
            np.any(demands < 0) or demands.sum() <= 0 or
            not np.array_equal(assignment_raw, assignment_raw.astype(np.int64))):
        raise ValueError("Nonfinite input, invalid demand or noninteger association")
    associations = assignment_raw.astype(np.int64)
    if np.any(associations < -1) or np.any(associations >= BEAMS):
        raise ValueError("Association outside [-1,99]")
    return ids, positions, demands, centers, associations


def expected_edge_components(users, demands):
    """Reconstruct all provable light components independently of certificates."""
    count = len(users)
    adjacent = np.empty((count, count), dtype=bool)
    for start in range(0, count, 128):
        distance = ((users[start:start+128, None, :] - users[None, :, :]) ** 2).sum(-1)
        adjacent[start:start+128] = distance <= chord_limit(.5) + 1e-14
    remaining = np.ones(count, bool)
    result = []
    while remaining.any():
        first = int(np.flatnonzero(remaining)[0])
        pending, members = [first], []
        remaining[first] = False
        while pending:
            current = pending.pop()
            members.append(current)
            neighbors = np.flatnonzero(adjacent[current] & remaining)
            remaining[neighbors] = False
            pending.extend(neighbors.tolist())
        members.sort()
        if not (0 < demands[members].sum() < demands.sum()/BEAMS*.5):
            continue
        points = users[members]
        if any(np.all(((points-point)**2).sum(-1) <= chord_limit(.25)-1e-14) for point in points):
            result.append(members)
    return sorted(result)


def audit_row(row, config=None, *, require_timing=True):
    issues = []
    config = config or {}
    coverage = config.get("coverage", config)
    opt = coverage.get("balance", {})
    if row.get("status") == "initialization_failed":
        source, diag = row["input"], row.get("balance_diagnostics", {})
        ids = source["aircraft_ids"]
        positions, demands = np.asarray(source["positions"], float), np.asarray(source["demands_bps"], float)
        if (len(ids) == 0 or len(set(ids)) != len(ids) or positions.shape != (len(ids), 2) or
                demands.shape != (len(ids),) or not np.isfinite(positions).all() or
                not np.isfinite(demands).all() or np.any(demands < 0)):
            issues.append("Invalid source arrays in declared initialization failure")
        if any(row.get(key) is not None for key in ("metrics", "final_positions", "final_associations", "beam_loads_bps")):
            issues.append("Initialization failure must not invent a layout or metric values")
        if not diag.get("failure_reason") or len(diag.get("common_start_sha256", "")) != 64:
            issues.append("Initialization failure lacks reason or frozen preparation source SHA")
        stage_seconds = diag.get("stage_seconds", {})
        total_seconds = diag.get("total_pipeline_seconds")
        if require_timing and (not stage_seconds or total_seconds is None or
                any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in stage_seconds.values()) or
                not math.isclose(sum(stage_seconds.values()), total_seconds, abs_tol=1e-6, rel_tol=1e-6)):
            issues.append("Initialization failure lacks actual consistent elapsed time")
        return {"scene_id": row["scene_id"], "passed": not issues, "issues": issues,
                "verification_scope": "explicit_failure_record_only_no_layout_claim",
                "business_sha256": canonical_sha(sorted(zip(ids, positions.tolist(), demands.tolist()))),
                "recomputed": None, "total_pipeline_seconds": total_seconds}
    ids, positions, demands, centers, associations = _arrays(row)
    users, beams = rays(positions), rays(centers)
    eligible = np.sum((users[:, None, :] - beams[None, :, :]) ** 2, axis=-1) <= chord_limit(.25) + 1e-14
    covered = eligible.any(axis=1)
    served = associations >= 0
    if not np.array_equal(covered, served):
        issues.append("Geometric coverage differs from complete unique association")
    if np.any(~eligible[np.flatnonzero(served), associations[served]]):
        issues.append("A user is assigned outside its beam footprint")
    if not visible(positions).all() or not visible(centers).all():
        issues.append("User or beam outside the fixed Europe visibility contract")
    loads = np.bincount(associations[served], weights=demands[served], minlength=BEAMS)
    saved_loads = np.asarray(row["beam_loads_bps"], dtype=np.float64)
    if saved_loads.shape != (BEAMS,) or not np.allclose(loads, saved_loads, atol=BPS_ATOL, rtol=RTOL):
        issues.append("Per-beam demand ledger differs from aircraft associations")
    if not math.isclose(float(loads.sum() + demands[~served].sum()), float(demands.sum()),
                        abs_tol=BPS_ATOL, rel_tol=RTOL):
        issues.append("Demand conservation failure")
    diag = row.get("balance_diagnostics", {})
    regions = diag.get("certified_regions")
    if regions is None:
        issues.append("Missing declared edge certificate list")
        regions = []
    declared_components = sorted(sorted(region["aircraft_indices"]) for region in regions)
    if declared_components != expected_edge_components(users, demands):
        issues.append("Edge certificate set omits or invents an eligible isolated component")
    seen = set()
    exempt = []
    for number, region in enumerate(regions):
        members = np.asarray(region["aircraft_indices"], dtype=int)
        if (len(members) == 0 or len(set(members.tolist())) != len(members) or
                np.any(members < 0) or np.any(members >= len(ids)) or seen.intersection(members.tolist())):
            issues.append(f"Invalid/overlapping edge certificate {number}")
            continue
        seen.update(members.tolist())
        if region.get("aircraft_ids") != [ids[i] for i in members]:
            issues.append(f"Edge certificate {number} aircraft identity mismatch")
        outside = np.setdiff1d(np.arange(len(ids)), members)
        if len(outside):
            distance = np.sum((users[members, None, :] - users[None, outside, :]) ** 2, axis=-1)
            if np.any(distance <= chord_limit(.5) + 1e-14):
                issues.append(f"Edge certificate {number} is not an isolated geometric component")
        witness = np.asarray(region["witness_center"], dtype=float).reshape(1, 2)
        if (not visible(witness)[0] or
                np.any(np.sum((users[members] - rays(witness)[0]) ** 2, axis=1) > chord_limit(.25) - 1e-14)):
            issues.append(f"Edge certificate {number} has no valid one-beam witness")
        demand = float(demands[members].sum())
        if not (0 < demand < float(demands.sum()) / BEAMS * .5):
            issues.append(f"Edge certificate {number} violates the fixed light-demand threshold")
        if not math.isclose(demand, float(region["demand_bps"]), abs_tol=BPS_ATOL, rel_tol=RTOL):
            issues.append(f"Edge certificate {number} demand mismatch")
        assigned = associations[members]
        if assigned[0] >= 0 and np.all(assigned == assigned[0]) and np.count_nonzero(associations == assigned[0]) == len(members):
            exempt.append(int(assigned[0]))
    exempt = sorted(set(exempt))
    mask = np.ones(BEAMS, bool)
    mask[exempt] = False
    subject = loads[mask]
    subject_mean = float(subject.mean()) if len(subject) else 0.0
    subject_cv = float(subject.std() / subject_mean) if subject_mean > 0 else None
    band = float(opt.get("subject_range_tolerance", .2))
    in_band = int(np.count_nonzero((subject >= (1-band)*subject_mean) &
                                  (subject <= (1+band)*subject_mean))) if subject_mean > 0 else 0
    total = float(demands.sum())
    dcov, ucov = float(demands[covered].sum() / total), float(covered.mean())
    near = dcov >= .999 and ucov >= .999
    strict = bool(np.all(covered))
    initial = np.asarray(row["input"]["initial_centers"], dtype=float)
    reference = np.asarray(row["input"].get("movement_reference_centers", initial), dtype=float)
    if not np.array_equal(reference, initial):
        issues.append("Movement reference was reset after common preparation")
    movement = np.rad2deg(2 * np.arcsin(np.clip(np.linalg.norm(beams-rays(initial), axis=1)/2, 0, 1)))
    movement_ok = bool(np.all(movement <= float(coverage.get("max_movement_deg", .25)) + MOVEMENT_ATOL_DEG))
    if not movement_ok:
        issues.append("Cumulative displacement exceeds the initial-center limit")
    geometry_ok = bool(visible(centers).all() and np.all(loads > 0) and movement_ok)
    edge_ok = len(exempt) <= int(opt.get("max_edge_beams", 10)) and len(regions) <= int(opt.get("max_edge_beams", 10))
    target_coverage = strict if opt.get("coverage_goal", "strict_100") == "strict_100" else near
    target = bool(target_coverage and geometry_ok and edge_ok and subject_cv is not None and
                  subject_cv <= float(opt.get("subject_cv_target", .2)) and
                  in_band >= int(opt.get("min_subject_beams_in_range", 90)))
    values = {
        "coverage_demand": dcov, "coverage_users": ucov,
        "uncovered_demand_bps": float(demands[~covered].sum()), "uncovered_users": int((~covered).sum()),
        "load_cv": float(loads.std()/loads.mean()) if loads.mean() > 0 else None,
        "load_std_bps": float(loads.std()), "max_load_bps": float(loads.max()),
        "empty_beams": int((loads <= 0).sum()), "active_beams": BEAMS,
        "subject_cv": subject_cv, "subject_std_bps": float(subject.std()),
        "subject_mean_bps": subject_mean, "subject_beams": int(mask.sum()),
        "subject_in_band_beams": in_band, "subject_in_band_fraction_all_100": in_band/BEAMS,
        "subject_max_mean_ratio": float(subject.max()/subject_mean) if subject_mean > 0 else None,
        "edge_beam_count": len(exempt), "certified_edge_region_count": len(regions),
        "edge_limit_met": edge_ok, "edge_demand_fraction": float(loads[exempt].sum()/total),
        "coverage_100": strict, "coverage_99_9": near, "feasible": geometry_ok and near,
        "balanced_target_met": target,
        "movement_mean_deg": float(movement.mean()), "movement_max_deg": float(movement.max()),
    }
    metrics = row.get("metrics", {})
    for key, expected in values.items():
        if key not in metrics:
            issues.append(f"Missing metric {key}")
            continue
        saved = metrics[key]
        if expected is None:
            if saved is not None:
                issues.append(f"{key}: undefined value must be null")
        elif isinstance(expected, bool):
            if saved not in (True, False, 0, 1) or bool(saved) != expected:
                issues.append(f"{key}: boolean differs")
        elif saved is None or not math.isfinite(float(saved)) or not math.isclose(float(saved), expected,
                abs_tol=BPS_ATOL if key.endswith("bps") else OTHER_ATOL, rel_tol=RTOL):
            issues.append(f"{key}: saved={saved}, recomputed={expected}")
    saved_edges = metrics.get("edge_beams", diag.get("qualified_edge_beam_ids"))
    if saved_edges is None or sorted(saved_edges) != exempt:
        issues.append("Qualified edge beam IDs differ")
    stage_seconds = diag.get("stage_seconds", {})
    total_seconds = row.get("total_pipeline_seconds", diag.get("total_pipeline_seconds"))
    if require_timing:
        if not stage_seconds or total_seconds is None:
            issues.append("Missing measured stage timing or complete pipeline time")
        elif any(not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in stage_seconds.values()):
            issues.append("Invalid stage timing")
        elif not math.isclose(sum(stage_seconds.values()), total_seconds, abs_tol=1e-6, rel_tol=1e-6):
            issues.append("Pipeline time differs from measured stage sum")
    return {"scene_id": row["scene_id"], "passed": not issues, "issues": issues,
            "business_sha256": canonical_sha(sorted(zip(ids, positions.tolist(), demands.tolist()))),
            "recomputed": values, "qualified_edge_beam_ids": exempt,
            "total_pipeline_seconds": total_seconds}


def audit_results(results, expected_scenes):
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "expected_scenes": expected_scenes,
              "independent_implementation": "NumPy only; no planning/environment imports",
              "tolerances": {"bps_absolute": BPS_ATOL, "other_metric_absolute": OTHER_ATOL,
                             "relative": RTOL, "movement_legality_allowance_deg": MOVEMENT_ATOL_DEG},
              "sources": {}, "methods": {}, "issues": []}
    reference = None
    for label, path in results.items():
        rows, config = read_bundle(path)
        report["sources"][label] = {"path": str(path), "sha256": sha(path)}
        ids = [row.get("scene_id") for row in rows]
        if len(rows) != expected_scenes or len(set(ids)) != len(ids):
            report["issues"].append(f"{label}: scene count or uniqueness failure")
        audits = []
        for index, row in enumerate(rows):
            try:
                audits.append(audit_row(row, config))
            except (KeyError, ValueError, TypeError, IndexError, FloatingPointError) as exc:
                audits.append({"scene_id": row.get("scene_id"), "passed": False,
                               "issues": [f"Cannot verify scene {index}: {exc}"]})
        report["methods"][label] = {"scene_count": len(rows), "passed_scenes": sum(a["passed"] for a in audits),
                                    "declared_initialization_failures": sum(a.get("recomputed", {}) is None for a in audits),
                                    "scene_audits": audits}
        businesses = {a["scene_id"]: a.get("business_sha256") for a in audits}
        if reference is None:
            reference = businesses
        elif businesses != reference:
            report["issues"].append(f"{label}: scenes or actual aircraft inputs differ between methods")
    report["passed"] = bool(results) and not report["issues"] and all(
        method["passed_scenes"] == expected_scenes for method in report["methods"].values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, help="LABEL=/path/results.json")
    parser.add_argument("--expected-scenes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_results(parse_results(args.result), args.expected_scenes)
    write_json(args.output / "audit.json", report)
    print(json.dumps({"passed": report["passed"], "methods": len(report["methods"]),
                      "issues": report["issues"]}, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
