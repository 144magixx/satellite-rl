"""Independent audit of the seven first-stage workbench evaluation results.

Read-only toward frozen inputs and workbench jobs. Recompute aircraft coverage
from saved positions, demand, terminal centers and active beams with NumPy
float64, then publish an audit ledger, raw distributions and paired plots.
No PPO/environment module or training entry is imported here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


LABELS = ("R0", "R1", "R2", "C0", "C1", "C2", "P0")
INITIALIZER = {label: "rule" if label.startswith("R") else
               "cluster" if label.startswith("C") else "empty" for label in LABELS}
METHOD = {"R0": "none", "R1": "local_search", "R2": "ppo",
          "C0": "none", "C1": "local_search", "C2": "ppo", "P0": "ppo"}
PAIRS = (("R2", "R0"), ("R2", "R1"), ("C2", "C0"), ("C2", "C1"),
         ("P0", "R0"), ("P0", "C0"), ("P0", "R2"), ("P0", "C2"))
METRICS = ("coverage_demand", "coverage_users", "uncovered_demand_bps",
           "uncovered_users", "load_cv", "load_std_bps", "max_load_bps",
           "empty_beams", "active_beams", "movement_mean_deg", "movement_max_deg",
           "movement_applicable", "feasible", "score")
SCORE_DIRECTIONS = {"coverage_demand": 1, "coverage_users": 1,
                    "uncovered_demand_bps": -1, "load_cv": -1, "feasible": 1}
EARTH_KM = 6371.0
ORBIT_KM = 42164.0
SAT_LON_DEG = 13.0
BEAM_COUNT = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_business_sha(aircraft_ids, positions, demands):
    """Results omit observation times; hash every saved ID/position/request."""
    records = sorted((identifier, float(position[0]), float(position[1]), float(demand))
                     for identifier, position, demand in zip(aircraft_ids, positions, demands))
    return hashlib.sha256(json.dumps(records, ensure_ascii=False, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def original_business_sha(records):
    rows = sorted((str(row["aircraft_id"]), str(row.get("timestamp_utc", "")),
                   float(row["lat"]), float(row["lon"]), float(row["demand_bps"]))
                  for row in records)
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def load_original_sources(directory, issues):
    directory = Path(directory)
    record_path = directory / "record.json"
    if not record_path.is_file():
        issues.append(f"original aircraft source record missing: {record_path}")
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    source = record.get("source") or {}
    if (record.get("count") != 100 or source.get("partition_role") != "test" or
            source.get("source_dates") != ["2024-01-22"] or
            len(source.get("snapshots", [])) != 100 or len(record.get("files", [])) != 100):
        issues.append("original aircraft record is not the frozen 100-scene 2024-01-22 test partition")
        return None
    originals = {}
    for index in range(100):
        path = directory / f"source_observations_{index:05d}.jsonl.gz"
        if not path.is_file():
            issues.append(f"original aircraft file missing: {path}")
            continue
        actual_sha = sha256(path)
        snapshot = source["snapshots"][index]
        reference = snapshot.get("source_observations_ref") or {}
        if (reference.get("sha256") != actual_sha or
                record["files"][index].get("sha256") != actual_sha or
                Path(reference.get("path", "")).resolve() != path.resolve()):
            issues.append(f"original aircraft scene {index}: compressed bytes/SHA differs from record")
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            if len(records) != reference.get("count"):
                issues.append(f"original aircraft scene {index}: row count differs from record")
            business_sha = original_business_sha(records)
            if (business_sha != snapshot.get("business_sha256") or
                    business_sha != record["files"][index].get("scene_id")):
                issues.append(f"original aircraft scene {index}: raw business SHA differs from record")
            originals[index] = {"sha256": actual_sha, "scene_id": business_sha,
                                "aircraft_ids": [row["aircraft_id"] for row in records],
                                "positions": np.asarray([[row["lat"], row["lon"]]
                                                         for row in records], dtype=np.float64),
                                "demands": np.asarray([row["demand_bps"] for row in records],
                                                      dtype=np.float64)}
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            issues.append(f"original aircraft scene {index}: cannot read raw business: {error}")
    return {"record_sha256": sha256(record_path), "scenes": originals}


def rays(locations):
    locations = np.asarray(locations, dtype=np.float64)
    lat = np.deg2rad(locations[..., 0])
    lon = np.deg2rad(locations[..., 1])
    surface = EARTH_KM * np.stack((np.cos(lat) * np.cos(lon),
                                  np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)
    sat_lon = np.deg2rad(SAT_LON_DEG)
    satellite = ORBIT_KM * np.array([np.cos(sat_lon), np.sin(sat_lon), 0.0])
    vectors = surface - satellite
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def view_angles(first, second):
    chord = np.linalg.norm(first - second, axis=-1)
    return np.rad2deg(2.0 * np.arcsin(np.minimum(chord / 2.0, 1.0)))


def visible(positions):
    positions = np.asarray(positions, dtype=np.float64)
    lat, lon = positions[:, 0], positions[:, 1]
    cosine = np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon - SAT_LON_DEG))
    elevation = np.rad2deg(np.arctan2(
        cosine - EARTH_KM / ORBIT_KM, np.sqrt(np.maximum(0.0, 1.0 - cosine ** 2))))
    return ((lat >= 25.0) & (lat <= 80.0) & (lon >= -40.0) & (lon <= 50.0) &
            (elevation >= 10.0))


def layouts_hash(centers):
    return hashlib.sha256(np.asarray(centers, dtype="<f8").tobytes()).hexdigest()


def require_close(issues, context, key, actual, expected, *, rtol=1e-11, atol=1e-6):
    if not np.isclose(actual, expected, rtol=rtol, atol=atol):
        issues.append(f"{context}: {key} saved={actual!r}, recomputed={expected!r}")


def load_frozen(path, expected_sha, issues, label):
    """Verify a frozen aircraft NPZ independently of the project loader."""
    path = Path(path)
    if not path.is_file():
        issues.append(f"{label}: frozen test NPZ missing: {path}")
        return None
    actual_sha = sha256(path)
    if actual_sha != expected_sha:
        issues.append(f"{label}: frozen test NPZ SHA differs: {actual_sha} != {expected_sha}")
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["format_version"]) != "coverage_aircraft_v1":
                raise ValueError("not an aircraft-level coverage archive")
            lengths = archive["lengths"]
            positions = archive["positions"]
            demands = archive["demands"]
            ids = json.loads(str(archive["aircraft_ids_json"]))
            metadata = json.loads(str(archive["metadata_json"]))
            scene_ids = archive["scene_ids"].tolist()
            source_hashes = archive["source_hashes"].tolist()
            centers = archive["centers"]
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as error:
        issues.append(f"{label}: frozen NPZ cannot be read: {error}")
        return None
    if (lengths.shape != (100,) or positions.ndim != 3 or positions.shape[0] != 100 or
            positions.shape[2] != 2 or demands.shape != positions.shape[:2] or
            centers.shape != (100, BEAM_COUNT, 2) or len(ids) != 100 or
            len(scene_ids) != 100 or len(source_hashes) != 100):
        issues.append(f"{label}: frozen NPZ does not contain 100 aligned aircraft scenes")
        return None
    return {"path": str(path), "sha256": actual_sha, "lengths": lengths,
            "positions": positions, "demands": demands, "ids": ids,
            "scene_ids": scene_ids, "source_hashes": source_hashes, "centers": centers,
            "metadata": metadata}


def validate_input(label, row, frozen, originals, issues):
    context = f"{label}/scene_{row.get('scene_index')}"
    supplied = row.get("input") or {}
    try:
        ids = supplied["aircraft_ids"]
        positions = np.asarray(supplied["positions"], dtype=np.float64)
        demands = np.asarray(supplied["demands_bps"], dtype=np.float64)
        index = int(row["scene_index"])
        if not 0 <= index < 100:
            raise ValueError("scene_index outside frozen 100")
        if (positions.shape != (len(ids), 2) or demands.shape != (len(ids),) or
                len(ids) == 0 or len(ids) != len(set(ids)) or
                not np.isfinite(positions).all() or not np.isfinite(demands).all() or
                np.any(demands < 0) or not np.all(visible(positions))):
            raise ValueError("invalid aircraft IDs, coordinates or requests")
        if frozen is not None:
            size = int(frozen["lengths"][index])
            if (size != len(ids) or ids != frozen["ids"][index] or
                    not np.array_equal(positions, frozen["positions"][index, :size]) or
                    not np.array_equal(demands, frozen["demands"][index, :size]) or
                    row["scene_id"] != frozen["scene_ids"][index] or
                    row["scene_sha256"] != frozen["source_hashes"][index]):
                issues.append(f"{context}: result aircraft business or source identity differs from frozen NPZ")
            initial = supplied.get("initial_centers")
            if initial is not None and not np.array_equal(
                    np.asarray(initial, dtype=np.float64), frozen["centers"][index]):
                issues.append(f"{context}: saved initial layout differs from frozen NPZ")
        if originals is not None:
            original = originals["scenes"].get(index)
            if original is None:
                issues.append(f"{context}: original aircraft business is unavailable")
            elif (row["scene_id"] != original["scene_id"] or
                  row["scene_sha256"] != original["sha256"] or
                  ids != original["aircraft_ids"] or
                  not np.array_equal(positions, original["positions"]) or
                  not np.array_equal(demands, original["demands"])):
                issues.append(f"{context}: result aircraft business differs from original gzip source")
        if row.get("source_hash") not in (None, row.get("scene_sha256")):
            issues.append(f"{context}: source_hash and scene_sha256 differ")
        return {"ids": ids, "positions": positions, "demands": demands,
                "business_sha256": canonical_business_sha(ids, positions, demands),
                "requested_demand_bps": float(demands.sum()), "index": index}
    except (KeyError, TypeError, ValueError) as error:
        issues.append(f"{context}: cannot validate input: {error}")
        return None


def physical_metrics(positions, demands, centers, active, initial, options):
    center_rays = rays(centers)
    active_ids = np.flatnonzero(active)
    assignment = np.full(len(demands), -1, dtype=np.int64)
    if len(active_ids):
        user_rays = rays(positions)
        squared = np.sum((user_rays[:, None, :] - center_rays[active_ids][None, :, :]) ** 2, axis=-1)
        local_nearest = np.argmin(squared, axis=1)
        nearest = active_ids[local_nearest]
        radius = float(options["beam_width_deg"]) / 2.0
        limit = 4.0 * np.sin(np.deg2rad(radius) / 2.0) ** 2 + 1e-14
        covered = squared[np.arange(len(demands)), local_nearest] <= limit
        assignment[covered] = nearest[covered]
    else:
        covered = np.zeros(len(demands), dtype=bool)
    loads = np.bincount(assignment[covered], weights=demands[covered], minlength=BEAM_COUNT)
    total = float(demands.sum())
    covered_total = float(demands[covered].sum())
    uncovered_total = float(demands[~covered].sum())
    mean = float(loads.mean())
    std = float(loads.std(ddof=0))
    cv = std / mean if mean > 0 else 0.0
    movement_applicable = options["initialization"] != "empty"
    movement = view_angles(center_rays, rays(initial)) if movement_applicable else np.zeros(BEAM_COUNT)
    coverage_demand = covered_total / total if total > 0 else 0.0
    coverage_users = float(np.count_nonzero(covered) / len(demands))
    empty = int(np.count_nonzero(loads <= 0))
    feasible = bool(coverage_demand >= options["min_coverage_demand"] and
                    coverage_users >= options["min_coverage_users"] and
                    empty == 0 and len(active_ids) == BEAM_COUNT)
    score = (options["score_coverage_demand_weight"] * coverage_demand +
             options["score_coverage_users_weight"] * coverage_users -
             options["score_load_cv_weight"] * cv -
             options["score_movement_weight"] * float(movement.mean()) / options["max_movement_deg"])
    return {"assignment": assignment, "loads": loads,
            "coverage_demand": coverage_demand,
            "coverage_users": coverage_users,
            "uncovered_demand_bps": uncovered_total,
            "uncovered_users": int(np.count_nonzero(~covered)),
            "load_cv": cv, "load_std_bps": std,
            "max_load_bps": float(loads.max()), "empty_beams": empty,
            "active_beams": int(len(active_ids)),
            "movement_mean_deg": float(movement.mean()),
            "movement_max_deg": float(movement.max()),
            "movement_applicable": movement_applicable,
            "feasible": feasible, "score": float(score),
            "total_demand_bps": total,
            "covered_demand_bps": covered_total}


def audit_terminal(label, row, source, options, issues):
    context = f"{label}/scene_{source['index']}"
    saved_metrics = row.get("metrics")
    failed = (saved_metrics is None or
              row.get("input", {}).get("metadata", {}).get("initialization_valid") is False)
    if saved_metrics is None:
        if row.get("status") != "initialization_failed":
            issues.append(f"{context}: missing terminal metrics without initialization_failed status")
        return {"failed": True, "physical": None, "summary": {
            "coverage_demand": 0.0, "coverage_users": 0.0,
            "uncovered_demand_bps": source["requested_demand_bps"],
            "uncovered_users": len(source["ids"]), "feasible": 0.0}}
    centers_raw = row.get("final_centers", row.get("final_positions"))
    associated_raw = row.get("associations", row.get("final_associations"))
    try:
        centers = np.asarray(centers_raw, dtype=np.float64)
        associated = np.asarray(associated_raw, dtype=np.int64)
        loads_saved = np.asarray(row["beam_loads_bps"], dtype=np.float64)
        initial_raw = row.get("input", {}).get("initial_centers")
        initial = np.asarray(initial_raw, dtype=np.float64)
        if (centers.shape != (BEAM_COUNT, 2) or initial.shape != (BEAM_COUNT, 2) or
                associated.shape != (len(source["ids"]),) or
                loads_saved.shape != (BEAM_COUNT,) or
                not np.isfinite(centers).all() or not np.isfinite(initial).all() or
                not np.all(visible(centers))):
            raise ValueError("invalid terminal centers, active beams, association or loads")
        if label == "P0":
            if "active_beam_mask" not in row:
                raise ValueError("P0 requires explicit active_beam_mask")
            active = np.asarray(row["active_beam_mask"], dtype=bool)
            if (active.shape != (BEAM_COUNT,) or not active.all() or
                    not np.array_equal(initial, np.tile([45.0, 13.0], (BEAM_COUNT, 1))) or
                    row["input"]["metadata"].get("initial_active_count") != 0):
                raise ValueError("P0 must finish 100 active placements from inactive parking")
            candidates = {tuple(value) for value in source["positions"][source["demands"] > 0]}
            selected = [tuple(value) for value in centers]
            if len(set(selected)) != BEAM_COUNT or any(value not in candidates for value in selected):
                raise ValueError("P0 contains duplicate or non-input positive-request center")
        else:
            active = np.asarray(row.get("active_beam_mask", [True] * BEAM_COUNT), dtype=bool)
            if active.shape != (BEAM_COUNT,) or not active.all():
                raise ValueError("R/C terminal must retain all 100 active beams")
            movement = view_angles(rays(centers), rays(initial))
            if float(movement.max()) > options["max_movement_deg"] + 1e-10:
                raise ValueError("microadjustment exceeds cumulative satellite-view movement bound")
        physical = physical_metrics(source["positions"], source["demands"],
                                    centers, active, initial, options)
        if not np.array_equal(associated, physical["assignment"]):
            issues.append(f"{context}: saved unique aircraft associations differ")
        if not np.allclose(loads_saved, physical["loads"], rtol=1e-12, atol=1e-5):
            issues.append(f"{context}: saved all-100-beam loads differ")
        if not np.isclose(float(loads_saved.sum() + physical["uncovered_demand_bps"]),
                          source["requested_demand_bps"], rtol=1e-12, atol=1e-5):
            issues.append(f"{context}: source demand is not conserved")
        for key in METRICS:
            if key not in saved_metrics:
                issues.append(f"{context}: missing metric {key}")
                continue
            expected = physical[key]
            actual = saved_metrics[key]
            if isinstance(expected, bool):
                if bool(actual) != expected:
                    issues.append(f"{context}: saved {key} differs from physical value")
            elif key in ("empty_beams", "active_beams", "uncovered_users"):
                if int(actual) != expected:
                    issues.append(f"{context}: saved {key} differs from physical count")
            else:
                require_close(issues, context, key, float(actual), float(expected),
                              rtol=1e-11, atol=1e-5 if key.endswith("bps") else 1e-9)
        for name, value in (("initial_layout_hash", initial), ("final_layout_hash", centers)):
            if row.get(name) is not None and row[name] != layouts_hash(value):
                issues.append(f"{context}: {name} differs from saved center bytes")
        expected_uncovered_ids = [source["ids"][i] for i in np.flatnonzero(physical["assignment"] < 0)]
        if "uncovered_aircraft_ids" in row and row["uncovered_aircraft_ids"] != expected_uncovered_ids:
            issues.append(f"{context}: uncovered aircraft IDs differ")
        if "uncovered_aircraft_indices" in row and row["uncovered_aircraft_indices"] != [
                int(i) for i in np.flatnonzero(physical["assignment"] < 0)]:
            issues.append(f"{context}: uncovered aircraft indices differ")
        summary = {key: (0.0 if failed else float(physical[key]))
                   for key in ("coverage_demand", "coverage_users", "feasible")}
        summary["uncovered_demand_bps"] = (source["requested_demand_bps"] if failed else
                                           physical["uncovered_demand_bps"])
        summary["uncovered_users"] = len(source["ids"]) if failed else physical["uncovered_users"]
        return {"failed": failed, "physical": physical, "summary": summary}
    except (KeyError, TypeError, ValueError) as error:
        issues.append(f"{context}: cannot audit terminal: {error}")
        return None


def audit_group(label, result, result_sha, originals, issues, *, require_frozen):
    metadata = result.get("metadata") or {}
    options = (metadata.get("config") or {}).get("coverage") or {}
    evaluation = (metadata.get("config") or {}).get("evaluation") or {}
    if (metadata.get("research_stage") != "coverage" or
            metadata.get("method") != METHOD[label] or
            options.get("initialization") != INITIALIZER[label] or
            options.get("beam_count") != BEAM_COUNT or
            options.get("beam_width_deg") != 0.5 or
            evaluation.get("repeats") != 1):
        issues.append(f"{label}: frozen group identity, 100 beams or one-repeat rule differs")
    record = (metadata.get("datasets") or {}).get("test") or {}
    frozen = None
    if require_frozen:
        if not record.get("path") or not record.get("sha256"):
            issues.append(f"{label}: frozen test NPZ path/SHA missing")
        else:
            frozen = load_frozen(record["path"], record["sha256"], issues, label)
            if (frozen is not None and originals is not None and
                    frozen["metadata"].get("source_record_sha256") != originals["record_sha256"]):
                issues.append(f"{label}: frozen NPZ points to a different original aircraft record")
    raw_rows = result.get("scenes") or []
    if len(raw_rows) != 100:
        issues.append(f"{label}: expected 100 independent test scenes, got {len(raw_rows)}")
    rows = {}
    for raw in raw_rows:
        source = validate_input(label, raw, frozen, originals, issues)
        if source is None:
            continue
        index = source["index"]
        if index in rows:
            issues.append(f"{label}: duplicate scene_index {index}")
            continue
        audited = audit_terminal(label, raw, source, options, issues)
        if audited is None:
            continue
        rows[index] = {"scene_index": index, "scene_id": raw["scene_id"],
                       "source_sha256": raw["scene_sha256"],
                       "business_input_sha256": source["business_sha256"],
                       "requested_demand_bps": source["requested_demand_bps"],
                       "aircraft_count": len(source["ids"]), "status": raw.get("status"),
                       "initializer_failed": audited["failed"],
                       "summary_metrics": audited["summary"],
                       "physical_metrics": ({key: float(audited["physical"][key])
                                             for key in METRICS}
                                            if audited["physical"] is not None else None)}
    if set(rows) != set(range(100)):
        issues.append(f"{label}: missing one or more of the frozen 100 scene indices")
    ordered = [rows[index] for index in sorted(rows)]
    failed = sum(row["initializer_failed"] for row in ordered)
    physical_rows = [row["physical_metrics"] for row in ordered
                     if row["physical_metrics"] is not None]
    movement_rows = [row for row in physical_rows if row["movement_applicable"]]
    total_requested = sum(row["requested_demand_bps"] for row in ordered)
    total_uncovered = sum(row["summary_metrics"]["uncovered_demand_bps"] for row in ordered)
    summary = {"scene_count": len(ordered), "failed_scene_count": failed,
               "valid_scene_count": len(ordered) - failed,
               "valid_metric_scene_count": len(physical_rows),
               "movement_metric_scene_count": len(movement_rows),
               "coverage_demand": float(np.mean([row["summary_metrics"]["coverage_demand"]
                                                 for row in ordered])) if ordered else 0.0,
               "coverage_users": float(np.mean([row["summary_metrics"]["coverage_users"]
                                                for row in ordered])) if ordered else 0.0,
               "feasible": float(np.mean([row["summary_metrics"]["feasible"]
                                          for row in ordered])) if ordered else 0.0,
               "requested_demand_bps": total_requested,
               "total_uncovered_demand_bps": total_uncovered,
               "total_uncovered_users": float(sum(row["summary_metrics"]["uncovered_users"]
                                                    for row in ordered)),
               "uncovered_demand_bps": total_uncovered / len(ordered) if ordered else 0.0,
               "uncovered_users": float(np.mean([row["summary_metrics"]["uncovered_users"]
                                                 for row in ordered])) if ordered else 0.0,
               "pooled_demand_coverage": (1.0 - total_uncovered / total_requested
                                          if total_requested > 0 else 0.0)}
    for key in ("load_cv", "load_std_bps", "max_load_bps", "empty_beams",
                "active_beams", "score"):
        if physical_rows:
            summary[key] = float(np.mean([row[key] for row in physical_rows]))
    for key in ("movement_mean_deg", "movement_max_deg"):
        if movement_rows:
            summary[key] = float(np.mean([row[key] for row in movement_rows]))
    observed = result.get("summary") or {}
    for key in summary:
        if key not in observed:
            issues.append(f"{label}: saved summary missing {key}")
            continue
        require_close(issues, label, f"summary.{key}", float(observed[key]),
                      float(summary[key]), rtol=1e-10,
                      atol=1e-5 if key.endswith("bps") else 1e-9)
    return {"result_sha256": result_sha, "frozen_dataset_sha256": frozen["sha256"] if frozen else None,
            "saved_summary": observed, "audited_summary": summary, "scenes": ordered}


def paired(grouped, issues):
    baseline = grouped["R0"]["scenes"]
    identities = {row["scene_index"]: (row["scene_id"], row["source_sha256"],
                                       row["business_input_sha256"], row["requested_demand_bps"],
                                       row["aircraft_count"]) for row in baseline}
    for label, group in grouped.items():
        for row in group["scenes"]:
            if identities.get(row["scene_index"]) != (
                    row["scene_id"], row["source_sha256"], row["business_input_sha256"],
                    row["requested_demand_bps"], row["aircraft_count"]):
                issues.append(f"{label}/scene_{row['scene_index']}: input business/SHA differs from R0")
    groups_by_index = {label: {row["scene_index"]: row for row in group["scenes"]}
                       for label, group in grouped.items()}
    result = {}
    for newer, older in PAIRS:
        records = []
        for index in range(100):
            one = groups_by_index[newer].get(index)
            zero = groups_by_index[older].get(index)
            if one is None or zero is None:
                continue
            delta = {}
            for key in SCORE_DIRECTIONS:
                if key == "load_cv":
                    a = one["physical_metrics"]
                    b = zero["physical_metrics"]
                    delta[key] = (a[key] - b[key] if a is not None and b is not None and
                                  not one["initializer_failed"] and not zero["initializer_failed"]
                                  else None)
                else:
                    delta[key] = (one["summary_metrics"][key] - zero["summary_metrics"][key])
            records.append({"scene_index": index, "scene_id": one["scene_id"],
                            "business_input_sha256": one["business_input_sha256"],
                            "delta_new_minus_old": delta,
                            "both_feasible": bool(one["summary_metrics"]["feasible"] and
                                                  zero["summary_metrics"]["feasible"])})
        aggregates = {}
        for key, direction in SCORE_DIRECTIONS.items():
            values = np.asarray([row["delta_new_minus_old"][key] for row in records
                                 if row["delta_new_minus_old"][key] is not None], dtype=np.float64)
            epsilon = 1e-5 if key == "uncovered_demand_bps" else 1e-12
            oriented = direction * values
            aggregates[key] = {"paired_count": int(len(values)),
                               "mean_delta_new_minus_old": float(values.mean()) if len(values) else None,
                               "improved_scenes": int(np.count_nonzero(oriented > epsilon)),
                               "declined_scenes": int(np.count_nonzero(oriented < -epsilon)),
                               "unchanged_scenes": int(np.count_nonzero(np.abs(oriented) <= epsilon))}
        feasible_cv = [row["delta_new_minus_old"]["load_cv"] for row in records
                       if row["both_feasible"] and row["delta_new_minus_old"]["load_cv"] is not None]
        aggregates["load_cv_when_both_feasible"] = {
            "paired_count": len(feasible_cv),
            "mean_delta_new_minus_old": float(np.mean(feasible_cv)) if feasible_cv else None}
        result[f"{newer}-{older}"] = {"new": newer, "old": older,
                                      "interpretation": ("full-route difference; action space and episode length differ"
                                                         if "P0" in (newer, older) else "same initializer contribution"),
                                      "metrics": aggregates, "scenes": records}
    return result


def figures(grouped, pairs, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, key, title in zip(axes, ("coverage_demand", "uncovered_demand_bps", "feasible"),
                              ("Demand coverage", "Uncovered demand (Mbps)", "Feasible scenes")):
        values = [[row["summary_metrics"][key] / (1e6 if key == "uncovered_demand_bps" else 1.0)
                   for row in grouped[label]["scenes"]] for label in LABELS]
        ax.boxplot([item or [np.nan] for item in values], labels=LABELS, showfliers=True)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = destination / "all_scene_distributions.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["all_scene_distributions"] = str(path)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for ax, pair_name in zip(axes.flat, ("R2-R0", "C2-C0", "P0-R2", "P0-C2")):
        rows = pairs[pair_name]["scenes"]
        ax.plot([row["scene_index"] for row in rows],
                [row["delta_new_minus_old"]["coverage_demand"] for row in rows],
                marker=".", markersize=3, linewidth=0.55)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(pair_name + " demand coverage delta")
        ax.set_ylabel("New minus old")
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Independent test scene index")
    fig.tight_layout()
    path = destination / "paired_coverage_delta.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["paired_coverage_delta"] = str(path)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for ax, pair_name in zip(axes.flat, ("R2-R0", "C2-C0", "P0-R2", "P0-C2")):
        rows = pairs[pair_name]["scenes"]
        ax.plot([row["scene_index"] for row in rows],
                [row["delta_new_minus_old"]["load_cv"] if
                 row["delta_new_minus_old"]["load_cv"] is not None else np.nan for row in rows],
                marker=".", markersize=3, linewidth=0.55)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(pair_name + " load CV delta (valid initializer pairs)")
        ax.set_ylabel("New minus old")
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Independent test scene index")
    fig.tight_layout()
    path = destination / "paired_load_cv_delta.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["paired_load_cv_delta"] = str(path)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = [[row["physical_metrics"]["load_cv"] for row in grouped[label]["scenes"]
             if row["physical_metrics"] is not None] for label in LABELS]
    ax.boxplot([item or [np.nan] for item in data], labels=LABELS, showfliers=True)
    ax.set_title("Load CV: layouts with physical terminal evidence (failures kept raw)")
    ax.set_ylabel("Std / mean of all 100 beam loads")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = destination / "load_cv_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["load_cv_distribution"] = str(path)
    return paths


def markdown(grouped, pairs, plots, issues):
    lines = ["# 第一阶段七组逐场景独立审计", "",
             "按保存的逐机业务、终态波束中心、活跃束和唯一归属，使用独立 NumPy float64 几何重算。",
             "各组共同100个独立考核场景；初始化失败保留在分母，覆盖与可行率记0，未覆盖需求计全部请求。",
             "P0从空布局放置100束；R/C对初始100束执行微调。P0无移动项，三路线动作空间与回合长度不同。",
             "", "|组|需求覆盖|可行比例|失败场景|冻结输入SHA|",
             "|---|---:|---:|---:|---|"]
    for label in LABELS:
        group = grouped[label]
        value = group["audited_summary"]
        lines.append(f"|{label}|{value['coverage_demand']:.6f}|{value['feasible']:.6f}|"
                     f"{value['failed_scene_count']}|`{group['frozen_dataset_sha256'] or 'unavailable'}`|")
    lines += ["", "|配对（新－旧）|需求覆盖平均差|改善|下降|不变|",
              "|---|---:|---:|---:|---:|"]
    for name, pair in pairs.items():
        metric = pair["metrics"]["coverage_demand"]
        mean = metric["mean_delta_new_minus_old"]
        mean_text = f"{mean:.6f}" if mean is not None else "未记录"
        lines.append(f"|{name}|{mean_text}|{metric['improved_scenes']}|"
                     f"{metric['declined_scenes']}|{metric['unchanged_scenes']}|")
    lines += ["", "![逐场景分布](figures/all_scene_distributions.png)",
              "", "![同场景需求覆盖差值](figures/paired_coverage_delta.png)",
              "", "![同场景负载离散度差值](figures/paired_load_cv_delta.png)",
              "", "![全部100束负载离散度](figures/load_cv_distribution.png)", "",
              f"审计问题数：{len(issues)}。详细输入SHA、逐场景指标、配对差值及错误见 audit.json。",
              "结果文件未保存原始观测UTC时刻；业务SHA可从保存的ID/位置/需求重算，"
              "原始source gz SHA通过冻结NPZ记录与任务结果交叉核对。",
              "P0与微调的差值是完整路线对比，不能只归因于初始化方式；单训练种子也不支持稳定性结论。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory with R0.json ... P0.json from the workbench suite")
    parser.add_argument("--output", type=Path, required=True,
                        help="New audit directory; results are never modified")
    parser.add_argument("--source-directory", type=Path,
                        help="Original source_observations_*.jsonl.gz and record.json; required for formal audit")
    parser.add_argument("--skip-frozen-input", action="store_true",
                        help="Engineering smoke only; formal audit requires frozen NPZ SHA verification")
    args = parser.parse_args()
    if not args.skip_frozen_input and args.source_directory is None:
        parser.error("formal audit requires --source-directory for original aircraft/SHA verification")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"audit output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    issues = []
    originals = load_original_sources(args.source_directory, issues) if args.source_directory else None
    grouped = {}
    for label in LABELS:
        path = args.results_dir / f"{label}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing seven-group result: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        grouped[label] = audit_group(label, value, sha256(path), originals, issues,
                                     require_frozen=not args.skip_frozen_input)
    paired_results = paired(grouped, issues)
    plots = figures(grouped, paired_results, output / "figures")
    audit = {"status": "passed" if not issues and not args.skip_frozen_input else
               "engineering_check_only" if not issues else "failed",
             "created_at": datetime.now(timezone.utc).isoformat(),
             "auditor_sha256": sha256(Path(__file__)),
             "results_directory": str(args.results_dir.resolve()),
             "frozen_input_verified": not args.skip_frozen_input,
             "original_aircraft_record_sha256": originals["record_sha256"] if originals else None,
             "metric_policy": {"beam_count": 100, "full_width_deg": 0.5,
                               "association": "nearest active satellite ray, lowest beam ID ties",
                               "failure_denominator": "all 100 scenes, failed initializer counts zero coverage/feasible",
                               "movement": "satellite-view angle to each initial center; P0 not applicable",
                               "pair_delta": "new minus old, one independent scene per index"},
             "issues": issues, "groups": grouped, "paired": paired_results, "plots": plots}
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2,
                                                  allow_nan=False) + "\n", encoding="utf-8")
    (output / "审计说明.md").write_text(markdown(grouped, paired_results, plots, issues),
                                           encoding="utf-8")
    print(json.dumps({"status": audit["status"], "issues": len(issues),
                      "audit": str(output / "audit.json"), "scenes_per_group": {
                          label: len(value["scenes"]) for label, value in grouped.items()}},
                     ensure_ascii=False))
    if issues:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
