"""Strict, repeatable S1-V10 analysis from locally synchronized workbench ledgers.

The program reads frozen controller identities and the SHA-verified local text
mirror.  It never calls a remote service, starts training, or fills unavailable
physical results with zero.  It can emit a factual partial-progress report.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from satellite_rl.power_refinement.physics import _sgm
from scripts.stage1_variable import controller, ops
from scripts.s2p19 import archive_sources


MULTIPLIERS = (0.75, 1.0, 1.25)
ROUTE_NAMES = {
    "R01": "容量压力逐步增束", "R02": "候选集整数覆盖", "R03": "固定中心需求再归属",
    "R04": "热点拆分与轻载合并", "R05": "可变宽度布局", "R06": "需求质量传输与整数修复",
    "R07": "区域拆除重建", "R08": "区域重构PPO", "R09": "空布局构造PPO",
    "R10": "下游反馈重构", "R11": "需求扰动鲁棒布局", "R12": "连续时刻重配置",
}
SCENE_COLUMNS = (
    "task_key", "method", "route", "role", "purpose", "run_id", "port",
    "source_commit", "output_kind", "scene_id",
    "scene_sha256", "sequence_id", "selected_time_utc", "temporal_mode",
    "demand_multiplier", "aircraft_count", "covered_users", "coverage_users",
    "coverage_demand", "coverage_100", "demand_conserved", "beam_count",
    "active_business_beams", "zero_demand_beams", "load_cv", "in_band_fraction",
    "peak_load_ratio", "reference_capacity_bps", "reference_unserved_bps",
    "total_demand_bps", "uncovered_demand_bps", "center_carried_fraction",
    "user_tdma_carried_fraction", "beam_mean_si", "beam_mean_sgm",
    "total_power_w", "complete_allocation", "timing_granularity", "timing_source",
    "method_seconds_before_common_downstream", "initialization_seconds",
    "algorithm_seconds", "geometry_seconds", "final_check_seconds",
    "downstream_interface_seconds", "allocation_seconds", "physical_verification_seconds",
    "downstream_total_seconds", "total_seconds", "geometry_calls",
    "geometry_budget_charges", "ray_pair_checks", "physical_allocation_calls",
    "allocation_decisions", "independent_physics_checks", "aircraft_link_checks",
)
MEASURES = (
    "coverage_users", "coverage_demand", "load_cv", "in_band_fraction",
    "peak_load_ratio", "beam_count", "center_carried_fraction",
    "user_tdma_carried_fraction", "beam_mean_si", "beam_mean_sgm",
    "method_seconds_before_common_downstream", "initialization_seconds",
    "algorithm_seconds", "final_check_seconds",
    "downstream_interface_seconds", "allocation_seconds", "physical_verification_seconds",
    "downstream_total_seconds", "total_seconds", "geometry_calls",
    "geometry_budget_charges", "ray_pair_checks", "physical_allocation_calls",
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source_ref(source):
    if isinstance(source, (str, Path)):
        return {"path": str(source)}
    if "source_ref" in source:
        return source["source_ref"]
    if "zip_path" in source:
        return source
    path = Path(source["path"])
    if "bytes" in source and path.stat().st_size != source["bytes"]:
        raise ValueError(f"plain source byte count differs: {path}")
    return {"path": str(path), "sha256": source.get("sha256")}


def _json_source(source):
    with archive_sources.open_text_source(_source_ref(source)) as stream:
        return json.load(stream)


def _iter_jsonl(source, *, with_line_number=False):
    """Parse one row at a time, retaining the exact bad-file line number."""
    ref = _source_ref(source)
    label = archive_sources.source_label(ref)
    with archive_sources.open_text_source(ref) as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL row must be an object")
                yield (number, row) if with_line_number else row
            except json.JSONDecodeError as error:
                raise ValueError(f"{label}:{number}: malformed JSONL: {error.msg}") from error
            except ValueError as error:
                raise ValueError(f"{label}:{number}: {error}") from error


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite(value, label, *, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{label} is missing or nonfinite")
    return float(value)


def _stats(values):
    finite = [float(x) for x in values if x is not None and np.isfinite(x)]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    array = np.asarray(finite, dtype=float)
    return {"count": len(array), "mean": float(array.mean()),
            "median": float(np.median(array)), "p95": float(np.percentile(array, 95))}


def _method(task):
    route = task["route"]
    suffix = ("（序列温启动）" if task["role"] == "sequence" and route == "R12" else
              "（序列独立帧）" if task["role"] == "sequence" else
              "（静态冷启动）" if route == "R12" else "")
    return f"{route} {ROUTE_NAMES[route]}{suffix}"


def _scene_row(task, raw):
    """Recalculate accounting from full user/beam arrays before aggregation."""
    key = task["key"]
    m = raw.get("metrics")
    down = raw.get("downstream")
    beams = raw.get("beams")
    users = raw.get("users")
    if not isinstance(m, dict) or not isinstance(down, dict) or not isinstance(beams, list) or not isinstance(users, list):
        raise ValueError(f"{key}: terminal row lacks full geometry/physical/user arrays")
    n = int(m["beam_count"])
    if len(beams) != n or len(users) != int(m["aircraft_count"]):
        raise ValueError(f"{key}: actual beam or aircraft count differs")
    ids = [u["aircraft_id"] for u in users]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{key}: duplicate aircraft identity")
    total = _finite(m["total_demand_bps"], "total demand")
    if not np.isclose(sum(_finite(u["demand_bps"], "user demand") for u in users),
                      total, rtol=1e-10, atol=1e-5):
        raise ValueError(f"{key}: user demand not conserved")
    layout = raw.get("layout")
    if (not isinstance(layout, dict) or layout.get("aircraft_ids") != ids or
            len(layout.get("assignment", [])) != len(users) or
            len(layout.get("beam_ids", [])) != n):
        raise ValueError(f"{key}: layout aircraft/beam identities differ from user ledger")
    assigned_demand = np.zeros(n, dtype=float)
    uncovered_from_users = 0.0
    for index, user in enumerate(users):
        owner = user.get("assigned_beam")
        if owner != layout["assignment"][index]:
            if not (owner is None and layout["assignment"][index] == -1):
                raise ValueError(f"{key}: user assignment differs from layout")
        demand = _finite(user["demand_bps"], "user demand")
        if demand < 0:
            raise ValueError(f"{key}: negative user demand")
        if owner is None:
            uncovered_from_users += demand
            if user.get("assigned_beam_stable_id") is not None:
                raise ValueError(f"{key}: uncovered user has beam identity")
        else:
            if type(owner) is not int or not 0 <= owner < n:
                raise ValueError(f"{key}: user owner is outside actual beam range")
            if user.get("assigned_beam_stable_id") != layout["beam_ids"][owner]:
                raise ValueError(f"{key}: user stable beam identity differs")
            assigned_demand[owner] += demand
    beam_d = np.asarray([_finite(b["demand_bps"], "beam demand") for b in beams])
    uncovered = _finite(m["uncovered_demand_bps"], "uncovered demand")
    if not np.isclose(beam_d.sum() + uncovered, total, rtol=1e-10, atol=1e-5):
        raise ValueError(f"{key}: beam and uncovered demand do not sum to all aircraft")
    if not np.allclose(beam_d, assigned_demand, rtol=1e-10, atol=1e-5) or not np.isclose(
            uncovered, uncovered_from_users, rtol=1e-10, atol=1e-5):
        raise ValueError(f"{key}: per-beam or uncovered demand differs from user ownership")
    if (m.get("beam_ids") != layout["beam_ids"] or
            not np.allclose(np.asarray(m.get("beam_demand_bps", []), dtype=float),
                            beam_d, rtol=1e-10, atol=1e-5)):
        raise ValueError(f"{key}: core metric beam identities or demand differ")
    if not bool(m["demand_conserved"]) or bool(m["coverage_100"]) != bool(raw["coverage_complete"]):
        raise ValueError(f"{key}: conservation or full-coverage flag differs")
    if int(m["covered_users"]) != sum(u.get("assigned_beam") is not None for u in users):
        raise ValueError(f"{key}: covered-user count differs from ownership")
    covered_count = int(m["covered_users"])
    if (bool(m["coverage_100"]) != (covered_count == len(users)) or
            not np.isclose(_finite(m["coverage_users"], "coverage users"),
                           covered_count / len(users) if users else 1.0, atol=1e-10) or
            not np.isclose(_finite(m["coverage_demand"], "coverage demand"),
                           1.0 - uncovered / total if total else 1.0, atol=1e-10)):
        raise ValueError(f"{key}: coverage denominators differ from all aircraft")
    beam_c = np.zeros(n, dtype=float)
    beam_si = []
    beam_sgm = []
    for index, b in enumerate(beams):
        if b.get("beam_id") != index or b.get("stable_beam_id") != layout["beam_ids"][index]:
            raise ValueError(f"{key}: beam processing identity differs")
        d = beam_d[index]
        c = _finite(b.get("capacity_bps"), "terminal beam capacity")
        carried = _finite(b.get("carried_bps"), "terminal beam carried")
        beam_c[index] = c
        if c < 0 or not np.isclose(carried, min(c, d), rtol=1e-8, atol=1e-5):
            raise ValueError(f"{key}: terminal min(C,D) differs")
        if d > 0:
            ratio = c / d
            if not np.isclose(_finite(b.get("unclipped_capacity_demand_ratio"), "C/D"),
                              ratio, rtol=1e-8, atol=1e-8):
                raise ValueError(f"{key}: terminal beam C/D differs")
            si = _finite(b.get("si"), "terminal beam SI")
            sgm = _finite(b.get("sgm"), "terminal beam SGM")
            if not np.isclose(si, min(ratio, 1.0), rtol=1e-8, atol=1e-8):
                raise ValueError(f"{key}: terminal beam SI differs from C/D")
            beam_si.append(si)
            beam_sgm.append(sgm)
        elif (any(b.get(name) is not None for name in
                  ("unclipped_capacity_demand_ratio", "si", "sgm")) or
              c != 0 or carried != 0 or b.get("slots") != 0 or b.get("power_w") != 0):
            raise ValueError(f"{key}: zero-demand beam has fabricated service or RF resources")
    positive = beam_d[beam_d > 0]
    if len(positive):
        expected_sgm, _ = _sgm(beam_c[beam_d > 0], positive)
        if not np.allclose(beam_sgm, expected_sgm, rtol=1e-8, atol=1e-8):
            raise ValueError(f"{key}: terminal beam SGM differs from capacity and demand")
    mean = float(positive.mean()) if len(positive) else None
    band = (float(np.mean((positive >= .8 * mean) & (positive <= 1.2 * mean)))
            if mean is not None else None)
    if not isinstance(down.get("metrics"), dict) or not isinstance(down.get("timing"), dict):
        raise ValueError(f"{key}: downstream summary or segmented timing absent")
    dm, dt, calls = down["metrics"], down["timing"], down.get("calls")
    if (not isinstance(calls, dict) or
            any(type(calls.get(field)) is not int or calls[field] < 0 for field in
                ("layout_allocation_calls", "allocation_decisions",
                 "independent_physics_checks", "aircraft_link_checks"))):
        raise ValueError(f"{key}: physical call ledger missing or invalid")
    if type(dm.get("complete_allocation")) is not bool or type(dm.get("processed_count")) is not int:
        raise ValueError(f"{key}: allocation completion or processed count missing")
    if not 0 <= dm["processed_count"] <= len(positive) or (
            dm["complete_allocation"] and dm["processed_count"] != len(positive)):
        raise ValueError(f"{key}: allocation completion differs from processed beams")
    center = _finite(dm.get("demand_weighted_carried_fraction"), "center carried", nullable=True)
    direct_center = sum(b["carried_bps"] for b in beams) / total if total else None
    if (center is None) != (direct_center is None) or (center is not None and not np.isclose(
            center, direct_center, rtol=1e-8, atol=1e-8)):
        raise ValueError(f"{key}: center carried does not match beam ledger")
    if not np.isclose(_finite(dm.get("total_demand_bps"), "downstream total demand"),
                      total, rtol=1e-10, atol=1e-5):
        raise ValueError(f"{key}: downstream denominator differs from all users")
    if (not np.isclose(_finite(dm.get("carried_bps"), "downstream carried"),
                       sum(b["carried_bps"] for b in beams), rtol=1e-8, atol=1e-5) or
            not np.isclose(_finite(dm.get("uncovered_demand_bps"), "downstream uncovered"),
                           uncovered, rtol=1e-8, atol=1e-5)):
        raise ValueError(f"{key}: downstream carried or uncovered total differs")
    if len(positive):
        if (not np.isclose(_finite(dm.get("beam_mean_si"), "beam mean SI"),
                           np.mean(beam_si), rtol=1e-8, atol=1e-8) or
                not np.isclose(_finite(dm.get("beam_mean_sgm"), "beam mean SGM"),
                               np.mean(beam_sgm), rtol=1e-8, atol=1e-8)):
            raise ValueError(f"{key}: beam mean SI/SGM differs from terminal beams")
    elif dm.get("beam_mean_si") is not None or dm.get("beam_mean_sgm") is not None:
        raise ValueError(f"{key}: no-business beam SI/SGM mean must be missing")
    power = [_finite(b.get("power_w"), "beam power") for b in beams]
    if any(p < 0 for p in power) or not np.isclose(
            _finite(dm.get("total_power_w"), "total power"), sum(power), rtol=1e-8, atol=1e-6):
        raise ValueError(f"{key}: total power differs from actual beams")
    if sum(power) > 3000 + 1e-6:
        raise ValueError(f"{key}: total power exceeds common budget")
    compact_episode = (task["kind"] == "train" and raw.get("completed_episode") is True and
                       raw.get("terminal_physics_status") ==
                       "evaluated_all_actual_beams_and_users" and
                       "aircraft_service" not in down)
    service = ({"capacity_bps": [u.get("capacity_bps") for u in users],
                "airtime_fraction": [u.get("airtime_fraction") for u in users],
                "carried_bps": [u.get("carried_bps") for u in users],
                "si": [u.get("si") for u in users],
                "demand_weighted_carried_fraction": down.get("user_tdma_carried_fraction"),
                "aircraft_link_checks": calls.get("aircraft_link_checks")}
               if compact_episode else down.get("aircraft_service", {}))
    if not isinstance(service, dict):
        raise ValueError(f"{key}: aircraft TDMA service arrays missing")
    user_carried = _finite(service.get("demand_weighted_carried_fraction"),
                           "aircraft TDMA carried", nullable=True)
    count_users = len(users)
    capacity_values = service.get("capacity_bps")
    share_values = service.get("airtime_fraction")
    if (not isinstance(capacity_values, list) or len(capacity_values) != count_users or
            not isinstance(share_values, list) or len(share_values) != count_users):
        raise ValueError(f"{key}: aircraft capacity or airtime array length differs")
    user_d = np.asarray([u["demand_bps"] for u in users], dtype=float)
    user_c = np.asarray([_finite(v, "aircraft capacity") for v in capacity_values])
    shares = np.asarray([_finite(v, "aircraft airtime") for v in share_values])
    if np.any(user_c < 0) or np.any((shares < 0) | (shares > 1 + 1e-10)):
        raise ValueError(f"{key}: negative capacity or invalid aircraft airtime")
    if total > 0:
        names = (("carried_bps", "si") if compact_episode else
                 ("full_time_rate_bps", "carried_bps", "unclipped_capacity_demand_ratio", "si"))
        if any(not isinstance(service.get(name), list) or len(service[name]) != count_users
               for name in names):
            raise ValueError(f"{key}: positive-demand aircraft terminal arrays missing")
        user_min = np.asarray([_finite(v, "aircraft min(C,D)") for v in service["carried_bps"]])
        if not compact_episode:
            full_time = np.asarray([_finite(v, "aircraft full-time rate")
                                    for v in service["full_time_rate_bps"]])
            if np.any(full_time < 0) or not np.allclose(user_c, full_time * shares,
                                                        rtol=1e-8, atol=1e-5):
                raise ValueError(f"{key}: aircraft full-time rate and TDMA capacity differ")
        if not np.allclose(user_min, np.minimum(user_c, user_d), rtol=1e-8, atol=1e-5):
            raise ValueError(f"{key}: aircraft TDMA capacity or min(C,D) differs")
        for index, demand in enumerate(user_d):
            ratio = (None if compact_episode else
                     service["unclipped_capacity_demand_ratio"][index])
            si = service["si"][index]
            if demand > 0:
                expected = user_c[index] / demand
                if ((not compact_episode and not np.isclose(
                        _finite(ratio, "aircraft C/D"), expected, rtol=1e-8, atol=1e-8)) or
                        not np.isclose(_finite(si, "aircraft SI"), min(expected, 1.0),
                                       rtol=1e-8, atol=1e-8)):
                    raise ValueError(f"{key}: aircraft C/D or SI differs")
            elif (not compact_episode and ratio is not None) or si is not None:
                raise ValueError(f"{key}: zero-demand aircraft has fabricated C/D or SI")
        expected_carried = float(user_min.sum() / total)
        if user_carried is None or not np.isclose(user_carried, expected_carried,
                                                  rtol=1e-8, atol=1e-8):
            raise ValueError(f"{key}: aircraft carried fraction differs from all-user denominator")
        if not compact_episode:
            valid = user_d > 0
            met = int(np.count_nonzero(valid & (user_c >= user_d)))
            if (service.get("positive_demand_users") != int(valid.sum()) or
                    service.get("positive_demand_users_met") != met or
                    service.get("all_positive_demand_users_met") != bool(np.all(user_c[valid] >= user_d[valid]))):
                raise ValueError(f"{key}: aircraft demand-met counts differ from all positive users")
    elif (user_carried is not None or np.any(user_c != 0) or np.any(shares != 0)):
        raise ValueError(f"{key}: no-demand aircraft service must be explicitly missing or zero")
    for index, beam in enumerate(beams):
        members = [u for u, row in enumerate(users) if row.get("assigned_beam") == index]
        member_share = float(shares[members].sum())
        active = beam_d[index] > 0 and int(beam.get("slots", 0)) > 0
        if active and not np.isclose(member_share, 1.0, rtol=0, atol=1e-10):
            raise ValueError(f"{key}: aircraft airtime does not sum to one in active beam")
        if not active and not np.isclose(member_share, 0.0, rtol=0, atol=1e-10):
            raise ValueError(f"{key}: unserved or zero-demand beam duplicates airtime")
    if any(shares[u] != 0 or user_c[u] != 0 for u, row in enumerate(users)
           if row.get("assigned_beam") is None or user_d[u] == 0):
        raise ValueError(f"{key}: uncovered or zero-demand aircraft receives resources")
    link_checks = sum(row.get("assigned_beam") is not None and user_d[u] > 0 and
                      int(beams[row["assigned_beam"]].get("slots", 0)) > 0
                      for u, row in enumerate(users))
    if (service.get("aircraft_link_checks") != link_checks or
            calls.get("aircraft_link_checks") != link_checks):
        raise ValueError(f"{key}: aircraft link-check call ledger differs")
    for name in ("initialization_seconds", "algorithm_seconds", "final_check_seconds", "total_seconds"):
        if _finite(raw.get(name), name) < 0:
            raise ValueError(f"{key}: negative measured time")
    for name in ("interface_seconds", "allocation_seconds", "verification_seconds", "total_seconds"):
        if _finite(dt.get(name), f"downstream {name}") < 0:
            raise ValueError(f"{key}: negative downstream time")
    if not np.isclose(dt["total_seconds"], dt["interface_seconds"] +
                      dt["allocation_seconds"] + dt["verification_seconds"],
                      rtol=1e-7, atol=1e-4):
        raise ValueError(f"{key}: downstream segmented time differs from total")
    if not np.isclose(raw["total_seconds"], raw["initialization_seconds"] +
                      raw["algorithm_seconds"] + raw["final_check_seconds"] +
                      dt["total_seconds"], rtol=1e-7, atol=1e-4):
        raise ValueError(f"{key}: segmented wall time differs from total")
    output_kind = raw.get("initialization_info", {}).get("output_kind")
    if output_kind is None:
        output_kind = ("policy_deterministic_endpoint" if task["route"] in ("R08", "R09")
                       else "feedback_selected_terminal_layout" if task["route"] == "R10" else None)
    if output_kind is None:
        raise ValueError(f"{key}: output-kind provenance absent")
    learned = task["route"] in ("R08", "R09")
    timing_granularity = ("separate_initialization_and_policy_execution" if learned else
                          "combined_initialization_search_and_feedback_physics" if task["route"] == "R10" else
                          "combined_initialization_and_search")
    row = {"task_key": key, "method": _method(task), "route": task["route"],
           "role": task["role"], "purpose": task["purpose"],
           "run_id": task["run_id"], "port": task["port"],
           "source_commit": task["source_commit"], "output_kind": output_kind,
           "scene_id": raw["scene_id"], "scene_sha256": raw["scene_sha256"],
           "sequence_id": raw.get("sequence_id"),
           "selected_time_utc": raw.get("selected_time_utc"),
           "temporal_mode": raw.get("temporal_mode"),
           "demand_multiplier": _finite(raw["demand_multiplier"], "multiplier"),
           "aircraft_count": int(m["aircraft_count"]),
           "covered_users": int(m["covered_users"]),
           "coverage_users": _finite(m["coverage_users"], "coverage users"),
           "coverage_demand": _finite(m["coverage_demand"], "coverage demand"),
           "coverage_100": bool(m["coverage_100"]),
           "demand_conserved": bool(m["demand_conserved"]),
           "beam_count": n, "active_business_beams": len(positive),
           "zero_demand_beams": n - len(positive), "load_cv": _finite(m["load_cv"], "CV"),
           "in_band_fraction": band,
           "peak_load_ratio": _finite(m["peak_load_ratio"], "peak load"),
           "reference_capacity_bps": _finite(m["reference_capacity_bps"], "reference capacity"),
           "reference_unserved_bps": _finite(m["reference_unserved_bps"], "reference unserved"),
           "total_demand_bps": total, "uncovered_demand_bps": uncovered,
           "center_carried_fraction": center,
           "user_tdma_carried_fraction": user_carried,
           "beam_mean_si": _finite(dm.get("beam_mean_si"), "beam SI", nullable=True),
           "beam_mean_sgm": _finite(dm.get("beam_mean_sgm"), "beam SGM", nullable=True),
           "total_power_w": _finite(dm.get("total_power_w"), "total power"),
           "complete_allocation": bool(dm.get("complete_allocation")),
           "timing_granularity": timing_granularity,
           "timing_source": f"frozen_workbench_runner:{task['source_commit']}",
           "method_seconds_before_common_downstream": float(raw["initialization_seconds"] +
                                                            raw["algorithm_seconds"]),
           "initialization_seconds": float(raw["initialization_seconds"]) if learned else None,
           "algorithm_seconds": float(raw["algorithm_seconds"]) if learned else None,
           "geometry_seconds": (_finite(raw.get("geometry_seconds"), "geometry time", nullable=True)
                                if learned else None),
           "final_check_seconds": float(raw["final_check_seconds"]),
           "downstream_interface_seconds": float(dt["interface_seconds"]),
           "allocation_seconds": float(dt["allocation_seconds"]),
           "physical_verification_seconds": float(dt["verification_seconds"]),
           "downstream_total_seconds": float(dt["total_seconds"]),
           "total_seconds": float(raw["total_seconds"]),
           "geometry_calls": int(raw["geometry_calls"]),
           "geometry_budget_charges": int(raw["geometry_budget_charges"]),
           "ray_pair_checks": raw.get("ray_pair_checks"),
           "physical_allocation_calls": calls.get("layout_allocation_calls"),
           "allocation_decisions": calls.get("allocation_decisions"),
           "independent_physics_checks": calls.get("independent_physics_checks"),
           "aircraft_link_checks": calls.get("aircraft_link_checks")}
    return row


def _grid(task, rows):
    """No scenario/multiplier can disappear from a completed method."""
    seen = set()
    mother = defaultdict(set)
    hashes = defaultdict(set)
    for row in rows:
        factor = row["demand_multiplier"]
        if factor not in MULTIPLIERS:
            raise ValueError(f"{task['key']}: unexpected demand multiplier {factor}")
        identity = ((row["sequence_id"], row["selected_time_utc"])
                    if task["role"] == "sequence" else row["scene_id"])
        key = (identity, factor)
        if key in seen:
            raise ValueError(f"{task['key']}: duplicate mother scene and demand tier")
        seen.add(key)
        mother[identity].add(factor)
        hashes[identity].add(row["scene_sha256"])
        if task["role"] == "sequence":
            expected_mode = ("independent_scene" if task["route"] == "R04" else
                             "sequence_first_frame" if row["selected_time_utc"] ==
                             min(x["selected_time_utc"] for x in rows
                                 if x["sequence_id"] == row["sequence_id"]) else
                             "sequence_warm_start")
            if row["temporal_mode"] != expected_mode:
                raise ValueError(f"{task['key']}: sequence mode differs at {identity}")
        elif task["route"] == "R12" and row["temporal_mode"] != "cold_init":
            raise ValueError(f"{task['key']}: static R12 must be a cold start")
    if any(len(values) != 1 for values in hashes.values()):
        raise ValueError(f"{task['key']}: one mother has multiple source SHA values")
    expected_count = task.get("expected_count")
    complete = expected_count is not None and len(rows) == expected_count
    if complete and (len(mother) * 3 != expected_count or
                     any(levels != set(MULTIPLIERS) for levels in mother.values())):
        raise ValueError(f"{task['key']}: complete task lacks full mother×demand grid")
    return {"observed_rows": len(rows), "expected_rows": expected_count,
            "observed_mothers": len(mother),
            "complete": bool(complete),
            "incomplete_mothers": sum(levels != set(MULTIPLIERS) for levels in mother.values())}


def _task_sources(state, mirror_root, sync_manifest):
    """Map controller-owned logical names to one SHA-pinned plain/ZIP source."""
    mapped = {}
    manifest_files = {}
    fixed_names = {"job.json", "config.json", "status.json", "artifacts.json",
                   "results.json", "results.partial.jsonl", "actions.jsonl",
                   "episodes.jsonl", "updates.jsonl", "evaluations.jsonl"}

    def recognized(name):
        return name in fixed_names or (name.startswith("validation_") and
                                       name.endswith(".jsonl") and
                                       name[len("validation_"):-len(".jsonl")].isdigit())

    def entry(record, name):
        expected, size = record.get("sha256"), record.get("bytes")
        if (not isinstance(expected, str) or len(expected) != 64 or
                any(ch not in "0123456789abcdef" for ch in expected.lower()) or
                type(size) is not int or size < 0):
            raise ValueError(f"{name}: manifest SHA or byte count invalid")
        ref = record.get("source_ref")
        if ref is None:
            if not record.get("path"):
                raise ValueError(f"{name}: plain manifest source lacks path")
            path = Path(record["path"]).resolve()
            if not path.is_file() or path.stat().st_size != size or _sha(path) != expected:
                raise ValueError(f"{name}: plain source bytes or SHA differ from manifest")
            return {"path": str(path), "source_ref": {"path": str(path),
                    "sha256": expected}, "sha256": expected, "bytes": size}
        if not isinstance(ref, dict) or ref.get("sha256") != expected or ref.get("bytes") != size:
            raise ValueError(f"{name}: ZIP member SHA/bytes differ from logical file")
        zip_path = Path(ref.get("zip_path", ""))
        if (not zip_path.is_absolute() or not zip_path.is_file() or
                not isinstance(ref.get("zip_sha256"), str) or
                len(ref["zip_sha256"]) != 64 or
                any(ch not in "0123456789abcdef" for ch in ref["zip_sha256"].lower())):
            raise ValueError(f"{name}: ZIP path or SHA invalid")
        if not isinstance(ref.get("member"), str):
            raise ValueError(f"{name}: ZIP member name missing")
        member = archive_sources.safe_member(ref["member"])
        if Path(member).name != name:
            raise ValueError(f"{name}: ZIP member has another logical filename")
        normalized = {"zip_path": str(zip_path.resolve()),
                      "zip_sha256": ref["zip_sha256"], "member": member,
                      "sha256": expected, "bytes": size}
        # Cache the whole-volume SHA only while size/mtime remain unchanged.
        archive_sources._check_zip(normalized)
        with zipfile.ZipFile(zip_path) as bundle:
            try:
                member_info = bundle.getinfo(member)
            except KeyError as error:
                raise ValueError(f"{name}: ZIP member missing") from error
            if member_info.is_dir() or member_info.file_size != size:
                raise ValueError(f"{name}: ZIP member size/type differs")
        result = {"source_ref": normalized, "sha256": expected, "bytes": size}
        if record.get("path"):
            path = Path(record["path"]).resolve()
            if path.is_file():
                if path.stat().st_size != size or _sha(path) != expected:
                    raise ValueError(f"{name}: optional plain mirror differs from ZIP source")
                result["path"] = str(path)
        return result

    if sync_manifest is not None:
        if sync_manifest.get("scientific_commit") != state["scientific_commit"]:
            raise ValueError("Synchronized analysis manifest scientific commit differs")
        seen_logical = set()
        for row in sync_manifest.get("files", []):
            name = row.get("name") or Path(row.get("path", "")).name
            if not isinstance(name, str) or not name:
                raise ValueError("Synchronized manifest has unsafe logical filename")
            archive_sources.safe_member(name)  # Source/weights may use nested logical names.
            key = (str(row["port"]), row["run_id"], name)
            if key in seen_logical:
                raise ValueError("Synchronized manifest repeats a logical analysis filename")
            seen_logical.add(key)
            if not recognized(name):
                continue
            manifest_files[key] = row
        for key, record in sync_manifest.get("tasks", {}).items():
            task = state.get("tasks", {}).get(key)
            if task is None and key.startswith("prepare_"):
                continue  # Preparation provenance belongs to the data ledger.
            if task is None or task["port"] != record["port"] or task["run_id"] != record["run_id"]:
                raise ValueError("Synchronized task identity differs from controller")
    for key, task in state.get("tasks", {}).items():
        port, run_id = str(task["port"]), task["run_id"]
        directory = mirror_root / port / run_id
        files = {}
        local_names = {name for name in fixed_names if (directory / name).is_file()}
        local_names.update(path.name for path in directory.glob("validation_*.jsonl")
                           if recognized(path.name))
        if sync_manifest is not None:
            declared = {name for p, r, name in manifest_files if p == port and r == run_id}
            extra_local = local_names - declared
            if extra_local:
                raise ValueError(f"{key}: local analysis files absent from synchronized manifest: {sorted(extra_local)}")
            for name in sorted(declared):
                files[name] = entry(manifest_files[(port, run_id, name)], name)
        else:
            for name in sorted(local_names):
                path = (directory / name).resolve()
                digest = _sha(path)
                files[name] = {"path": str(path), "source_ref": {"path": str(path),
                               "sha256": digest}, "sha256": digest,
                               "bytes": path.stat().st_size}
        mapped[key] = {"task": task | {"source_commit": controller.expected_task_source(state, key)},
                       "directory": str(directory), "files": files}
    return mapped


def _task_rows(key, source):
    task = source["task"] | {"key": key}
    files = source["files"]
    if task["kind"] != "evaluate" or "results.partial.jsonl" not in files:
        return [], None
    raw_source = files["results.partial.jsonl"]
    raw_label = archive_sources.source_label(_source_ref(raw_source))
    if "results.json" in files:
        result = _json_source(files["results.json"])
        if result.get("run_id") != task["run_id"] or result.get("rows_sha256") != files["results.partial.jsonl"]["sha256"]:
            raise ValueError(f"{key}: finalized results SHA or run identity differs")
    rows = []
    for number, raw in _iter_jsonl(raw_source, with_line_number=True):
        try:
            rows.append(_scene_row(task, raw))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{raw_label}:{number}: {error}") from error
    grid = _grid(task, rows)
    if grid["complete"] and "results.json" not in files:
        raise ValueError(f"{key}: full raw rows lack finalized results.json")
    return rows, grid


def _train_rows(key, source):
    task = source["task"] | {"key": key}
    if task["kind"] != "train":
        return None
    files = source["files"]
    updates, episodes, evaluations = [], [], []
    trace_index = defaultdict(lambda: {"count": 0, "first": None, "last": None})
    action_count = 0
    if "actions.jsonl" in files:
        source_ref = files["actions.jsonl"]
        for number, action in _iter_jsonl(source_ref, with_line_number=True):
            try:
                action_count += 1
                if action["env_step"] != action_count:
                    raise ValueError(f"{key}: action ledger decision step is not continuous")
                key_tuple = (action["stream_id"], action["stream_episode_serial"])
                value = trace_index[key_tuple]
                value["count"] += 1
                if value["first"] is None:
                    value["first"] = action_count
                value["last"] = action_count
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{archive_sources.source_label(_source_ref(source_ref))}:{number}: {error}") from error
    if "updates.jsonl" in files:
        source_ref = files["updates.jsonl"]
        for number, row in _iter_jsonl(source_ref, with_line_number=True):
            try:
                updates.append({"env_step": int(row["env_step"]),
                                "policy_loss": _finite(row.get("policy_loss"), "policy_loss"),
                                "entropy": _finite(row.get("entropy"), "entropy")})
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{archive_sources.source_label(_source_ref(source_ref))}:{number}: {error}") from error
    steps = [row["env_step"] for row in updates]
    if any(b <= a or b - a != 2048 for a, b in zip([0] + steps[:-1], steps)):
        raise ValueError(f"{key}: PPO update step ledger is not 2048-decision spaced")
    completed_episodes = partial_episodes = 0
    if "episodes.jsonl" in files:
        source_ref = files["episodes.jsonl"]
        for number, row in _iter_jsonl(source_ref, with_line_number=True):
            try:
                if int(row["terminal_env_step"]) > (steps[-1] if steps else 0):
                    raise ValueError("episode terminates beyond observed training steps")
                trace = row.get("action_trace")
                if not isinstance(trace, dict) or trace.get("file") != "actions.jsonl":
                    raise ValueError("episode lacks separate action-trace reference")
                found = trace_index.get((trace["stream_id"], trace["stream_episode_serial"]))
                if not found or (found["count"], found["first"], found["last"]) != (
                        trace["count"], trace["first_env_step"], trace["last_env_step"]):
                    raise ValueError("episode trace reference differs from raw actions")
                if row.get("completed_episode"):
                    if (row.get("terminal_physics_status") != "evaluated_all_actual_beams_and_users" or
                            any(beam.get("capacity_bps") is None for beam in row.get("beams", []))):
                        raise ValueError("completed training episode lacks final physical beams")
                    _scene_row(task, row)  # Audit and discard full user/layout arrays.
                    completed_episodes += 1
                elif row.get("terminal_physics_status") != "not_run_incomplete_budget_tail":
                    raise ValueError("unfinished episode lacks explicit physics omission")
                else:
                    partial_episodes += 1
                coverage_value = row.get("coverage_100", row.get("metrics", {}).get("coverage_100"))
                if type(coverage_value) is not bool:
                    raise ValueError("episode lacks explicit terminal full-coverage flag")
                episodes.append({"terminal_env_step": int(row["terminal_env_step"]),
                                 "episode_return": _finite(row.get("episode_return"), "episode return"),
                                 "coverage_100": coverage_value,
                                 "completed_episode": bool(row.get("completed_episode"))})
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{archive_sources.source_label(_source_ref(source_ref))}:{number}: {error}") from error
    interval = 2048 if task["purpose"] == "verification" else 16384
    due = list(range(interval, (steps[-1] if steps else 0) + 1, interval))
    if "evaluations.jsonl" in files:
        source_ref = files["evaluations.jsonl"]
        for number, row in _iter_jsonl(source_ref, with_line_number=True):
            try:
                summary = row["summary"]
                evaluations.append({"env_step": int(row["env_step"]),
                                    "summary": {
                                        "mean_coverage_users": _finite(summary.get("mean_coverage_users"), "validation coverage"),
                                        "mean_load_cv": _finite(summary.get("mean_load_cv"), "validation CV")},
                                    "rows_sha256": row["rows_sha256"],
                                    "downstream_allocation_calls": int(row["downstream_allocation_calls"])})
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{archive_sources.source_label(_source_ref(source_ref))}:{number}: {error}") from error
    nodes = [row["env_step"] for row in evaluations]
    if len(nodes) != len(set(nodes)) or any(node not in due for node in nodes):
        raise ValueError(f"{key}: duplicate or unscheduled validation node")
    for node in nodes:
        name = f"validation_{node}.jsonl"
        if name not in files:
            raise ValueError(f"{key}: validation node lacks 20 mothers × 3 demands")
        mother = defaultdict(set)
        count = 0
        source_ref = files[name]
        for number, row in _iter_jsonl(source_ref, with_line_number=True):
            try:
                count += 1
                if row["demand_multiplier"] not in MULTIPLIERS:
                    raise ValueError("validation node has unknown demand tier")
                _scene_row(task, row)  # Audit terminal arrays, retain only grid keys.
                if row.get("downstream", {}).get("calls", {}).get("layout_allocation_calls") != 1:
                    raise ValueError("validation node physical allocation call ledger differs")
                mother[(row["scene_id"], row["scene_sha256"])].add(row["demand_multiplier"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{archive_sources.source_label(_source_ref(source_ref))}:{number}: {error}") from error
        if count != 60:
            raise ValueError(f"{key}: validation node has {count}/60 mother×tier rows")
        if len(mother) != 20 or any(levels != set(MULTIPLIERS) for levels in mother.values()):
            raise ValueError(f"{key}: validation node mother×tier grid differs")
        ledger = next(row for row in evaluations if row["env_step"] == node)
        if ledger.get("rows_sha256") != files[name]["sha256"]:
            raise ValueError(f"{key}: validation node SHA differs from ledger")
        if ledger.get("downstream_allocation_calls") != 60:
            raise ValueError(f"{key}: validation physical allocation call ledger differs")
    budget = 2048 if task["purpose"] == "verification" else 262144
    completed = bool(steps and steps[-1] == budget and nodes == due and
                     action_count == budget and episodes)
    return {"budget": budget, "observed_steps": steps[-1] if steps else 0,
            "observed_actions": action_count,
            "updates": len(updates), "completed_episodes": completed_episodes,
            "partial_episodes": partial_episodes,
            "validation_nodes": nodes, "expected_validation_nodes": len(range(interval, budget + 1, interval)),
            "validation_downstream_allocation_calls": len(nodes) * 60,
            "validation_downstream_allocation_call_budget": (60 if task["purpose"] == "verification"
                                                             else 960),
            "complete": completed, "episodes_raw": episodes,
            "updates_raw": updates, "evaluations_raw": evaluations}


def _group_stats(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["role"], row["route"], row["demand_multiplier"])].append(row)
    result = []
    for (role, route, factor), items in sorted(grouped.items()):
        result.append({"role": role, "route": route,
                       "method": items[0]["method"], "demand_multiplier": factor,
                       "scene_rows": len(items), "mother_scenes": len({x["scene_id"] for x in items}),
                       "coverage_100_count": sum(x["coverage_100"] for x in items),
                       "coverage_100_fraction": sum(x["coverage_100"] for x in items) / len(items),
                       "failed_coverage_count": sum(not x["coverage_100"] for x in items),
                       "statistics": {name: _stats([x[name] for x in items]) for name in MEASURES}})
    return result


def _paired_sequences(rows):
    relevant = [row for row in rows if row["role"] == "sequence" and row["route"] in ("R04", "R12")]
    by_route = {route: {(r["sequence_id"], r["selected_time_utc"], r["demand_multiplier"]): r
                        for r in relevant if r["route"] == route} for route in ("R04", "R12")}
    common = sorted(by_route["R04"].keys() & by_route["R12"].keys())
    pairs = []
    for key in common:
        base, warm = by_route["R04"][key], by_route["R12"][key]
        if base["scene_sha256"] != warm["scene_sha256"]:
            raise ValueError("R04/R12 sequence pair uses different aircraft scene SHA")
        pairs.append({"sequence_id": key[0], "selected_time_utc": key[1],
                      "demand_multiplier": key[2], "scene_sha256": base["scene_sha256"],
                      "r04_user_tdma_carried_fraction": base["user_tdma_carried_fraction"],
                      "r12_user_tdma_carried_fraction": warm["user_tdma_carried_fraction"],
                      "delta_user_tdma_carried_fraction": (
                          warm["user_tdma_carried_fraction"] - base["user_tdma_carried_fraction"]
                          if warm["user_tdma_carried_fraction"] is not None and
                          base["user_tdma_carried_fraction"] is not None else None),
                      "delta_load_cv": warm["load_cv"] - base["load_cv"],
                      "delta_beam_count": warm["beam_count"] - base["beam_count"],
                      "delta_total_seconds": warm["total_seconds"] - base["total_seconds"]})
    by_sequence = defaultdict(list)
    for pair in pairs:
        by_sequence[(pair["sequence_id"], pair["demand_multiplier"])].append(pair)
    sequence_units = [{"sequence_id": seq, "demand_multiplier": factor,
                       "frames": len(items),
                       "mean_delta_user_tdma_carried_fraction": _stats(
                           [x["delta_user_tdma_carried_fraction"] for x in items])["mean"],
                       "mean_delta_load_cv": _stats([x["delta_load_cv"] for x in items])["mean"],
                       "mean_delta_total_seconds": _stats([x["delta_total_seconds"] for x in items])["mean"]}
                      for (seq, factor), items in sorted(by_sequence.items())]
    return {"matched_frames": len(pairs), "r04_frames": len(by_route["R04"]),
            "r12_frames": len(by_route["R12"]), "sequence_units": sequence_units,
            "paired_rows": pairs,
            "interpretation": "相邻帧属于同一母序列；统计单位是序列，逐帧曲线仅展示轨迹。"}


def _plot_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next((x for x in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei",
                              "Noto Sans CJK JP") if x in available), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [family]
    plt.rcParams["axes.unicode_minus"] = False
    return plt, family


def _smooth(values, window):
    array = np.asarray(values, dtype=float)
    return [float(array[max(0, i - window + 1):i + 1].mean()) for i in range(len(array))]


def _training_plot(outdir, route, data, *, kind, partial):
    updates = data["updates_raw"]
    episodes = data["episodes_raw"]
    evaluations = data["evaluations_raw"]
    if not updates and not episodes:
        return None
    plt, font = _plot_setup()
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    fig.suptitle(f"{route} {kind}训练原始记录与平滑曲线" +
                 ("（阶段数据）" if partial else ""))
    if episodes:
        x = [int(r["terminal_env_step"]) for r in episodes]
        for ax, name, title in ((axes[0, 0], "episode_return", "回合回报"),
                                (axes[0, 1], "coverage_100", "终态严格全覆盖")):
            values = [float(r[name]) if name in r else float(r["metrics"][name])
                      for r in episodes]
            ax.plot(x, values, alpha=.22, linewidth=.6, label="原始回合")
            ax.plot(x, _smooth(values, 64), linewidth=1.7, label="64回合移动均值")
            ax.set(title=title, xlabel="真实训练决策步", ylabel=title)
            ax.legend()
    if updates:
        x = [int(r["env_step"]) for r in updates]
        for ax, name, title in ((axes[1, 0], "policy_loss", "策略损失"),
                                (axes[1, 1], "entropy", "动作熵")):
            values = [_finite(r.get(name), name) for r in updates]
            ax.plot(x, values, alpha=.3, linewidth=.7, label="原始更新")
            ax.plot(x, _smooth(values, 8), linewidth=1.7, label="8轮移动均值")
            ax.set(title=title, xlabel="真实训练决策步", ylabel=title)
            ax.legend()
    if evaluations:
        x = [int(r["env_step"]) for r in evaluations]
        for ax, field, title in ((axes[2, 0], "mean_coverage_users", "验证飞机覆盖比例"),
                                 (axes[2, 1], "mean_load_cv", "验证需求CV")):
            values = [_finite(r["summary"].get(field), field) for r in evaluations]
            ax.plot(x, values, marker="o", linewidth=1.1, label="固定验证节点")
            ax.set(title=title, xlabel="真实训练决策步", ylabel=title)
            ax.legend()
    else:
        for ax in axes[2]:
            ax.text(.5, .5, "尚无已同步验证节点", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_xlabel("真实训练决策步")
    fig.tight_layout()
    path = outdir / f"训练曲线_{route}_{kind}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": str(path), "font": font, "route": route, "kind": kind,
            "raw_episode_count": len(episodes), "raw_update_count": len(updates),
            "validation_node_count": len(evaluations),
            "smoothing": {"episodes": 64, "updates": 8}}


def _static_plot(outdir, rows, *, partial):
    items = [row for row in rows if row["role"] == "test"]
    if not items:
        return None
    plt, font = _plot_setup()
    routes = sorted({row["route"] for row in items})
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("新母场景三档需求：各方法原终态" + ("（阶段数据）" if partial else ""))
    colors = {0.75: "#2282A1", 1.0: "#E38E31", 1.25: "#7746A8"}
    for factor in MULTIPLIERS:
        x, coverage, carried, count = [], [], [], []
        for route in routes:
            sample = [r for r in items if r["route"] == route and r["demand_multiplier"] == factor]
            x.append(routes.index(route))
            coverage.append(sum(r["coverage_100"] for r in sample) / len(sample) if sample else np.nan)
            valid = [r["user_tdma_carried_fraction"] for r in sample
                     if r["user_tdma_carried_fraction"] is not None]
            carried.append(float(np.mean(valid)) if valid else np.nan)
            count.append(float(np.mean([r["beam_count"] for r in sample])) if sample else np.nan)
        offset = (MULTIPLIERS.index(factor) - 1) * .24
        axes[0, 0].bar(np.asarray(x) + offset, coverage, .22, color=colors[factor], label=f"×{factor:g}")
        axes[0, 1].plot(x, carried, marker="o", color=colors[factor], label=f"×{factor:g}")
        axes[1, 0].plot(x, count, marker="o", color=colors[factor], label=f"×{factor:g}")
    cv = [[r["load_cv"] for r in items if r["route"] == route] for route in routes]
    axes[1, 1].boxplot(cv, labels=routes, showfliers=True)
    for ax in axes.flat:
        ax.set_xticks(range(len(routes)), routes, rotation=35)
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].set(ylabel="严格全覆盖场景比例", ylim=(0, 1.05))
    axes[0, 1].set(ylabel="逐机TDMA需求加权有效承载")
    axes[1, 0].set(ylabel="实际业务束数N")
    axes[1, 1].set(ylabel="全体有效业务束需求CV")
    for ax in axes.flat[:3]:
        ax.legend()
    fig.tight_layout()
    path = outdir / "静态方法三档对照.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": str(path), "font": font, "route_count": len(routes),
            "scene_rows": len(items), "partial": partial}


def _sequence_plot(outdir, paired):
    units = paired["sequence_units"]
    if not units:
        return None
    from matplotlib.colors import TwoSlopeNorm

    plt, font = _plot_setup()
    sequences = sorted({unit["sequence_id"] for unit in units})
    short = {sequence_id: f"S{index:02d}" for index, sequence_id in enumerate(sequences, 1)}
    grid = {(unit["sequence_id"], unit["demand_multiplier"]): unit for unit in units}
    mapping_path = outdir / "连续序列标签映射.csv"
    _csv(mapping_path, [{"short_id": short[sequence_id], "sequence_id": sequence_id}
                        for sequence_id in sequences], ("short_id", "sequence_id"))
    fig, axes = plt.subplots(1, 2, figsize=(13, max(6, .38 * len(sequences) + 1.5)),
                             constrained_layout=True)
    fig.suptitle(f"R12－R04：{len(sequences)}条已配对母序列 × 三需求档；每格为该序列内已配对帧均值")
    for ax, field, title, number_format in (
            (axes[0], "mean_delta_user_tdma_carried_fraction", "逐机TDMA有效承载差值", ".3f"),
            (axes[1], "mean_delta_total_seconds", "同帧总耗时差值（秒）", ".2f")):
        values = np.asarray([[grid.get((sequence_id, factor), {}).get(field)
                              for factor in MULTIPLIERS] for sequence_id in sequences], dtype=object)
        values = np.asarray([[np.nan if value is None else float(value) for value in row]
                             for row in values], dtype=float)
        finite = values[np.isfinite(values)]
        scale = max(float(np.max(np.abs(finite))) if len(finite) else 0.0, 1e-9)
        palette = plt.get_cmap("RdBu_r").copy()
        palette.set_bad("#edf0f2")
        picture = ax.imshow(np.ma.masked_invalid(values), cmap=palette,
                            norm=TwoSlopeNorm(vcenter=0, vmin=-scale, vmax=scale),
                            aspect="auto", interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks(range(len(MULTIPLIERS)), [f"×{factor:g}" for factor in MULTIPLIERS])
        ax.set_yticks(range(len(sequences)), [short[sequence_id] for sequence_id in sequences])
        ax.set_xlabel("同母场景需求倍率")
        ax.set_ylabel("母序列短标签；完整ID见映射CSV")
        for y in range(len(sequences)):
            for x in range(len(MULTIPLIERS)):
                value = values[y, x]
                label = format(value, number_format) if np.isfinite(value) else "—"
                ax.text(x, y, label, ha="center", va="center", fontsize=7,
                        color="white" if np.isfinite(value) and abs(value) > scale * .6 else "#19232b")
        fig.colorbar(picture, ax=ax, shrink=.83, pad=.02, label="R12－R04")
    path = outdir / "连续序列配对对照.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": str(path), "font": font,
            "sequence_units": len(units), "sequence_count": len(sequences),
            "statistical_unit": "mother_sequence", "demand_tiers": list(MULTIPLIERS),
            "sequence_label_map_path": str(mapping_path),
            "sequence_label_map": short, "matched_frames": paired["matched_frames"]}


def _markdown(summary):
    ready = summary["formal_ready"]
    lines = ["# S1-V10 可变波束数全覆盖与负载均衡：报告08数据材料", "",
             f"分析生成：{summary['generated_utc']}。科学提交：`{summary['scientific_commit']}`。",
             f"当前状态：{'正式原始结果齐备，可供主任务核写' if ready else '阶段数据；正式结果尚未齐备'}。",
             "", "## 固定统计口径", "",
             "严格全覆盖按逐机唯一合法归属判定，失败场景留在每法全部母场景分母。",
             "需求CV与带内比例使用当前布局所有正需求业务束；带内为各束需求落在本布局正需求束均值±20%的比例。",
             "中心容量有效承载和逐机比例需求TDMA有效承载分别列示；两者均以原场景全部飞机需求为分母。",
             "总时间包含方法执行、终态核验和下游真实分配/物理核验；非学习路线的原始方法耗时把初始化与搜索合并计量，原始初始化0仅是未拆分，报表单列为缺失。",
             "方法前段合并耗时可比；R10该段还包含最多16次内部物理反馈，另列最终共同分配。",
             "本材料对已保存容量和TDMA份额作代数、守恒核验，未由原始逐机位置独立重算离轴链路。",
             "同母场景三倍率与连续序列的帧不是独立样本；训练仅seed51001。", "",
             "## 已记录的静态测试分组", "",
             "| 方法 | 倍率 | 已记/计划景 | 严格全覆盖 | CV均值 | N均值 | 中心承载 | 逐机承载 | 总耗时均值s |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    if summary.get("throughput_source_overrides"):
        changes = summary["throughput_source_overrides"]
        repaired = sorted(key for key in changes if key.startswith("train_"))
        lines[4:4] = ["", "执行来源接续：" + "、".join(
            f"{key}从`{summary['scientific_commit'][:12]}`同ID断点接续到`{changes[key]['repair_commit'][:12]}`"
            for key in repaired) + "。学习任务与旧非学习基线可能来自不同构造器/CPU供给版本，"
            "训练及逐景墙钟差不能归因于PPO纯算法速度；旧冻结基线未重评。"]
    for group in summary["groups"]:
        if group["role"] != "test":
            continue
        s = group["statistics"]
        def show(name):
            value = s[name]["mean"]
            return "未记录" if value is None else f"{value:.4f}"
        lines.append(f"| {group['method']} | {group['demand_multiplier']:g} | "
                     f"{group['scene_rows']}/100 | {group['coverage_100_count']}/{group['scene_rows']} | "
                     f"{show('load_cv')} | {show('beam_count')} | "
                     f"{show('center_carried_fraction')} | {show('user_tdma_carried_fraction')} | "
                     f"{show('total_seconds')} |")
    if not any(g["role"] == "test" for g in summary["groups"]):
        lines.append("| 尚无已同步的正式测试结果 | — | 0/100 | — | — | — | — | — | — |")
    lines += ["", "## 训练与连续序列", ""]
    for route in ("R08", "R09"):
        key = f"train_{route}"
        info = summary["training"].get(key)
        lines.append(f"- {route}正式训练：" +
                     (f"{info['observed_steps']}/{info['budget']}真实决策，"
                      f"{info['updates']}条更新记录，验证节点{len(info['validation_nodes'])}/"
                      f"{info['expected_validation_nodes']}。" if info else "尚无已同步训练记录。"))
    account = summary["training_budget_accounting"]
    lines.append(f"- 正式训练决策账：R08区域重构宏动作"
                 f"{account['R08']['observed']}/{account['R08']['budget']}；"
                 f"R09从空布局逐束构造决策{account['R09']['observed']}/"
                 f"{account['R09']['budget']}。两条路线预算合计"
                 f"{account['combined_budget']}仅作资源账，动作单位不同，不据此比较学习效率。")
    lines.append("- 两模型正式验证的下游完整分配调用上限为2×16节点×60景＝1920次；"
                 "训练回合终态物理核验与共同测试另列，不混入验证调用账。")
    seq = summary["sequence"]
    lines.append(f"- R12与R04同帧匹配：{seq['matched_frames']}/360条倍率帧；"
                 f"统计单位为{len(seq['sequence_units'])}条序列×倍率。")
    lines += ["", "## 资料与限制", "",
              "逐景CSV、序列配对CSV、summary.json、inputs.json和原始日志SHA位于本分析目录。",
              "图来自各任务保存的原始终态、episode、update和validation记录；平滑窗口见plots.json。",
              "阶段结果仅表示本地已同步范围，不把未完成路线或缺少的物理字段补为0。"]
    if summary["issues"]:
        lines += ["", "待核事项："] + [f"- {item}" for item in summary["issues"]]
    return "\n".join(lines) + "\n"


def build_report(state_path, synced_root, output_dir, *, strict=False):
    """Build all derived artifacts; strict=True fails when formal evidence is incomplete."""
    state_path, synced_root, output_dir = map(Path, (state_path, synced_root, output_dir))
    state = _json(state_path)
    commit = state.get("scientific_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("Controller state lacks a frozen 40-character scientific commit")
    declared_registry = state.get("dataset_registry")
    if declared_registry is not None and (not isinstance(declared_registry, str) or
                                          not Path(declared_registry).is_absolute()):
        raise ValueError("Controller dataset registry reference must be an absolute path")
    registry_path = Path(declared_registry) if declared_registry else state_path.parent / "dataset_registry.json"
    registry = _json(registry_path) if registry_path.is_file() else None
    if registry is not None:
        expected_registry_sha = state.get("dataset_registry_file_sha256")
        if expected_registry_sha is not None and _sha(registry_path) != expected_registry_sha:
            raise ValueError("Controller dataset registry file SHA differs")
        if registry.get("scientific_commit") != commit:
            raise ValueError("Dataset registry scientific commit differs")
        if (state.get("dataset_sha256") is not None and
                registry.get("canonical_sha256") != state["dataset_sha256"]):
            raise ValueError("Controller dataset SHA registry differs")
    graph = controller.TASKS
    unexpected = set(state.get("tasks", {})) - set(graph)
    if unexpected:
        raise ValueError(f"Controller includes unknown task keys: {sorted(unexpected)}")
    manifest_path = synced_root / "manifest.json"
    sync_manifest = _json(manifest_path) if manifest_path.is_file() else None
    sources = _task_sources(state, synced_root, sync_manifest)
    all_rows, training, grids, issues = [], {}, {}, []
    if registry is None:
        issues.append(f"冻结dataset_registry.json不存在：{registry_path}")
    for key, spec in graph.items():
        if key not in sources:
            issues.append(f"{key}：工作台任务尚未在controller登记")
            continue
        source = sources[key]
        task = source["task"]
        for name in ("route", "role", "purpose", "kind", "port"):
            if task[name] != spec[name]:
                raise ValueError(f"{key}: controller task differs from frozen graph ({name})")
        if registry is not None:
            lane = registry.get("lanes", {}).get(str(task["port"]), {})
            if not task.get("dataset_sha256"):
                issues.append(f"{key}：controller任务缺冻结数据SHA账")
            for role, sha in task.get("dataset_sha256", {}).items():
                if lane.get(role, {}).get("sha256") != sha:
                    raise ValueError(f"{key}: dataset SHA differs from lane registry")
        files = source["files"]
        for required in ("job.json", "config.json", "status.json", "artifacts.json"):
            if required not in files:
                issues.append(f"{key}：本地缺少{required}")
        if "job.json" in files:
            frozen_job = _json_source(files["job.json"])
            if (frozen_job.get("run_id") != task["run_id"] or
                    frozen_job.get("git_commit") != controller.expected_task_source(state, key)):
                raise ValueError(f"{key}: job ID or frozen code commit differs")
            override = state.get("throughput_source_overrides", {}).get(key)
            if override and key in ("train_R08", "train_R09"):
                proof = frozen_job.get("variable_throughput_repair") or {}
                if (proof.get("origin_commit") != commit or
                        proof.get("repair_commit") != task["source_commit"] or
                        proof.get("validation_receipt_sha256") !=
                        override.get("validation_receipt_sha256") or
                        proof.get("checkpoint_sha256") !=
                        override.get("origin_checkpoint_sha256")):
                    raise ValueError(f"{key}: throughput repair source receipt differs")
            if override and key in ("test_R08", "test_R09"):
                if frozen_job.get("variable_dataset_contract", {}).get("sha256") != override.get("dataset_contract_sha256"):
                    raise ValueError(f"{key}: learned test uses another canonical input contract")
        if "config.json" in files:
            frozen_config = _json_source(files["config.json"])
            actual = {role: record.get("sha256") for role, record in
                      frozen_config.get("datasets", {}).items()}
            if task.get("dataset_sha256") and actual != task["dataset_sha256"]:
                raise ValueError(f"{key}: config dataset SHA differs from controller")
        if "status.json" in files:
            status = _json_source(files["status.json"])
            if key in state.get("audits", {}) and status.get("stage") != "completed":
                raise ValueError(f"{key}: audited task local status is not completed")
        if "artifacts.json" in files:
            artifacts = _json_source(files["artifacts.json"])
            if artifacts.get("run_id") not in (None, task["run_id"]):
                raise ValueError(f"{key}: artifact run ID differs from controller")
        rows, grid = _task_rows(key, source)
        if grid is not None:
            grids[key] = grid
            all_rows.extend(rows)
            if not grid["complete"]:
                issues.append(f"{key}：逐景网格{grid['observed_rows']}/{grid['expected_rows']}")
        elif task["kind"] == "evaluate":
            issues.append(f"{key}：本地尚无逐景原始结果")
        train = _train_rows(key, source)
        if train is not None:
            training[key] = train
            if not train["complete"]:
                issues.append(f"{key}：训练/验证记录未达冻结终点")
        if key not in state.get("audits", {}):
            issues.append(f"{key}：controller尚未完成逐任务核验")
        elif rows and state["audits"][key].get("sha256") and \
                state["audits"][key]["sha256"] != source["files"]["results.partial.jsonl"]["sha256"]:
            raise ValueError(f"{key}: controller raw-result audit SHA differs")
    # Every method must face the same frozen test mother×tier grid.  Partial
    # data remain reportable, but only full equality can make formal_ready.
    test_sets = {key: {(r["scene_id"], r["scene_sha256"], r["demand_multiplier"])
                       for r in all_rows if r["task_key"] == key}
                 for key in graph if key.startswith("test_")}
    full_test_sets = [value for value in test_sets.values() if len(value) == 300]
    if full_test_sets and any(value != full_test_sets[0] for value in full_test_sets):
        raise ValueError("Formal test methods do not share identical mother scenes and demand tiers")
    paired = _paired_sequences(all_rows)
    if paired["r04_frames"] and paired["r12_frames"] and paired["matched_frames"] != min(
            paired["r04_frames"], paired["r12_frames"]):
        issues.append("R04/R12序列存在未配对的已同步倍率帧")
    ready = (state.get("phase") == "completed_pending_report_archive" and
             set(state.get("tasks", {})) == set(graph) and
             set(state.get("audits", {})) == set(graph) and
             sync_manifest is not None and registry is not None and not issues and
             all(grids.get(key, {}).get("complete") for key in graph if graph[key]["kind"] == "evaluate") and
             all(training.get(key, {}).get("complete") for key in graph if graph[key]["kind"] == "train") and
             len(full_test_sets) == 12 and paired["matched_frames"] == 360)
    if strict and not ready:
        raise RuntimeError("Formal report evidence is incomplete; inspect progress summary without --strict")
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_inputs = output_dir / "inputs.json"
    source_map = {key: controller.expected_task_source(state, key)
                  for key in state.get("tasks", {})}
    if previous_inputs.is_file():
        prior = _json(previous_inputs)
        if prior.get("scientific_commit") != commit or prior.get("task_source_commits") != source_map:
            raise ValueError("Refusing to overwrite report artifacts from another task-source mapping")
    input_manifest = {"scientific_commit": commit, "controller_state": {"path": str(state_path.resolve()),
                       "sha256": _sha(state_path)},
                      "task_source_commits": source_map,
                      "throughput_source_overrides": state.get("throughput_source_overrides", {}),
                      "dataset_registry": {"path": str(registry_path.resolve()),
                                           "sha256": _sha(registry_path)} if registry else None,
                      "synced_manifest": {"path": str(manifest_path.resolve()),
                                          "sha256": _sha(manifest_path)} if sync_manifest else None,
                      "tasks": {key: {"port": value["task"]["port"],
                                      "run_id": value["task"]["run_id"],
                                      "files": value["files"]} for key, value in sources.items()}}
    _save(previous_inputs, input_manifest)
    _csv(output_dir / "per_scene.csv", all_rows, SCENE_COLUMNS)
    paired_columns = ("sequence_id", "selected_time_utc", "demand_multiplier", "scene_sha256",
                      "r04_user_tdma_carried_fraction", "r12_user_tdma_carried_fraction",
                      "delta_user_tdma_carried_fraction", "delta_load_cv",
                      "delta_beam_count", "delta_total_seconds")
    _csv(output_dir / "sequence_pairs.csv", paired["paired_rows"], paired_columns)
    plots = []
    for route in ("R08", "R09"):
        for key, kind in ((f"verify_{route}", "工程验证"),
                          (f"train_{route}", "正式")):
            info = training.get(key)
            if info:
                plot = _training_plot(output_dir, route, info, kind=kind,
                                      partial=not info["complete"])
                if plot:
                    plots.append(plot)
    static = _static_plot(output_dir, all_rows, partial=not ready)
    if static:
        plots.append(static)
    sequence = _sequence_plot(output_dir, paired)
    if sequence:
        plots.append(sequence)
    _save(output_dir / "plots.json", plots)
    grouped = _group_stats(all_rows)
    clean_training = {key: {k: v for k, v in value.items() if not k.endswith("_raw")}
                      for key, value in training.items()}
    budget_account = {
        route: {"decision_unit": ("region_reconstruction_macro_action" if route == "R08"
                                  else "constructive_beam_or_stop_decision"),
                "observed": clean_training.get(f"train_{route}", {}).get("observed_steps", 0),
                "budget": 262144,
                "verification_observed": clean_training.get(f"verify_{route}", {}).get(
                    "observed_steps", 0),
                "verification_budget": 2048}
        for route in ("R08", "R09")}
    budget_account["combined_budget"] = 524288
    budget_account["formal_validation_downstream_allocation_call_budget"] = 1920
    budget_account["combined_is_only_resource_accounting"] = True
    clean_sequence = {k: v for k, v in paired.items() if k != "paired_rows"}
    summary = {"scientific_commit": commit,
               "task_source_commits": source_map,
               "throughput_source_overrides": state.get("throughput_source_overrides", {}),
               "generated_utc": datetime.now(timezone.utc).isoformat(),
               "formal_ready": bool(ready), "controller_phase": state.get("phase"),
               "task_count": len(state.get("tasks", {})), "expected_task_count": len(graph),
               "audited_task_count": len(state.get("audits", {})),
               "synced_manifest_verified": sync_manifest is not None,
               "dataset_registry_verified": registry is not None,
               "issues": issues, "grids": grids,
               "groups": grouped, "training": clean_training,
               "training_budget_accounting": budget_account,
               "sequence": clean_sequence, "plots": plots,
               "methods": {key: _method(spec) for key, spec in graph.items()},
               "metric_notes": {"in_band_fraction": "正需求业务束D相对本布局正需求束均值±20%",
                                "peak_load_ratio": "max(逐束需求D)/600 Mbps参考容量；是峰值参考容量压力，不是max(D)/mean(D)",
                                "center_carried_fraction": "束中心C，sum(min(C,D))/原场景所有飞机需求",
                                "user_tdma_carried_fraction": "逐机离轴容量加需求比例TDMA，sum(min(C_u,D_u))/所有飞机需求",
                                "physical_audit_scope": "对保存的终态束/用户容量、需求、有效承载、SI、SGM、TDMA份额和总分母作独立代数与守恒核验；未从原始逐机位置重新计算离轴链路容量",
                                "training_compact_user_limit": "完整训练回合保留每机C/min(C,D)/SI/airtime；未保存逐机full-time率与C/D，报告仅由C和D重算比值，不伪称对缺失字段作原始值核验",
                                "timing_granularity": "非学习路线原始initialization_seconds=0表示计时未拆分，报表填null；method_seconds_before_common_downstream是初始化+搜索合并实测，R10还含内部物理反馈；总时间不回填或扣除缺项",
                                "timing_source_comparability": "任务执行源码逐项记录；学习提速接续与旧冻结基线的CPU供给/构造器可能不同，耗时差不归因为PPO纯算法速度",
                                "sequence_unit": "同序列多帧和三倍率不视作独立母序列"}}
    _save(output_dir / "summary.json", summary)
    (output_dir / "报告08_数据材料.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path,
                        default=ops.LOCAL / "controller" / "state.json")
    parser.add_argument("--synced-root", type=Path,
                        default=ops.LOCAL / "synced_jobs")
    parser.add_argument("--output", type=Path,
                        default=ops.LOCAL / "analysis" / "report08")
    parser.add_argument("--strict", action="store_true",
                        help="Require all formal frozen tasks and verified synchronized raw results")
    args = parser.parse_args()
    result = build_report(args.state, args.synced_root, args.output, strict=args.strict)
    print(json.dumps({"formal_ready": result["formal_ready"],
                      "task_count": result["task_count"],
                      "issues": result["issues"], "output": str(args.output)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
