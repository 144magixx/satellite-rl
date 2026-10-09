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

import numpy as np

from scripts.stage1_variable import controller, ops


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
    "total_power_w", "complete_allocation", "initialization_seconds",
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
    "initialization_seconds", "algorithm_seconds", "final_check_seconds",
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


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


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
    beam_d = np.asarray([_finite(b["demand_bps"], "beam demand") for b in beams])
    uncovered = _finite(m["uncovered_demand_bps"], "uncovered demand")
    if not np.isclose(beam_d.sum() + uncovered, total, rtol=1e-10, atol=1e-5):
        raise ValueError(f"{key}: beam and uncovered demand do not sum to all aircraft")
    if not bool(m["demand_conserved"]) or bool(m["coverage_100"]) != bool(raw["coverage_complete"]):
        raise ValueError(f"{key}: conservation or full-coverage flag differs")
    if int(m["covered_users"]) != sum(u.get("assigned_beam") is not None for u in users):
        raise ValueError(f"{key}: covered-user count differs from ownership")
    for index, b in enumerate(beams):
        if b.get("beam_id") != index:
            raise ValueError(f"{key}: beam processing identity differs")
        d = beam_d[index]
        c = _finite(b.get("capacity_bps"), "terminal beam capacity")
        carried = _finite(b.get("carried_bps"), "terminal beam carried")
        if c < 0 or not np.isclose(carried, min(c, d), rtol=1e-8, atol=1e-5):
            raise ValueError(f"{key}: terminal min(C,D) differs")
        if d > 0 and not np.isclose(_finite(b.get("unclipped_capacity_demand_ratio"), "C/D"),
                                    c / d, rtol=1e-8, atol=1e-8):
            raise ValueError(f"{key}: terminal C/D differs")
    positive = beam_d[beam_d > 0]
    mean = float(positive.mean()) if len(positive) else None
    band = (float(np.mean((positive >= .8 * mean) & (positive <= 1.2 * mean)))
            if mean is not None else None)
    if not isinstance(down.get("metrics"), dict) or not isinstance(down.get("timing"), dict):
        raise ValueError(f"{key}: downstream summary or segmented timing absent")
    dm, dt, calls = down["metrics"], down["timing"], down.get("calls", {})
    center = _finite(dm.get("demand_weighted_carried_fraction"), "center carried", nullable=True)
    direct_center = sum(b["carried_bps"] for b in beams) / total if total else None
    if center is not None and not np.isclose(center, direct_center, rtol=1e-8, atol=1e-8):
        raise ValueError(f"{key}: center carried does not match beam ledger")
    service = down.get("aircraft_service", {})
    user_carried = _finite(service.get("demand_weighted_carried_fraction"),
                           "aircraft TDMA carried", nullable=True)
    if user_carried is not None and not 0 <= user_carried <= 1 + 1e-8:
        raise ValueError(f"{key}: aircraft TDMA carried outside [0,1]")
    for name in ("initialization_seconds", "algorithm_seconds", "final_check_seconds", "total_seconds"):
        if _finite(raw.get(name), name) < 0:
            raise ValueError(f"{key}: negative measured time")
    for name in ("interface_seconds", "allocation_seconds", "verification_seconds", "total_seconds"):
        if _finite(dt.get(name), f"downstream {name}") < 0:
            raise ValueError(f"{key}: negative downstream time")
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
           "initialization_seconds": float(raw["initialization_seconds"]),
           "algorithm_seconds": float(raw["algorithm_seconds"]),
           "geometry_seconds": _finite(raw.get("geometry_seconds"), "geometry time", nullable=True),
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
    """Map only controller-owned IDs to locally mirrored, SHA-verified files."""
    mapped = {}
    manifest_files = {}
    if sync_manifest is not None:
        if sync_manifest.get("scientific_commit") != state["scientific_commit"]:
            raise ValueError("Synchronized analysis manifest scientific commit differs")
        for row in sync_manifest.get("files", []):
            key = (str(row["port"]), row["run_id"], Path(row["path"]).name)
            if key in manifest_files:
                raise ValueError("Synchronized manifest repeats an analysis filename")
            manifest_files[key] = row
        for key, record in sync_manifest.get("tasks", {}).items():
            task = state.get("tasks", {}).get(key)
            if task is None or task["port"] != record["port"] or task["run_id"] != record["run_id"]:
                raise ValueError("Synchronized task identity differs from controller")
    for key, task in state.get("tasks", {}).items():
        port, run_id = str(task["port"]), task["run_id"]
        directory = mirror_root / port / run_id
        files = {}
        for name in ("job.json", "config.json", "status.json", "artifacts.json",
                     "results.json", "results.partial.jsonl", "actions.jsonl", "episodes.jsonl",
                     "updates.jsonl", "evaluations.jsonl"):
            path = directory / name
            if not path.is_file():
                continue
            digest = _sha(path)
            receipt = manifest_files.get((port, run_id, name))
            if sync_manifest is not None and (receipt is None or
                                               Path(receipt["path"]).resolve() != path.resolve() or
                                               digest != receipt["sha256"]):
                raise ValueError(f"{key}: local mirror SHA differs from verified archive")
            files[name] = {"path": str(path.resolve()), "sha256": digest,
                           "bytes": path.stat().st_size}
        # Validation nodes are raw inputs and receive the same SHA check.
        for path in sorted(directory.glob("validation_*.jsonl")):
            digest = _sha(path)
            receipt = manifest_files.get((port, run_id, path.name))
            if sync_manifest is not None and (receipt is None or
                                               Path(receipt["path"]).resolve() != path.resolve() or
                                               digest != receipt["sha256"]):
                raise ValueError(f"{key}: validation node SHA differs")
            files[path.name] = {"path": str(path.resolve()), "sha256": digest,
                                "bytes": path.stat().st_size}
        mapped[key] = {"task": task | {"source_commit": state["scientific_commit"]},
                       "directory": str(directory), "files": files}
    return mapped


def _task_rows(key, source):
    task = source["task"] | {"key": key}
    files = source["files"]
    if task["kind"] != "evaluate" or "results.partial.jsonl" not in files:
        return [], None
    raw_path = Path(files["results.partial.jsonl"]["path"])
    raw_rows = _rows(raw_path)
    if "results.json" in files:
        result = _json(files["results.json"]["path"])
        if result.get("run_id") != task["run_id"] or result.get("rows_sha256") != _sha(raw_path):
            raise ValueError(f"{key}: finalized results SHA or run identity differs")
    rows = [_scene_row(task, raw) for raw in raw_rows]
    grid = _grid(task, rows)
    if grid["complete"] and "results.json" not in files:
        raise ValueError(f"{key}: full raw rows lack finalized results.json")
    return rows, grid


def _train_rows(key, source):
    task = source["task"]
    if task["kind"] != "train":
        return None
    files = source["files"]
    updates = _rows(files["updates.jsonl"]["path"]) if "updates.jsonl" in files else []
    episodes = _rows(files["episodes.jsonl"]["path"]) if "episodes.jsonl" in files else []
    evaluations = _rows(files["evaluations.jsonl"]["path"]) if "evaluations.jsonl" in files else []
    trace_index = defaultdict(lambda: {"count": 0, "first": None, "last": None})
    action_count = 0
    if "actions.jsonl" in files:
        with Path(files["actions.jsonl"]["path"]).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                action = json.loads(line)
                action_count += 1
                if action["env_step"] != action_count:
                    raise ValueError(f"{key}: action ledger decision step is not continuous")
                key_tuple = (action["stream_id"], action["stream_episode_serial"])
                value = trace_index[key_tuple]
                value["count"] += 1
                if value["first"] is None:
                    value["first"] = action_count
                value["last"] = action_count
    steps = [int(row["env_step"]) for row in updates]
    if any(b <= a or b - a != 2048 for a, b in zip([0] + steps[:-1], steps)):
        raise ValueError(f"{key}: PPO update step ledger is not 2048-decision spaced")
    if any(int(row["terminal_env_step"]) > (steps[-1] if steps else 0) for row in episodes):
        raise ValueError(f"{key}: episode terminates beyond observed training steps")
    for row in episodes:
        trace = row.get("action_trace")
        if not isinstance(trace, dict) or trace.get("file") != "actions.jsonl":
            raise ValueError(f"{key}: episode lacks separate action-trace reference")
        found = trace_index.get((trace["stream_id"], trace["stream_episode_serial"]))
        if not found or (found["count"], found["first"], found["last"]) != (
                trace["count"], trace["first_env_step"], trace["last_env_step"]):
            raise ValueError(f"{key}: episode trace reference differs from raw actions")
        if row.get("completed_episode"):
            if (row.get("terminal_physics_status") != "evaluated_all_actual_beams_and_users" or
                    any(beam.get("capacity_bps") is None for beam in row.get("beams", []))):
                raise ValueError(f"{key}: completed training episode lacks final physical beams")
        elif row.get("terminal_physics_status") != "not_run_incomplete_budget_tail":
            raise ValueError(f"{key}: unfinished episode lacks explicit physics omission")
    interval = 2048 if task["purpose"] == "verification" else 16384
    due = list(range(interval, (steps[-1] if steps else 0) + 1, interval))
    nodes = [int(row["env_step"]) for row in evaluations]
    if len(nodes) != len(set(nodes)) or any(node not in due for node in nodes):
        raise ValueError(f"{key}: duplicate or unscheduled validation node")
    for node in nodes:
        name = f"validation_{node}.jsonl"
        if name not in files or len(_rows(files[name]["path"])) != 60:
            raise ValueError(f"{key}: validation node lacks 20 mothers × 3 demands")
        node_rows = _rows(files[name]["path"])
        mother = defaultdict(set)
        for row in node_rows:
            if row["demand_multiplier"] not in MULTIPLIERS:
                raise ValueError(f"{key}: validation node has unknown demand tier")
            down = row.get("downstream", {})
            if down.get("calls", {}).get("layout_allocation_calls") != 1 or any(
                    beam.get("capacity_bps") is None or beam.get("carried_bps") is None or
                    (beam["demand_bps"] > 0 and
                     (beam.get("unclipped_capacity_demand_ratio") is None or
                      beam.get("si") is None or beam.get("sgm") is None))
                    for beam in row.get("beams", [])):
                raise ValueError(f"{key}: validation terminal lacks true downstream service")
            mother[(row["scene_id"], row["scene_sha256"])].add(row["demand_multiplier"])
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
            "updates": len(updates), "completed_episodes": sum(bool(row.get("completed_episode", True))
                                                   for row in episodes),
            "partial_episodes": sum(not bool(row.get("completed_episode", True))
                                    for row in episodes),
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
    plt, font = _plot_setup()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [f"{x['sequence_id']} ×{x['demand_multiplier']:g}" for x in units]
    x = np.arange(len(units))
    for ax, field, title in ((axes[0], "mean_delta_user_tdma_carried_fraction", "逐机TDMA承载差值"),
                             (axes[1], "mean_delta_total_seconds", "同帧总耗时差值（秒）")):
        values = [u[field] if u[field] is not None else np.nan for u in units]
        ax.bar(x, values, color="#408A7C")
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(title=title, ylabel="R12－R04；先在每条序列内均值")
        ax.set_xticks(x, labels, rotation=75)
    fig.tight_layout()
    path = outdir / "连续序列配对对照.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": str(path), "font": font,
            "sequence_units": len(units), "matched_frames": paired["matched_frames"]}


def _markdown(summary):
    ready = summary["formal_ready"]
    lines = ["# S1-V10 可变波束数全覆盖与负载均衡：报告08数据材料", "",
             f"分析生成：{summary['generated_utc']}。科学提交：`{summary['scientific_commit']}`。",
             f"当前状态：{'正式原始结果齐备，可供主任务核写' if ready else '阶段数据；正式结果尚未齐备'}。",
             "", "## 固定统计口径", "",
             "严格全覆盖按逐机唯一合法归属判定，失败场景留在每法全部母场景分母。",
             "需求CV与带内比例使用当前布局所有正需求业务束；带内为各束需求落在本布局正需求束均值±20%的比例。",
             "中心容量有效承载和逐机比例需求TDMA有效承载分别列示；两者均以原场景全部飞机需求为分母。",
             "总时间由布局初始化、算法执行、终态核验和下游真实分配/物理核验分段实测组成。",
             "同母场景三倍率与连续序列的帧不是独立样本；训练仅seed51001。", "",
             "## 已记录的静态测试分组", "",
             "| 方法 | 倍率 | 已记/计划景 | 严格全覆盖 | CV均值 | N均值 | 中心承载 | 逐机承载 | 总耗时均值s |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
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
    registry_path = state_path.parent / "dataset_registry.json"
    registry = _json(registry_path) if registry_path.is_file() else None
    if registry is not None:
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
        issues.append("本地尚无冻结dataset_registry.json")
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
            frozen_job = _json(files["job.json"]["path"])
            if (frozen_job.get("run_id") != task["run_id"] or
                    frozen_job.get("git_commit") != commit):
                raise ValueError(f"{key}: job ID or frozen code commit differs")
        if "config.json" in files:
            frozen_config = _json(files["config.json"]["path"])
            actual = {role: record.get("sha256") for role, record in
                      frozen_config.get("datasets", {}).items()}
            if task.get("dataset_sha256") and actual != task["dataset_sha256"]:
                raise ValueError(f"{key}: config dataset SHA differs from controller")
        if "status.json" in files:
            status = _json(files["status.json"]["path"])
            if key in state.get("audits", {}) and status.get("stage") != "completed":
                raise ValueError(f"{key}: audited task local status is not completed")
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
    if previous_inputs.is_file() and _json(previous_inputs).get("scientific_commit") != commit:
        raise ValueError("Refusing to overwrite report artifacts from another scientific commit")
    input_manifest = {"scientific_commit": commit, "controller_state": {"path": str(state_path.resolve()),
                       "sha256": _sha(state_path)},
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
                                "center_carried_fraction": "束中心C，sum(min(C,D))/原场景所有飞机需求",
                                "user_tdma_carried_fraction": "逐机离轴容量加需求比例TDMA，sum(min(C_u,D_u))/所有飞机需求",
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
