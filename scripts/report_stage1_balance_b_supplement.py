"""Read-only S1-02 B delivery supplement from six archived method results.

The six-method, 100-scene gate is the default.  ``--allow-partial`` permits a
clearly labelled structural smoke check of complete menu pairs.  Neither mode
loads a policy or executes a coverage planner.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

try:
    from .audit_stage1_balance import audit_row, read_bundle, sha
except ImportError:  # Direct ``python scripts/...`` execution.
    from audit_stage1_balance import audit_row, read_bundle, sha


MENUS = ("X0", "X1", "X4")
KINDS = ("search", "ppo")
REQUIRED = {f"{menu}_{kind}" for menu in MENUS for kind in KINDS}
TIME_FIELDS = ("c1", "edge_preprocess", "repair", "assignment", "adjustment",
               "environment_setup", "policy_and_adjustment", "final_verification",
               "final_verification_and_record")
COMPARE_FIELDS = ("coverage_demand", "coverage_users", "subject_cv", "load_cv",
                  "subject_in_band_beams", "edge_beam_count", "balanced_target_met")
RAW_FIELDS = {"metrics": "raw_terminal_metrics",
              "final_positions": "raw_terminal_positions",
              "final_associations": "raw_terminal_associations",
              "beam_loads_bps": "raw_terminal_beam_loads_bps"}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def input_sha(row):
    source = row["input"]
    return canonical_sha({key: source[key] for key in
                          ("aircraft_ids", "positions", "demands_bps", "initial_centers")})


def align_results(bundles, expected_scenes, allow_partial=False):
    labels = set(bundles)
    if not allow_partial and labels != REQUIRED:
        raise ValueError(f"Formal B delivery requires exactly {sorted(REQUIRED)}")
    if not labels or labels - REQUIRED:
        raise ValueError("Unknown or empty B method labels")
    for menu in MENUS:
        present = {f"{menu}_{kind}" for kind in KINDS} & labels
        if present and present != {f"{menu}_{kind}" for kind in KINDS}:
            raise ValueError(f"{menu}: PPO and same-menu search must be paired")
    reference = None
    aligned = {}
    for label, rows in bundles.items():
        if len(rows) != expected_scenes:
            raise ValueError(f"{label}: expected {expected_scenes} original scenes, got {len(rows)}")
        by_id = {row["scene_id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"{label}: duplicate scene IDs")
        identity = {scene_id: (row["scene_sha256"], input_sha(row))
                    for scene_id, row in by_id.items()}
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError(f"{label}: scene IDs, source SHA or original aircraft input differs")
        aligned[label] = [by_id[key] for key in sorted(by_id)]
    return aligned


def _qualified_edges(regions, associations):
    assignment = np.asarray(associations, dtype=np.int64)
    qualified = []
    for region in regions:
        members = np.asarray(region["aircraft_indices"], dtype=np.int64)
        if members.size == 0:
            continue
        beam = int(assignment[members[0]])
        if (beam >= 0 and np.all(assignment[members] == beam) and
                np.count_nonzero(assignment == beam) == members.size):
            qualified.append(beam)
    return sorted(set(qualified))


def raw_audit_view(row):
    """Swap saved raw fields in memory; never modify or write the result row."""
    if row.get("PPO_adjustment_performed") is False:
        return None
    missing = [source for source in RAW_FIELDS.values() if row.get(source) is None]
    if missing:
        raise ValueError(f"{row['scene_id']}: PPO result lacks raw fields {missing}")
    view = dict(row)
    for target, source in RAW_FIELDS.items():
        view[target] = row[source]
    diag = dict(row.get("balance_diagnostics", {}))
    diag["qualified_edge_beam_ids"] = _qualified_edges(
        diag.get("certified_regions", []), view["final_associations"])
    view["balance_diagnostics"] = diag
    return view


def _numeric(row, key):
    metrics = row.get("metrics")
    value = metrics.get(key) if isinstance(metrics, dict) else None
    return float(value) if isinstance(value, (float, int)) and math.isfinite(float(value)) else None


def _change(left, right, key):
    a, b = _numeric(left, key), _numeric(right, key)
    return None if a is None or b is None else b - a


def _describe(values):
    data = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not data:
        return {"count": 0, "min": None, "mean": None, "median": None, "p95": None, "max": None}
    return {"count": len(data), "min": data[0], "mean": statistics.mean(data),
            "median": statistics.median(data), "p95": float(np.percentile(data, 95)),
            "max": data[-1]}


def _direction(changes, *, higher_is_better):
    oriented = [value if higher_is_better else -value for value in changes if value is not None]
    return {"improved": sum(value > 1e-10 for value in oriented),
            "equal": sum(abs(value) <= 1e-10 for value in oriented),
            "worse": sum(value < -1e-10 for value in oriented),
            "paired_valid_layout_scenes": len(oriented)}


def _time_row(label, row, scene_index):
    diag = row.get("balance_diagnostics", {})
    stages = diag.get("stage_seconds", {})
    total = row.get("total_pipeline_seconds", diag.get("total_pipeline_seconds"))
    if not stages or total is None:
        raise ValueError(f"{label} scene {scene_index}: measured stage/total time missing")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
           for value in stages.values()):
        raise ValueError(f"{label} scene {scene_index}: invalid stage time")
    if not math.isclose(sum(stages.values()), float(total), rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"{label} scene {scene_index}: measured stages do not sum to saved total")
    result = {"method": label, "scene_id": row["scene_id"],
              "scene_sha256": row["scene_sha256"], "input_sha256": input_sha(row),
              "status": row["status"], "cluster_initializer_seconds": None,
              "recorded_total_pipeline_seconds": float(total),
              "evaluation_call_seconds": row.get("evaluation_call_seconds"),
              "common_preparation_seconds": row.get("common_preparation_seconds")}
    result.update({f"{field}_seconds": stages.get(field) for field in TIME_FIELDS})
    result["other_stage_seconds_json"] = json.dumps(
        {key: value for key, value in stages.items() if key not in TIME_FIELDS},
        sort_keys=True, separators=(",", ":"))
    return result


def analyze(results, *, expected_scenes=100, allow_partial=False, audit_path=None):
    receipts, rows, configs = {}, {}, {}
    for label, path in results.items():
        path = Path(path)
        rows[label], configs[label] = read_bundle(path)
        receipts[label] = {"path": str(path.resolve()), "sha256": sha(path),
                           "row_count": len(rows[label])}
        menu, kind = label.split("_", 1)
        meta = json.loads(path.read_text(encoding="utf-8"))["metadata"]
        if meta.get("action_menu") != menu:
            raise ValueError(f"{label}: frozen menu differs from label")
        if kind == "ppo" and meta.get("route") != "ppo":
            raise ValueError(f"{label}: not a frozen PPO result")
        if kind == "search" and meta.get("route") != menu:
            raise ValueError(f"{label}: not the same-menu search result")
    if not allow_partial:
        if audit_path is None:
            raise ValueError("Formal six-method supplement requires the independent audit path")
        audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
        if (not audit.get("passed") or audit.get("expected_scenes") != expected_scenes or
                set(audit.get("sources", {})) != set(results)):
            raise ValueError("Independent audit does not cover exactly these six methods")
        for label in results:
            if audit["sources"][label]["sha256"] != receipts[label]["sha256"]:
                raise ValueError(f"{label}: result changed after independent audit")
    aligned = align_results(rows, expected_scenes, allow_partial)
    timing = []
    raw_best = []
    paired = []
    method_status = {}
    for label, ordered in aligned.items():
        method_status[label] = dict(Counter(row["status"] for row in ordered))
        timing.extend(_time_row(label, row, index) for index, row in enumerate(ordered))
    for menu in MENUS:
        ppo_key, search_key = f"{menu}_ppo", f"{menu}_search"
        if ppo_key not in aligned:
            continue
        for index, (ppo, search) in enumerate(zip(aligned[ppo_key], aligned[search_key])):
            if ppo["scene_id"] != search["scene_id"]:
                raise ValueError(f"{menu}: same-menu pair is misaligned")
            raw = raw_audit_view(ppo)
            if raw is not None:
                # The existing independent NumPy audit sees a temporary view;
                # no original PPO result is rewritten or re-evaluated.
                evidence = audit_row(raw, configs[ppo_key], require_timing=False)
                if not evidence["passed"]:
                    raise ValueError(f"{menu} raw scene {index}: {evidence['issues']}")
            raw_best.append({
                "menu": menu, "scene_id": ppo["scene_id"],
                "scene_sha256": ppo["scene_sha256"],
                "input_sha256": input_sha(ppo),
                "PPO_adjustment_performed": ppo.get("PPO_adjustment_performed", raw is not None),
                "raw_state_recorded": raw is not None,
                "raw_status": raw["status"] if raw else None,
                "best_status": ppo["status"],
                "raw_layout_sha256": canonical_sha(raw["final_positions"]) if raw else None,
                "best_layout_sha256": canonical_sha(ppo["final_positions"]) if ppo.get("final_positions") is not None else None,
                "raw_vs_best_same_layout_and_association":
                    (raw["final_positions"] == ppo["final_positions"] and
                     raw["final_associations"] == ppo["final_associations"]) if raw else None,
                **{f"raw_{key}": _numeric(raw, key) if raw else None for key in COMPARE_FIELDS},
                **{f"best_{key}": _numeric(ppo, key) for key in COMPARE_FIELDS},
                **{f"best_minus_raw_{key}": _change(raw, ppo, key) if raw else None
                   for key in COMPARE_FIELDS},
            })
            paired.append({
                "menu": menu, "scene_id": ppo["scene_id"],
                "scene_sha256": ppo["scene_sha256"], "input_sha256": input_sha(ppo),
                "PPO_status": ppo["status"], "search_status": search["status"],
                **{f"PPO_minus_search_{key}": _change(search, ppo, key)
                   for key in COMPARE_FIELDS},
                "PPO_minus_search_recorded_total_seconds":
                    _time_row(ppo_key, ppo, index)["recorded_total_pipeline_seconds"] -
                    _time_row(search_key, search, index)["recorded_total_pipeline_seconds"],
            })
    summary = {
        "scope": "partial_smoke_not_formal_six_method_delivery" if allow_partial else
                 "full_six_method_archived_result_supplement",
        "expected_original_scene_count": expected_scenes,
        "same_scene_raw_and_best_are_paired_not_independent_samples": True,
        "cluster_initializer_seconds": None,
        "cluster_initializer_limitation": "Prepared cluster centers predate these jobs; no per-scene generation time is saved. Recorded total is the sum of saved C1/edge/repair/assignment/adjustment or policy/verification stages.",
        "audit_path": str(Path(audit_path).resolve()) if audit_path else None,
        "audit_sha256": sha(audit_path) if audit_path else None,
        "results": receipts, "method_status": method_status,
        "raw_best": {menu: {
            "original_scenes": sum(row["menu"] == menu for row in raw_best),
            "raw_state_rows": sum(row["menu"] == menu and row["raw_state_recorded"] for row in raw_best),
            "raw_and_best_different_rows": sum(row["menu"] == menu and
                                                row["raw_vs_best_same_layout_and_association"] is False
                                                for row in raw_best),
            "best_minus_raw": {key: _describe(row[f"best_minus_raw_{key}"] for row in raw_best
                                              if row["menu"] == menu)
                               for key in COMPARE_FIELDS},
        } for menu in MENUS if f"{menu}_ppo" in aligned},
        "PPO_vs_same_menu_search": {menu: {
            "original_paired_scenes": sum(row["menu"] == menu for row in paired),
            "PPO_minus_search": {key: {
                "distribution": _describe(row[f"PPO_minus_search_{key}"] for row in paired
                                          if row["menu"] == menu),
                "direction": _direction((row[f"PPO_minus_search_{key}"] for row in paired
                                         if row["menu"] == menu),
                                        higher_is_better=key in
                                        ("coverage_demand", "coverage_users",
                                         "subject_in_band_beams", "balanced_target_met")),
            } for key in COMPARE_FIELDS},
            "recorded_total_seconds_delta": _describe(
                row["PPO_minus_search_recorded_total_seconds"] for row in paired
                if row["menu"] == menu),
        } for menu in MENUS if f"{menu}_ppo" in aligned},
        "timing_by_method": {
            label: {
                field: _describe(row.get(field) for row in timing if row["method"] == label)
                for field in ("recorded_total_pipeline_seconds", "evaluation_call_seconds",
                              "common_preparation_seconds", "c1_seconds", "edge_preprocess_seconds",
                              "repair_seconds", "assignment_seconds", "adjustment_seconds",
                              "environment_setup_seconds", "policy_and_adjustment_seconds",
                              "final_verification_seconds", "final_verification_and_record_seconds")
            } for label in aligned},
    }
    return summary, timing, raw_best, paired


def _csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True,
                        help="LABEL=/absolute/path/results.json; labels X0_search/X0_ppo etc")
    parser.add_argument("--audit", type=Path, help="Passed six-method independent audit JSON")
    parser.add_argument("--expected-scenes", type=int, default=100)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Explicit structural smoke of complete menu pairs; never formal ranking")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for item in args.result:
        label, separator, filename = item.partition("=")
        if not separator or not label or label in results or not filename:
            raise ValueError(f"Invalid/duplicate --result {item!r}")
        results[label] = Path(filename)
    summary, timing, raw_best, paired = analyze(
        results, expected_scenes=args.expected_scenes,
        allow_partial=args.allow_partial, audit_path=args.audit)
    args.output.mkdir(parents=True, exist_ok=True)
    _csv(args.output / "per_scene_recorded_time.csv", timing)
    _csv(args.output / "per_scene_PPO_raw_vs_best.csv", raw_best)
    _csv(args.output / "per_scene_PPO_vs_same_menu_search.csv", paired)
    summary["script_sha256"] = sha(Path(__file__))
    (args.output / "supplement.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({"scope": summary["scope"], "methods": len(results),
                      "timing_rows": len(timing), "raw_best_rows": len(raw_best),
                      "paired_rows": len(paired)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
