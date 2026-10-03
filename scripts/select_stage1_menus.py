"""Choose X0 and two menus using only the frozen 20 validation scenes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from audit_stage1_balance import audit_results, parse_results, read_bundle, sha, write_json

MENUS = ("X0", "X1", "X2", "X3", "X4")


def summarize(rows):
    if not rows or any(r.get("metrics") is None and r.get("status") != "initialization_failed" for r in rows):
        raise ValueError("All scenes need actual metrics or an explicit audited initialization failure")
    metrics = [r["metrics"] for r in rows if r.get("metrics") is not None]
    def average(key):
        values = [m.get(key) for m in metrics]
        if not values or any(v is None or not math.isfinite(float(v)) for v in values):
            return None
        return float(np.mean(values))
    times = [r.get("total_pipeline_seconds", r.get("balance_diagnostics", {}).get("total_pipeline_seconds")) for r in rows]
    if any(v is None or not math.isfinite(float(v)) or float(v) < 0 for v in times):
        raise ValueError("Full per-scene pipeline timings are required for efficiency selection")
    return {
        "scene_count": len(rows),
        "valid_layout_scenes": len(metrics),
        "initialization_failed_scenes": len(rows) - len(metrics),
        "full_coverage_legal_scenes": sum(bool(m["coverage_100"]) and int(m["empty_beams"]) == 0 for m in metrics),
        "near_coverage_legal_scenes": sum(bool(m["feasible"]) for m in metrics),
        "balanced_target_scenes": sum(bool(m["balanced_target_met"]) for m in metrics),
        "subject_cv_mean": average("subject_cv"),
        "subject_in_band_beams_mean": average("subject_in_band_beams"),
        "subject_max_mean_ratio_mean": average("subject_max_mean_ratio"),
        "all_cv_mean": average("load_cv"),
        "full_pipeline_seconds_mean": float(np.mean(times)),
        "full_pipeline_seconds_median": float(np.median(times)),
        "full_pipeline_seconds_p95": float(np.percentile(times, 95)),
    }


def quality_key(label, summary):
    def finite(key):
        value = summary[key]
        return float(value) if value is not None else math.inf
    return (-summary["full_coverage_legal_scenes"], -summary["near_coverage_legal_scenes"],
            -summary["balanced_target_scenes"], finite("subject_cv_mean"),
            finite("subject_max_mean_ratio_mean"), summary["full_pipeline_seconds_mean"], label)


def choose(summaries):
    if set(summaries) != set(MENUS):
        raise ValueError("Selection requires X0-X4 exactly, with X0 always retained")
    quality_order = sorted(MENUS[1:], key=lambda label: quality_key(label, summaries[label]))
    quality = quality_order[0]
    baseline = summaries["X0"]
    remaining = [label for label in quality_order if label != quality]
    eligible = []
    for label in remaining:
        s = summaries[label]
        if (s["full_coverage_legal_scenes"] >= baseline["full_coverage_legal_scenes"] and
                s["near_coverage_legal_scenes"] >= baseline["near_coverage_legal_scenes"] and
                s["balanced_target_scenes"] >= baseline["balanced_target_scenes"] and
                s["subject_cv_mean"] is not None and baseline["subject_cv_mean"] is not None and
                s["subject_cv_mean"] <= baseline["subject_cv_mean"] + 1e-12 and
                s["subject_in_band_beams_mean"] >= baseline["subject_in_band_beams_mean"]):
            eligible.append(label)
    if eligible:
        other = min(eligible, key=lambda label: (summaries[label]["full_pipeline_seconds_mean"],
                                                quality_key(label, summaries[label])))
        reason = "fastest_remaining_menu_noninferior_to_X0_on_registered_coverage_and_balance_metrics"
    else:
        other, reason = remaining[0], "no_efficiency_eligible_menu_use_next_quality_rank_without_rerun"
    return {"selected_menus": ["X0", quality, other], "quality_menu": quality,
            "second_menu": other, "second_menu_reason": reason,
            "quality_order": quality_order, "efficiency_eligible": eligible}


def select_results(results):
    if set(results) != set(MENUS):
        raise ValueError("Exactly five X0-X4 validation result files are required")
    audited = audit_results(results, 20)
    if not audited["passed"]:
        raise ValueError("Validation result audit failed; cannot choose models from unchecked metrics")
    summaries = {}
    for label, path in results.items():
        rows, config = read_bundle(path)
        datasets = config.get("datasets", {})
        record = datasets.get("validation")
        if not record or set(datasets) != {"validation"}:
            raise ValueError(f"{label}: selection may only use the validation partition")
        dates = record.get("dates", record.get("source", {}).get("source_dates", []))
        if dates != ["2024-01-12"]:
            raise ValueError(f"{label}: not the pre-registered Jan12 validation scenes")
        if config.get("coverage", {}).get("balance", {}).get("action_menu") != label:
            raise ValueError(f"{label}: action menu does not match frozen task configuration")
        summaries[label] = summarize(rows)
    decision = choose(summaries)
    return {"experiment_id": "S1-02", "status": "passed", "menus": decision["selected_menus"],
            "selection_source": "Jan12_validation_only", "created_at": datetime.now(timezone.utc).isoformat(),
            "selection_scope": "Jan12 20 validation scenes; never A/B independent test", "passed": True,
            "selection_rule_version": "quality_and_efficiency_v1", **decision, "summaries": summaries,
            "sources": {label: {"path": str(path), "sha256": sha(path)} for label, path in results.items()},
            "training_steps_per_menu": 409600, "total_training_steps": 1228800, "seed": 51001,
            "limitations": ["Nonlearning menu screening does not establish PPO performance",
                            "Single training seed; results do not establish global optimality"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_results(parse_results(args.result))
    write_json(args.output, result)
    print(json.dumps({"selected_menus": result["selected_menus"], "passed": True,
                      "reason": result["second_menu_reason"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
