"""Read-only paired comparison from archived complete stage 3 case CSV files."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import fmean, median


ROOT = Path("D:/paper/outputs/experiments/stage3_all_routes_20261003")
ARCHIVE = ROOT / "remote_archive"
OUT = ROOT / "completion"
PORT_MODES = {20418: "total", 20424: "initial"}
METHODS = ("none", "discrete", "slsqp", "sca", "de", "pso", "wmmse")
METRICS = ("final_si_mean", "final_sgm_mean", "basic_met_count", "all_basic_met", "power_w")
TIMING_KEYS = ("initialization_seconds", "physics_setup_seconds", "optimizer_wall_seconds",
               "optimizer_callback_io_seconds", "optimizer_excluding_callback_seconds",
               "verification_seconds", "total_pipeline_seconds")


def quantile95(values: list[float]) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    low = int(position)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[min(low + 1, len(ordered) - 1)] * fraction


def distribution(values: list[float]) -> dict:
    return {"mean": fmean(values), "median": median(values), "p95": quantile95(values)}


def metric_number(raw: str, name: str) -> float:
    if name == "all_basic_met":
        if raw == "True":
            return 1.0
        if raw == "False":
            return 0.0
        raise ValueError("all_basic_met must be explicit True/False in saved CSV")
    return float(raw)


def self_check() -> None:
    assert metric_number("True", "all_basic_met") == 1.0
    assert metric_number("False", "all_basic_met") == 0.0
    assert metric_number("0.875", "final_si_mean") == 0.875
    try:
        metric_number("1", "all_basic_met")
    except ValueError:
        pass
    else:
        raise AssertionError("Ambiguous boolean CSV value was accepted")
    assert paired([1.0, 0.0, -1.0]) == {
        "mean": 0.0, "positive": 1, "tie": 1, "negative": 1, "scene_count": 3}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(port: int) -> dict:
    manifest = json.loads((ARCHIVE / str(port) / "archive_manifest.json").read_text(encoding="utf-8"))
    matching = [item for item in manifest["files"]
                if item["label"] == "job" and item["relative"] == "power_results.csv"]
    if len(matching) != 1:
        raise RuntimeError(f"Port {port} is missing the complete case CSV")
    reference = matching[0]
    path = Path(reference["local_path"])
    if path.stat().st_size != reference["bytes"] or sha(path) != reference["sha256"]:
        raise RuntimeError(f"Port {port} archived case CSV SHA changed")
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 2100:
        raise RuntimeError(f"Port {port} has {len(rows)} instead of 2100 complete cases")
    result = {}
    for row in rows:
        if row["power_budget_mode"] != PORT_MODES[port] or row["method"] not in METHODS:
            raise RuntimeError("Unexpected budget mode or method in saved cases")
        key = (row["source_scene_sha256"], int(row["repeat"]), row["method"])
        if key in result:
            raise RuntimeError("Duplicate case identity in saved CSV")
        result[key] = {**row, **{metric: metric_number(row[metric], metric) for metric in METRICS},
                       **{field: float(row[field]) for field in TIMING_KEYS}}
    scenes = {key[0] for key in result}
    if len(scenes) != 100 or any(
            (scene, repeat, method) not in result
            for scene in scenes for repeat in range(3) for method in METHODS):
        raise RuntimeError("Incomplete scene/repeat/method grid")
    return result


def paired(values: list[float]) -> dict:
    tolerance = 1e-12
    return {"mean": fmean(values),
            "positive": sum(x > tolerance for x in values),
            "tie": sum(abs(x) <= tolerance for x in values),
            "negative": sum(x < -tolerance for x in values),
            "scene_count": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("stage3_analysis_self_check_passed")
        return
    data = {PORT_MODES[port]: load(port) for port in PORT_MODES}
    total, initial = data["total"], data["initial"]
    if set(total) != set(initial):
        raise RuntimeError("Two budget modes did not use the same scene/repeat/method grid")
    allocation_keys = {(scene, repeat) for scene, repeat, _ in total}
    allocation_sha = {}
    for scene, repeat in allocation_keys:
        hashes = {data[mode][scene, repeat, method]["initial_allocation_sha256"]
                  for mode in data for method in METHODS}
        if len(hashes) != 1:
            raise RuntimeError(f"Initial allocation SHA differs at {scene} repeat {repeat}")
        allocation_sha[f"{scene}:{repeat}"] = hashes.pop()
        for metric in METRICS:
            if abs(total[scene, repeat, "none"][metric]
                   - initial[scene, repeat, "none"][metric]) > 1e-10:
                raise RuntimeError("No-adjustment result differs across budget modes")
    scenes = sorted({key[0] for key in total})
    scene_rows = []
    summaries = {}
    for mode, rows in data.items():
        summaries[mode] = {}
        for method in METHODS:
            selected = [row for key, row in rows.items() if key[2] == method]
            summaries[mode][method] = {metric: fmean(row[metric] for row in selected)
                                       for metric in METRICS}
            summaries[mode][method].update(
                allocations=300,
                fallback_count=sum(row["fallback_used"].lower() == "true" for row in selected),
                mean_solver_seconds=fmean(float(row["solver_seconds"]) for row in selected))
            summaries[mode][method]["timing"] = {}
            for field in TIMING_KEYS:
                per_action = [row[field] for row in selected]
                per_scene = [sum(rows[scene, repeat, method][field] for repeat in range(3))
                             for scene in scenes]
                summaries[mode][method]["timing"][field] = {
                    "single_allocation": distribution(per_action),
                    "scene_three_actions_sum": distribution(per_scene),
                }
        summaries[mode]["paired_vs_none"] = {}
        for method in METHODS[1:]:
            comparisons = {}
            for metric in METRICS:
                deltas = []
                for scene in scenes:
                    deltas.append(fmean(rows[scene, repeat, method][metric]
                                        - rows[scene, repeat, "none"][metric]
                                        for repeat in range(3)))
                    scene_rows.append({"mode": mode, "scene_sha256": scene,
                                       "method": method, "metric": metric,
                                       "delta_vs_none": deltas[-1]})
                comparisons[metric] = paired(deltas)
            summaries[mode]["paired_vs_none"][method] = comparisons
    cross = {}
    for method in METHODS:
        cross[method] = {}
        for metric in METRICS:
            deltas = [fmean(total[scene, repeat, method][metric]
                            - initial[scene, repeat, method][metric] for repeat in range(3))
                      for scene in scenes]
            cross[method][metric] = paired(deltas)
    result = {"status": "read_only_paired_summary_from_sha_archived_complete_cases",
              "computed_utc": datetime.now(timezone.utc).isoformat(),
              "scene_count": 100, "action_samples_per_scene": 3,
              "methods": list(METHODS), "training_interactions": 0,
              "initial_allocation_sha_equal_across_modes_and_methods": True,
              "initial_allocation_count": len(allocation_sha),
              "none_metrics_equal_across_modes": True,
              "summaries": summaries, "total_minus_initial_by_scene": cross,
              "limitations": "One frozen T0 training seed, 150k training steps, and one Jan26 date; repeated actions averaged within each scene. GPU execution may affect wall-time comparisons."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "paired_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "paired_scene_differences.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scene_rows[0]))
        writer.writeheader()
        writer.writerows(scene_rows)
    print(json.dumps({"status": result["status"], "scene_count": 100,
                      "initial_allocation_count": len(allocation_sha),
                      "summary_path": str(OUT / "paired_summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
