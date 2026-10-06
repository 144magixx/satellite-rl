"""Aggregate S2-P18 comparison results and build the Chinese report figures.

Reads the completed evaluation / refinement jobs from both instances, so it works
the same way for the pilot (144 validation scenes) and for the final stage
(440 common assessment scenes). It reports which fields the saved rows actually
carry instead of assuming a schema, and every figure is generated from those
saved rows only.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ops18  # noqa: E402

# The three arms plus the two baselines carried by each assessment job.
METHODS = ("V2", "V3", "G0", "fixed", "greedy")
RUN_PREFIX = "s2p18-"

CANDIDATE_METRICS = {
    "si": ("final_si_mean", "si_mean", "mean_si"),
    "sgm": ("final_sgm_mean", "sgm_mean", "mean_sgm"),
    "basic": ("basic_fraction", "basic_met_fraction", "basic_service_fraction"),
    "slots": ("actual_slots", "allocated_slots", "slots_total"),
    "power": ("power_w", "total_power_w"),
    "carried": ("effective_carried_bps", "throughput_bps", "total_rate_bps"),
    "carried_useful": ("useful_carried_bps", "min_carried_bps"),
    "jain": ("jain", "jain_index"),
    "complete": ("complete", "complete_fraction"),
}

FETCH_RESULTS = r'''
import json,pathlib
directory=pathlib.Path(@DIR@)
path=directory/"results.json"
out={"exists":path.is_file()}
if path.is_file():
    data=json.loads(path.read_text(encoding="utf-8"))
    out["top_keys"]=sorted(data.keys())
    scenes=data.get("scenes") or []
    out["scene_count"]=len(scenes)
    if scenes:
        out["scene_keys"]=sorted(scenes[0].keys())
        metric_keys=set()
        for row in scenes[:5]:
            metrics=row.get("metrics") or {}
            metric_keys |= set(metrics.keys())
        out["metric_keys"]=sorted(metric_keys)
    baselines=data.get("baselines") or {}
    out["baseline_keys"]=sorted(baselines.keys())
    out["summary"]=data.get("summary")
print(json.dumps(out,ensure_ascii=False)[:12000])
'''


def run_index(port: int) -> list:
    """All S2-P18 jobs on one instance, newest first."""
    runs = ops18.api(port, "/runs", timeout=60).get("runs", [])
    out = []
    for item in runs:
        request = str(item.get("request_id", ""))
        if not request.startswith(RUN_PREFIX):
            continue
        detail = ops18.api(port, "/runs/" + item["run_id"], timeout=60)
        status = detail.get("status") or {}
        out.append({"run_id": item["run_id"], "request_id": request,
                    "kind": item.get("kind"),
                    "stage": status.get("stage") or item.get("stage"),
                    "env_step": status.get("env_step"),
                    "directory": detail.get("path")})
    return out


def probe_results(port: int, directory: str) -> dict:
    body = FETCH_RESULTS.replace("@DIR@", repr(directory))
    result = ops18.remote_python(body, port, timeout=300)
    lines = [line for line in result["stdout"].strip().splitlines() if line.strip()]
    if not lines:
        return {"exists": False, "error": result["stderr"][-200:]}
    return json.loads(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true",
                        help="只探测已保存结果的字段结构，不做汇总")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = {}
    for port in ops18.PORTS:
        for run in run_index(port):
            if run["kind"] not in ("evaluate", "refine"):
                continue
            if run["stage"] not in ("completed", "failed", "interrupted", "training"):
                continue
            entry = {"port": port, **{k: v for k, v in run.items() if k != "directory"}}
            if args.probe:
                entry["probe"] = probe_results(port, run["directory"])
            report[run["run_id"]] = entry
    if args.write:
        ops18.write_json(ops18.LOCAL / "analysis_inputs" / "evaluation_inventory.json",
                         {"checked_utc": ops18.utc_now(), "jobs": report})
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
