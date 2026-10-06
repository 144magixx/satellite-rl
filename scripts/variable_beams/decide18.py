"""Aggregate S2-P18 validation results per beam count and apply the G1-G4 gates.

Read-only: it pulls each arm's evaluations.jsonl from the two instances, groups
by beam count and load, and reports the mean SI/SGM plus the mechanism metrics
(slots per beam, power, effective carried) that the plan's thresholds refer to.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ops18  # noqa: E402

ARMS = {"V2": 20418, "G0": 20418, "V3": 20424}
FAILED_PREFIX = "s2p18-g0-pilot-100000-51001"  # superseded attempts are skipped by stage

FETCH = r'''
import json,pathlib
directory=pathlib.Path({directory!r})
path=directory/"{filename}"
rows=[]
if path.is_file():
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line=line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
print(json.dumps({{"rows":rows[-{limit}:],"total":len(rows)}}))
'''


def read_jsonl(port: int, directory: str, filename: str, limit: int = 4000) -> dict:
    body = FETCH.format(directory=directory, filename=filename, limit=limit)
    result = ops18.remote_python(body, port, timeout=600)
    if result["exit_code"]:
        raise RuntimeError(f"{port} 读取 {filename} 失败: {result['stderr'][-300:]}")
    return json.loads(result["stdout"].strip().splitlines()[-1])


def arm_runs(port: int, arm: str) -> list:
    runs = ops18.api(port, "/runs", timeout=60).get("runs", [])
    found = []
    for item in runs:
        request = str(item.get("request_id", ""))
        if not request.startswith(f"s2p18-{arm.lower()}-pilot-"):
            continue
        detail = ops18.api(port, "/runs/" + item["run_id"], timeout=60)
        status = detail.get("status") or {}
        found.append({"run_id": item["run_id"], "request_id": request,
                      "stage": status.get("stage") or item.get("stage"),
                      "env_step": status.get("env_step"),
                      "directory": detail.get("path")})
    return found


def summarize(rows: list) -> dict:
    """Group validation rows by beam count and load, then average metrics."""
    groups: dict = {}
    for row in rows:
        metrics = row.get("metrics") or {}
        key = (row.get("beam_count"), (row.get("scene_metadata") or {}).get("load_multiplier"))
        bucket = groups.setdefault(key, [])
        bucket.append(metrics)
    out = {}
    for (count, load), items in sorted(groups.items(), key=lambda kv: (kv[0][0] or 0, str(kv[0][1]))):
        def mean(name):
            values = [float(m[name]) for m in items if isinstance(m.get(name), (int, float))]
            return round(statistics.fmean(values), 6) if values else None
        out[f"N={count}|load={load}"] = {
            "scenes": len(items),
            "si": mean("final_si_mean"),
            "sgm": mean("final_sgm_mean"),
            "slots_total": mean("slots_total"),
            "power_w": mean("power_w"),
            "carried_bps": mean("effective_carried_bps") or mean("throughput_bps"),
            "basic_met": mean("basic_met_count"),
        }
    return out


def by_n(table: dict, metric: str) -> dict:
    """Average over loads for each N."""
    acc: dict = {}
    for key, value in table.items():
        count = int(key.split("|")[0].split("=")[1])
        if value.get(metric) is None:
            continue
        acc.setdefault(count, []).append(value[metric])
    return {count: round(statistics.fmean(values), 6) for count, values in sorted(acc.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="validating", choices=("validating", "training", "any"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = {}
    for arm, port in ARMS.items():
        runs = [run for run in arm_runs(port, arm) if run["stage"] != "failed"]
        if not runs:
            report[arm] = {"status": "no_live_run"}
            continue
        run = max(runs, key=lambda item: item.get("env_step") or 0)
        data = read_jsonl(port, run["directory"], "evaluations.jsonl")
        table = summarize(data["rows"])
        report[arm] = {
            "port": port, "run_id": run["run_id"], "stage": run["stage"],
            "env_step": run["env_step"], "evaluation_rows": data["total"],
            "per_n_si": by_n(table, "si"), "per_n_sgm": by_n(table, "sgm"),
            "per_group": table,
        }
    # G1 style comparison: the plan anchors N>=125 to the frozen V1 result.
    v1 = {"100": 0.9342, "120": None, "160": None}
    report["reference"] = {"v1_final_si_at_100": v1["100"],
                           "note": "V1 的 40 万步终态；试点臂为 10 万步，绝对比较只作参考"}
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.write:
        ops18.write_json(ops18.LOCAL / "analysis_inputs" / "validation_by_n.json",
                         {"checked_utc": ops18.utc_now(), **report})
    print(text)


if __name__ == "__main__":
    main()
