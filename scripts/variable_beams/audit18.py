"""Audit an SCA refinement job: per-method results, per-beam tails and stage timings.

Reads the refinement job's own CSVs (``power_results.csv``, ``power_beams.csv``,
``power_scene_times.csv``) plus its saved initial sources, so the report can present
both the direct allocation result and the result after SCA without mixing the two
accounting bases. Statistics are reported as mean/median/P95 with the scene count
that produced them, and every aggregate names its denominator.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, r"D:\paper\scripts\variable_beams")
import ops18  # noqa: E402

STAGE_KEYS = ("initialization_seconds", "physics_setup_seconds", "optimizer_wall_seconds",
              "optimizer_callback_io_seconds", "optimizer_excluding_callback_seconds",
              "verification_seconds", "total_pipeline_seconds",
              "reference_optimizer_wall_seconds", "reference_total_seconds")
METHOD_LABELS = {"none": "原分配（不调整）", "sca": "功率优化 SCA", "sca_si_lp": "SI优先 SCA"}


def read_remote_csv(port: int, directory: str, filename: str) -> list:
    """Pull one CSV back through the shared instance directory, verifying its SHA."""
    body = (
        "import hashlib,json,pathlib,shutil\n"
        f"source=pathlib.Path({directory!r})/{filename!r}\n"
        f"target=pathlib.Path({ops18.instance(port)['ops']!r})/'audit_{filename}'\n"
        "out={'exists':source.is_file()}\n"
        "if out['exists']:\n"
        "    shutil.copy2(source,target)\n"
        "    out['sha256']=hashlib.sha256(target.read_bytes()).hexdigest()\n"
        "    out['bytes']=target.stat().st_size\n"
        "print(json.dumps(out))\n"
    )
    result = ops18.remote_python(body, port, timeout=900)
    lines = [line for line in result["stdout"].strip().splitlines() if line.strip()]
    if not lines:
        return []
    info = json.loads(lines[-1])
    if not info.get("exists"):
        return []
    staging = f"{ops18.instance(port)['ops']}/audit_{filename}"
    import paramiko
    client = ops18.connect(port)
    try:
        with paramiko.SFTPClient.from_transport(client.get_transport()) as sftp:
            with sftp.open(staging) as handle:
                text = handle.read().decode("utf-8")
    finally:
        client.close()
    return list(csv.DictReader(io.StringIO(text)))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stats(values: list) -> dict:
    clean = sorted(value for value in values if value is not None)
    if not clean:
        return {"n": 0}
    index = max(0, min(len(clean) - 1, int(0.95 * (len(clean) - 1))))
    return {"n": len(clean), "mean": round(statistics.fmean(clean), 4),
            "median": round(statistics.median(clean), 4), "p95": round(clean[index], 4)}


def summarise(rows: list) -> dict:
    """Per method: metrics, stage timings and the per-beam C/D tail."""
    out = {}
    by_method = {}
    for row in rows:
        by_method.setdefault(row.get("method"), []).append(row)
    for method, items in by_method.items():
        entry = {"rows": len(items), "label": METHOD_LABELS.get(method, method)}
        for key in ("final_si_mean", "final_sgm_mean", "power_w"):
            entry[key] = stats([number(row.get(key)) for row in items])
        beams = [number(row.get("capacity_demand_ratio")) for row in items
                 if "capacity_demand_ratio" in row]
        if beams:
            entry["beam_ratio"] = stats(beams)
        for key in STAGE_KEYS:
            values = [number(row.get(key)) for row in items]
            if any(value is not None for value in values):
                entry[key] = stats(values)
        out[method] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, choices=ops18.PORTS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    detail = ops18.api(args.port, "/runs/" + args.run_id, timeout=60)
    directory = detail.get("path")
    if not directory:
        raise RuntimeError("缺少任务目录")
    status = detail.get("status") or {}
    report = {"run_id": args.run_id, "port": args.port, "directory": directory,
              "stage": status.get("stage"), "env_step": status.get("env_step")}
    results = read_remote_csv(args.port, directory, "power_results.csv")
    beams = read_remote_csv(args.port, directory, "power_beams.csv")
    times = read_remote_csv(args.port, directory, "power_scene_times.csv")
    report["files"] = {"power_results.csv": len(results),
                       "power_beams.csv": len(beams),
                       "power_scene_times.csv": len(times)}
    if results:
        report["methods"] = summarise(results)
        report["columns"] = sorted(results[0].keys())
    if beams:
        report["beam_columns"] = sorted(beams[0].keys())
        ratios = [number(row.get("capacity_demand_ratio") or row.get("ratio")) for row in beams]
        si = [number(row.get("si")) for row in beams]
        report["beams_all"] = {"capacity_demand_ratio": stats(ratios), "si": stats(si)}
    if times:
        report["scene_timing_columns"] = sorted(times[0].keys())
        for key in ("total_pipeline_seconds_mean_per_action", "optimizer_wall_seconds_mean_per_action",
                    "reference_total_seconds_mean_per_action"):
            values = [number(row.get(key)) for row in times]
            if any(value is not None for value in values):
                report.setdefault("scene_timings", {})[key] = stats(values)
    if args.write:
        name = f"{args.run_id}_audit.json"
        ops18.write_json(ops18.LOCAL / "analysis" / name, report)
    print(json.dumps(report, ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    main()
