"""Summarise a refinement job's scene-level timings without moving the huge per-row CSV.

``power_scene_times.csv`` carries one row per scene and method with the stage sums, so
it is small enough to stream over SFTP. ``power_results.csv`` is far larger and is left
on the instance; the aggregates that matter for the report come from the scene table.
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

LABELS = {"none": "原分配（不调整）", "sca": "功率优化 SCA", "sca_si_lp": "SI优先 SCA"}
STAGES = ("initialization_seconds", "physics_setup_seconds", "optimizer_wall_seconds",
          "verification_seconds", "total_pipeline_seconds",
          "reference_total_seconds")


def fetch_text(port: int, directory: str, filename: str, limit_bytes: int = 8_000_000) -> str:
    staging = f"{ops18.instance(port)['ops']}/audit_{filename}"
    body = (
        "import json,pathlib,shutil\n"
        f"source=pathlib.Path({directory!r})/{filename!r}\n"
        f"target=pathlib.Path({staging!r})\n"
        "out={'exists':source.is_file()}\n"
        "if out['exists']:\n"
        "    shutil.copy2(source,target)\n"
        "    out['bytes']=target.stat().st_size\n"
        "print(json.dumps(out))\n"
    )
    result = ops18.remote_python(body, port, timeout=900)
    lines = [line for line in result["stdout"].strip().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(result["stderr"][-300:])
    info = json.loads(lines[-1])
    if not info.get("exists"):
        return ""
    if info.get("bytes", 0) > limit_bytes:
        print(f"提示：{filename} 为 {info['bytes']} 字节，超过阈值，仍继续流式读取")
    import paramiko
    client = ops18.connect(port)
    try:
        with paramiko.SFTPClient.from_transport(client.get_transport()) as sftp:
            with sftp.open(staging) as handle:
                handle.prefetch()
                return handle.read().decode("utf-8", errors="replace")
    finally:
        client.close()


def stats(values):
    clean = sorted(value for value in values if isinstance(value, (int, float)))
    if not clean:
        return None
    index = max(0, min(len(clean) - 1, int(0.95 * (len(clean) - 1))))
    return {"n": len(clean), "mean": round(statistics.fmean(clean), 4),
            "median": round(statistics.median(clean), 4), "p95": round(clean[index], 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, choices=ops18.PORTS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    detail = ops18.api(args.port, "/runs/" + args.run_id, timeout=120)
    directory = detail.get("path")
    if not directory:
        raise RuntimeError("缺少任务目录")
    text = fetch_text(args.port, directory, "power_scene_times.csv")
    if not text:
        print("没有 power_scene_times.csv")
        return
    rows = list(csv.DictReader(io.StringIO(text)))
    print("场景计时行数:", len(rows))
    if rows:
        print("列:", sorted(rows[0].keys())[:14])
    report = {"run_id": args.run_id, "port": args.port, "rows": len(rows)}
    by_method = {}
    for row in rows:
        by_method.setdefault(str(row.get("method")), []).append(row)
    for method, items in sorted(by_method.items()):
        entry = {"scenes": len(items)}
        for stage in STAGES:
            key = f"{stage}_mean_per_action"
            values = []
            for row in items:
                try:
                    values.append(float(row.get(key)))
                except (TypeError, ValueError):
                    continue
            if values:
                entry[stage] = stats(values)
        report[method] = entry
    print(json.dumps(report, ensure_ascii=False, indent=1)[:2500])
    if args.write:
        ops18.write_json(ops18.LOCAL / "analysis" / f"timing_{args.run_id}.json", report)


if __name__ == "__main__":
    main()
