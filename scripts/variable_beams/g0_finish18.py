"""Wait for G0's SCA, save its per-method table and per-beam tails, then fill the report.

This replaces an earlier chain whose polling window could expire before the refinement
finished. It waits on the job's own status, writes both result files, runs the tail
aggregation for G0, and finally invokes the report filler so the numbers come from the
saved files rather than from typing.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, r"D:\paper\scripts\variable_beams")
import ops18  # noqa: E402
import sca_results18  # noqa: E402

PORT = 20418
REQUEST = "s2p18-g0-sca-fix2"
PY = r"C:\Users\Admin\anaconda3\python.exe"
OUT = Path(r"D:\paper\outputs\experiments\variable_beams_s2p18_20261006\analysis")
LABELS = {"none": "原分配（不调整）", "sca": "功率优化 SCA", "sca_si_lp": "SI优先 SCA"}


def find_job() -> dict:
    for item in ops18.api(PORT, "/runs", timeout=60).get("runs", []):
        if str(item.get("request_id", "")) != REQUEST:
            continue
        detail = ops18.api(PORT, "/runs/" + item["run_id"], timeout=60)
        status = detail.get("status") or {}
        return {"run_id": item["run_id"], "stage": status.get("stage"),
                "units": status.get("env_step"), "budget": status.get("budget_env_steps"),
                "error": status.get("error")}
    return {}


def main() -> None:
    deadline = time.time() + 3 * 3600
    while time.time() < deadline:
        state = find_job()
        print(time.strftime("%H:%M:%S"), state, flush=True)
        if state.get("error"):
            print("G0 的 SCA 出错，停止", flush=True)
            return
        if state.get("stage") == "completed":
            break
        time.sleep(180)
    else:
        print("等待超时", flush=True)
        return

    run_id = state["run_id"]
    text = sca_results18.stream_csv(PORT, run_id, "power_results.csv")
    rows = list(csv.DictReader(io.StringIO(text)))
    by_method = {}
    for row in rows:
        by_method.setdefault(str(row.get("method")), []).append(row)
    table = {}
    for method, items in sorted(by_method.items()):
        def col(key):
            out = []
            for row in items:
                try:
                    out.append(float(row[key]))
                except (KeyError, TypeError, ValueError):
                    continue
            return out
        table[method] = {"label": LABELS.get(method, method), "rows": len(items),
                         "si": round(statistics.fmean(col("final_si_mean")), 4),
                         "sgm": round(statistics.fmean(col("final_sgm_mean")), 4),
                         "power_w": round(statistics.fmean(col("power_w")), 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "g0_sca_per_method.json").write_text(
        json.dumps({"checked_utc": ops18.utc_now(), "run_id": run_id, "methods": table},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("已写入 g0_sca_per_method.json:", json.dumps(table, ensure_ascii=False), flush=True)

    print("开始逐束尾部聚合（远端分块）", flush=True)
    tail = subprocess.run([PY, r"D:\paper\scripts\variable_beams\beam_tails18.py", "--arm", "G0"],
                          capture_output=True, text=True)
    print((tail.stdout or "")[-500:], flush=True)
    print((tail.stderr or "")[-300:], flush=True)

    print("填入报告", flush=True)
    fill = subprocess.run([PY, r"D:\paper\scripts\variable_beams\fill_g0_sca18.py"],
                          capture_output=True, text=True)
    print((fill.stdout or "")[-500:], flush=True)
    print((fill.stderr or "")[-300:], flush=True)
    print("G0 全流程收尾完成", flush=True)


if __name__ == "__main__":
    main()
