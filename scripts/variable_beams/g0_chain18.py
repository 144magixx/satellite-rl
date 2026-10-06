"""Wait for G0's assessment, submit its SCA, then collect both result sets.

Chained explicitly so nothing depends on a later hand-off: the SCA is only submitted
once the assessment reports completed, and its per-method metrics are read from the
refinement's own CSV export.
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
ASSESSMENT_REQUEST = "s2p18-g0-jan29-eval-fix2"
PY = r"C:\Users\Admin\anaconda3\python.exe"
DRIVER = r"D:\paper\scripts\variable_beams\submit18.py"
OUT = Path(r"D:\paper\outputs\experiments\variable_beams_s2p18_20261006\analysis")


def assessment_state() -> dict:
    for item in ops18.api(PORT, "/runs", timeout=60).get("runs", []):
        if str(item.get("request_id", "")) != ASSESSMENT_REQUEST:
            continue
        detail = ops18.api(PORT, "/runs/" + item["run_id"], timeout=60)
        status = detail.get("status") or {}
        return {"run_id": item["run_id"], "stage": status.get("stage"),
                "units": status.get("env_step"), "budget": status.get("budget_env_steps"),
                "error": status.get("error")}
    return {}


def main() -> None:
    deadline = time.time() + 5400
    while time.time() < deadline:
        state = assessment_state()
        print(time.strftime("%H:%M:%S"), state, flush=True)
        if state.get("error"):
            print("考核出错，停止", flush=True)
            return
        if state.get("stage") == "completed":
            break
        time.sleep(120)
    else:
        print("等待超时", flush=True)
        return

    print("=== 提交 G0 的统一 SCA ===", flush=True)
    result = subprocess.run([PY, DRIVER, "--arm", "G0", "--port", str(PORT),
                             "--sca", "--tag", "fix2", "--execute"],
                            capture_output=True, text=True)
    print((result.stdout or "")[-600:], flush=True)
    print((result.stderr or "")[-300:], flush=True)

    print("=== 读取 G0 考核总体与基线 ===", flush=True)
    run_id = state["run_id"]
    data = ops18.api(PORT, f"/runs/{run_id}/results", timeout=2400)
    summary = data.get("summary") or {}
    print(json.dumps({key: summary.get(key) for key in
                      ("final_si_mean", "final_sgm_mean", "actual_slots", "power_w")},
                     ensure_ascii=False), flush=True)
    for name, block in (data.get("baselines") or {}).items():
        base = block.get("summary") or {}
        print(f"  基线 {name}: SI={base.get('final_si_mean')} 槽={base.get('actual_slots')}", flush=True)
    scenes = [row for row in (data.get("scenes") or [])
              if str(row.get("group", "model")).startswith("model")]
    per_n = {}
    for count in sorted({row.get("beam_count") for row in scenes if row.get("beam_count")}):
        rows = [row for row in scenes if row.get("beam_count") == count]
        si = statistics.fmean(row["metrics"]["final_si_mean"] for row in rows)
        slots = statistics.fmean(row["metrics"]["actual_slots"] for row in rows) / count
        per_n[str(count)] = {"si": round(si, 4), "slots_per_beam": round(slots, 2),
                             "rows": len(rows)}
        print(f"  N={count}: SI={si:.4f} 槽/束={slots:.2f} 行数={len(rows)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "g0_assessment.json").write_text(
        json.dumps({"checked_utc": ops18.utc_now(), "run_id": run_id,
                    "overall": {key: summary.get(key) for key in
                                ("final_si_mean", "final_sgm_mean", "actual_slots", "power_w")},
                    "baselines": {name: (block.get("summary") or {}).get("final_si_mean")
                                  for name, block in (data.get("baselines") or {}).items()},
                    "per_n": per_n}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("已写入 g0_assessment.json", flush=True)
    print("=== 读取 G0 的 SCA 逐方法结果 ===", flush=True)
    for _ in range(20):
        time.sleep(60)
        for item in ops18.api(PORT, "/runs", timeout=60).get("runs", []):
            if "g0-sca" not in str(item.get("request_id", "")):
                continue
            detail = ops18.api(PORT, "/runs/" + item["run_id"], timeout=60)
            status = detail.get("status") or {}
            print("  SCA", status.get("stage"), status.get("env_step"), "/",
                  status.get("budget_env_steps"), flush=True)
            if status.get("stage") == "completed":
                text = sca_results18.stream_csv(PORT, item["run_id"], "power_results.csv")
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
                    table[method] = {"rows": len(items),
                                     "si": round(statistics.fmean(col("final_si_mean")), 4),
                                     "sgm": round(statistics.fmean(col("final_sgm_mean")), 4),
                                     "power_w": round(statistics.fmean(col("power_w")), 1)}
                (OUT / "g0_sca_per_method.json").write_text(
                    json.dumps({"checked_utc": ops18.utc_now(), "methods": table},
                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(table, ensure_ascii=False), flush=True)
                print("G0 全流程完成", flush=True)
                return
    print("SCA 未在窗口内完成", flush=True)


if __name__ == "__main__":
    main()
