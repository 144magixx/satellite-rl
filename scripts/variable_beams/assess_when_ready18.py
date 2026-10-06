"""Wait for each S2-P18 arm to finish 400k steps, then submit its 440-scene assessment.

Waiting by hand wastes rounds: the assessment gate accepts a model only at the
400k-step endpoint, so the useful trigger is exactly "the extension reached its
budget and the run settled". This script polls quietly, then submits through the
same guarded driver used elsewhere (``submit18.py --evaluate``), which selects the
``final`` export and refuses anything else.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, r"D:\paper\scripts\variable_beams")
import ops18  # noqa: E402

PY = r"C:\Users\Admin\anaconda3\python.exe"
DRIVER = r"D:\paper\scripts\variable_beams\submit18.py"
ARMS = {"V3": (20424, "train_sac_20261006_084726_c004cd09"),
        "V2": (20418, "train_sac_20261006_084454_365eff71"),
        "G0": (20418, "train_sac_20261006_092142_142c6953")}
POLL = 180
DEADLINE_HOURS = 14


def state(port: int, run_id: str) -> dict:
    detail = ops18.api(port, "/runs/" + run_id, timeout=60)
    return detail.get("status") or {}


def submit(arm: str, port: int, mode: str) -> dict:
    """Submit either the 440-scene assessment or the SCA refinement for one arm."""
    result = subprocess.run([PY, DRIVER, "--arm", arm, "--port", str(port), f"--{mode}", "--execute"],
                            capture_output=True, text=True)
    text = (result.stdout or "").strip()
    try:
        return json.loads(text[text.index("{"):]) if "{" in text else {"raw": text}
    except Exception:
        return {"raw": text[-600:], "stderr": (result.stderr or "")[-300:]}


def assessment_state(arm: str, port: int) -> dict:
    """Latest assessment job for this arm, so SCA can be chained after it completes."""
    runs = ops18.api(port, "/runs", timeout=60).get("runs", [])
    best = {}
    for item in runs:
        request = str(item.get("request_id", ""))
        if not request.startswith(f"s2p18-{arm.lower()}-") or "eval" not in request:
            continue
        detail = ops18.api(port, "/runs/" + item["run_id"], timeout=60)
        status = detail.get("status") or {}
        row = {"run_id": item["run_id"], "request_id": request,
               "stage": status.get("stage") or item.get("stage"),
               "env_step": status.get("env_step"), "error": status.get("error")}
        if not best or (row["env_step"] or 0) >= (best.get("env_step") or 0):
            best = row
    return best


def main() -> None:
    done, refined = set(), set()
    deadline = time.time() + DEADLINE_HOURS * 3600
    while time.time() < deadline and len(refined) < len(ARMS):
        for arm, (port, run_id) in ARMS.items():
            if arm in done:
                continue
            status = state(port, run_id)
            stage = status.get("stage")
            step = status.get("env_step")
            budget = status.get("budget_env_steps")
            print(json.dumps({"beijing": time.strftime("%H:%M:%S"), "arm": arm,
                              "stage": stage, "env_step": step, "budget": budget,
                              "error": status.get("error")}, ensure_ascii=True), flush=True)
            if status.get("error"):
                print(f"{arm} 出错，停止跟进", flush=True)
                return
            if stage == "completed" and step == budget == 400000:
                print(f"{arm} 已到40万步终点，提交440景考核", flush=True)
                print(json.dumps(submit(arm, port, "evaluate"), ensure_ascii=False), flush=True)
                done.add(arm)
            if arm in done:
                # 考核完成后立刻串接统一SCA；SCA以已完成的考核为冻结初解来源。
                assessment = assessment_state(arm, port)
                if assessment.get("stage") == "completed" and arm not in refined:
                    print(f"{arm} 考核完成，提交统一SCA", flush=True)
                    print(json.dumps(submit(arm, port, "sca"), ensure_ascii=False), flush=True)
                    refined.add(arm)
                elif assessment.get("error"):
                    print(f"{arm} 考核出错：{assessment['error']}", flush=True)
                    return
        time.sleep(POLL)
    print("等待结束，已提交考核:", sorted(done), "已提交SCA:", sorted(refined), flush=True)


if __name__ == "__main__":
    main()
