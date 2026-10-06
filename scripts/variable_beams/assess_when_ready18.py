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


def submit(arm: str, port: int) -> dict:
    result = subprocess.run([PY, DRIVER, "--arm", arm, "--port", str(port), "--evaluate", "--execute"],
                            capture_output=True, text=True)
    text = (result.stdout or "").strip()
    try:
        return json.loads(text[text.index("{"):]) if "{" in text else {"raw": text}
    except Exception:
        return {"raw": text[-600:], "stderr": (result.stderr or "")[-300:]}


def main() -> None:
    done = set()
    deadline = time.time() + DEADLINE_HOURS * 3600
    while time.time() < deadline and len(done) < len(ARMS):
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
                print(json.dumps(submit(arm, port), ensure_ascii=False), flush=True)
                done.add(arm)
        time.sleep(POLL)
    print("等待结束，已提交:", sorted(done), flush=True)


if __name__ == "__main__":
    main()
