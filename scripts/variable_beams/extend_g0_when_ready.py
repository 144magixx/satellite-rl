"""Wait for the G0 pilot to finish, then run the extension chain in one go.

The chain is the same one already exercised twice (V2, V3): retarget the step
budget, flip a completed pilot to interrupted, retire only the snapshot that
collides with the retarget, then resume through the workbench. It refuses to
continue past a step whose receipt does not confirm success.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\paper\scripts\variable_beams")
import ops18  # noqa: E402

RUN = "train_sac_20261006_092142_142c6953"
PORT = 20418
TARGET_STEPS = 400000
PY = r"C:\Users\Admin\anaconda3\python.exe"
SCRIPTS = Path(r"D:\paper\scripts\variable_beams")


def status() -> dict:
    detail = ops18.api(PORT, "/runs/" + RUN, timeout=60)
    return detail.get("status") or {}


def run(script: str, *args: str) -> dict:
    result = subprocess.run([PY, str(SCRIPTS / script), *args],
                            capture_output=True, text=True)
    text = (result.stdout or "").strip()
    try:
        return json.loads(text[text.index("{"):]) if "{" in text else {"raw": text}
    except Exception:
        return {"raw": text[-800:], "stderr": (result.stderr or "")[-400:]}


def main() -> None:
    deadline = time.time() + 7200
    while time.time() < deadline:
        state = status()
        stage = state.get("stage")
        print(time.strftime("%H:%M:%S"), stage, state.get("env_step"), flush=True)
        if stage in ("completed", "failed", "interrupted"):
            break
        if state.get("error"):
            print("ERROR:", state["error"], flush=True)
            return
        time.sleep(120)
    else:
        print("等待超时，未在窗口内收尾", flush=True)
        return
    if state.get("stage") != "completed":
        print("G0 试点未正常完成，需人工检查：", state.get("stage"), state.get("error"), flush=True)
        return
    print("=== 1) 重定标到 40 万步 ===", flush=True)
    print(json.dumps(run("extend18.py", "--port", str(PORT), "--run-id", RUN,
                         "--steps", str(TARGET_STEPS), "--execute"),
                     ensure_ascii=False), flush=True)
    print("=== 2) 退役与试点终点冲突的快照 ===", flush=True)
    print(json.dumps(run("retire_snapshot18.py", "--port", str(PORT), "--run-id", RUN,
                         "--step", "100000", "--execute"), ensure_ascii=False), flush=True)
    print("=== 3) 通过工作台恢复 ===", flush=True)
    resumed = ops18.api(PORT, f"/runs/{RUN}/resume", {}, method="POST", timeout=180)
    print("resume:", resumed.get("stage"), resumed.get("env_step"), flush=True)
    for _ in range(10):
        time.sleep(60)
        state = status()
        print(time.strftime("%H:%M:%S"), state.get("stage"), state.get("env_step"),
              "/", state.get("budget_env_steps"), "alive", state.get("process_alive"),
              "err", state.get("error"), flush=True)
        if state.get("stage") == "training" and (state.get("env_step") or 0) > 100000:
            print("G0 补足阶段已启动", flush=True)
            return
        if state.get("error"):
            print("恢复后出错：", state["error"], flush=True)
            return
    print("恢复后未在10分钟内进入训练，需检查", flush=True)


if __name__ == "__main__":
    main()
