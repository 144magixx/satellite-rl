"""Hourly S2-P18 progress log with immediate alerts on stage changes.

Design per the user's instruction (2026-10-06): record progress **once an hour**.
The process polls quietly every 10 minutes but only writes a progress line when the
hourly mark is reached; it writes immediately if an arm changes stage or reports an
error, because that needs attention rather than a delayed note. Every line is
appended to a durable log next to the run manifest.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, r"D:\paper\scripts\variable_beams")
import ops18  # noqa: E402

LOG = ops18.LOCAL / "followup" / "hourly_progress.jsonl"
POLL_SECONDS = 600
HOUR_SECONDS = 3600
MAX_HOURS = 14
SKIP = ("retry1", "retry2")


def snapshot() -> dict:
    arms = {}
    for port in ops18.PORTS:
        for item in ops18.api(port, "/runs", timeout=60).get("runs", []):
            request = str(item.get("request_id", ""))
            if not request.startswith("s2p18-") or any(tag in request for tag in SKIP):
                continue
            detail = ops18.api(port, "/runs/" + item["run_id"], timeout=60)
            status = detail.get("status") or {}
            stage = status.get("stage") or item.get("stage")
            if stage == "failed" and "retry3" not in request:
                continue
            arms[request.split("-")[1].upper()] = {
                "port": port, "run_id": item["run_id"], "stage": stage,
                "env_step": status.get("env_step"),
                "budget_env_steps": status.get("budget_env_steps"),
                "steps_per_second": round(status.get("steps_per_second") or 0.0, 2),
                "process_alive": status.get("process_alive"),
                "error": status.get("error"),
            }
    return arms


def record(arms: dict, note: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"utc": ops18.utc_now(), "beijing": time.strftime("%Y-%m-%d %H:%M:%S"),
             "note": note, "arms": arms}
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    summary = " | ".join(
        f"{arm}:{data['stage']}:{data['env_step']}/{data['budget_env_steps']}"
        for arm, data in sorted(arms.items()))
    print(f"[{entry['beijing']}] {note} {summary}", flush=True)


def main() -> None:
    last_progress = 0.0
    last_stages = {}
    started = time.time()
    for hour in range(MAX_HOURS):
        arms = snapshot()
        stages = {arm: data["stage"] for arm, data in arms.items()}
        errors = {arm: data["error"] for arm, data in arms.items() if data["error"]}
        changed = stages != last_stages
        if errors:
            record(arms, "错误需处理")
            return
        if changed and last_stages:
            record(arms, "阶段变化")
        now = time.time()
        if now - last_progress >= HOUR_SECONDS or last_progress == 0.0:
            record(arms, "每小时记录")
            last_progress = now
        last_stages = stages
        if not arms:
            record({}, "无在跑任务，监控结束")
            return
        time.sleep(POLL_SECONDS)
    record(snapshot(), "监控窗口结束")


if __name__ == "__main__":
    main()
