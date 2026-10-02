"""Persistent local watcher for two authorized stage 3 workbench tasks.

Observes API, PID birth and frozen source; archives only after both jobs finish.
It does not submit, stop, resume or create evaluation tasks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time


from remote_observe_stage3 import observe


HERE = Path(__file__).resolve().parent
ROOT = Path("D:/paper/outputs/experiments/stage3_power_20261003")
PREFLIGHT = ROOT / "preflight"
COMPLETION = ROOT / "completion"
PORTS = (20418, 20424)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(8):
        try:
            pending.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.05 * 2 ** attempt, 1.0))


def append_history(path: Path, value: dict) -> None:
    for attempt in range(8):
        try:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.05 * 2 ** attempt, 1.0))


def observation(commit: str) -> dict:
    result = {"observed_utc": now(), "instances": {}}
    for port in PORTS:
        try:
            result["instances"][str(port)] = observe(port, commit)
        except Exception as exc:
            result["instances"][str(port)] = {"error": type(exc).__name__ + ": " + str(exc)[:400]}
    return result


def complete(result: dict) -> bool:
    return all(
        item.get("api_consistent") is True and item.get("source_frozen") is True
        and item.get("status", {}).get("stage") == "completed"
        and item.get("status", {}).get("env_step") == 900
        for item in result["instances"].values()
    )


def finalize(commit: str, observed: dict) -> dict:
    commands = (
        [sys.executable, str(HERE / "remote_completion_archive.py"), "--commit", commit],
        [sys.executable, str(HERE / "audit_stage3_beams.py")],
        [sys.executable, str(HERE / "analyze_stage3_results.py")],
    )
    outputs = []
    for command in commands:
        process = subprocess.run(command, capture_output=True, text=True, timeout=7200)
        outputs.append({"command": command, "exit_code": process.returncode,
                        "stdout_tail": process.stdout[-2500:], "stderr_tail": process.stderr[-2500:]})
        if process.returncode:
            raise RuntimeError("Completion step failed: " + json.dumps(outputs[-1], ensure_ascii=False))
    manifest_paths = [ROOT / "remote_archive" / str(port) / "archive_manifest.json" for port in PORTS]
    summary = COMPLETION / "paired_summary.json"
    paired_csv = COMPLETION / "paired_scene_differences.csv"
    audit_path = COMPLETION / "beam_audit.json"
    if not all(path.is_file() for path in (*manifest_paths, audit_path, summary, paired_csv)):
        raise RuntimeError("Completion outputs missing after successful scripts")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit.get("status") != "independent_all_1800_cases_180000_beams_verified"
            or audit.get("total_cases") != 1800 or audit.get("total_beams") != 180000):
        raise RuntimeError("Independent all-beam audit completion gate failed")
    outcome = {"status": "both_completed_archived_and_paired",
               "completed_utc": now(), "commit": commit,
               "observation": observed,
               "archive_receipt": str(ROOT / "remote_archive" / "completion_archive_receipt.json"),
               "source_bundle": str(PREFLIGHT / f"stage3_{commit}.bundle"),
               "data_receipt": str(PREFLIGHT / "remote_jan25_data_receipt.json"),
               "files": {str(path): sha(path) for path in (*manifest_paths, audit_path, summary, paired_csv)},
               "independent_beam_audit": str(audit_path),
               "steps": outputs,
               "training_interactions": 0,
               "formal_evaluation_units": 1800}
    save(COMPLETION / "receipt.json", outcome)
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--interval-seconds", type=int, default=60)
    args = parser.parse_args()
    commit = args.commit.lower()
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Full frozen commit required")
    if not 30 <= args.interval_seconds <= 3600:
        raise ValueError("Watcher interval must be 30–3600 seconds")
    PREFLIGHT.mkdir(parents=True, exist_ok=True)
    state_path = PREFLIGHT / "watch_state.json"
    history_path = PREFLIGHT / "watch_history.jsonl"
    attempts = 0
    while True:
        result = observation(commit)
        append_history(history_path, result)
        stages = {port: item.get("status", {}).get("stage")
                  for port, item in result["instances"].items()}
        state = {"status": "watching", "updated_utc": now(), "watcher_pid": __import__("os").getpid(),
                 "commit": commit, "stages": stages, "last_observation": result,
                 "completion_attempts": attempts}
        if any(stage in ("failed", "stopped", "cancelled", "interrupted") for stage in stages.values()):
            state["status"] = "needs_attention"
        if complete(result):
            attempts += 1
            state["status"] = "archiving"
            state["completion_attempts"] = attempts
            save(state_path, state)
            try:
                receipt = finalize(commit, result)
                state.update(status="completed", completed_utc=receipt["completed_utc"],
                             completion_receipt=str(COMPLETION / "receipt.json"))
                save(state_path, state)
                return
            except Exception as exc:
                state.update(status="archive_retry_pending", error=type(exc).__name__ + ": " + str(exc)[-1800:])
                save(state_path, state)
                time.sleep(min(300 * attempts, 1800))
                continue
        save(state_path, state)
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
