"""Single persistent S1-D12A coordinator from source prepare through test.

It owns no scientific worker. Every long task is submitted to the matching
frozen workbench. A nonblocking OS file lock prevents duplicate launchers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import sys
import time

import psutil

from scripts.stage1_drl_r04 import ops
from scripts.stage1_drl_r04 import controller, data_controller


LOCK = ops.LOCAL / "controller" / "launcher.lock"
RECEIPT = ops.LOCAL / "controller" / "launcher.json"
MANIFEST = ops.LOCAL / "run_manifest.json"
PLAN = ops.LOCAL / "run_plan.json"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def lock_once():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("S1-D12A persistent launcher already owns the OS lock") from exc
        try:
            yield stream
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def _task_status(port, run_id):
    detail = ops.api(port, f"/api/runs/{run_id}")
    job, status = detail["job"], detail["status"]
    if job.get("run_id") != run_id or job.get("experiment_id") != "S1-D12A":
        raise ValueError("workbench task identity differs during launcher observation")
    return {"run_id": run_id, "port": port, "request_id": job.get("request_id"),
            "kind": job.get("kind"), "variant": job.get("variant"),
            "git_commit": job.get("git_commit"), "stage": status.get("stage"),
            "env_step": status.get("env_step"), "budget_env_steps": status.get("budget_env_steps"),
            "update_step": status.get("update_step"), "pid": status.get("pid"),
            "heartbeat_at": status.get("heartbeat_at"),
            "stop_reason": status.get("stop_reason"),
            "scientific_stop": status.get("scientific_stop", False)}


def sync_manifest(commit, phase, prepare, state=None, attention=None):
    plan_sha = sha(PLAN)
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if document.get("experiment_id") != "S1-D12A" or Path(document.get("run_plan", "")).resolve() != PLAN.resolve():
        raise ValueError("S1-D12A immutable run-plan reference differs")
    if document.get("run_plan_sha256") not in (None, plan_sha):
        raise ValueError("S1-D12A run plan changed after launcher start")
    tasks = dict(document.get("tasks") or {})
    for day, item in (prepare.get("tasks") or {}).items():
        tasks["prepare_" + day] = _task_status(item["port"], item["run_id"])
    for key, item in ((state or {}).get("tasks") or {}).items():
        tasks[key] = _task_status(item["port"], item["run_id"])
    document.update(status="running" if phase not in (
        "attention_required", "scientific_stop_unreachable",
        "scientific_stop_unreachable_validation", "test_complete_report_archive_pending") else
        "test_completed_report_pending" if phase == "test_complete_report_archive_pending" else
        "attention_required" if phase == "attention_required" else "scientifically_stopped",
        phase=phase, scientific_commit=commit, run_plan_sha256=plan_sha,
        tasks=tasks, updated_utc=now(),
        audits={**document.get("audits", {}),
                "reachability": (state or {}).get("reachability"),
                "validation_reachability": (state or {}).get("validation_reachability"),
                "pilot": (state or {}).get("pilot_audits", {})},
        need_attention=attention or document.get("need_attention", {}),
        controller_state=str(controller.STATE),
        launcher_receipt=str(RECEIPT))
    data_controller.save(MANIFEST, document)
    return document


def _prepare_snapshot(preparation):
    rows = {}
    for day, item in preparation["tasks"].items():
        rows[day] = _task_status(item["port"], item["run_id"])
        if rows[day]["git_commit"] != preparation["scientific_commit"]:
            raise ValueError("prepared data task uses another source commit")
    return rows


def run(commit, poll_seconds):
    from contextlib import contextmanager
    guard = contextmanager(lock_once)
    with guard():
        identity = psutil.Process(os.getpid())
        receipt = {"experiment_id": "S1-D12A", "status": "starting",
                   "source_commit": commit, "pid": os.getpid(),
                   "pid_create_time_unix": identity.create_time(),
                   "command": [sys.executable, *sys.argv],
                   "lock_path": str(LOCK), "started_utc": now(),
                   "run_plan_sha256": sha(PLAN), "phase": "service_check"}
        data_controller.save(RECEIPT, receipt)
        preparation = {"tasks": {}}
        state = None
        try:
            for port in ops.SSH_PORTS:
                data_controller.assert_service(port, commit)
            receipt.update(status="running", phase="official_source_metadata")
            data_controller.save(RECEIPT, receipt)
            data_controller.upload_sources()
            preparation = data_controller.submit_preparations(commit)
            while True:
                prepared = _prepare_snapshot(preparation)
                failures = {day: row for day, row in prepared.items()
                            if row["stage"] in ("failed", "interrupted", "stopped")}
                if failures:
                    receipt.update(status="attention_required", phase="data_preparation_failed",
                                   failure={"tasks": failures}, observed_utc=now())
                    data_controller.save(RECEIPT, receipt)
                    sync_manifest(commit, "attention_required", preparation,
                                  attention={"data_preparation": failures})
                    return 2
                if all(row["stage"] == "completed" for row in prepared.values()):
                    break
                receipt.update(phase="data_preparation", observed_utc=now(),
                               preparation_status={day: row["stage"] for day, row in prepared.items()})
                data_controller.save(RECEIPT, receipt)
                sync_manifest(commit, "data_preparation", preparation)
                time.sleep(poll_seconds)
            receipt.update(phase="register_datasets", observed_utc=now())
            data_controller.save(RECEIPT, receipt)
            data_controller.register_datasets(commit)
            state = json.loads(controller.STATE.read_text(encoding="utf-8")) if controller.STATE.is_file() else {
                "experiment_id": "S1-D12A", "scientific_commit": commit,
                "phase": "awaiting_data_registry", "tasks": {}}
            if state.get("scientific_commit") != commit:
                raise ValueError("controller state belongs to another frozen commit")
            while True:
                state = controller.tick(state)
                state["observed_utc"] = now()
                data_controller.save(controller.STATE, state)
                sync_manifest(commit, state["phase"], preparation, state,
                              attention=state.get("problem"))
                receipt.update(phase=state["phase"], observed_utc=now(),
                               controller_task_count=len(state["tasks"]))
                data_controller.save(RECEIPT, receipt)
                if state["phase"] in ("attention_required", "scientific_stop_unreachable",
                                      "scientific_stop_unreachable_validation",
                                      "test_complete_report_archive_pending"):
                    receipt.update(status="completed" if state["phase"] ==
                                   "test_complete_report_archive_pending" else "attention_required",
                                   ended_utc=now())
                    data_controller.save(RECEIPT, receipt)
                    return 0 if receipt["status"] == "completed" else 2
                time.sleep(poll_seconds)
        except BaseException as exc:
            receipt.update(status="failed", phase=receipt.get("phase"),
                           error=type(exc).__name__ + ": " + str(exc), ended_utc=now())
            data_controller.save(RECEIPT, receipt)
            try:
                sync_manifest(commit, "attention_required", preparation, state,
                              attention={"launcher": receipt["error"]})
            except Exception as secondary:
                receipt["manifest_sync_error"] = type(secondary).__name__ + ": " + str(secondary)
                data_controller.save(RECEIPT, receipt)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if len(args.commit) != 40 or not 15 <= args.poll_seconds <= 600:
        parser.error("frozen 40-character commit and 15–600 second interval required")
    raise SystemExit(run(args.commit, args.poll_seconds))


if __name__ == "__main__":
    main()
