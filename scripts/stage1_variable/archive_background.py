"""One bounded, single-owner S1-V10 completed-job archive sweep.

This process only reads completed workbench jobs and writes isolated archive
ZIPs/receipts.  It never starts, stops or resumes scientific workers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import psutil

from scripts.stage1_variable import archive_jobs, ops


REPO = Path(__file__).resolve().parents[2]
ROOT = ops.LOCAL / "job_archives" / "background"
LOCK = ROOT / "single_owner.lock"
STATUS = ROOT / "status.json"
RECEIPT = ROOT / "batch_receipt.json"
WATCHED_FILES = ("scripts/stage1_variable/archive_jobs.py",
                 "scripts/stage1_variable/archive_inventory.py",
                 "scripts/stage1_variable/archive_background.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".part")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def _source_gate(expected_commit):
    if not isinstance(expected_commit, str) or len(expected_commit) != 40:
        raise ValueError("exact reviewed S1-V10 ops commit required")
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO), *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    actual = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", *WATCHED_FILES)
    if actual != expected_commit or dirty:
        raise RuntimeError("S1-V10 archive sweep source commit/files differ")
    return actual


def run(expected_commit, limit):
    if not 1 <= limit <= 32:
        raise ValueError("S1-V10 archive batch limit must be 1..32")
    _source_gate(expected_commit)
    with archive_jobs._lock(LOCK):
        process = psutil.Process(os.getpid())
        owner = {"pid": process.pid, "birth": process.create_time(),
                 "command": process.cmdline(), "ops_commit": expected_commit,
                 "started_utc": now(), "limit": limit}
        save(ROOT / "launch.json", owner)
        status = {**owner, "stage": "archiving_completed_jobs", "completed": {},
                  "last_observed_utc": now()}
        save(STATUS, status)
        def record(key, result, completed, planned):
            status["completed"][key] = {"run_id": result["run_id"],
                                        "port": result["port"],
                                        "compressed_bytes": result["total_compressed_bytes"],
                                        "receipt": result["local_receipt_path"]}
            status.update(last_observed_utc=now(), completed_count=completed,
                          planned_count=planned)
            save(STATUS, status)
        try:
            result = archive_jobs.archive_all_completed(limit=limit, on_task=record)
        except Exception as error:
            status.update(stage="failed", last_observed_utc=now(),
                          error=str(error), traceback=traceback.format_exc())
            save(STATUS, status)
            raise
        receipt = {**owner, "completed_utc": now(), "stage": "completed",
                   "purpose": "archive_only_completed_exited_s1v10_jobs_no_scientific_execution",
                   "result": result,
                   "formal_final_archive_closed": False,
                   "final_audit_and_report_required": True}
        save(RECEIPT, receipt)
        status.update(stage="completed", last_observed_utc=now(),
                      batch_receipt=str(RECEIPT))
        save(STATUS, status)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops-commit", required=True)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(run(args.ops_commit, args.limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
