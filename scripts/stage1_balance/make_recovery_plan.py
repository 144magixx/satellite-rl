"""Freeze the sole allowed 20-scene A2 reuse after a zero-scene A failure."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

from remote_sources import LOCAL


REPO = Path(__file__).resolve().parents[2]
OLD_COMMIT = "77d18aac49e36e0b4490d82689230d9aa9bbee52"
OLD_PLAN_SHA = "5895e5d2712eb8db84942bbdfd1719af2d75ccfd32e390ca72282f15d46ef8f4"
PREPARE = "evaluate_ppo_20261004_042121_08092f17"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, default=LOCAL / "plan_recovery.json")
    args = parser.parse_args()
    commit = args.commit.lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or commit == OLD_COMMIT:
        raise ValueError("A new full scientific bug-fix commit is required")
    actual = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                     text=True).strip()
    if actual != commit:
        raise RuntimeError("Current repo differs from recovery science commit")
    old_plan_path = LOCAL / "plan_frozen.json"
    old_state_path = LOCAL / "preflight/a_dev_failure_20261004/controller_state.json"
    failure_path = LOCAL / "preflight/a_dev_failure_receipt.json"
    if sha(old_plan_path) != OLD_PLAN_SHA:
        raise RuntimeError("Original frozen plan changed")
    old_plan, old_state, failure = read(old_plan_path), read(old_state_path), read(failure_path)
    if old_state["phase"] != "needs_attention" or old_state["plan_sha256"] != OLD_PLAN_SHA:
        raise RuntimeError("Original failed controller state differs")
    old_jobs = {label: info["run_id"] for label, info in failure["failed_jobs"].items()}
    if set(old_jobs) != {"A_dev_A0", "A_dev_A1", "A_dev_A2", "A_dev_A3"}:
        raise RuntimeError("Recovery requires exactly four zero-scene A jobs")
    job_dir = LOCAL / "jobs" / PREPARE
    job, config, status, artifacts = (read(job_dir / name) for name in
                                      ("job.json", "config.json", "status.json", "artifacts.json"))
    starts = job_dir / "starts.jsonl"
    snapshot = LOCAL / "source_snapshots" / OLD_COMMIT / "source_manifest.json"
    if (job["run_id"] != PREPARE or job["git_commit"] != OLD_COMMIT or
            status["stage"] != "completed" or status["evaluated_scenes"] != 20 or
            config["coverage"]["balance"]["route"] != "prepare" or
            artifacts["balance_starts_sha256"] != sha(starts) or
            read(snapshot)["git_commit"] != OLD_COMMIT):
        raise RuntimeError("Sole allowed prior prepare evidence differs")
    dataset = config["datasets"]["validation"]
    if (old_state["datasets"]["validation"] != dataset["id"] or
            failure["successful_validation_prepare"]["starts_sha256"] !=
            artifacts["balance_starts_sha256"]):
        raise RuntimeError("Original validation dataset/start identity differs")
    allow = {"run_id": PREPARE, "request_id": job["request_id"],
             "source_commit": OLD_COMMIT,
             "source_manifest_sha256": sha(snapshot),
             "job_sha256": sha(job_dir / "job.json"),
             "config_sha256": sha(job_dir / "config.json"),
             "status_sha256": sha(job_dir / "status.json"),
             "artifacts_sha256": sha(job_dir / "artifacts.json"),
             "dataset_id": dataset["id"], "dataset_sha256": dataset["sha256"],
             "starts_sha256": artifacts["balance_starts_sha256"]}
    new_plan = deepcopy(old_plan)
    new_plan["source_commit"] = commit
    new_plan["created_utc"] = datetime.now(timezone.utc).isoformat()
    new_plan["recovery_context"] = {
        "version": "zero_scene_startup_recovery_v1",
        "previous_plan_sha256": OLD_PLAN_SHA,
        "previous_state_sha256": sha(old_state_path),
        "previous_failed_jobs": old_jobs,
        "allowed_completed_prepare": {"validation": allow},
        "scientific_scope": "same four A methods and budgets; prior four IDs failed at zero scenes",
    }
    output = args.output.resolve()
    if output.exists():
        new_plan["created_utc"] = read(output).get("created_utc")
    encoded = (json.dumps(new_plan, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if output.exists():
        if output.read_bytes() != encoded:
            raise RuntimeError("Frozen recovery plan differs")
    else:
        output.write_bytes(encoded)
    print(json.dumps({"status": "recovery_plan_frozen", "path": str(output),
                      "sha256": hashlib.sha256(encoded).hexdigest(),
                      "new_commit": commit, "reused_prepare": PREPARE,
                      "replaced_failed_jobs": old_jobs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
