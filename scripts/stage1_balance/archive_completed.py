"""Incrementally mirror completed S1-02 jobs, weights, reports and source SHA.

Run with --watch as a persistent local background process after formal launch.
Only completed workbench jobs and passed report directories are copied.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

from remote_sources import LOCAL, REMOTE, connect
import remote_stage1


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def remote_json(sftp, path: str, default=None):
    try:
        with sftp.open(path, "rb") as stream:
            return json.loads(stream.read())
    except FileNotFoundError:
        return default


def remote_inventory(client, relative_roots: list[str]) -> list[dict]:
    code = f'''
import hashlib,json,pathlib
root=pathlib.Path({REMOTE!r}).resolve();selected={relative_roots!r};entries=[]
for rel in selected:
 folder=(root/rel).resolve()
 if folder!=root and root not in folder.parents:raise RuntimeError('Archive root escaped S1-02')
 if not folder.is_dir():continue
 for path in sorted(folder.rglob('*')):
  if path.is_symlink():raise RuntimeError('Unexpected symlink in frozen completed artifact: '+str(path))
  if not path.is_file():continue
  h=hashlib.sha256()
  with path.open('rb') as stream:
   for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
  entries.append({{'relative':str(path.relative_to(root)),'sha256':h.hexdigest(),
                  'bytes':path.stat().st_size}})
print(json.dumps(entries))
'''
    encoded = base64.b64encode(code.encode()).decode()
    command = "python3 -B -c 'exec(__import__(\"base64\").b64decode(\"" + encoded + "\"))'"
    return json.loads(remote_stage1.execute(client, command))


def target(relative: str) -> Path:
    path = Path(relative)
    parts = path.parts
    if ".." in parts or path.is_absolute():
        raise ValueError("Archive path escaped S1-02")
    if parts[:3] == ("models", "runs", "workbench"):
        return Path("D:/paper/models/runs/first_stage_s1_02_20261004") / Path(*parts[3:])
    if parts[:4] == ("outputs", "experiments", "first_stage_s1_02_20261004", "analysis"):
        return LOCAL / "analysis" / Path(*parts[4:])
    if parts[:3] == ("outputs", "workbench", "jobs"):
        return LOCAL / "jobs" / Path(*parts[3:])
    if parts[:3] == ("outputs", "workbench", "source_snapshots"):
        return LOCAL / "source_snapshots" / Path(*parts[3:])
    return LOCAL / "remote_archive" / path


def copy_file(sftp, item: dict, inventory_by_relative: dict[str, dict]) -> dict:
    rel = item["relative"]
    source = REMOTE + "/" + rel.replace("\\", "/")
    destination = target(rel)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == item["bytes"] and sha(destination) == item["sha256"]:
            action = "verified_existing"
        else:
            prefix = "outputs/experiments/first_stage_s1_02_20261004/analysis/A_development/"
            name = rel[len(prefix):] if rel.startswith(prefix) else ""
            if name not in ("audit.json", "audit.stdout.log", "audit.stderr.log"):
                raise RuntimeError("Local archived artifact changed: " + str(destination))
            history_rel = prefix + "history/initial_failed_" + name
            historical = inventory_by_relative.get(history_rel)
            if historical is None or sha(destination) != historical["sha256"]:
                raise RuntimeError("Earlier failed audit is not represented by remote SHA history")
            historical_target = target(history_rel)
            historical_target.parent.mkdir(parents=True, exist_ok=True)
            if historical_target.exists():
                if sha(historical_target) != historical["sha256"]:
                    raise RuntimeError("Local failed-audit history changed")
            else:
                shutil.copyfile(destination, historical_target)
                if sha(historical_target) != historical["sha256"]:
                    raise RuntimeError("Failed-audit history copy differs")
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            sftp.get(source, str(temporary))
            if temporary.stat().st_size != item["bytes"] or sha(temporary) != item["sha256"]:
                raise RuntimeError("Corrected audit transfer SHA differs")
            temporary.replace(destination)
            action = "corrected_audit_with_prior_sha_history"
    else:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        sftp.get(source, str(temporary))
        if temporary.stat().st_size != item["bytes"] or sha(temporary) != item["sha256"]:
            raise RuntimeError("Completed remote artifact transfer SHA differs: " + source)
        temporary.replace(destination)
        action = "copied_verified"
    return {"relative": rel, "local_path": str(destination), "sha256": item["sha256"],
            "bytes": item["bytes"], "action": action}


def archive_once() -> dict:
    client = connect()
    try:
        with client.open_sftp() as sftp:
            route = remote_json(sftp, REMOTE + "/ops/current_controller_route.json")
            if route:
                if (route.get("experiment_id") != "S1-02" or
                        route.get("plan_relative") != "ops/controller_plan_recovery.json" or
                        route.get("state_relative") != "ops/controller_state_recovery.json"):
                    raise RuntimeError("Recovery route escaped its isolated S1-02 contract")
                plan_path = REMOTE + "/ops/controller_plan_recovery.json"
                state_path = REMOTE + "/ops/controller_state_recovery.json"
            else:
                plan_path = REMOTE + "/ops/controller_plan.json"
                state_path = REMOTE + "/ops/controller_state.json"
            state = remote_json(sftp, state_path, {})
            plan = remote_json(sftp, plan_path, {})
            if not state or not plan:
                return {"status": "awaiting_controller", "checked_utc": now(), "files": []}
            if route:
                with sftp.open(plan_path, "rb") as stream:
                    actual_plan_sha = hashlib.sha256(stream.read()).hexdigest()
                if (actual_plan_sha != route.get("plan_sha256") or
                        plan.get("source_commit") != route.get("source_commit") or
                        state.get("plan_sha256") != actual_plan_sha):
                    raise RuntimeError("Recovery route plan SHA/source/state binding differs")
            jobs = state.get("jobs", {})
            completed = {}
            roots = []
            for label, job in sorted(jobs.items()):
                run_id = job["run_id"]
                remote_dir = f"outputs/workbench/jobs/{run_id}"
                progress = remote_json(sftp, REMOTE + "/" + remote_dir + "/status.json", {})
                if progress.get("stage") == "completed":
                    completed[label] = run_id
                    roots.append(remote_dir)
                    if job.get("kind") == "train":
                        roots.append(f"models/runs/workbench/{run_id}")
            if completed:
                roots.append("outputs/workbench/source_snapshots/" + plan["source_commit"])
                reuse = (plan.get("recovery_context", {}).get("allowed_completed_prepare", {})
                         .get("validation"))
                if reuse and completed.get("prepare_validation") == reuse.get("run_id"):
                    roots.append("outputs/workbench/source_snapshots/" + reuse["source_commit"])
            analysis = "outputs/experiments/first_stage_s1_02_20261004/analysis"
            for phase in ("A_development", "A_report", "B_menu_selection", "B_report"):
                gate = remote_json(sftp, REMOTE + "/" + analysis + "/" + phase + "/gate.json")
                if phase == "A_development":
                    if all("A_dev_" + method in completed for method in ("A0", "A1", "A2", "A3")):
                        roots.append(analysis + "/" + phase)
                elif gate and gate.get("status") == "passed":
                    roots.append(analysis + "/" + phase)
            inventory = remote_inventory(client, roots)
            inventory_by_relative = {item["relative"]: item for item in inventory}
            files = [copy_file(sftp, item, inventory_by_relative) for item in inventory]
            receipt = {"experiment_id": "S1-02", "status": "completed_artifacts_sha_mirrored",
                       "checked_utc": now(), "controller_phase": state.get("phase"),
                       "controller_route": route,
                       "source_commit": plan["source_commit"], "completed_jobs": completed,
                       "pending_jobs": sorted(set(jobs) - set(completed)),
                       "files": files, "file_count": len(files),
                       "copied_bytes": sum(item["bytes"] for item in files
                                           if item["action"] == "copied_verified")}
            save(LOCAL / "preflight/archive_ledger.json", receipt)
            snapshot = LOCAL / "preflight/controller_state.latest.json"
            save(snapshot, state)
            return {key: receipt[key] for key in ("status", "checked_utc", "controller_phase",
                                                    "completed_jobs", "pending_jobs",
                                                    "file_count", "copied_bytes")}
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=180)
    args = parser.parse_args()
    if args.interval_seconds < 30 or args.interval_seconds > 3600:
        raise ValueError("Watcher interval must be 30–3600 seconds")
    status_file = LOCAL / "preflight/archive_watch_status.json"
    failures = 0
    while True:
        try:
            result = archive_once()
            failures = 0
            save(status_file, {"status": "running" if args.watch else "single_check_done",
                               "checked_utc": now(), "result": result})
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if not args.watch:
                return
            if result.get("controller_phase") == "needs_attention":
                if result.get("controller_route") is None:
                    save(status_file, {"status": "awaiting_recovery", "checked_utc": now(),
                                       "reason": "original_zero_scene_failure_preserved", "result": result})
                else:
                    save(status_file, {"status": "needs_attention", "checked_utc": now(),
                                       "reason": "recovery_controller_needs_attention", "result": result})
                    return
            if (result.get("controller_phase") == "awaiting_final_delivery"
                    and not result.get("pending_jobs")):
                save(status_file, {"status": "mirror_complete_pending_final_delivery",
                                   "checked_utc": now(), "result": result})
                return
        except BaseException as error:
            failures += 1
            save(status_file, {"status": "retrying" if failures < 3 else "needs_attention",
                               "checked_utc": now(), "consecutive_failures": failures,
                               "error": type(error).__name__ + ": " + str(error)})
            if failures >= 3 or not args.watch:
                raise
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
