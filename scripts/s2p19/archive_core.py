"""Gate and serially copy the 12 final and 3 training S2-P19 jobs.

Default/--check only reads live workbench GETs. --execute first rechecks every
job, then uses archive_completed one job at a time so the shared source index
cannot lose entries. Failure/ops evidence belongs to the separate supplemental
archive plan; this tool never marks that work complete.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from . import archive, ops


ROOT = ops.LOCAL
DEFAULT_PLAN = ROOT / "controller/migrations/four_lane_plan.json"
DEFAULT_STATE = ROOT / "controller/migrations/four_lane_state.json"
DEFAULT_HISTORICAL_PLAN = ROOT / "controller/plan.json"
DEFAULT_MANIFEST = ROOT / "run_manifest.json"
DEFAULT_LEDGER = ROOT / "remote_archive/core_archive_ledger.json"

FINAL_UNITS = {
    "V2_final_direct_20418": 2640,
    "V2_final_B_20418": 7920,
    "baseline_test_20418": 4400,
    "baseline_greedy_B_20418": 2640,
    "baseline_rule_B_20418": 2640,
    "baseline_random_B_20418": 7920,
    "C0_final_direct_20473": 2640,
    "C0_final_B_20473": 7920,
    "C1_final_direct_20424": 2640,
    "C1_final_B_20424": 7920,
    "I0_final_direct_20474": 2640,
    "I0_final_B_20474": 7920,
}
TRAINING = {"C0": (20418, 400000, 400000),
            "C1": (20424, 400000, 400000),
            "I0": (20418, 397112, 400000)}


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".tmp")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def planned_jobs(active: dict, state: dict, historical: dict, manifest: dict) -> list[dict]:
    if (active.get("schema") != "s2p19.four_lane.v1" or
            active.get("experiment_id") != "S2-P19" or
            historical.get("experiment_id") != "S2-P19" or
            manifest.get("experiment_id") != "S2-P19" or
            active.get("final_mothers") != 40 or active.get("final_scenes") != 880 or
            active.get("complete_B_budget") != 14960 or
            state.get("plan_sha256") is None):
        raise ValueError("S2-P19 core plan identity missing")
    old = {task["id"]: task for task in historical.get("tasks", [])}
    current = {task["id"]: task for task in active.get("tasks", [])}
    legacy = active.get("completed_legacy") or {}
    if set(legacy) | set(current) != set(FINAL_UNITS) or set(legacy) & set(current):
        raise ValueError("final 12 task set differs from frozen scope")
    jobs = []
    for task_id, units in FINAL_UNITS.items():
        if task_id in legacy:
            task = old.get(task_id)
            run_id = legacy[task_id]
            if not task or task.get("port") != 20418:
                raise ValueError(f"historical task identity missing: {task_id}")
        else:
            task = current[task_id]
            record = (state.get("tasks") or {}).get(task_id) or {}
            run_id = record.get("run_id")
            if not run_id or (task.get("adopted_run_id") and task["adopted_run_id"] != run_id):
                raise ValueError(f"active task run identity missing: {task_id}")
            if record.get("request_id") != task.get("request_id"):
                raise ValueError(f"active task request identity differs: {task_id}")
        jobs.append({"task_id": task_id, "port": task["port"], "run_id": run_id,
                     "request_id": task["request_id"],
                     "git_commit": task["expected_commit"],
                     "kind": "final", "env_step": units, "budget_env_steps": units})
    for arm, (port, step, budget) in TRAINING.items():
        record = (manifest.get("tasks") or {}).get(arm) or {}
        if (record.get("port") != port or record.get("env_step") != step or
                record.get("budget_env_steps") != budget):
            raise ValueError(f"training frozen endpoint differs: {arm}")
        jobs.append({"task_id": arm, "port": port, "run_id": record["run_id"],
                     "request_id": record["request_id"],
                     "git_commit": record["git_commit"],
                     "kind": "training", "env_step": step, "budget_env_steps": budget})
    if len({item["run_id"] for item in jobs}) != 15 or len({item["request_id"] for item in jobs}) != 15:
        raise ValueError("core runs or requests are not unique")
    if any(item["port"] not in ops.SSH_PORTS for item in jobs):
        raise ValueError("core plan contains unauthorized instance")
    return jobs


def live_gate(jobs: list[dict], api_get=ops.api) -> dict:
    """Read all fifteen real GETs; never infer completion from local state."""
    observed, pending = {}, []
    for item in jobs:
        task_id, port, run_id = item["task_id"], item["port"], item["run_id"]
        try:
            detail = api_get(port, "/api/runs/" + run_id)
            job, status = detail.get("job") or {}, detail.get("status") or {}
            reasons = []
            if detail.get("read_only") or job.get("run_id") != run_id or status.get("run_id") != run_id:
                reasons.append("workbench run identity")
            if (job.get("experiment_id") != "S2-P19" or
                    job.get("archive_root") != ops.instance(port)["archive"]):
                reasons.append("experiment/archive ownership")
            if (job.get("request_id") != item["request_id"] or
                    job.get("git_commit") != item["git_commit"]):
                reasons.append("frozen request/source")
            if status.get("stage") != "completed":
                reasons.append("not completed")
            for key in ("env_step", "budget_env_steps"):
                if status.get(key) != item[key]:
                    reasons.append(f"{key} budget")
            for key in ("completed_cases", "total_cases"):
                if key in status and status[key] != item["env_step"] and item["kind"] == "final":
                    reasons.append(f"{key} count")
            if item["task_id"] == "I0" and (status.get("completed_due_to") != "episode_cap" or
                                                status.get("episodes") != 12500):
                reasons.append("I0 episode-cap endpoint")
            observed[task_id] = {"port": port, "run_id": run_id,
                                 "stage": status.get("stage"),
                                 "env_step": status.get("env_step"),
                                 "budget_env_steps": status.get("budget_env_steps"),
                                 "reasons": reasons}
            if reasons:
                pending.append(task_id + ": " + ", ".join(reasons))
        except Exception as error:
            observed[task_id] = {"port": port, "run_id": run_id,
                                 "reasons": ["GET failed: " + str(error)]}
            pending.append(task_id + ": GET failed")
    return {"schema": "s2p19.core_archive_gate.v1",
            "checked_utc": datetime.now(timezone.utc).isoformat(),
            "ready": not pending and len(observed) == 15,
            "completed": sum(not row["reasons"] for row in observed.values()),
            "total": 15, "pending": pending, "jobs": observed}


def verify_prior(task: dict, record: dict, source_index: Path) -> None:
    receipt_path = Path(record["receipt"])
    receipt = read_json(receipt_path)
    local = Path(receipt["local_receipt_path"])
    package = read_json(local)
    if (receipt.get("task_id") != task["task_id"] or
            receipt.get("port") != task["port"] or
            receipt.get("run_id") != task["run_id"] or
            receipt.get("all_volume_and_member_shas_verified") is not True or
            archive.digest(local) != receipt.get("remote_receipt_sha256") or
            package.get("port") != task["port"] or
            package.get("run_id") != task["run_id"] or
            package.get("job_stage") != "completed"):
        raise ValueError("prior core archive receipt identity/integrity differs")
    volumes = package.get("volumes") or []
    if not volumes or any(archive.digest(local.parent / archive.safe_member(volume["name"])) !=
                          volume["sha256"] for volume in volumes):
        raise ValueError("prior core archive ZIP volume SHA differs")
    index = read_json(source_index)
    group = "training" if task["kind"] == "training" else "jobs"
    spec = (index.get(group) or {}).get(task["task_id"]) or {}
    if (spec.get("run_id") != task["run_id"] or spec.get("port") != task["port"] or
            spec.get("archive_inventory_sha256") != package.get("inventory_sha256")):
        raise ValueError("prior core archive source index differs")


def execute(jobs: list[dict], gate: dict, *, ledger_path: Path = DEFAULT_LEDGER,
            source_index: Path = archive.SOURCE_INDEX,
            destination: Path = archive.DEFAULT_DESTINATION,
            archive_job=archive.archive_completed,
            archive_options: dict | None = None) -> dict:
    if (not gate.get("ready") or gate.get("completed") != 15 or
            gate.get("pending") or len(jobs) != 15 or
            any((gate.get("jobs") or {}).get(item["task_id"], {}).get("reasons") != [] or
                (gate.get("jobs") or {}).get(item["task_id"], {}).get("run_id") != item["run_id"] or
                (gate.get("jobs") or {}).get(item["task_id"], {}).get("port") != item["port"]
                for item in jobs)):
        raise ValueError("all fifteen live workbench completion/budget gates are required")
    ledger_path = Path(ledger_path)
    ledger = read_json(ledger_path) if ledger_path.is_file() else {
        "schema": "s2p19.core_archive_ledger.v1", "completed": {}, "failures": []}
    if ledger.get("schema") != "s2p19.core_archive_ledger.v1":
        raise ValueError("unknown core archive ledger")
    completed = ledger.setdefault("completed", {})
    for task in jobs:
        task_id = task["task_id"]
        if task_id in completed:
            verify_prior(task, completed[task_id], source_index)
            continue
        try:
            result = archive_job(task["port"], task["run_id"], task_id,
                                 destination=destination, source_index=source_index,
                                 **(archive_options or {}))
            record = {"receipt": result["receipt"], "port": task["port"],
                      "run_id": task["run_id"]}
            verify_prior(task, record, source_index)
            completed[task_id] = record
            write_json(ledger_path, ledger)
        except Exception as error:
            ledger.setdefault("failures", []).append({
                "task_id": task_id, "port": task["port"], "run_id": task["run_id"],
                "at_utc": datetime.now(timezone.utc).isoformat(), "error": str(error)})
            write_json(ledger_path, ledger)
            raise
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--historical-plan", type=Path, default=DEFAULT_HISTORICAL_PLAN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--source-index", type=Path, default=archive.SOURCE_INDEX)
    parser.add_argument("--destination", type=Path, default=archive.DEFAULT_DESTINATION)
    parser.add_argument("--part-mib", type=int, default=256)
    parser.add_argument("--max-gib", type=int, default=20)
    parser.add_argument("--list", action="store_true", help="local plan only; no workbench GET")
    parser.add_argument("--execute", action="store_true", help="copy after all live GET gates pass")
    args = parser.parse_args()
    if not 1 <= args.part_mib <= 1024 or not 1 <= args.max_gib <= 64:
        parser.error("archive part/size bound outside archive.py limits")
    active = read_json(args.plan)
    state = read_json(args.state)
    if state.get("plan_sha256") != archive.digest(args.plan):
        raise ValueError("active state is bound to another plan SHA")
    jobs = planned_jobs(active, state, read_json(args.historical_plan),
                        read_json(args.manifest))
    if args.list:
        print(json.dumps({"jobs": jobs, "serial": True,
                          "supplemental_archive": "separate_pending"}, ensure_ascii=False))
        return
    gate = live_gate(jobs)
    print(json.dumps(gate, ensure_ascii=False))
    if args.execute:
        if not gate["ready"]:
            raise SystemExit("core archive gate pending; no copy attempted")
        ledger = execute(jobs, gate, ledger_path=args.ledger,
                         source_index=args.source_index, destination=args.destination,
                         archive_options={"part_mib": args.part_mib,
                                          "max_gib": args.max_gib})
        print(json.dumps({"core_archive_completed": len(ledger["completed"]),
                          "ledger": str(args.ledger)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
