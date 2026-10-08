"""Review and serially archive 16 completed S2-P19 ancillary workbench jobs.

Default/--list/--check are local-only. --derive-contract freezes independent
local evidence into an explicit contract. Only --execute can issue workbench
GETs, stat-only inventory probes, and copy-only completed-job archives. Failed
jobs and ops evidence belong to a separate supplemental archive plan.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from . import archive, archive_core, archive_waiter, ops


ROOT = ops.LOCAL
RAW_PLAN = ROOT / "analysis/supplemental_completed_jobs_plan.json"
LIVE_IDENTITY = ROOT / "preflight/heartbeat_20261008T141435Z/supplemental_live_identity.json"
HISTORICAL_PLAN = ROOT / "controller/plan.json"
HISTORICAL_STATE = ROOT / "controller/state.json"
VALIDATION_JOBS = ROOT / "controller/validation_jobs.json"
MANIFEST = ROOT / "run_manifest.json"
C0_AUDIT = ROOT / "pilot_audit/C0_audit.json"
C1_AUDIT = ROOT / "pilot_audit/C1_audit.json"
DEFAULT_CONTRACT = Path(__file__).with_name("supplemental_completed_contract.json")
DEFAULT_LEDGER = ROOT / "remote_archive/supplemental_completed_ledger.json"
DEFAULT_LOCK = ROOT / "remote_archive/supplemental_completed.lock"
DEFAULT_INVENTORY = ROOT / "remote_archive/supplemental_completed_inventory"

SMOKE_RUN = "evaluate_sac_20261007_173305_3a1ccaff"
SMOKE_KEY = "validation_greedy_direct"
SOURCE_13EA = "13ea014f1c1709c4a82059fc89ba3092e75010e1"
SOURCE_29DEE = "29dee3248893d42d7569aa57e004cf3db0d0ee65"
FROZEN_UNITS = {
    "prepare_prepare_20261007_152010_778973d6": 144,
    "prepare_prepare_20261007_152018_209a01c9": 576,
    "prepare_prepare_20261007_154940_dd12808c": 40,
    "prepare_prepare_20261007_171407_3c757c98": 880,
    "evaluate_sac_20261007_153239_e442d472": 216,
    "refine_sac_20261007_154539_38e92936": 216,
    "refine_sac_20261007_154541_e3178010": 216,
    "refine_sac_20261007_154543_8adc79ad": 216,
    SMOKE_RUN: 72,
    "refine_sac_20261007_173438_2d6aca5b": 216,
    "refine_sac_20261007_173502_95ffdf86": 216,
    "refine_sac_20261007_173525_dbf2fd92": 216,
    "refine_sac_20261007_191743_e37373af": 216,
    "refine_sac_20261008_004138_1239ba3f": 216,
    "refine_sac_20261008_003602_2b0fbb0a": 216,
    "refine_sac_20261008_010934_f19c8d82": 216,
}
PILOT_REQUESTS = {
    "refine_sac_20261007_173502_95ffdf86": "s2p19-c0-pilot-B",
    "refine_sac_20261007_173525_dbf2fd92": "s2p19-c1-pilot-B",
}
SECONDARY_COMMITS = {
    "refine_sac_20261007_173502_95ffdf86": SOURCE_13EA,
    "refine_sac_20261007_173525_dbf2fd92": SOURCE_13EA,
    "refine_sac_20261007_191743_e37373af": SOURCE_13EA,
}


def evidence_paths() -> dict[str, Path]:
    return {"raw_plan": RAW_PLAN, "live_identity": LIVE_IDENTITY,
            "historical_plan": HISTORICAL_PLAN,
            "historical_state": HISTORICAL_STATE,
            "validation_jobs": VALIDATION_JOBS, "manifest": MANIFEST,
            "C0_audit": C0_AUDIT, "C1_audit": C1_AUDIT}


def source_fingerprints(paths: dict[str, Path]) -> dict[str, str]:
    result = {name: archive.digest(path) for name, path in paths.items()
              if name != "manifest"}
    pilot = (archive_core.read_json(paths["manifest"]).get("tasks") or {}).get("I0_pilot_B")
    if not isinstance(pilot, dict):
        raise ValueError("I0 pilot manifest evidence missing")
    payload = json.dumps(pilot, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    result["manifest_I0_pilot_B_record"] = hashlib.sha256(payload).hexdigest()
    return result


def _kind(run_id: str) -> str:
    if run_id.startswith("prepare_prepare_"):
        return "prepare"
    if run_id.startswith("evaluate_sac_"):
        return "evaluate"
    if run_id.startswith("refine_sac_"):
        return "refine"
    raise ValueError("supplemental run kind outside frozen set")


def derive_contract(paths: dict[str, Path] | None = None) -> dict:
    """Freeze independent plan/audit identities; live snapshot is corroboration."""
    paths = paths or evidence_paths()
    raw = archive_core.read_json(paths["raw_plan"])
    live = json.loads(Path(paths["live_identity"]).read_text(encoding="utf-8"))
    historical = archive_core.read_json(paths["historical_plan"])
    old_state = archive_core.read_json(paths["historical_state"])
    validation = archive_core.read_json(paths["validation_jobs"])
    manifest = archive_core.read_json(paths["manifest"])
    audits = {arm: archive_core.read_json(paths[arm + "_audit"])
              for arm in ("C0", "C1")}
    if (raw.get("schema") != "s2p19_supplemental_completed_jobs_plan_v1" or
            raw.get("experiment_id") != "S2-P19" or
            historical.get("service_commit") != SOURCE_13EA or
            len(raw.get("supplemental_jobs") or []) != 16 or
            raw.get("failed_mother_source_separate", {}).get("not_completed_job") is not True or
            not isinstance(live, list) or len(live) != 16):
        raise ValueError("raw/live supplemental scope differs")
    old_tasks = {task["id"]: task for task in historical["tasks"]}
    validation_by_run = {record.get("run_id"): record
                         for record in validation.get("jobs", {}).values()}
    live_by_run = {item.get("run_id"): item for item in live}
    raw_jobs = raw["supplemental_jobs"]
    if (set(live_by_run) != set(FROZEN_UNITS) or
            {item["run_id"] for item in raw_jobs} != set(FROZEN_UNITS)):
        raise ValueError("sixteen frozen supplemental run IDs differ")
    jobs = []
    requests = set()
    for item in raw_jobs:
        run_id, port = item["run_id"], item["owner_port"]
        observed = live_by_run[run_id]
        job, status = observed["job"], observed["status"]
        if (observed.get("port") != port or observed.get("read_only") or
                job.get("run_id") != run_id or job.get("experiment_id") != "S2-P19" or
                job.get("archive_root") != ops.instance(port)["archive"] or
                job.get("kind") != _kind(run_id) or status.get("run_id") != run_id or
                status.get("stage") != "completed" or
                status.get("env_step") != FROZEN_UNITS[run_id] or
                status.get("budget_env_steps") != FROZEN_UNITS[run_id]):
            raise ValueError("live snapshot contradicts frozen supplemental scope: " + run_id)
        old_id = (item.get("task_key") or "").split(" / ", 1)[0]
        old = old_tasks.get(old_id)
        validation_ref = validation_by_run.get(run_id)
        if old:
            state_ref = (old_state.get("tasks") or {}).get(old_id) or {}
            if (state_ref.get("run_id") != run_id or old.get("port") != port or
                    old.get("request_id") != item.get("request_id")):
                raise ValueError("historical task/request evidence differs: " + run_id)
            request_id, commit, basis = old["request_id"], old["expected_commit"], "historical_plan+state"
        elif validation_ref:
            detail_job = (validation_ref.get("detail") or {}).get("job") or {}
            if (validation_ref.get("port") != port or
                    validation_ref.get("request_id") != item.get("request_id") or
                    detail_job.get("run_id") != run_id):
                raise ValueError("validation job evidence differs: " + run_id)
            request_id, commit, basis = (validation_ref["request_id"],
                                         detail_job["git_commit"], "validation_jobs")
            if commit != (SOURCE_29DEE if run_id.endswith("f19c8d82") else SOURCE_13EA):
                raise ValueError("validation source commit differs")
        elif run_id in PILOT_REQUESTS:
            arm = "C0" if "173502" in run_id else "C1"
            if audits[arm].get("source_runs", {}).get(arm + "_B") != run_id:
                raise ValueError("pilot audit lacks run identity: " + run_id)
            request_id, commit, basis = (PILOT_REQUESTS[run_id],
                                         SECONDARY_COMMITS[run_id],
                                         "pilot_audit+frozen_pilot_request")
        else:
            pilot = (manifest.get("tasks") or {}).get("I0_pilot_B") or {}
            if pilot.get("run_id") != run_id or pilot.get("port") != port:
                raise ValueError("I0 pilot manifest identity differs")
            request_id, commit, basis = (pilot["request_id"],
                                         SECONDARY_COMMITS[run_id], "run_manifest+frozen_source")
        if (not request_id or job.get("request_id") != request_id or
                job.get("git_commit") != commit or request_id in requests):
            raise ValueError("request/source identity not independently established: " + run_id)
        requests.add(request_id)
        task_id = SMOKE_KEY if run_id == SMOKE_RUN else "supp_" + run_id
        jobs.append({"task_id": task_id, "run_id": run_id, "port": port,
                     "request_id": request_id, "git_commit": commit,
                     "kind": _kind(run_id), "env_step": FROZEN_UNITS[run_id],
                     "budget_env_steps": FROZEN_UNITS[run_id],
                     "reuse_existing": run_id == SMOKE_RUN,
                     "identity_basis": basis, "purpose": item["purpose"]})
    return {"schema": "s2p19.supplemental_completed_contract.v1",
            "experiment_id": "S2-P19", "core_dependency": "complete_15_and_waiter_inactive",
            "failed_and_ops_evidence": "separate_pending",
            "source_sha256": source_fingerprints(paths),
            "jobs": jobs}


def load_contract(path: Path = DEFAULT_CONTRACT,
                  sources: dict[str, Path] | None = None) -> dict:
    contract = archive_core.read_json(path)
    if (contract.get("schema") != "s2p19.supplemental_completed_contract.v1" or
            contract.get("experiment_id") != "S2-P19" or
            len(contract.get("jobs") or []) != 16 or
            {item["run_id"] for item in contract["jobs"]} != set(FROZEN_UNITS)):
        raise ValueError("supplemental execution contract is incomplete")
    paths = sources or evidence_paths()
    if contract.get("source_sha256") != source_fingerprints(paths):
        raise ValueError("frozen supplemental evidence SHA differs")
    rebuilt = derive_contract(paths)
    if contract != rebuilt:
        raise ValueError("supplemental contract differs from independent source evidence")
    return contract


def core_local_gate(*, status_path=archive_waiter.DEFAULT_STATUS,
                    ledger_path=archive_core.DEFAULT_LEDGER,
                    core_lock=archive_waiter.DEFAULT_LOCK) -> dict:
    status = archive_core.read_json(status_path) if Path(status_path).is_file() else {}
    ledger = archive_core.read_json(ledger_path) if Path(ledger_path).is_file() else {}
    return {"phase": status.get("phase"), "core_count": status.get("core_count"),
            "ledger_count": len(ledger.get("completed") or {}),
            "ready_snapshot": (status.get("phase") == "core_complete" and
                               status.get("core_count") == 15 and
                               len(ledger.get("completed") or {}) == 15),
            "core_lock": str(core_lock),
            "checked_utc": datetime.now(timezone.utc).isoformat(),
            "lock_not_probed": True}


def live_gate(jobs: list[dict], api_get=ops.api) -> dict:
    observed, pending = {}, []
    for item in jobs:
        task_id, port, run_id = item["task_id"], item["port"], item["run_id"]
        try:
            detail = api_get(port, "/api/runs/" + run_id)
            observed_utc = datetime.now(timezone.utc).isoformat()
            job, status = detail.get("job") or {}, detail.get("status") or {}
            reasons = []
            if (detail.get("read_only") or job.get("run_id") != run_id or
                    status.get("run_id") != run_id or job.get("experiment_id") != "S2-P19" or
                    job.get("archive_root") != ops.instance(port)["archive"]):
                reasons.append("run/owner identity")
            if (job.get("request_id") != item["request_id"] or
                    job.get("git_commit") != item["git_commit"] or
                    job.get("kind") != item["kind"]):
                reasons.append("frozen request/source/kind")
            if status.get("stage") != "completed":
                reasons.append("not completed")
            if (status.get("env_step") != item["env_step"] or
                    status.get("budget_env_steps") != item["budget_env_steps"]):
                reasons.append("saved workload budget")
            for key in ("completed_cases", "total_cases"):
                if status.get(key) is not None and status[key] != item["env_step"]:
                    reasons.append(key + " count")
            observed[task_id] = {"run_id": run_id, "port": port,
                                 "observed_utc": observed_utc, "reasons": reasons,
                                 "stage": status.get("stage"),
                                 "env_step": status.get("env_step")}
            if reasons:
                pending.append(task_id + ": " + ", ".join(reasons))
        except Exception as error:
            observed[task_id] = {"run_id": run_id, "port": port,
                                 "observed_utc": datetime.now(timezone.utc).isoformat(),
                                 "reasons": ["GET failed: " + str(error)]}
            pending.append(task_id + ": GET failed")
    return {"ready": not pending and len(observed) == 16,
            "checked_utc": datetime.now(timezone.utc).isoformat(),
            "jobs": observed, "pending": pending,
            "completed": sum(not row["reasons"] for row in observed.values()),
            "total": 16}


def verify_package(item: dict, receipt_path: Path,
                   *, source_index: Path = archive.SOURCE_INDEX) -> dict:
    """Validate an existing local copy/reference without reusing remote work."""
    download = archive_core.read_json(receipt_path)
    package_path = Path(download["local_receipt_path"])
    package = archive_core.read_json(package_path)
    if (download.get("task_id") != item["task_id"] or
            download.get("run_id") != item["run_id"] or
            download.get("port") != item["port"] or
            download.get("all_volume_and_member_shas_verified") is not True or
            archive.digest(package_path) != download.get("remote_receipt_sha256") or
            package.get("schema") != "s2p19.archive.v1" or
            package.get("run_id") != item["run_id"] or
            package.get("port") != item["port"] or
            package.get("job_stage") != "completed" or
            package.get("job_git_commit") != item["git_commit"] or
            package.get("job_kind") != item["kind"]):
        raise ValueError("supplemental receipt identity/integrity differs")
    volumes = package.get("volumes") or []
    if not volumes or any(archive.digest(package_path.parent / archive.safe_member(v["name"])) !=
                          v["sha256"] for v in volumes):
        raise ValueError("supplemental ZIP volume SHA differs")
    index = archive_core.read_json(source_index)
    spec = (index.get("jobs") or {}).get(item["task_id"])
    if spec:
        if (spec.get("run_id") != item["run_id"] or
                spec.get("archive_inventory_sha256") != package.get("inventory_sha256") or
                (spec.get("port") is not None and spec["port"] != item["port"])):
            raise ValueError("supplemental source-index identity differs")
    elif item["kind"] != "prepare":
        raise ValueError("supplemental evaluation missing source-index entry")
    return {"receipt": str(receipt_path), "port": item["port"],
            "run_id": item["run_id"], "inventory_sha256": package["inventory_sha256"]}


def verify_smoke(item: dict, *, source_index: Path = archive.SOURCE_INDEX) -> dict:
    if not item.get("reuse_existing") or item["run_id"] != SMOKE_RUN:
        raise ValueError("only frozen validation greedy direct may reuse smoke")
    index = archive_core.read_json(source_index)
    spec = (index.get("jobs") or {}).get(SMOKE_KEY) or {}
    receipt = Path(spec.get("archive_receipt_path") or "")
    if (spec.get("run_id") != SMOKE_RUN or not receipt.is_file() or
            spec.get("archive_inventory_sha256") is None):
        raise ValueError("verified smoke source-index entry missing")
    record = verify_package(item, receipt.with_name("download_receipt.json"),
                            source_index=source_index)
    if record["inventory_sha256"] != spec["archive_inventory_sha256"]:
        raise ValueError("smoke archive inventory differs")
    return {**record, "mode": "existing_verified_no_recopy"}


def inventory_gate(jobs: list[dict], *, source_index: Path = archive.SOURCE_INDEX,
                   destination: Path = archive.DEFAULT_DESTINATION,
                   probe=archive_waiter.probe_inventory) -> dict:
    """Inventory only 15 new copies; reference the verified existing smoke."""
    smoke = [item for item in jobs if item.get("reuse_existing")]
    if len(jobs) != 16 or len(smoke) != 1:
        raise ValueError("supplemental inventory scope must be 15 new + one smoke")
    saved = verify_smoke(smoke[0], source_index=source_index)
    download = archive_core.read_json(saved["receipt"])
    package = archive_core.read_json(download["local_receipt_path"])
    fresh = [item for item in jobs if not item.get("reuse_existing")]
    result = archive_waiter.inventory_gate(fresh, probe=probe,
        destination=destination, expected_count=15)
    result["jobs"][smoke[0]["task_id"]] = {
        "run_id": smoke[0]["run_id"], "port": smoke[0]["port"],
        "member_count": package["member_count"] if "member_count" in package else None,
        "total_input_bytes": package.get("total_input_bytes"),
        "inventory_sha256": package["inventory_sha256"],
        "source": "existing_verified_local_ZIP_no_new_copy",
        "observed_utc": datetime.now(timezone.utc).isoformat(), "reasons": []}
    result["ready"] = result["ready"] and len(result["jobs"]) == 16
    result["existing_references"] = 1
    return result


def execute(contract: dict, gate: dict, inventory: dict, *,
            source_index: Path = archive.SOURCE_INDEX,
            destination: Path = archive.DEFAULT_DESTINATION,
            ledger_path: Path = DEFAULT_LEDGER,
            archive_job=archive.archive_completed) -> dict:
    jobs = contract["jobs"]
    if (not gate.get("ready") or gate.get("completed") != 16 or gate.get("pending") or
            not inventory.get("ready") or inventory.get("pending") or
            len(inventory.get("jobs") or {}) != 16 or
            any((gate.get("jobs") or {}).get(item["task_id"], {}).get("run_id") != item["run_id"] or
                (gate.get("jobs") or {}).get(item["task_id"], {}).get("port") != item["port"] or
                (gate.get("jobs") or {}).get(item["task_id"], {}).get("reasons") != [] or
                (inventory.get("jobs") or {}).get(item["task_id"], {}).get("run_id") != item["run_id"] or
                (inventory.get("jobs") or {}).get(item["task_id"], {}).get("port") != item["port"] or
                (inventory.get("jobs") or {}).get(item["task_id"], {}).get("reasons") != []
                for item in jobs)):
        raise ValueError("sixteen live completion and capacity gates required")
    contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    ledger_path = Path(ledger_path)
    ledger = archive_core.read_json(ledger_path) if ledger_path.is_file() else {
        "schema": "s2p19.supplemental_completed_ledger.v1",
        "contract_sha256": contract_sha, "completed": {}, "failures": []}
    if (ledger.get("schema") != "s2p19.supplemental_completed_ledger.v1" or
            ledger.get("contract_sha256") != contract_sha):
        raise ValueError("unknown supplemental completed-job ledger")
    completed = ledger.setdefault("completed", {})
    for item in jobs:
        task_id = item["task_id"]
        try:
            if task_id in completed:
                saved = completed[task_id]
                if item.get("reuse_existing"):
                    verified = verify_smoke(item, source_index=source_index)
                else:
                    verified = verify_package(item, Path(saved["receipt"]),
                                              source_index=source_index)
                if verified["run_id"] != saved["run_id"] or verified["port"] != saved["port"]:
                    raise ValueError("saved supplemental ledger differs")
                continue
            if item.get("reuse_existing"):
                record = verify_smoke(item, source_index=source_index)
            else:
                result = archive_job(item["port"], item["run_id"], task_id,
                    destination=destination, source_index=source_index,
                    part_mib=1024, max_gib=256, compression_level=6,
                    remote_timeout_seconds=43200)
                record = verify_package(item, Path(result["receipt"]),
                                        source_index=source_index)
                record["mode"] = "new_copy_verified"
            completed[task_id] = record
            archive_core.write_json(ledger_path, ledger)
        except Exception as error:
            ledger.setdefault("failures", []).append({
                "task_id": task_id, "run_id": item["run_id"],
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "error": str(error)})
            archive_core.write_json(ledger_path, ledger)
            raise
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--source-index", type=Path, default=archive.SOURCE_INDEX)
    parser.add_argument("--destination", type=Path, default=archive.DEFAULT_DESTINATION)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--core-lock", type=Path, default=archive_waiter.DEFAULT_LOCK)
    parser.add_argument("--inventory-dir", type=Path, default=DEFAULT_INVENTORY)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--derive-contract", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.derive_contract:
        contract = derive_contract()
        archive_core.write_json(args.contract, contract)
        print(json.dumps({"contract": str(args.contract), "jobs": len(contract["jobs"]),
                          "sha256": archive.digest(args.contract)}, ensure_ascii=False))
        return
    contract = load_contract(args.contract)
    if args.list:
        print(json.dumps({"jobs": contract["jobs"], "remote_called": False},
                         ensure_ascii=False))
        return
    core = core_local_gate(core_lock=args.core_lock)
    if not args.execute:
        print(json.dumps({"core": core, "contract": str(args.contract),
                          "jobs": len(contract["jobs"]), "remote_called": False},
                         ensure_ascii=False))
        return
    with archive_waiter.exclusive_lock(args.core_lock), \
            archive_waiter.exclusive_lock(args.lock):
        core = core_local_gate(core_lock=args.core_lock)
        if not core["ready_snapshot"]:
            raise SystemExit("core archive incomplete; supplemental copy refused")
        active = archive_core.read_json(archive_core.DEFAULT_PLAN)
        state = archive_core.read_json(archive_core.DEFAULT_STATE)
        if state.get("plan_sha256") != archive.digest(archive_core.DEFAULT_PLAN):
            raise ValueError("core plan/state SHA differs")
        core_jobs = archive_core.planned_jobs(active, state,
            archive_core.read_json(archive_core.DEFAULT_HISTORICAL_PLAN),
            archive_core.read_json(archive_core.DEFAULT_MANIFEST))
        core_ledger = archive_core.read_json(archive_core.DEFAULT_LEDGER)
        for item in core_jobs:
            archive_core.verify_prior(item, core_ledger["completed"][item["task_id"]],
                                      args.source_index)
        gate = live_gate(contract["jobs"])
        print(json.dumps({"live_gate": gate}, ensure_ascii=False))
        if not gate["ready"]:
            raise SystemExit("supplemental live identity/completion gate pending")
        inventory = inventory_gate(contract["jobs"], source_index=args.source_index,
                                   destination=args.destination)
        archive_core.write_json(args.inventory_dir / "summary.json",
            {key: value for key, value in inventory.items() if key != "jobs"})
        for task_id, item in inventory["jobs"].items():
            archive_core.write_json(args.inventory_dir / (task_id + ".json"), item)
        if not inventory["ready"]:
            raise SystemExit("supplemental metadata/capacity gate pending")
        ledger = execute(contract, gate, inventory, source_index=args.source_index,
                         destination=args.destination, ledger_path=args.ledger)
        print(json.dumps({"supplemental_completed_jobs_archived": len(ledger["completed"]),
                          "failed_and_ops_evidence": "separate_pending"},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
