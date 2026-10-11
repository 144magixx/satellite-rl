"""Narrow client and offline proof boundary for the signed ND12 source repair."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

from satellite_rl.workbench.nondrl_source_repair import (
    ALLOWED_RUNS, DIFF_ALLOWLIST, FORMAT, FUTURE_ROLES, ORIGIN_COMMIT,
    REPAIR_ID, validate_job_lineage,
)


def contract_bytes(payload, expected):
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("source repair contract raw SHA differs")
    value = json.loads(payload)
    if (value.get("format") != FORMAT or value.get("repair_id") != REPAIR_ID or
            value.get("origin_commit") != ORIGIN_COMMIT or
            not re.fullmatch(r"[0-9a-f]{40}", value.get("execution_commit", "")) or
            value["execution_commit"] == ORIGIN_COMMIT):
        raise ValueError("source repair identity differs")
    ids, roles = value.get("allowed_run_ids", []), value.get("allowed_future_roles", [])
    if (not isinstance(ids, list) or len(ids) != len(set(ids)) or set(ids) - set(ALLOWED_RUNS) or
            set(value.get("runs", {})) != set(ids) or
            any((r.get("route"), r.get("kind"), r.get("preparation_mode")) not in FUTURE_ROLES for r in roles)):
        raise ValueError("source repair scope exceeds authorization")
    files = value.get("diff_files", [])
    if (not files or len({f["path"] for f in files}) != len(files) or
            any(f["path"] not in DIFF_ALLOWLIST for f in files) or not value.get("validation_receipts")):
        raise ValueError("source repair diff or validation receipt missing")
    return value


def role_allowed(contract, body):
    route = body["config"]["coverage"]["nondrl"]["route"]
    role = (route, body["kind"], body.get("preparation_mode") if body["kind"] == "prepare" else None)
    return role in {(r["route"], r["kind"], r.get("preparation_mode"))
                    for r in contract["allowed_future_roles"]}


def receipt_member(item):
    return "source_lineage/validation_receipts/" + item["sha256"] + "/" + Path(item["path"]).name


def audit_lineage(job, evidence):
    """Verify archived original bytes and every actual execution source member."""
    if job.get("git_commit") == ORIGIN_COMMIT and not job.get("source_repair_contract"):
        return {"origin_commit": ORIGIN_COMMIT, "execution_commit": ORIGIN_COMMIT, "repaired": False}
    validate_job_lineage(job, evidence.location, verify_source_snapshot=False, read_artifact=evidence.read)
    item = job["source_repair_contract"]
    contract = contract_bytes(evidence.read("source_lineage/contract.json"), item["contract_sha256"])
    source_bytes = evidence.read("source/source_manifest.json")
    if hashlib.sha256(source_bytes).hexdigest() != job["source_history"][-1]["source_manifest_sha256"]:
        raise ValueError("archived execution source manifest SHA differs")
    source = json.loads(source_bytes)
    if source.get("git_commit") != job["git_commit"] or not source.get("files"):
        raise ValueError("archived execution source identity differs")
    for name, sha in source["files"].items():
        if hashlib.sha256(evidence.read("source/" + name)).hexdigest() != sha:
            raise ValueError("archived execution source member differs: " + name)
    differences = {f["path"]: f for f in contract["diff_files"]}
    for name, pin in differences.items():
        if source["files"].get(name) != pin["execution_sha256"]:
            raise ValueError("execution source differs from signed file pins")
    original_bytes=evidence.read("source_lineage/origin/source_manifest.json")
    original = json.loads(original_bytes)
    if item.get("comparison_scope") == "new_execution_task":
        pin=evidence.json("source_lineage/origin_snapshot_pin.json")
        expected=contract.get("runs",{}).get(pin.get("run_id"),{}).get("origin_source_manifest_sha256")
        if expected!=pin.get("sha256") or hashlib.sha256(original_bytes).hexdigest()!=expected:
            raise ValueError("future task original snapshot is not pinned by the signed contract")
    if original.get("git_commit") != ORIGIN_COMMIT:
        raise ValueError("original snapshot source identity differs")
    changed = {name for name in set(original["files"]) | set(source["files"])
               if original["files"].get(name) != source["files"].get(name)}
    if changed != set(differences) or any(original["files"].get(n) != p.get("origin_sha256") for n,p in differences.items()):
        raise ValueError("exact original/execution file diff differs from signed contract")
    if item.get("comparison_scope") != "new_execution_task":
        for artifact in item["artifacts"]:
            relative=artifact["relative_path"]
            baseline=evidence.read("source_lineage/origin/artifacts/"+relative)
            if relative.endswith(".json.gz") and evidence.read(relative)!=baseline:
                raise ValueError("immutable origin scientific shard changed after repair")
            if relative.endswith("manifest.json"):
                before=json.loads(baseline)
                if before.get("format") not in ("s1_nondrl_teacher_v1","s1nd12_scene_shards_v1"):
                    continue
                current=evidence.json(relative)
                if current.get("provenance")!=before.get("provenance"):
                    raise ValueError("repair rewrote original manifest provenance")
                for key in ("rows","shards","problem_receipts"):
                    prefix=before.get(key)
                    if isinstance(prefix,list) and current.get(key,[])[:len(prefix)]!=prefix:
                        raise ValueError("repair rewrote committed origin manifest prefix")
                    if isinstance(prefix,list):
                        for new_item in current.get(key,[])[len(prefix):]:
                            if verify_marker(job,new_item.get("execution_source"))!=job["git_commit"]:
                                raise ValueError("new evidence was relabelled as an origin shard")
                for key in ("rows","problems","trials","elapsed_seconds","teacher_compute_seconds"):
                    value=before.get(key,0)
                    if not isinstance(value,list) and current.get(key,0)<value:
                        raise ValueError("repair rolled back a scientific budget counter")
                if current!=before and not current.get("execution_segments"):
                    raise ValueError("changed evidence lacks explicit execution segment")
    for receipt in contract["validation_receipts"]:
        payload = evidence.read(receipt_member(receipt))
        if hashlib.sha256(payload).hexdigest() != receipt["sha256"]:
            raise ValueError("archived repair validation receipt SHA differs")
        result = json.loads(payload)
        if result.get("passed") is not True and result.get("status") != "passed":
            raise ValueError("archived repair validation did not pass")
    history_status = evidence.json("source_lineage/origin/status.json") if item["comparison_scope"]!="new_execution_task" else None
    return {"origin_commit": ORIGIN_COMMIT, "execution_commit": job["git_commit"],
            "repaired": True, "contract_sha256": item["contract_sha256"],
            "comparison_scope": item["comparison_scope"], "source_history": job["source_history"],
            "original_terminal_status": {k:history_status.get(k) for k in ("stage","scientific_stop","stop_reason","error","ended_at")}
                if history_status is not None else None}


def manifest_origin(job, manifest):
    """A repaired mutable manifest retains its original header and explicit segments."""
    origin = job.get("source_repair_contract", {}).get("origin_provenance")
    provenance = manifest.get("provenance", {})
    if provenance.get("run_id") != job["run_id"]:
        raise ValueError("manifest run provenance differs")
    if origin is None:
        if provenance.get("git_commit") != job["git_commit"]:
            raise ValueError("manifest execution provenance differs")
    elif provenance != origin:
        raise ValueError("retained manifest origin provenance differs")
    for segment in manifest.get("execution_segments", []):
        verify_marker(job, segment.get("execution_source", {}))
    if job.get("source_repair_contract",{}).get("comparison_scope")=="new_execution_task":
        if not manifest.get("execution_segments"):
            raise ValueError("new repaired manifest lacks explicit execution segment")
        for key in ("rows","shards","problem_receipts"):
            entries=manifest.get(key)
            if isinstance(entries,list):
                for entry in entries:
                    verify_marker(job,entry.get("execution_source"))
    return provenance["git_commit"]


def verify_marker(job, marker):
    if not marker:
        if job.get("source_repair_contract",{}).get("comparison_scope")=="new_execution_task":
            raise ValueError("new repaired task row lacks its actual execution marker")
        return job.get("source_repair_contract", {}).get("origin_commit", job["git_commit"])
    item = job.get("source_repair_contract", {})
    origin = item.get("origin_provenance") or {}
    if (marker.get("git_commit") == origin.get("git_commit") == ORIGIN_COMMIT and
            marker.get("source_sha256") == origin.get("source_sha256") and
            set(marker) <= {"git_commit", "source_sha256"}):
        return ORIGIN_COMMIT
    if (not item or marker.get("git_commit") != job["git_commit"] or
            any(marker.get(k) != item.get(k) for k in ("repair_id", "contract_sha256", "origin_commit")) or
            marker.get("source_sha256") != job["source_history"][-1]["source_manifest_sha256"]):
        raise ValueError("row/sample execution marker differs from signed job source")
    return marker["git_commit"]
