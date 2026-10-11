"""Explicit same-ID engineering repair lineage; never relabel origin evidence.

The workbench additionally enforces its exact task/role whitelist. This pure
evidence layer requires original configuration and resume artifact byte hashes.
It does not select a source, launch a job, or grant permission to repair one.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re

FORMAT = "s1_nd12_source_repair_v1"
REPAIR_ID = "nd12_native_deadline_teacher_reference_v1"
ORIGIN_COMMIT = "105d5f57ec241cce0c484b95f6341d6dc418b3fe"


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def value_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def execution_source(provenance, contract=None):
    result = {"git_commit": provenance.get("git_commit"),
              "source_sha256": provenance.get("source_sha256")}
    if contract:
        result.update(repair_id=contract["repair_id"], contract_sha256=contract["contract_sha256"],
                      origin_commit=contract["origin_commit"])
    return result


def validate_repair(contract, provenance, *, verify_artifacts=True):
    """Validate the limited repair contract against actual execution provenance."""
    if not isinstance(contract, dict) or contract.get("format") != FORMAT or contract.get("repair_id") != REPAIR_ID:
        raise ValueError("unrecognized source repair contract")
    if contract.get("origin_commit") != ORIGIN_COMMIT:
        raise ValueError("source repair origin is outside the frozen group")
    for key, size in (("execution_commit", 40), ("contract_sha256", 64), ("config_sha256", 64)):
        if not re.fullmatch(r"[0-9a-f]{%d}" % size, str(contract.get(key, ""))):
            raise ValueError("source repair hash is invalid: " + key)
    if contract["execution_commit"] == ORIGIN_COMMIT or provenance.get("git_commit") != contract["execution_commit"]:
        raise ValueError("source repair execution commit differs")
    if provenance.get("run_id") != contract.get("run_id") or provenance.get("experiment_id") != "S1-ND12":
        raise ValueError("source repair task identity differs")
    if not contract.get("reason") or not contract.get("comparison_scope"):
        raise ValueError("source repair requires an explicit scientific comparison boundary")
    config_path = Path(contract["config_path"])
    if file_sha(config_path) != contract["config_sha256"]:
        raise ValueError("source repair original configuration bytes differ")
    origin = contract.get("origin_provenance")
    if origin is not None:
        if origin.get("git_commit") != ORIGIN_COMMIT:
            raise ValueError("source repair provenance origin differs")
        for key in set(origin) | set(provenance):
            if key not in ("git_commit", "source_sha256") and origin.get(key) != provenance.get(key):
                raise ValueError("source repair changes frozen provenance: " + key)
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("source repair lacks original artifact bindings")
    if not any(item.get("kind") == "config" and Path(item.get("path", "")) == config_path
               and item.get("sha256") == contract["config_sha256"] for item in artifacts):
        raise ValueError("source repair lacks frozen configuration artifact")
    if verify_artifacts:
        for item in artifacts:
            if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) or file_sha(item["path"]) != item["sha256"]:
                raise ValueError("source repair original artifact bytes differ: " + str(item.get("path")))
    return deepcopy(contract)


def attach_execution(manifest, provenance, contract, path, *, resume):
    """Return current execution marker and append one immutable origin segment.

    Each changed manifest must be bound directly by the approved contract on
    its first repair. Later resumes verify the saved origin bytes and segment;
    mutable current manifests are not compared to their pre-repair byte hash.
    Existing shards/checkpoints remain the caller's mandatory validation duty.
    """
    original = manifest.get("provenance", {})
    if original == provenance and not contract:
        return execution_source(provenance)
    if not contract:
        raise ValueError("resume execution provenance differs without source repair")
    path = Path(path)
    job_root = Path(contract.get("job_root", Path(contract["config_path"]).parent)).resolve()
    if job_root != Path(contract["config_path"]).parent.resolve() or not path.resolve().is_relative_to(job_root):
        raise ValueError("source repair evidence path is outside the bound job root")
    segments = manifest.setdefault("execution_segments", [])
    marker = execution_source(provenance, contract)
    previous = next((s for s in segments if s.get("execution_source") == marker), None)
    # The workbench checks the complete artifact set before launch. Here each
    # writer checks its own bound manifest: a sibling writer may already have
    # legitimately advanced another current manifest in the same execution.
    validate_repair(contract, provenance, verify_artifacts=False)
    if original != provenance:
        if not resume or original != contract.get("origin_provenance"):
            raise ValueError("source repair cannot rewrite original evidence provenance")
        if previous:
            origin_path = path.parent / previous["origin_manifest_path"]
            if file_sha(origin_path) != previous["origin_manifest_sha256"]:
                raise ValueError("saved pre-repair manifest differs")
            baseline = json.loads(origin_path.read_text(encoding="utf-8"))
            for key in ("rows", "shards", "problem_receipts"):
                prefix = baseline.get(key)
                if isinstance(prefix, list) and manifest.get(key, [])[:len(prefix)] != prefix:
                    raise ValueError("source repair changed a committed origin prefix: " + key)
            for key in ("rows", "problems", "trials", "elapsed_seconds", "teacher_compute_seconds"):
                old_value = baseline.get(key, 0)
                if not isinstance(old_value, list) and manifest.get(key, 0) < old_value:
                    raise ValueError("source repair rolled back a budget counter: " + key)
        else:
            raw = path.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            if not any(Path(a["path"]) == path and a["sha256"] == sha for a in contract["artifacts"]):
                raise ValueError("resumed manifest is not directly bound to the repair contract")
            relative = "lineage/origin_" + sha + ".json"
            saved = path.parent / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            if saved.exists() and saved.read_bytes() != raw:
                raise ValueError("origin manifest snapshot collision")
            if not saved.exists():
                partial = saved.with_suffix(".partial")
                with partial.open("wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                partial.replace(saved)
            segments.append({"execution_source": marker, "origin_manifest_path": relative,
                             "origin_manifest_sha256": sha, "start_rows": (len(manifest["rows"]) if isinstance(manifest.get("rows"), list) else manifest.get("rows", 0)),
                             "start_problems": manifest.get("problems", 0), "start_trials": manifest.get("trials", 0),
                             "start_elapsed_seconds": manifest.get("elapsed_seconds", 0.),
                             "start_active_sha256": value_sha(manifest.get("active")),
                             "start_reservations_sha256": value_sha(manifest.get("reservations"))})
    elif not previous:
        if path.exists() or resume:
            raise ValueError("new repair task cannot impersonate an old origin")
        segments.append({"execution_source": marker, "comparison_scope": contract["comparison_scope"],
                         "created_after_repair": True,
                         "start_rows": 0, "start_problems": 0, "start_trials": 0})
    return marker
