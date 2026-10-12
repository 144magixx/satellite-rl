"""Narrow client and offline proof boundary for the signed ND12 source repair."""
from __future__ import annotations
import hashlib
import gzip
import json
from pathlib import Path, PurePosixPath
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


def compact_receipt(value):
    """At most20 mother summaries and60 scalar rows; canonical raw stays external."""
    def scalars(record):
        return {k:v for k,v in record.items() if v is None or isinstance(v,(str,int,float,bool))}
    result={k:v for k,v in value.items() if k not in ("statistics","rows","result","physical","trace")}
    statistics=value.get("statistics",{})
    reduced=scalars(statistics)
    for key,entry in statistics.items():
        if isinstance(entry,list) and all(v is None or isinstance(v,(str,int,float,bool)) for v in entry):
            if len(entry)<=60:reduced[key]=entry
    reduced["mother_metrics"]=[scalars(r) for r in statistics.get("mother_metrics",[])[:20]]
    if statistics.get("problem_metrics"):
        rows=[]
        for row in statistics["problem_metrics"][:60]:
            small=scalars(row);large=row.get("result",{})
            if large:
                for name,key in (("initial_cv","initial_metrics_all"),("final_cv","metrics_all")):
                    small[name]=large.get(key,{}).get("load_cv_all")
                for name,key in (("initial_carry","initial_physical"),("final_carry","physical")):
                    small[name]=large.get(key,{}).get("aircraft_service",{}).get("demand_weighted_carried_fraction")
                small["reference_geometry_feasible"]=all(large.get("metrics_all",{}).get(k) is True for k in
                    ("geometry_feasible","unique_assignment","demand_conserved"))
            if isinstance(row.get("search_scope"),dict):small["search_scope"]=scalars(row["search_scope"])
            rows.append(small)
        reduced["problem_metrics"]=rows
        reduced["legacy_search_problems_without_explicit_scope"]=sum(not r.get("search_scope") for r in rows)
        reduced["explicit_complete_search_problems"]=sum(r.get("search_scope",{}).get("searched") is True and
            r.get("search_scope",{}).get("complete_search") is True for r in rows)
        if "complete_search_problems" in reduced:
            reduced["helper_reported_complete_search_problems"]=reduced.pop("complete_search_problems")
    result["statistics"]=reduced
    return result


def compact_attention(value):
    result={k:v for k,v in value.items() if k not in ("receipt","status","previous_attention")}
    if isinstance(value.get("receipt"),dict):result["receipt"]=compact_receipt(value["receipt"])
    if isinstance(value.get("status"),dict):
        result["status"]={k:v for k,v in value["status"].items() if k in (
            "run_id","stage","scientific_stop","stop_reason","error","pid","process_start_time","git_commit","heartbeat_at")}
    if isinstance(value.get("previous_attention"),dict):result["previous_attention"]=compact_attention(value["previous_attention"])
    return result


def compact_state_history(path, state):
    """Preserve original bytes once, then reduce historical payloads on restart."""
    if not Path(path).is_file():return
    changed=False
    for key in ("attention","route_attention"):
        if key=="attention" and isinstance(state.get(key),dict):
            trimmed=compact_attention(state[key]);changed|=trimmed!=state[key];state[key]=trimmed
        elif key=="route_attention":
            for route,value in state.get(key,{}).items():
                trimmed=compact_attention(value);changed|=trimmed!=value;state[key][route]=trimmed
    history=state.get("source_repair_attention_history",[])
    for index,value in enumerate(history):
        trimmed=compact_attention(value);changed|=trimmed!=value;history[index]=trimmed
    for route,gate in state.get("routes",{}).items():
        for key in ("teacher_reachability",):
            if isinstance(gate.get(key),dict):
                trimmed=compact_receipt(gate[key]);changed|=trimmed!=gate[key];gate[key]=trimmed
    if changed:
        path=Path(path);signature=hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda:stream.read(8<<20),b""):signature.update(block)
        sha=signature.hexdigest();target=None
        reused=preserved_payload_ref(path,sha,path.stat().st_size)
        if reused:target=Path(reused["path"])
        if target is None:target=path.parent/"history"/("payload_before_compaction_"+sha+".json.gz")
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            partial=target.with_suffix(".partial")
            with path.open("rb") as source, gzip.open(partial,"wb",compresslevel=6) as output:
                for block in iter(lambda:source.read(8<<20),b""):output.write(block)
            partial.replace(target)
        state.setdefault("payload_compaction_history",[]).append({"path":str(target),"original_sha256":sha,
            "original_bytes":path.stat().st_size,"gzip_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),
            "reason":"full original state preserved; compact canonical-evidence summaries for bounded coordinator"})


def preserved_payload_ref(path,expected,bytes_count):
    path=Path(path)
    preserved=path.parent.parent/"preflight/repair_epoch/postops_censored/preservation.json"
    roundtrip=preserved.with_name("history_roundtrip.json")
    if not preserved.is_file() or not roundtrip.is_file():return None
    proof=json.loads(roundtrip.read_bytes())
    for old in json.loads(preserved.read_bytes()):
        if old.get("source_sha256")!=expected or Path(old.get("source",".")).resolve()!=path.resolve():continue
        checks=[r for r in proof.get("checks",[]) if r.get("gzip")==old["gzip"] and r.get("passed") is True and
            r.get("uncompressed_sha256")==expected and r.get("bytes")==bytes_count]
        candidate=Path(old["gzip"])
        if (proof.get("passed") is True and checks and candidate.is_file() and
                hashlib.sha256(candidate.read_bytes()).hexdigest()==old["gzip_sha256"]):
            return {"path":str(candidate),"original_sha256":expected,"original_bytes":bytes_count,"gzip_sha256":old["gzip_sha256"]}
    raise ValueError("existing payload preservation is not an exact verified copy")


def validate_disposition(scope, route, receipt):
    """Explicit operator closure of the two exhausted, unknown teacher probes."""
    item=scope.get("entries",{}).get(route,{})
    source=receipt.get("task_source",{});statistics=receipt.get("statistics",{})
    ids={"02":"prepare_nondrl_20261012_045228_1d87b8e0","03":"prepare_nondrl_20261012_045232_8dca8863"}
    if (scope.get("format")!="s1_nd12_bounded_dispositions_v1" or scope.get("origin_commit")!=ORIGIN_COMMIT or
            scope.get("scientific_stop") is not False or set(scope.get("entries",{}))-set(ids) or
            route not in ids or item.get("run_id")!=ids[route] or source.get("run_id")!=ids[route] or
            item.get("status")!="budget_limited_inconclusive" or not item.get("reason") or
            source.get("contract_sha256")!=scope.get("contract_sha256") or
            source.get("wrapper_sha256")!=item.get("teacher_wrapper_sha256")):
        raise ValueError("bounded disposition task/source scope differs")
    manifest_sha=source.get("evidence_manifest_sha256") or next((s["sha256"] for s in receipt.get("sources",[])
        if s.get("path","").endswith("/teacher/manifest.json")),None)
    if manifest_sha!=item.get("teacher_evidence_manifest_sha256"):
        raise ValueError("bounded disposition canonical teacher SHA differs")
    if (receipt.get("status")!="attention_required" or receipt.get("passed") is not False or
            receipt.get("scientific_stop") is not False or
            receipt.get("reason")!="offline_teacher_search_evidence_gap: bounded unknown/reference-only problems remain" or
            statistics.get("actual_complete_problems")!=60 or statistics.get("legal_reference_mothers")!=list(range(20)) or
            len(statistics.get("improved_mothers",[]))>=6 or statistics.get("unknown_search_problems",0)<=0 or
            statistics.get("recorded_teacher_seconds",0)<{"02":600.,"03":240.}[route] or
            not 0<=statistics.get("true_trials",-1)<={"02":38400,"03":7680}[route]):
        raise ValueError("bounded disposition has not met its evidence-insufficient terminal scope")
    return item


def role_allowed(contract, body):
    route = body["config"]["coverage"]["nondrl"]["route"]
    role = (route, body["kind"], body.get("preparation_mode") if body["kind"] == "prepare" else None)
    return role in {(r["route"], r["kind"], r.get("preparation_mode"))
                    for r in contract["allowed_future_roles"]}


def receipt_member(item):
    return "source_lineage/validation_receipts/" + item["sha256"] + "/" + Path(item["path"]).name


def deferred_origin_fold(job, evidence, manifest_name, before, field, entry, pins):
    """Prove a pre-repair worker artifact was only folded into its parent later.

    Parallel teachers reserve workers before dispatch. A failed coordinator can
    leave immutable child shards/receipts committed but not folded. An entry
    added to the repaired parent is therefore not necessarily newly executed.
    This exception requires BOTH signed original artifact bytes and their exact
    reference in a signed original child manifest, under the original pending
    reservation. It never permits unmarked new rows or merely matching layouts.
    """
    if (before.get("format") != "s1_nondrl_teacher_v1" or field not in ("shards", "problem_receipts")
            or job.get("preparation_mode") != "teacher" or job.get("variant") != "01"):
        raise ValueError("new evidence was relabelled as an origin shard")
    path = entry.get("path", "")
    if (not isinstance(path, str) or path.startswith("/") or "\\" in path or
            any(part in ("", ".", "..") for part in path.split("/"))):
        raise ValueError("deferred origin fold has an unsafe artifact path")
    parts = PurePosixPath(path).parts
    if len(parts) < 3 or parts[0] != "workers" or not re.fullmatch(r"problem_[0-9]{6}", parts[1]):
        raise ValueError("deferred origin fold is not a reserved teacher worker")
    worker_index = int(parts[1][8:])
    if (str(worker_index) not in before.get("reservations", {}) or
            worker_index < before.get("problems", 0)):
        raise ValueError("deferred origin fold has no original pending reservation")
    base = PurePosixPath(manifest_name).parent
    target = str(base / path)
    child_name = str(base / parts[0] / parts[1] / "manifest.json")
    if target not in pins or child_name not in pins or entry.get("sha256") != pins[target]["sha256"]:
        raise ValueError("deferred origin fold lacks signed original artifact/manifest pins")
    baseline = evidence.read("source_lineage/origin/artifacts/" + target)
    current = evidence.read(target)
    if (hashlib.sha256(baseline).hexdigest() != pins[target]["sha256"] or current != baseline):
        raise ValueError("deferred origin scientific artifact bytes changed")
    child_bytes = evidence.read("source_lineage/origin/artifacts/" + child_name)
    if hashlib.sha256(child_bytes).hexdigest() != pins[child_name]["sha256"]:
        raise ValueError("deferred origin worker manifest SHA differs")
    child = json.loads(child_bytes)
    if (child.get("format") != "s1_nondrl_teacher_v1" or child.get("route") != job["variant"] or
            child.get("provenance") != job["source_repair_contract"]["origin_provenance"]):
        raise ValueError("deferred origin worker provenance differs")
    relative_child = str(PurePosixPath(*parts[2:]))
    matches = [r for r in child.get(field, []) if r.get("path") == relative_child]
    if len(matches) != 1 or {**matches[0], "path": path} != entry:
        raise ValueError("deferred origin entry differs from its original worker reference")
    if verify_marker(job, matches[0].get("execution_source")) != ORIGIN_COMMIT:
        raise ValueError("deferred artifact did not have original execution provenance")
    return {"manifest": manifest_name, "field": field, "path": target,
            "sha256": pins[target]["sha256"], "original_worker_manifest_sha256": pins[child_name]["sha256"],
            "original_worker_index": worker_index, "execution_commit": ORIGIN_COMMIT,
            "scope": "pre_repair_immutable_worker_artifact_folded_after_repair"}


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
    deferred_folds = []
    if item.get("comparison_scope") != "new_execution_task":
        origin_pins = {a["relative_path"]: a for a in item["artifacts"]}
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
                                deferred_folds.append(deferred_origin_fold(job, evidence, relative, before,
                                                                          key, new_item, origin_pins))
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
            "deferred_original_worker_folds": deferred_folds,
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
