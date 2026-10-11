"""Bounded persistent ND12 closure, independent of the science coordinator.

CLI: python -m scripts.stage1_nondrl12.finalizer --root <group-output>
No scientific task, documentation publication, or automation is started here.
Wait at most48 hours, polling durable controller/preparation facts every60s.
Launcher absence alone is not failure. Attention is recorded once and exits.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import zipfile

from satellite_rl.coverage.nondrl.runner import worker_lock
from satellite_rl.workbench.runtime import process_start_time
from . import archive, ops, report, strict_audit
from . import placement_transport

MODEL_ROOT=Path("D:/paper/models/runs/first_stage_nondrl12_20261012")


class Attention(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reserve(path, additional=0):
    if archive._disk_free(path)<archive.RESERVE+additional:
        raise Attention("local canonical volume below8GiB reserve")


def atomic_json(path,value):
    """Stream JSON without a second large encoded copy; preserve8GiB reserve."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    reserve(path.parent,1<<20)
    temporary=path.with_suffix(path.suffix+".partial")
    encoder=json.JSONEncoder(ensure_ascii=False,allow_nan=False,indent=2)
    last_check=0
    with temporary.open("w",encoding="utf-8",newline="\n") as stream:
        for block in encoder.iterencode(value):
            stream.write(block)
            if stream.tell()-last_check>8<<20:
                reserve(path.parent,1<<20);last_check=stream.tell()
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def input_paths(root):
    paths = {"plan":root/"run_plan.json","state":root/"controller/state.json",
            "preparations":root/"controller/data_preparation.json","manifest":root/"run_manifest.json",
            "registry":root/"controller/dataset_registry.json"}
    policy_path = root/"controller/source_policy.json"
    if policy_path.is_file():
        from .execution_sources import contract_bytes
        policy = read(policy_path)
        contract_path = Path(policy["local_contract_path"])
        contract = contract_bytes(contract_path.read_bytes(),policy["contract_sha256"])
        paths.update(source_policy=policy_path,source_contract=contract_path)
        for index,item in enumerate(contract["validation_receipts"]):
            source = Path(policy.get("validation_path_map",{}).get(item["path"],item.get("local_path",item["path"])))
            if archive.digest(source)!=item["sha256"]:
                raise Attention("repair validation final-input SHA differs")
            paths["source_validation_%03d"%index]=source
    for name in ("reused_january_archive_receipt", "reused_raw_source_refs"):
        path = root/"source"/(name+".json")
        if path.is_file():paths[name]=path
    audit = root/"preflight/reused_source_reference_audit.json"
    if audit.is_file():paths["reused_source_reference_audit"]=audit
    for name,relative in (("bounded_dispositions","controller/bounded_dispositions.json"),
            ("postops_preservation","preflight/repair_epoch/postops_censored/preservation.json"),
            ("postops_history_roundtrip","preflight/repair_epoch/postops_censored/history_roundtrip.json")):
        path=root/relative
        if path.is_file():paths[name]=path
    return paths


def readiness(root,paths):
    state=read(paths["state"]) if paths["state"].is_file() else None
    prepared=read(paths["preparations"]) if paths["preparations"].is_file() else None
    launcher=root/"controller/launcher.json"
    live_launcher=read(launcher) if launcher.is_file() else None
    for record in (state,live_launcher):
        if record and record.get("phase")=="attention_required":
            raise Attention("science coordinator requires attention: "+str(record.get("attention",record.get("error"))))
        if record and record.get("phase")=="stopped":
            raise Attention("science coordinator explicitly stopped")
    if prepared and any(task.get("stage")=="failed" and task.get("resume_attempts",0)>=2
                        for task in prepared.get("tasks",{}).values()):
        raise Attention("official preparation exhausted recorded recovery cap")
    # No launcher PID/liveness inference: science may survive client exits.
    return bool(state and state.get("phase")=="report_pending" and prepared and
                prepared.get("all_completed") is True and len(prepared.get("tasks",{}))==3 and
                all(task.get("stage")=="completed" for task in prepared["tasks"].values()))


def freeze_inputs(root,paths):
    destination=root/"analysis/final_input";destination.mkdir(parents=True,exist_ok=True)
    manifest_path=destination/"snapshot.json"
    if manifest_path.is_file():
        snapshot=read(manifest_path)
        for item in snapshot["files"].values():
            if archive.digest(item["path"])!=item["sha256"]:
                raise Attention("immutable final_input changed")
        return snapshot
    files={}
    for name,source in paths.items():
        if not source.is_file():raise Attention("required final input missing: "+str(source))
        target=destination/(name+".json")
        reserve(destination,source.stat().st_size)
        if target.exists() and archive.digest(target)!=archive.digest(source):
            raise Attention("interrupted final_input differs; refusing overwrite")
        if not target.exists():shutil.copyfile(source,target)
        sha=archive.digest(target)
        if archive.digest(source)!=sha:raise Attention("input changed during final snapshot")
        files[name]={"path":str(target.resolve()),"source_path":str(source.resolve()),"sha256":sha,"bytes":target.stat().st_size}
    # Freeze small generated gates too, rather than following mutable absolute
    # controller paths later. Raw observations/checkpoints remain in ZIPs.
    for category in ("gates","mechanisms","teacher_reachability"):
        source_dir=root/"controller"/category
        if not source_dir.is_dir():continue
        for source in sorted(source_dir.glob("*.json")):
            target=destination/"controller_evidence"/category/source.name
            target.parent.mkdir(parents=True,exist_ok=True);reserve(target.parent,source.stat().st_size)
            if category=="teacher_reachability" and source.stat().st_size>1<<20:
                from .execution_sources import compact_receipt
                summary=compact_receipt(read(source))
                summary["original_receipt_source"]={"path":str(source.resolve()),"sha256":archive.digest(source),"bytes":source.stat().st_size}
                encoded=json.dumps(summary,ensure_ascii=False,allow_nan=False,sort_keys=True).encode("utf-8")
                if target.exists() and target.read_bytes()!=encoded:
                    raise Attention("compact final helper receipt changed during snapshot")
                if not target.exists():target.write_bytes(encoded)
                files[f"{category}/{source.name}"]={"path":str(target.resolve()),"source_path":str(source.resolve()),
                    "sha256":archive.digest(target),"bytes":len(encoded),"original_receipt_source":summary["original_receipt_source"]}
                continue
            if target.exists() and archive.digest(target)!=archive.digest(source):
                raise Attention("generated final evidence changed during snapshot")
            if not target.exists():shutil.copyfile(source,target)
            files[f"{category}/{source.name}"]={"path":str(target.resolve()),"source_path":str(source.resolve()),
                "sha256":archive.digest(target),"bytes":target.stat().st_size}
    plan,state,prepared,registry=(read(files[name]["path"]) for name in ("plan","state","preparations","registry"))
    if (plan.get("experiment_id")!="S1-ND12" or state.get("experiment_id")!="S1-ND12" or
            prepared.get("experiment_id")!="S1-ND12" or registry.get("experiment_id")!="S1-ND12" or
            state.get("phase")!="report_pending" or prepared.get("all_completed") is not True or
            state.get("plan_sha256")!=report.stable_sha(plan) or
            plan.get("scientific_commit")!=prepared.get("scientific_commit") or
            plan.get("scientific_commit")!=registry.get("scientific_commit")):
        raise Attention("frozen final input identity/terminal source differs")
    if state.get("registry_sha256") and state["registry_sha256"]!=report.stable_sha(registry):
        raise Attention("frozen registry differs from controller identity")
    snapshot={"experiment_id":"S1-ND12","frozen_at":now(),"scientific_commit":plan["scientific_commit"],"files":files}
    atomic_json(manifest_path,snapshot)
    return snapshot


def receipt_path(root,receipt):
    return root/"job_archives"/str(receipt["port"])/receipt["run_id"]/"receipt.json"


def dependency_map(root,receipts,snapshot,cache):
    mapping={}
    for receipt in receipts:
        path=receipt_path(root,receipt)
        evidence=strict_audit.Evidence(path,cache)
        job=evidence.json("job.json")
        original=ops.BASE+f"/{receipt['port']}/archive/outputs/workbench/jobs/{receipt['run_id']}/"
        mapping[original]={"receipt":str(path.resolve()),"member_prefix":""}
        if job.get("kind")=="train" and job.get("model_dir"):
            mapping[job["model_dir"].rstrip("/")+"/"]={"receipt":str(path.resolve()),"member_prefix":"models/"}
    for category in ("gates","mechanisms","teacher_reachability"):
        mapping[str((root/"controller"/category).resolve())+os.sep]=str((root/"analysis/final_input/controller_evidence"/category).resolve())+os.sep
    for item in snapshot.get("files",{}).values():
        if item.get("source_path") and item.get("path"):mapping[item["source_path"]]=item["path"]
    return mapping


def export_endpoints(root,receipts,cache,model_root=None):
    outputs=[]
    destination=(Path(model_root) if model_root is not None else MODEL_ROOT if
                 root.resolve()==ops.LOCAL.resolve() else root/"models/runs/first_stage_nondrl12_20261012")
    destination.mkdir(parents=True,exist_ok=True)
    for receipt in receipts:
        evidence=strict_audit.Evidence(receipt_path(root,receipt),cache)
        job=evidence.json("job.json");status=evidence.json("status.json")
        if job.get("kind")!="train" or status.get("stage")!="completed":continue
        relative="models/final/policy.pt"
        member=evidence.prefix+relative
        if member not in evidence.members:raise Attention("completed training endpoint absent from canonical ZIP")
        volume,expected=evidence.members[member]
        target=destination/receipt["run_id"]/"final/policy.pt";target.parent.mkdir(parents=True,exist_ok=True)
        if expected["bytes"]>256*(1<<20):raise Attention("endpoint export is not a small model; preserve canonical ZIP")
        reserve(target.parent,expected["bytes"])
        if target.exists():
            if archive.digest(target)!=expected["sha256"]:raise Attention("existing endpoint differs")
        else:
            temporary=target.with_suffix(".pt.partial");h=hashlib.sha256();size=0
            with zipfile.ZipFile(volume) as bundle,bundle.open(member) as source,temporary.open("wb") as out:
                for block in iter(lambda:source.read(8<<20),b""):
                    h.update(block);size+=len(block);out.write(block)
                out.flush();os.fsync(out.fileno())
            if size!=expected["bytes"] or h.hexdigest()!=expected["sha256"]:raise Attention("endpoint member SHA differs")
            temporary.replace(target)
        model=evidence.read("models/final/model.json")
        model_record=json.loads(model)
        if model_record.get("policy_sha256")!=expected["sha256"] or model_record.get("selection")!="fixed_budget_endpoint":
            raise Attention("endpoint metadata is not the fixed-budget canonical policy")
        metadata=target.parent/"model.json"
        if metadata.exists() and metadata.read_bytes()!=model:raise Attention("existing endpoint metadata differs")
        if not metadata.exists():metadata.write_bytes(model)
        outputs.append({"run_id":receipt["run_id"],"path":str(target.resolve()),**expected,
                        "model_json_sha256":archive.digest(metadata),"canonical_receipt":str(receipt_path(root,receipt).resolve())})
    return outputs


def finalize(root,paths,guard,progress):
    guard();snapshot=freeze_inputs(root,paths)
    frozen={name:read(item["path"]) for name,item in snapshot["files"].items() if name in paths}
    planned=archive.plan(frozen["state"],frozen["preparations"])
    plan_path=root/"analysis/final_input/archive_plan.json"
    if plan_path.exists() and read(plan_path)!=planned:raise Attention("frozen archive plan differs")
    if not plan_path.exists():atomic_json(plan_path,planned)
    progress("archiving_source",tasks=len(planned["tasks"]))
    guard();source_receipt=archive.one_source(planned)
    receipts=[]
    for index,(key,entry) in enumerate(sorted(planned["tasks"].items())):
        guard();receipt=archive.one_task(entry);receipts.append(receipt)
        progress("archiving_tasks",completed=index+1,total=len(planned["tasks"]),latest_key=key)
    guard();cache=set();source_path=receipt_path(root,source_receipt)
    source_members=archive.verify_source_receipt(source_receipt,source_path.parent,cache)
    closures=[]
    for receipt in receipts:
        guard()
        closure=archive.verify_receipt(receipt,receipt_path(root,receipt).parent,source_receipt,source_path.parent,
                                       receipts,verified_volume_cache=cache)
        if closure.get("closed") is not True:raise Attention("dependency closure incomplete")
        closures.append(closure)
    atomic_json(root/"analysis/final_input/archive_closure.json",{"source_members":len(source_members),
                 "tasks":closures,"unique_verified_volumes":len(cache),"complete":True})
    mapping=dependency_map(root,receipts,snapshot,cache)
    atomic_json(root/"analysis/final_input/path_map.json",mapping)
    progress("building_strict_report",archived_tasks=len(receipts))
    guard();result=report.build(frozen["plan"],frozen["state"],root,path_map=mapping,
        gate_directory=root/"analysis/final_input/controller_evidence/gates",verified_volume_cache=cache)
    destination=root/"analysis/final_report";destination.mkdir(parents=True,exist_ok=True)
    if result.get("formal_ready") is not True:
        atomic_json(destination/"report.json",result)
        raise Attention("strict report not ready: "+"; ".join(result.get("issues",[])[:8]))
    result["endpoints"]=export_endpoints(root,receipts,cache)
    guard();reserve(destination,32<<20);result["figures"]=report.figures(result,destination/"figures")
    atomic_json(destination/"report.json",result)
    reserve(destination,1<<20)
    (destination/"report.md").write_text(report.markdown(result),encoding="utf-8")
    guard()
    receipt={"experiment_id":"S1-ND12","formal_ready":True,"completed_at":now(),
        "scientific_commit":snapshot["scientific_commit"],"tasks":len(receipts),"source_members":len(source_members),
        "report_sha256":archive.digest(destination/"report.json"),"markdown_sha256":archive.digest(destination/"report.md"),
        "figure_count":len(result["figures"]),"endpoint_count":len(result["endpoints"]),
        "docs_published":False,"scope":"canonical archive and local evidence report; delivery publication remains root responsibility"}
    atomic_json(destination/"completion.json",receipt)
    progress("complete",**{k:receipt[k] for k in ("tasks","figure_count","endpoint_count","report_sha256")})
    return receipt


def run(root=ops.LOCAL,*,wait_hours=48,poll_seconds=60,resume_after_attention=False,placement_file=None):
    root=Path(root).resolve()
    selected=Path(placement_file).resolve() if placement_file is not None else root/"controller/execution_placement.json"
    transport=placement_transport.Transport.from_file(selected) if placement_file is not None or selected.is_file() else None
    original_transport=archive.ops
    if transport is not None:archive.ops=transport
    try:
        return _run(root,wait_hours=wait_hours,poll_seconds=poll_seconds,
                    resume_after_attention=resume_after_attention,transport=transport)
    finally:
        archive.ops=original_transport


def _run(root=ops.LOCAL,*,wait_hours=48,poll_seconds=60,resume_after_attention=False,transport=None):
    root=Path(root).resolve()
    if not 0<wait_hours<=48 or poll_seconds!=60:raise ValueError("wait bounded at48h; polling fixed at60s")
    directory=root/"controller/finalizer";directory.mkdir(parents=True,exist_ok=True)
    from . import execution_sources
    from satellite_rl.workbench import nondrl_source_repair
    from satellite_rl.coverage.nondrl import source_lineage
    modules=[Path(__file__),Path(archive.__file__),Path(report.__file__),Path(strict_audit.__file__),Path(ops.__file__),
             Path(execution_sources.__file__),Path(nondrl_source_repair.__file__),Path(source_lineage.__file__)]
    disposition_path=root/"controller/bounded_dispositions.json"
    if disposition_path.is_file():modules.append(disposition_path)
    if transport is not None:modules.extend((Path(placement_transport.__file__),transport.path))
    guards={str(p.resolve()):archive.digest(p) for p in modules}
    status_path=directory/"status.json";prior=read(status_path) if status_path.is_file() else None
    if prior and prior.get("phase")=="complete":return prior
    if prior and prior.get("phase")=="attention_required" and not resume_after_attention:return prior
    state={"experiment_id":"S1-ND12","pid":os.getpid(),"process_start_time":process_start_time(os.getpid()),
           "started_at":prior.get("started_at",now()) if prior else now(),
           "deadline_unix":prior.get("deadline_unix",time.time()+wait_hours*3600) if prior else time.time()+wait_hours*3600,
           "module_sha256":guards,"phase":"waiting"}
    if transport is not None:
        state.update(placement_file=str(transport.path),placement_sha256=transport.mapping_sha256,
                     logical_to_physical=transport.mapping,physical_ports=sorted(set(transport.mapping.values())))
    paths=input_paths(root)
    monotonic_deadline=time.monotonic()+min(max(0.,state["deadline_unix"]-time.time()),48*3600)
    def progress(phase,**fields):
        state.update(phase=phase,observed_at=now(),**fields);atomic_json(status_path,state)
        print(json.dumps({"at":state["observed_at"],"phase":phase,**fields},ensure_ascii=False),flush=True)
    def guard():
        if transport is not None:transport.verify_unchanged()
        if any(archive.digest(p)!=sha for p,sha in guards.items()):raise Attention("finalizer module changed")
        if (directory/"stop.request").exists():raise Attention("explicit finalizer stop requested")
        if time.time()>=state["deadline_unix"] or time.monotonic()>=monotonic_deadline:
            raise Attention("48h bounded wait/execution deadline expired")
    with worker_lock(directory):
        try:
            if prior and prior.get("module_sha256")!=guards and not resume_after_attention:
                raise Attention("finalizer restart source guard differs; explicit diagnosed resume required")
            if prior and resume_after_attention:
                state["prior_attention"]={k:prior.get(k) for k in ("phase","error","observed_at","module_sha256")}
            progress("waiting")
            while True:
                guard()
                if readiness(root,paths):break
                progress("waiting")
                time.sleep(poll_seconds)
            receipt=finalize(root,paths,guard,progress)
            return {**state,"completion":receipt}
        except Exception as error:
            progress("attention_required",error=repr(error),traceback=traceback.format_exc())
            return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ops.LOCAL)
    parser.add_argument("--wait-hours",type=float,default=48)
    parser.add_argument("--resume-after-attention",action="store_true")
    parser.add_argument("--placement-file",type=Path)
    args=parser.parse_args();value=run(args.root,wait_hours=args.wait_hours,resume_after_attention=args.resume_after_attention,
                                      placement_file=args.placement_file)
    raise SystemExit(0 if value.get("phase")=="complete" else 2)


if __name__=="__main__":main()
