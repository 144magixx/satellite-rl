"""Single-ID workbench activation and provenance for an equivalent datafeed.

No active worker is changed or signalled. Activation requires a quiet complete
checkpoint after validation, preserves it, and queues the ordinary same-ID resume.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil

from satellite_rl.training.run_monitor import atomic_json
from .storage import digest, now, read_json
from . import nondrl_datafeed_bootstrap as bootstrap


def audit_checkpoint(job_dir):
    """Read-only safe-stop proof. Caller must first stop via the ordinary API."""
    from .runtime import pid_alive
    import torch
    directory=Path(job_dir).resolve();job=read_json(directory/"job.json",{})
    status,launch=read_json(directory/"status.json",{}),read_json(directory/"launch.json",{})
    if (job.get("run_id")!=bootstrap.RUN_ID or job.get("kind")!="train" or job.get("variant")!="01" or
            job.get("git_commit")!=bootstrap.BASE_COMMIT or status.get("stage")!="stopped" or
            status.get("scientific_stop") or not launch.get("pid") or
            not isinstance(launch.get("process_start_time"),(int,float)) or
            pid_alive(launch["pid"],launch["process_start_time"])):
        raise ValueError("extension audit requires the authorized task stopped and original birth exited")
    model=Path(job["model_dir"]).resolve();checkpoint=model/"checkpoint.pt"
    before=digest(checkpoint);state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    update=state.get("updates",-1);config=read_json(directory/"config.json",{})
    if (type(update)is not int or not 1000<update<10000 or not 25<=update%1000<=900 or
            state.get("total_updates")!=10000 or state.get("format")!="s1_nondrl_checkpoint_v1" or
            config["training"].get("updates")!=10000 or not state.get("model") or
            not state.get("optimizer",{}).get("state") or state.get("cursor",-1)!=update*32 or
            set(state.get("rng",{}))!={"python","numpy","torch","cuda","sampler"} or not state["rng"]["cuda"]):
        raise ValueError("checkpoint is incomplete or outside the completed-validation safe window")
    losses=model/"losses.jsonl";loss_raw=losses.read_bytes()
    if len(loss_raw)!=state["loss_bytes"] or hashlib.sha256(loss_raw).hexdigest()!=state["loss_sha256"]:
        raise ValueError("quiet checkpoint loss curve differs or has uncommitted tails")
    rows=[json.loads(line) for line in loss_raw.splitlines() if line.strip()]
    if len(rows)!=update or [r["update"] for r in rows]!=list(range(1,update+1)):
        raise ValueError("loss curve update prefix has a gap or duplicate")
    bootstrap.validate_loss_rows(rows,state)
    if any(float(item.get("step",-1))!=update for item in state["optimizer"]["state"].values()):
        raise ValueError("Adam step counter differs from checkpoint updates")
    expected_nodes=list(range(1000,update//1000*1000+1,1000))
    nodes=state.get("validation",[])
    if [node.get("updates") for node in nodes]!=expected_nodes or read_json(model/"validation.json")!=nodes:
        raise ValueError("checkpoint validation curve has a missing/unfinished node")
    manifests=[];expected_keys={f"{i}:{m:g}" for i in range(20) for m in (.75,1.,1.25)}
    for node in nodes:
        path=directory/"validation"/f"update_{node['updates']:06d}"/"manifest.json"
        record=read_json(path,{})
        if (node["metrics"].get("raw_manifest_sha256")!=digest(path) or
                {r["key"] for r in record.get("rows",[])}!=expected_keys or len(record.get("rows",[]))!=60):
            raise ValueError("fixed validation does not retain all60 scene records")
        for row in record["rows"]:
            source=(path.parent/row["file"]).resolve()
            if not source.is_relative_to(path.parent) or digest(source)!=row["sha256"]:
                raise ValueError("fixed validation shard differs")
        manifests.append({"path":str(path),"sha256":digest(path),"updates":node["updates"],"rows":60})
    validation_root=directory/"validation"
    if (any(p.is_dir() and p.name.startswith("update_") and int(p.name[7:])>expected_nodes[-1]
            for p in validation_root.iterdir()) or any(validation_root.rglob("*.partial"))):
        raise ValueError("unfinished next validation is retained; extension must not skip it")
    if digest(checkpoint)!=before:raise ValueError("checkpoint changed during quiet audit")
    return {"format":"s1_nd12_datafeed_safe_checkpoint_v1","passed":True,"run_id":bootstrap.RUN_ID,
        "updates":update,"checkpoint_sha256":before,"config_sha256":digest(directory/"config.json"),
        "loss_sha256":state["loss_sha256"],"loss_bytes":state["loss_bytes"],"loss_rows":len(rows),
        "validation_manifests":manifests,"complete_optimizer_rng_cursor":True,"complete_validation_nodes":True,
        "original_pid":launch["pid"],"original_process_start_time":launch["process_start_time"],
        "base_learning_contract":state["contract"]}


def validate_job_extension(job,job_dir,*,read_artifact=None):
    entries=job.get("runtime_extensions",[])
    if not entries:return None
    if len(entries)!=1:raise ValueError("only one exact datafeed extension is permitted")
    entry=entries[0];root=Path(job_dir)/"runtime_extensions"/bootstrap.EXTENSION_ID
    if read_artifact is not None:
        def payload(name,expected):
            value=read_artifact("runtime_extensions/"+bootstrap.EXTENSION_ID+"/"+name)
            if hashlib.sha256(value).hexdigest()!=expected:raise ValueError("archived runtime artifact differs")
            return value
        contract=json.loads(payload("contract.json",entry["contract_sha256"]))
        runtime=bootstrap.marker(contract,entry["contract_sha256"])
        if (job.get("run_id")!=bootstrap.RUN_ID or job.get("git_commit")!=bootstrap.BASE_COMMIT or
                contract.get("run_id")!=bootstrap.RUN_ID or contract.get("format")!=bootstrap.FORMAT or
                contract.get("extension_id")!=bootstrap.EXTENSION_ID or entry.get("marker")!=runtime or
                contract["provider"]["sha256"]!=bootstrap.PROVIDER_SHA or
                contract["cache_manifest"]["sha256"]!=bootstrap.CACHE_SHA):
            raise ValueError("archived composite runtime identity differs")
        for name,key in (("provider.py","provider"),("bootstrap.py","bridge"),("checkpoint_audit.json","checkpoint_audit"),
                         ("cache/manifest.json","cache_manifest"),("origin_checkpoint.pt","checkpoint")):
            payload(name,contract[key]["sha256"])
        cache=json.loads(payload("cache/manifest.json",contract["cache_manifest"]["sha256"]))
        for name,item in cache["files"].items():payload("cache/"+name,item["sha256"])
        payload("origin_losses_prefix.bin",contract["loss_prefix"]["sha256"])
        for i,item in enumerate(contract["equivalence_proofs"]):payload(f"proofs/{i:03d}.json",item["sha256"])
        return runtime
    contract,current,runtime=bootstrap.validate(root/"contract.json",entry["contract_sha256"],job_dir)
    if current.get("runtime_extensions")!=entries or entry.get("marker")!=runtime:
        raise ValueError("job composite runtime marker differs")
    for name,key in (("provider.py","provider"),("bootstrap.py","bridge"),("checkpoint_audit.json","checkpoint_audit"),
                     ("cache/manifest.json","cache_manifest"),("origin_checkpoint.pt","checkpoint")):
        bootstrap.pin({"path":str(root/name),"sha256":contract[key]["sha256"]})
    bootstrap.current_checkpoint(job,contract,runtime)
    return runtime


def worker_command(job,job_dir,python):
    if job.get("runtime_extension_pending"):
        raise ValueError("runtime extension activation transaction is not committed")
    runtime=validate_job_extension(job,job_dir)
    if runtime is None:return None
    entry=job["runtime_extensions"][0]
    return [python,"-u",str(Path(job_dir)/"runtime_extensions"/bootstrap.EXTENSION_ID/"bootstrap.py"),
            "--job-dir",str(job_dir),"--contract",entry["contract_path"],
            "--contract-sha256",entry["contract_sha256"],"--resume"]


def validate_model_consumer(source_job,source_dir,model_metadata,policy_path):
    """Read markers from the actual fixed model; never grant bootstrap to its consumer."""
    if not source_job.get("runtime_extensions"):
        if model_metadata.get("runtime_extensions"):
            raise ValueError("model has a runtime marker without an authorized producer")
        return []
    runtime=validate_job_extension(source_job,source_dir)
    import torch
    policy=torch.load(policy_path,map_location="cpu",weights_only=False)
    if (source_job.get("run_id")!=bootstrap.RUN_ID or
            model_metadata.get("runtime_extensions")!=[runtime] or policy.get("runtime_extensions")!=[runtime]):
        raise ValueError("model producer/endpoint composite runtime markers differ")
    return [runtime]


def preserve_checkpoint_tail(job,job_dir):
    """Before ordinary resume, preserve orphan loss bytes without truncating them."""
    runtime=validate_job_extension(job,job_dir)
    if runtime is None:return
    entry=job["runtime_extensions"][0];contract=read_json(entry["contract_path"])
    state=bootstrap.current_checkpoint(job,contract,runtime)
    path=Path(job["model_dir"])/"losses.jsonl"
    with path.open("rb") as stream:
        stream.seek(state["loss_bytes"]);tail=stream.read()
    if tail:
        digest_tail=hashlib.sha256(tail).hexdigest()
        target=Path(job_dir)/"runtime_extensions"/bootstrap.EXTENSION_ID/"recovery_tails"/f"loss_after_{state['updates']}_{digest_tail}.jsonl"
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=tail:raise ValueError("saved recovery tail differs")
        if not target.exists():target.write_bytes(tail)


def extend_job(scheduler,run_id,body):
    if run_id!=bootstrap.RUN_ID or set(body)!={"contract_path","contract_sha256"}:
        raise ValueError("datafeed extension only accepts the one approved run and frozen contract")
    with scheduler.lock,scheduler.store.lock:
        directory=scheduler.store.job_dir(run_id)
        contract,job,runtime=bootstrap.validate(body["contract_path"],body["contract_sha256"],directory,
                                               initial=not read_json(directory/"job.json",{}).get("runtime_extensions"))
        if job.get("runtime_extensions"):
            validate_job_extension(job,directory)
            if job["runtime_extensions"][0]["contract_sha256"]!=body["contract_sha256"]:
                raise ValueError("existing extension contract cannot change")
            if job.get("runtime_extension_pending"):
                status=read_json(directory/"status.json",{})
                status.update(stage="queued",pid=None,error=None,heartbeat_at=now(),runtime_extensions=[runtime])
                atomic_json(directory/"status.json",status)
                job["runtime_extension_pending"]=False;atomic_json(directory/"job.json",job)
                return status
            return read_json(directory/"status.json",{})
        actual=audit_checkpoint(directory);expected=read_json(contract["checkpoint_audit"]["path"])
        if actual!=expected:raise ValueError("current full-state audit differs from the approved quiet proof")
        root=directory/"runtime_extensions"/bootstrap.EXTENSION_ID;root.mkdir(parents=True,exist_ok=True)
        files={"contract.json":body["contract_path"],"origin_checkpoint.pt":contract["checkpoint"]["path"],
               "origin_status.json":directory/"status.json","provider.py":contract["provider"]["path"],
               "bootstrap.py":contract["bridge"]["path"],"checkpoint_audit.json":contract["checkpoint_audit"]["path"],
               "cache/manifest.json":contract["cache_manifest"]["path"]}
        cache=read_json(contract["cache_manifest"]["path"])
        for name in ("data.bin","index.sqlite"):
            files["cache/"+name]=Path(contract["cache_manifest"]["path"]).parent/name
        for i,item in enumerate(contract["equivalence_proofs"]):files[f"proofs/{i:03d}.json"]=item["path"]
        for name,path in files.items():
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and digest(target)!=digest(path):raise ValueError("frozen extension file collision")
            if not target.exists():shutil.copyfile(path,target)
        with Path(contract["loss_prefix"]["path"]).open("rb") as source:
            prefix=source.read(contract["loss_prefix"]["bytes"])
        (root/"origin_losses_prefix.bin").write_bytes(prefix)
        job.update(runtime_extensions=[{"contract_path":str(root/"contract.json"),"contract_sha256":body["contract_sha256"],
                                       "marker":runtime}],resume=True,
                   runtime_extension_pending=True,
                   runtime_history=[{"at":now(),"marker":runtime,"safe_start_update":contract["start_update"],
                                     "origin_checkpoint_sha256":contract["checkpoint"]["sha256"],"original_pid":actual["original_pid"],
                                     "original_process_start_time":actual["original_process_start_time"]}])
        atomic_json(directory/"job.json",job)
        validate_job_extension(job,directory)
        (directory/"stop.request").unlink(missing_ok=True)
        status=read_json(directory/"status.json",{})
        status.update(stage="queued",pid=None,error=None,heartbeat_at=now(),queue_reason="冻结等价数据供给扩展同ID恢复",runtime_extensions=[runtime])
        atomic_json(directory/"status.json",status)
        job["runtime_extension_pending"]=False;atomic_json(directory/"job.json",job)
        return status
