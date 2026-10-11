"""F20 execution-source extensions preserving the original scientific identity."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

from satellite_rl.training.run_monitor import atomic_json
from .storage import digest, inside, now, read_json, validate_id

ORIGIN_COMMIT = "2fc16884078ad64b97333013737fef5cd9294626"
FORMAT = "fair20_execution_source_v1"
REPAIR_ID = "fair20_strict_B_acceptance_direct_fallback_v1"
PROOF_FORMAT = "fair20_validation_only_resume_v1"
MODES = {"train", "evaluate", "matrix", "refine"}
DIFF_ALLOWLIST = {
    "src/satellite_rl/workbench/fair20_runner.py",
    "src/satellite_rl/workbench/fair20_source_repair.py",
    "src/satellite_rl/workbench/fair20_service.py",
    "src/satellite_rl/workbench/runtime.py",
    "src/satellite_rl/workbench/service.py",
    "src/satellite_rl/workbench/static/fair20.js",
    "src/satellite_rl/workbench/static/fair20.html",
}


def _hash(value, length=64):
    if not isinstance(value,str) or not re.fullmatch(r"[0-9a-f]{"+str(length)+"}",value):
        raise ValueError("invalid F20 execution-source hash")
    return value


def _pin(path, expected):
    path=Path(path).resolve(strict=True)
    if not path.is_file() or digest(path)!=_hash(expected):
        raise ValueError("F20 source repair SHA differs: "+str(path))
    return path


def _git(repo,*args):
    return subprocess.run(["git",*args],cwd=repo,capture_output=True,check=True).stdout


def validate_contract(store,path,sha256):
    path=_pin(path,sha256);contract=read_json(path,{})
    if (contract.get("format")!=FORMAT or contract.get("repair_id")!=REPAIR_ID or
            contract.get("scientific_commit")!=ORIGIN_COMMIT):
        raise ValueError("unknown F20 execution-source contract")
    execution=_hash(contract.get("execution_commit"),40)
    if execution==ORIGIN_COMMIT:raise ValueError("execution repair must have an honest distinct commit")
    lane=store.archive_root.parent
    repository=inside(contract.get("execution_source_repo",""),lane)
    scientific=inside(contract.get("scientific_source_repo",""),lane)
    for repo,revision in ((repository,execution),(scientific,ORIGIN_COMMIT)):
        if (not repo.is_dir() or _git(repo,"rev-parse","HEAD").decode().strip()!=revision or
                _git(repo,"status","--porcelain","--","src","data","pyproject.toml").strip()):
            raise ValueError("F20 scientific/execution checkout must be clean at its exact commit")
    actual=set(_git(repository,"diff","--name-only",ORIGIN_COMMIT,execution,"--","src","data","pyproject.toml").decode().splitlines())
    files=contract.get("diff_files",[])
    if (not actual or actual-DIFF_ALLOWLIST or {row.get("path") for row in files}!=actual or
            len(files)!=len(actual)):
        raise ValueError("F20 repair diff changes an unapproved algorithm/environment/solver/data file")
    for row in files:
        for revision,key in ((ORIGIN_COMMIT,"origin_sha256"),(execution,"execution_sha256")):
            shown=subprocess.run(["git","show",revision+":"+row["path"]],cwd=repository,capture_output=True)
            if shown.returncode:
                if key=="execution_sha256" or row.get(key) is not None:
                    raise ValueError("F20 repair file identity missing")
            elif hashlib.sha256(shown.stdout).hexdigest()!=_hash(row.get(key)):
                raise ValueError("F20 repair exact diff file hash differs")
    modes=contract.get("allowed_future_modes",[])
    if not isinstance(modes,list) or not modes or len(modes)!=len(set(modes)) or set(modes)-MODES:
        raise ValueError("future F20 execution-source scope is invalid")
    runs=contract.get("runs",{})
    if not isinstance(runs,dict):raise ValueError("F20 per-run repair pins must be an object")
    for identifier in runs:validate_id(identifier)
    receipts=contract.get("validation_receipts",[])
    if not receipts:raise ValueError("F20 source repair needs passed frozen validation receipts")
    for receipt in receipts:
        value=read_json(_pin(receipt["path"],receipt["sha256"]),{})
        if value.get("passed") is not True and value.get("status")!="passed":
            raise ValueError("F20 source repair validation has not passed")
    if (contract.get("learning_and_environment_unchanged") is not True or
            contract.get("optimizer_and_budgets_unchanged") is not True or
            contract.get("acceptance_policy")!="strict_SI_SGM_service_resource_else_verified_direct_no_extra_solve"):
        raise ValueError("F20 repair must preserve strict service protection, learning and solver budgets")
    return contract,path


def verify_snapshot(snapshot,revision,manifest_sha256=None):
    snapshot=Path(snapshot).resolve(strict=True)
    manifest_path=snapshot/"source_manifest.json"
    if manifest_sha256 is not None:_pin(manifest_path,manifest_sha256)
    manifest=read_json(manifest_path,{})
    if manifest.get("git_commit")!=revision or not manifest.get("files"):
        raise ValueError("F20 execution snapshot manifest identity differs")
    actual={str(path.relative_to(snapshot)).replace("\\","/") for path in snapshot.rglob("*")
            if path.is_file() and path.name!="source_manifest.json" and "__pycache__" not in path.parts}
    if actual!=set(manifest["files"]):raise ValueError("F20 execution snapshot file set differs")
    for name,expected in manifest["files"].items():_pin(inside(snapshot/name,snapshot),expected)
    return manifest


def freeze_execution(store,contract):
    revision=contract["execution_commit"]
    destination=store.root/"source_snapshots"/revision
    if not (destination/"source_manifest.json").is_file():
        archive=_git(contract["execution_source_repo"],"archive","--format=zip",revision,"src","data","pyproject.toml")
        destination.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            for item in bundle.infolist():inside(destination/item.filename,destination)
            bundle.extractall(destination)
        atomic_json(destination/"source_manifest.json",{"git_commit":revision,"created_at":now(),"format_version":1,
            "scientific_commit":ORIGIN_COMMIT,"files":{str(path.relative_to(destination)).replace("\\","/"):digest(path)
                for path in destination.rglob("*") if path.is_file()}})
    verify_snapshot(destination,revision)
    return destination


def _reference(contract,path,sha256,snapshot):
    return {"format":FORMAT,"scientific_commit":ORIGIN_COMMIT,"execution_commit":contract["execution_commit"],
            "snapshot_path":str(snapshot),"manifest_sha256":digest(snapshot/"source_manifest.json"),
            "contract_path":str(path),"contract_sha256":sha256}


def register_group(scheduler,body):
    if set(body)!={"contract_path","contract_sha256"}:raise ValueError("execution source accepts only a frozen contract reference")
    with scheduler.lock,scheduler.store.lock:
        store=scheduler.store;contract,path=validate_contract(store,body["contract_path"],body["contract_sha256"])
        registration=store.root/"fair20_execution_source.json"
        old=read_json(registration)
        if old and old["contract_sha256"]!=body["contract_sha256"]:
            raise ValueError("F20 registered repair cannot silently change")
        target=store.root/"fair20_source_lineage"/"contract.json";target.parent.mkdir(exist_ok=True)
        if not target.exists():shutil.copyfile(path,target)
        _pin(target,body["contract_sha256"])
        snapshot=freeze_execution(store,contract)
        reference=_reference(contract,target,body["contract_sha256"],snapshot)
        reference["scientific_source_repo"]=contract["scientific_source_repo"]
        atomic_json(registration,reference)
        store.source_root=Path(contract["scientific_source_repo"]).resolve()
        return reference


def registered_source(store):
    reference=read_json(store.root/"fair20_execution_source.json")
    if not reference:return None
    contract,_=validate_contract(store,reference["contract_path"],reference["contract_sha256"])
    verify_snapshot(reference["snapshot_path"],reference["execution_commit"],reference["manifest_sha256"])
    store.source_root=Path(contract["scientific_source_repo"]).resolve()
    return reference


def _bind(store,job,directory,contract,path,sha256,snapshot,comparison_scope):
    if job.get("git_commit")!=ORIGIN_COMMIT or job.get("experiment_id")!="S2-F20":
        raise ValueError("F20 execution source cannot replace the original scientific identity")
    target=directory/"source_lineage"/"contract.json";target.parent.mkdir(exist_ok=True)
    if not target.exists():shutil.copyfile(path,target)
    _pin(target,sha256)
    job["execution_source"]=_reference(contract,target,sha256,snapshot)
    job["execution_source"]["config_sha256"]=digest(directory/"config.json")
    job["source_history"]=job.get("source_history",[])+[{"scientific_commit":ORIGIN_COMMIT,
        "execution_commit":contract["execution_commit"],"repair_id":REPAIR_ID,"contract_sha256":sha256,
        "manifest_sha256":job["execution_source"]["manifest_sha256"],"at":now(),"comparison_scope":comparison_scope}]
    validate_execution_job(job,directory)
    atomic_json(directory/"job.json",job)
    return job


def bind_future_job(store,job):
    reference=registered_source(store)
    if not reference or job.get("mode") not in MODES:return job
    directory=store.job_dir(job["run_id"])
    if job.get("execution_source"):
        validate_execution_job(job,directory);return job
    status=read_json(directory/"status.json",{})
    if status.get("stage") not in ("queued","queued_resources") or (directory/"launch.json").exists():
        raise ValueError("already executed F20 jobs require the explicit same-ID source-repair transaction")
    contract,path=validate_contract(store,reference["contract_path"],reference["contract_sha256"])
    if job["mode"] not in contract["allowed_future_modes"]:raise ValueError("new F20 task execution role is outside repair scope")
    return _bind(store,job,directory,contract,path,reference["contract_sha256"],Path(reference["snapshot_path"]),"new_task_same_science")


def validate_execution_job(job,directory):
    reference=job.get("execution_source")
    if not reference:return
    if (reference.get("format")!=FORMAT or reference.get("scientific_commit")!=ORIGIN_COMMIT or
            job.get("git_commit")!=ORIGIN_COMMIT or job.get("experiment_id")!="S2-F20"):
        raise ValueError("mixed F20 scientific/execution identity")
    contract=read_json(_pin(reference["contract_path"],reference["contract_sha256"]),{})
    if (contract.get("scientific_commit")!=ORIGIN_COMMIT or contract.get("execution_commit")!=reference["execution_commit"] or
            contract.get("repair_id")!=REPAIR_ID or contract.get("format")!=FORMAT):
        raise ValueError("F20 execution source contract identity differs")
    _pin(Path(directory)/"config.json",reference["config_sha256"])
    verify_snapshot(job["source_root"],ORIGIN_COMMIT)
    verify_snapshot(reference["snapshot_path"],reference["execution_commit"],reference["manifest_sha256"])
    entries=job.get("source_history",[])
    if (not entries or entries[-1].get("contract_sha256")!=reference["contract_sha256"] or
            entries[-1].get("execution_commit")!=reference["execution_commit"] or
            entries[-1].get("scientific_commit")!=ORIGIN_COMMIT):
        raise ValueError("F20 execution source history missing")
    if job.get("mode") not in contract.get("allowed_future_modes",[]):
        raise ValueError("F20 execution role is outside frozen repair scope")
    if entries[-1].get("comparison_scope")=="same_id_validation_only_repair":
        item=contract.get("runs",{}).get(job["run_id"])
        if not item:raise ValueError("same-ID F20 repair run is not allowlisted")
        history=Path(directory)/"source_lineage"/"origin"
        for name in ("job","status","config","launch"):
            if item.get(name+"_sha256"):_pin(history/(name+".json"),item[name+"_sha256"])
        _pin(history/"source_manifest.json",item["origin_manifest_sha256"])
        for artifact in item.get("artifacts",[]):
            _pin(inside(history/"artifacts"/artifact["relative_path"],history),artifact["sha256"])
    elif entries[-1].get("comparison_scope")!="new_task_same_science":
        raise ValueError("unknown F20 execution-source comparison scope")


def execution_view(job):
    view=dict(job)
    if job.get("execution_source"):
        reference=job["execution_source"]
        view.update(git_commit=reference["execution_commit"],source_root=reference["snapshot_path"])
    return view


def repair_job(scheduler,run_id,body):
    from .runtime import pid_alive
    if set(body)!={"contract_path","contract_sha256"}:raise ValueError("source repair accepts only immutable contract reference")
    with scheduler.lock,scheduler.store.lock:
        store=scheduler.store;contract,path=validate_contract(store,body["contract_path"],body["contract_sha256"])
        directory=store.job_dir(validate_id(run_id));job=read_json(directory/"job.json",{});status=read_json(directory/"status.json",{})
        if job.get("execution_source"):
            validate_execution_job(job,directory)
            if job["execution_source"]["contract_sha256"]!=body["contract_sha256"]:raise ValueError("same task cannot change repair contract")
            if job.get("source_repair_pending"):
                status.update(stage="queued_resources" if job.get("resource_hold") else "queued",pid=None,error=None,
                              heartbeat_at=now(),queue_reason="继续已绑定的F20来源修复事务")
                atomic_json(directory/"status.json",status)
                job["source_repair_pending"]=False;atomic_json(directory/"job.json",job)
            return status
        item=contract.get("runs",{}).get(run_id)
        if not item or status.get("stage") not in ("failed","stopped","interrupted","queued","queued_resources"):
            raise ValueError("existing task is not within the explicit repair/state scope")
        launch=read_json(directory/"launch.json",{})
        if launch and (launch.get("pid")!=item.get("original_pid") or
                launch.get("process_start_time")!=item.get("original_process_start_time") or
                pid_alive(launch.get("pid"),launch.get("process_start_time"))):
            raise ValueError("original worker identity must be pinned and exited")
        if not launch and status.get("stage") not in ("queued","queued_resources"):
            raise ValueError("original failed worker launch evidence missing")
        if job.get("mode") not in contract["allowed_future_modes"]:raise ValueError("existing task mode outside F20 repair scope")
        for name in ("job","status","config"):_pin(directory/(name+".json"),item[name+"_sha256"])
        if launch:_pin(directory/"launch.json",item["launch_sha256"])
        verify_snapshot(job["source_root"],ORIGIN_COMMIT,item["origin_manifest_sha256"])
        artifacts=item.get("artifacts",[])
        declared={artifact["relative_path"] for artifact in artifacts}
        mandatory={str(p.relative_to(directory)).replace("\\","/") for p in directory.rglob("*") if p.is_file() and
            "source_lineage" not in p.parts and "__pycache__" not in p.parts and (p.name in
            ("checkpoint.pt","failure.json","worker.log","scene_order.json","traces.jsonl","episodes.jsonl","updates.jsonl","evaluations.jsonl","state.json","model.json") or
            p.parent.name.startswith("validation_") or p.parent.name=="final" or p.name.startswith("policy"))}
        if mandatory-declared:raise ValueError("original checkpoint/training/validation/failure artifacts must all be pinned")
        history=directory/"source_lineage"/"origin";history.mkdir(parents=True,exist_ok=True)
        for name in ("job","status","config","launch"):
            if (directory/(name+".json")).exists():shutil.copyfile(directory/(name+".json"),history/(name+".json"))
        shutil.copyfile(Path(job["source_root"])/"source_manifest.json",history/"source_manifest.json")
        for artifact in artifacts:
            source=_pin(inside(directory/artifact["relative_path"],directory),artifact["sha256"])
            target=inside(history/"artifacts"/artifact["relative_path"],history)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);_pin(target,artifact["sha256"])
        proof_ref=item.get("validation_only_resume_proof")
        if status.get("stage") in ("failed","interrupted") and job.get("kind")=="train":
            if not proof_ref:raise ValueError("validation-only failure requires frozen no-extra-interaction proof")
            proof=read_json(_pin(proof_ref["path"],proof_ref["sha256"]),{})
            if (proof.get("version")!=PROOF_FORMAT or proof.get("run_id")!=run_id or
                    proof.get("scientific_commit")!=ORIGIN_COMMIT or proof.get("execution_commit")!=contract["execution_commit"] or
                    proof.get("checkpoint_step")!=status.get("env_step") or
                    proof.get("agent_interactions")!=proof.get("checkpoint_step") or
                    proof.get("failed_phase")!=status.get("phase") or
                    proof.get("failed_phase")!=f"validation_{proof.get('checkpoint_step')}" or
                    proof.get("failed_status_sha256")!=item["status_sha256"] or
                    proof.get("failure_sha256")!=digest(directory/"failure.json") or
                    proof.get("checkpoint_sha256")!=digest(directory/"checkpoint.pt")):
                raise ValueError("validation-only proof is not bound to the original failure/node")
            expected_records={name:history/(name+".json") for name in ("job","status","config","launch")}
            expected_records.update(checkpoint=history/"artifacts"/"checkpoint.pt",failure=history/"artifacts"/"failure.json")
            records=proof.get("origin_records",{})
            if set(records)!=set(expected_records):raise ValueError("validation-only proof origin records are incomplete")
            for name,expected_path in expected_records.items():
                reference=records[name]
                if Path(reference.get("path","")).resolve()!=expected_path.resolve():
                    raise ValueError("validation-only proof must reference immutable preserved origin bytes")
                _pin(expected_path,reference["sha256"])
            logs=proof.get("training_logs",{});sizes=proof.get("training_log_sizes",{})
            expected_logs={"traces.jsonl","episodes.jsonl","updates.jsonl"}
            if set(logs)!=expected_logs or set(sizes)!=expected_logs:
                raise ValueError("validation-only proof training log prefix records are incomplete")
            for name in expected_logs:
                row=logs[name];log_path=directory/name
                if (Path(row.get("path","")).resolve()!=log_path.resolve() or
                        type(sizes[name]) is not int or sizes[name]!=log_path.stat().st_size or
                        row.get("saved_size")!=sizes[name]):
                    raise ValueError("validation-only proof training log lengths or paths differ")
                _pin(log_path,row["sha256"])
            frozen=directory/"source_lineage"/"validation_only_resume.json";shutil.copyfile(proof_ref["path"],frozen)
            job["validation_only_resume_proof"]={"proof_path":str(frozen),"proof_sha256":proof_ref["sha256"]}
        snapshot=freeze_execution(store,contract)
        job["source_repair_origin"]={"job_sha256":item["job_sha256"],"status_sha256":item["status_sha256"],
                                    "config_sha256":item["config_sha256"],"manifest_sha256":item["origin_manifest_sha256"]}
        job["source_repair_pending"]=True
        if status.get("stage") not in ("queued","queued_resources"):
            job.update(resume=True,resume_origin_stage=status["stage"],resume_origin_env_step=status.get("env_step"),
                       resume_attempt=job.get("resume_attempt",0)+1)
        job=_bind(store,job,directory,contract,path,body["contract_sha256"],snapshot,"same_id_validation_only_repair")
        (directory/"stop.request").unlink(missing_ok=True)
        status.update(stage="queued_resources" if job.get("resource_hold") else "queued",pid=None,error=None,
                      source_repair=REPAIR_ID,execution_source=job["execution_source"],heartbeat_at=now(),
                      queue_reason="同ID验证验收执行源修复；原科学提交、训练状态和预算保持")
        atomic_json(directory/"status.json",status)
        job["source_repair_pending"]=False;atomic_json(directory/"job.json",job)
        return status
