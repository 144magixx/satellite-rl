"""Control-only queued DQN migration and SHA-bound read-only peer catalog."""
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import shutil

from satellite_rl.training.run_monitor import atomic_json
from .storage import digest,inside,now,read_json,validate_id

SCIENCE="2fc16884078ad64b97333013737fef5cd9294626"
EXECUTION="fbf79047ea859b91d8c7ee1decddd53a1436c359"
FORMAT="fair20_queued_dqn_migration_v1"


def port(store):
    declared=os.environ.get("SATELLITE_RL_F20_EXECUTOR_PORT")
    actual=int(store.archive_root.parent.name) if store.archive_root.parent.name in ("20418","20474") else None
    value=int(declared) if declared is not None else actual or 20474
    if value not in (20418,20474) or actual is not None and value!=actual:
        raise ValueError("F20 executor port contradicts its archive lane")
    return value


def base(store):
    return store.archive_root.parent.parent if store.archive_root.parent.name in ("20418","20474") else store.archive_root.parent


def ownership_root(store):
    return base(store)/"fair20_ownership"


@contextmanager
def owner_lock(store,identifier):
    root=ownership_root(store);root.mkdir(parents=True,exist_ok=True)
    path=root/(validate_id(identifier)+".lock")
    with path.open("a+b") as handle:
        if os.name=="nt":
            import msvcrt
            if path.stat().st_size==0:handle.write(b"0");handle.flush()
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_LOCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX)
        try:yield root/(identifier+".json")
        finally:
            if os.name=="nt":handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


@contextmanager
def launch_ownership(store,job):
    """Lock extends through Popen and launch publication, closing export races."""
    with owner_lock(store,job["run_id"]) as path:
        if port(store)==20474 and (store.job_dir(job["run_id"])/"migration"/"export_receipt.json").is_file():
            raise ValueError("source export journal permanently revokes this lane's scheduling")
        owner=read_json(path)
        if owner is None:
            owner={"format":"fair20_owner_v1","run_id":job["run_id"],"owner_port":port(store),
                   "owner_archive_root":str(store.archive_root),"state":"owned","epoch":0,"at":now()}
            atomic_json(path,owner)
        if owner.get("owner_port")!=port(store) or owner.get("state")!="owned" or owner.get("owner_archive_root")!=str(store.archive_root):
            raise ValueError("F20 task scheduling belongs to another lane or an incomplete migration")
        yield owner


def rejected_stage(store,job):
    owner=read_json(ownership_root(store)/(validate_id(job["run_id"])+".json"),{})
    return "queued_resources" if owner.get("owner_port")==port(store) and owner.get("state")!="owned" else "migrated"


def _pin(path,sha):
    path=Path(path).resolve(strict=True)
    if not path.is_file() or digest(path)!=sha:raise ValueError("migration/peer source SHA differs: "+str(path))
    return path


def contract(store,body):
    value=read_json(_pin(body["contract_path"],body["contract_sha256"]),{})
    if (value.get("format")!=FORMAT or value.get("source_port")!=20474 or value.get("target_port")!=20418 or
            value.get("scientific_commit")!=SCIENCE or value.get("execution_commit") not in (SCIENCE,EXECUTION)):
        raise ValueError("migration is limited to the original queued DQN A from74 to18")
    for key,expected in (("source_archive_root",base(store)/"20474"/"archive"),("target_archive_root",base(store)/"20418"/"archive")):
        if Path(value.get(key,"")).resolve()!=expected.resolve():raise ValueError("migration archive paths leave the isolated F20 lanes")
    validate_id(value["run_id"]);validate_id(value["request_id"])
    return value


def export_job(scheduler,body):
    from .runtime import pid_alive
    if set(body)!={"contract_path","contract_sha256"}:raise ValueError("export accepts only frozen migration contract reference")
    store=scheduler.store;value=contract(store,body)
    if port(store)!=20474:raise ValueError("only source74 can export the queued task")
    directory=store.job_dir(value["run_id"])
    with scheduler.lock,store.lock,owner_lock(store,value["run_id"]) as owner_path:
        receipt_path=directory/"migration"/"export_receipt.json"
        prior=read_json(receipt_path)
        if prior:
            if prior["contract_sha256"]!=body["contract_sha256"]:raise ValueError("migration contract cannot change")
            owner=read_json(owner_path,{})
            if owner.get("contract_sha256")!=body["contract_sha256"] or owner.get("owner_port")!=20418:
                atomic_json(owner_path,{"format":"fair20_owner_v1","run_id":value["run_id"],"owner_port":20418,
                    "owner_archive_root":value["target_archive_root"],"state":"adoption_pending","epoch":owner.get("epoch",0)+1,
                    "contract_sha256":body["contract_sha256"],"at":now()})
            status=read_json(directory/"status.json",{})
            if status.get("stage")!="migrated":
                status.update(stage="migrated",migration_target_port=20418,migration_contract_sha256=body["contract_sha256"],
                              queue_reason="继续已冻结的queued0迁移撤权事务",heartbeat_at=now())
                atomic_json(directory/"status.json",status)
            return {**prior,"export_receipt_path":str(receipt_path),"export_receipt_sha256":digest(receipt_path)}
        job=read_json(_pin(directory/"job.json",value["source_job_sha256"]))
        status=read_json(_pin(directory/"status.json",value["source_status_sha256"]))
        controls=read_json(_pin(directory/"config.json",value["config_sha256"]))
        launch=read_json(directory/"launch.json",{})
        if (job.get("experiment_id")!="S2-F20" or job.get("algorithm")!="dqn" or job.get("candidate")!="A" or
                job.get("mode")!="shorttune" or controls.get("candidate")!="A" or controls.get("mode")!="shorttune" or
                job.get("request_id")!=value["request_id"] or job.get("git_commit")!=SCIENCE or
                job.get("execution_source",{}).get("execution_commit",job.get("git_commit"))!=value["execution_commit"] or
                job.get("frozen_input_sha256")!=value["frozen_input_sha256"] or
                status.get("stage") not in ("queued","queued_resources") or status.get("env_step",0)!=0 or
                status.get("updates",status.get("update_step",0))!=0 or launch or pid_alive(status.get("pid")) or
                (directory/"checkpoint.pt").exists()):
            raise ValueError("DQN migration requires untouched queued0 task, no worker or learning state")
        for name in ("traces.jsonl","episodes.jsonl","updates.jsonl","worker.log"):
            if (directory/name).exists() and (directory/name).stat().st_size:raise ValueError("queued migration has executed work")
        owner=read_json(owner_path)
        if owner and (owner.get("owner_port")!=20474 or owner.get("state")!="owned"):raise ValueError("source has already lost task ownership")
        preserved=directory/"migration"/"origin";preserved.mkdir(parents=True,exist_ok=True)
        for name in ("job.json","status.json","config.json"):shutil.copyfile(directory/name,preserved/name)
        frozen_contract=directory/"migration"/"contract.json";shutil.copyfile(body["contract_path"],frozen_contract)
        receipt={"format":FORMAT,"run_id":job["run_id"],"request_id":job["request_id"],"contract_sha256":body["contract_sha256"],
                 "source_archive_root":str(store.archive_root),"target_archive_root":value["target_archive_root"],
                 "source_port":20474,"target_port":20418,"source_dispatch_revoked":True,"at":now(),
                 "original_job_path":str(preserved/"job.json"),"original_job_sha256":digest(preserved/"job.json"),
                 "original_status_path":str(preserved/"status.json"),"original_status_sha256":digest(preserved/"status.json"),
                 "config_sha256":value["config_sha256"],"frozen_input_sha256":value["frozen_input_sha256"]}
        receipt["scientific_commit"]=SCIENCE;receipt["execution_commit"]=value["execution_commit"]
        atomic_json(receipt_path,receipt)
        # Shared owner moves to a non-dispatchable target state before source
        # state changes. A crash here cannot leave a source worker dispatchable.
        atomic_json(owner_path,{"format":"fair20_owner_v1","run_id":job["run_id"],"owner_port":20418,
                    "owner_archive_root":value["target_archive_root"],"state":"adoption_pending","epoch":(owner or {}).get("epoch",0)+1,
                    "contract_sha256":body["contract_sha256"],"at":now()})
        status.update(stage="migrated",executor_port=20474,migration_target_port=20418,migration_contract_sha256=body["contract_sha256"],
                      queue_reason="DQN queued0迁移已撤销74调度权，等待18接管",heartbeat_at=now())
        atomic_json(directory/"status.json",status)
        return {**receipt,"export_receipt_path":str(receipt_path),"export_receipt_sha256":digest(receipt_path)}


def adopt_job(scheduler,body):
    if set(body)!={"contract_path","contract_sha256","export_receipt_path","export_receipt_sha256"}:
        raise ValueError("adopt needs fixed contract and source dispatch revocation receipt")
    store=scheduler.store;value=contract(store,body)
    if port(store)!=20418:raise ValueError("only target18 can adopt this queued task")
    expected=Path(value["source_archive_root"])/"outputs"/"workbench"/"jobs"/value["run_id"]
    if Path(body["export_receipt_path"]).resolve()!=(expected/"migration"/"export_receipt.json").resolve():raise ValueError("export receipt path differs")
    receipt=read_json(_pin(body["export_receipt_path"],body["export_receipt_sha256"]))
    if receipt.get("contract_sha256")!=body["contract_sha256"] or receipt.get("source_dispatch_revoked") is not True:
        raise ValueError("source scheduling has not been revoked")
    with scheduler.lock,store.lock,owner_lock(store,value["run_id"]) as owner_path:
        owner=read_json(owner_path,{})
        if owner.get("owner_port")!=20418 or owner.get("contract_sha256")!=body["contract_sha256"]:
            raise ValueError("target does not hold the migration ownership token")
        destination=store.job_dir(value["run_id"])
        existing=read_json(destination/"job.json")
        if existing:
            if existing.get("migration",{}).get("contract_sha256")!=body["contract_sha256"]:raise ValueError("target task ID already belongs to different work")
            if owner.get("state")=="adoption_pending":
                owner.update(state="owned",adopted_job_sha256=digest(destination/"job.json"),at=now());atomic_json(owner_path,owner)
            return {"job":existing,"status":read_json(destination/"status.json",{})}
        if owner.get("state")!="adoption_pending" or read_json(expected/"status.json",{}).get("stage")!="migrated":
            raise ValueError("migration is not at its exported non-dispatchable boundary")
        original=read_json(_pin(receipt["original_job_path"],value["source_job_sha256"]))
        _pin(expected/"config.json",value["config_sha256"])
        destination.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(expected/"config.json",destination/"config.json")
        model_dir=store.models_root/value["run_id"];model_dir.mkdir(exist_ok=True)
        job=deepcopy(original)
        job.update(archive_root=str(store.archive_root),model_dir=str(model_dir),executor_port=20418,resource_hold=True)
        job["migration"]={**receipt,"export_receipt_path":body["export_receipt_path"],"export_receipt_sha256":body["export_receipt_sha256"],
                          "model_dir_before":original["model_dir"],"input_sources":"original74 read-only SHA-bound frozen files"}
        atomic_json(destination/"migration.json",job["migration"])
        atomic_json(destination/"status.json",{"run_id":job["run_id"],"stage":"queued_resources","env_step":0,"updates":0,
                    "update_step":0,"episode":0,"budget_env_steps":32768,"executor_port":20418,"heartbeat_at":now(),
                    "queue_reason":"74原ID DQN A由18接管，等待真实18资源放行"})
        atomic_json(destination/"job.json",job)
        owner.update(state="owned",adopted_job_sha256=digest(destination/"job.json"),at=now());atomic_json(owner_path,owner)
        return {"job":job,"status":read_json(destination/"status.json")}


def _peer_path(store,identifier):return store.root/"fair20_peer_catalog"/(validate_id(identifier)+".json")


def import_peer(store,body):
    if set(body)!={"source_ref"}:raise ValueError("peer import accepts only a SHA-bound source_ref")
    reference=deepcopy(body["source_ref"]);identifier=validate_id(reference["run_id"])
    directory=peer_directory(store,reference)
    record={"source_ref":reference,"read_only":True,"imported_at":now(),"path":str(directory)}
    path=_peer_path(store,identifier);path.parent.mkdir(exist_ok=True)
    old=read_json(path)
    if old and old["source_ref"]!=reference:raise ValueError("peer task provenance cannot silently change")
    if not old:atomic_json(path,record)
    return old or record


def peer_directory(store,reference):
    archive=Path(reference["archive_root"]).resolve()
    if archive not in {(base(store)/str(value)/"archive").resolve() for value in (20418,20474)} or archive==store.archive_root:
        raise ValueError("peer must be the other approved isolated F20 archive")
    directory=archive/"outputs"/"workbench"/"jobs"/validate_id(reference["run_id"])
    for name in ("job","status","config","results"):_pin(directory/(name+".json"),reference[name+"_sha256"])
    job=read_json(directory/"job.json");status=read_json(directory/"status.json");config=read_json(directory/"config.json")
    if (job.get("run_id")!=reference["run_id"] or status.get("run_id")!=reference["run_id"] or
            job.get("experiment_id")!="S2-F20" or job.get("git_commit")!=SCIENCE or status.get("stage")!="completed"):
        raise ValueError("peer is not a completed original-science F20 dependency")
    role=reference.get("role")
    if role=="prepare":valid=job.get("mode") in ("prepare","sourceprep")
    elif role=="shorttune":valid=job.get("mode")=="shorttune" and job.get("algorithm")=="dqn" and status.get("env_step")==32768
    elif role=="model":valid=job.get("mode")=="train" and job.get("algorithm")=="dqn" and status.get("env_step")==400000
    else:valid=False
    if not valid:raise ValueError("peer role is outside the DQN parallel dependency scope")
    actual_execution=job.get("execution_source",{}).get("execution_commit",job.get("git_commit"))
    if role=="shorttune" and actual_execution!=SCIENCE:raise ValueError("shorttune peer must preserve original 2fc execution")
    if role=="model" and actual_execution!=EXECUTION:raise ValueError("formal model peer must use approved fbf execution")
    if config.get("research_stage")!="fair20":raise ValueError("peer configuration is not F20")
    return directory


def resolve_peer(store,identifier):
    record=read_json(_peer_path(store,identifier))
    return peer_directory(store,record["source_ref"]) if record else None


def peer_records(store):
    result=[]
    for path in sorted((store.root/"fair20_peer_catalog").glob("*.json")):
        record=read_json(path);directory=peer_directory(store,record["source_ref"])
        result.append({"job":read_json(directory/"job.json"),"status":read_json(directory/"status.json"),
                       "read_only":True,"source_ref":record["source_ref"]})
    return result
