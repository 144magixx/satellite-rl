"""Durable geometry-only workbench worker, packed NPZ supply and mother batching."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time
import traceback
import numpy as np
from satellite_rl.training.run_monitor import atomic_json
from .storage import digest, read_json, now
from .coverage_geoload import GROUP, pin, validate_config

_MODEL_CACHE = {}

def cached_model(reference):
    """One immutable model load per pool worker, outside online search budget."""
    import torch
    from satellite_rl.coverage.geoload.models import make_model, MODEL_FORMAT
    path=Path(reference['path']).resolve();key=(str(path),reference['sha256'],'cuda')
    load_seconds=0.
    if key not in _MODEL_CACHE:
        started=time.perf_counter()
        if not torch.cuda.is_available():raise RuntimeError('formal MLM inference requires CUDA; no CPU fallback')
        state=torch.load(pin(reference),map_location='cpu',weights_only=False)
        if state.get('format')!=MODEL_FORMAT or state.get('update')!=10000 or state.get('from_scratch') is not True:
            raise ValueError('evaluation requires new fixed10000 endpoint')
        model=make_model(state.get('network'));model.load_state_dict(state['model']);model.to('cuda');model.eval()
        torch.cuda.synchronize();load_seconds=time.perf_counter()-started
        _MODEL_CACHE.clear();_MODEL_CACHE[key]=model
    return _MODEL_CACHE[key],load_seconds

@contextmanager
def exclusive_lock(path, *, shared=False, blocking=False):
    """Kernel lock outlives no process; a stale PID cannot confer ownership."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0); handle.write(b"\0"); handle.flush(); handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | (0 if blocking else fcntl.LOCK_NB))
        try: yield
        finally:
            if os.name == "nt":
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(handle, fcntl.LOCK_UN)

def load_cases(reference):
    path = pin(reference)
    with np.load(path, allow_pickle=False) as values:
        required = {"positions", "demands", "offsets", "case_ids", "mother_ids", "demand_profiles"}
        if not required.issubset(values.files):
            raise ValueError("G16 packed dataset fields missing")
        positions = values["positions"].copy(); demands = values["demands"].copy()
        offsets = values["offsets"].copy(); ids = values["case_ids"].astype(str)
        mothers = values["mother_ids"].astype(str); profiles = values["demand_profiles"].astype(str)
    count = len(ids)
    if (count != reference["count"] or len(set(ids)) != count or len(mothers) != count or
        len(profiles) != count or offsets.shape != (count+1,) or offsets[0] != 0 or
        offsets[-1] != len(demands) or np.any(np.diff(offsets) < 0) or
        positions.shape != (len(demands), 2)):
        raise ValueError("G16 packed dataset identity/shape/count invalid")
    groups = {}
    for i, identifier in enumerate(ids):
        a,b = int(offsets[i]),int(offsets[i+1])
        case = {"case_id": identifier, "mother_id": mothers[i], "demand_profile": profiles[i],
                "positions": positions[a:b], "demands": demands[a:b], "profile": "europe_v1"}
        groups.setdefault(mothers[i], []).append(case)
    return list(groups.values())

def _run_mother(cases, methods, search, mode, teacher_root=None, model_ref=None, output_root=None):
    # Each mother is copied to one worker; all profiles/methods share its initial
    # geometry but never share mutable algorithm state. No gzip parsing per case.
    from satellite_rl.coverage.geoload.core import Scene, initial_layout, evaluate_case, metrics
    from satellite_rl.coverage.geoload.config import GeometryConfig
    outputs = []
    initial = None
    for case in cases:
        scene = Scene(case["positions"], case["demands"], case["profile"])
        began = time.perf_counter()
        try:
            if initial is None: initial = initial_layout(scene, GeometryConfig())
            initial_seconds = time.perf_counter() - began
        except ValueError as error:
            for method in methods:
                outputs.append({"case_id": case["case_id"], "mother_id": case["mother_id"],
                    "demand_profile": case["demand_profile"], "method": method,
                    "outcome": "not_found", "error": str(error), "scientific_failure": False,
                    "initialization_seconds": time.perf_counter()-began})
            continue
        for method in methods:
            case_key=result_key(case["case_id"],method)
            if output_root:
                started=Path(output_root).parent/"started"/(case_key+".json")
                started.parent.mkdir(exist_ok=True)
                atomic_json(started,{"case_id":case["case_id"],"method":method,"started_at":now(),
                    "pid":os.getpid(),"search_limit":search,"status":"started"})
            began = time.perf_counter()
            model_load_seconds=0.
            if mode == "variable":
                from satellite_rl.coverage.geoload.classical import run_classical
                from satellite_rl.coverage.geoload.config import SearchConfig
                n0=len(initial.centers); cap=min(200,len(scene.demands));entries=[]
                quota0=run_classical(scene,initial,"QUOTA",SearchConfig(**search)).to_record()
                entries.append({"stage":"same_N_quota","target_N":n0,"outcome":"completed",**quota0})
                targets=sorted({min(cap,max(1,int(np.ceil(scale*n0)))) for scale in (.8,1.2)}-{n0})
                for target_n in targets:
                    constructed=run_classical(scene,initial,"VARIABLE",SearchConfig(**search,allow_variable_n=True,target_n=target_n)).to_record()
                    reached=len(constructed['layout']['centers'])==target_n
                    entries.append({"stage":"construct","target_N":target_n,"outcome":"completed" if reached else "not_found",
                                    "certified_infeasible":False,**constructed})
                    if reached:
                        from satellite_rl.coverage.geoload.core import Layout
                        quota=run_classical(scene,Layout.from_record(constructed['layout']),"QUOTA",SearchConfig(**search)).to_record()
                        entries.append({"stage":"target_quota","target_N":target_n,"outcome":"completed",**quota})
                    else:
                        entries.append({"stage":"target_quota","target_N":target_n,"outcome":"not_run_constructor_not_found",
                                        "budgets":{"attempted":0,"seconds":0}})
                result={"entries":entries,"N0":n0,"metrics":metrics(scene,initial),"layout":initial.to_record()}
            elif mode == "teacher":
                from satellite_rl.coverage.geoload.teacher import teacher_case
                target = Path(teacher_root) / result_key(case["case_id"], method)
                result = teacher_case({**case,"initial":initial.to_record()},target,search)
                for record in result["samples"]:
                    record["path"] = str(Path(target.name) / record["path"]).replace("\\", "/")
                result["teacher_shard"] = {"path": str(target/"manifest.json"), "sha256": digest(target/"manifest.json")}
            elif mode == "initial":
                result = {"layout": initial.to_record(), "metrics": metrics(scene,initial),
                          "journal": [], "budgets": {"attempted": 0}}
            elif method in ("MLM", "rule", "random"):
                from satellite_rl.coverage.geoload.classical import run_classical
                from satellite_rl.coverage.geoload.config import SearchConfig
                from satellite_rl.coverage.geoload.learning import orderer
                model = None
                if method == "MLM":
                    model,model_load_seconds=cached_model(model_ref)
                rank_journal=[]
                result = run_classical(scene,initial,"JOINT",SearchConfig(**search),
                     candidate_order=orderer(method,model=model,seed=51001,journal=rank_journal,
                                             device='cuda' if method=='MLM' else 'cpu')).to_record()
                result["ranking_journal"] = rank_journal
                result['inference_device']='cuda' if method=='MLM' else 'cpu'
            else:
                result = evaluate_case({**case, "initial": initial.to_record()}, method, search)
            row={"case_id": case["case_id"], "mother_id": case["mother_id"],
                "demand_profile": case["demand_profile"], "method": method, "outcome": "completed",
                "initialization_seconds": initial_seconds, "online_seconds": time.perf_counter()-began-model_load_seconds,
                "model_cold_load_seconds":model_load_seconds,"model_cache_hit":method=='MLM' and model_load_seconds==0.,
                "end_to_end_seconds":initial_seconds+time.perf_counter()-began,
                **result}
            if output_root:
                atomic_json(Path(output_root)/(case_key+".json"),row)
                atomic_json(started,{"case_id":case["case_id"],"method":method,"finished_at":now(),
                    "pid":os.getpid(),"status":"completed","result_sha256":digest(Path(output_root)/(case_key+".json"))})
            outputs.append(row)
    return outputs

def _train(directory, job, config, update, resume):
    from satellite_rl.coverage.geoload.learning import train, orderer
    from satellite_rl.coverage.geoload.classical import run_classical
    from satellite_rl.coverage.geoload.core import Scene, initial_layout
    from satellite_rl.coverage.geoload.config import SearchConfig
    groups=load_cases(config["datasets"]["validation"])
    if sum(len(g) for g in groups)!=80: raise ValueError("MLM fixed validation requires80 cases")
    protocol={**config["training"],"scientific_commit":job["scientific_commit"],
              "teacher_source_sha256":job["references"]["teacher"]["sha256"],
              "validation_sha256":config["datasets"]["validation"]["sha256"]}
    def validation(model,node,node_dir):
        node_dir.mkdir(parents=True,exist_ok=True)
        index=[]
        for group in groups:
            for case in group:
                scene=Scene(case["positions"],case["demands"]);initial=initial_layout(scene)
                rank=[]
                result=run_classical(scene,initial,"JOINT",SearchConfig(**config["search"]),
                    candidate_order=orderer("MLM",model=model,journal=rank,device="cuda")).to_record()
                key=result_key(case["case_id"],"MLM")
                atomic_json(node_dir/(key+".json"),{**result,"case_id":case["case_id"],
                    "mother_id":case["mother_id"],"demand_profile":case["demand_profile"],"ranking_journal":rank})
                index.append({"case_id":case["case_id"],"sha256":digest(node_dir/(key+".json"))})
        atomic_json(node_dir/"node_done.json",{"update":node,"count":80,"index":index,
                    "validation_sha256":config["datasets"]["validation"]["sha256"]})
    def progress(cursor,loss):
        update(update_step=cursor,loss=loss,budget_updates=10000)
        if (directory/"stop.request").exists():
            # train saves a complete checkpoint every100 updates; only stop at
            # those committed boundaries, preserving optimizer/RNG/log prefix.
            if cursor%100==0: raise InterruptedError("workbench stop at checkpoint boundary")
    result=train(job["references"]["teacher"]["path"],Path(job["model_dir"]),protocol,
                 device="cuda",resume=resume,validation=validation,progress=progress)
    final=Path(job["model_dir"])/"final.pt"
    atomic_json(directory/"artifacts.json",{"group":GROUP,"completed":True,"updates":10000,
        "model":str(final),"model_sha256":digest(final),"training":result,
        "scientific_commit":job["scientific_commit"]})
    update(stage="completed",update_step=10000,completed_at=now())

def result_key(case_id, method):
    import hashlib
    return hashlib.sha256((case_id+"\0"+method).encode()).hexdigest()

def run_job(directory, *, resume=False):
    directory = Path(directory).resolve()
    job = read_json(directory / "job.json", {}); config = validate_config(read_json(directory / "config.json", {}))
    if job.get("experiment_id") != GROUP: raise ValueError("wrong geometry worker group")
    status = read_json(directory / "status.json", {})
    guard = threading.RLock(); quit_heartbeat = threading.Event()
    def update(**fields):
        with guard:
            status.update(fields, pid=os.getpid(), heartbeat_at=now())
            atomic_json(directory / "status.json", status)
    def heartbeat():
        while not quit_heartbeat.wait(10): update()
    try:
        # Shared lane lock allows bounded train/teacher/variable parallelism;
        # final test obtains an exclusive lock, including all child processes.
        # This is a kernel lock, not a stale marker or guessed PID condition.
        thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
        update(stage="starting",queue_reason="等待本端计算排他锁" if job["dataset_role"]=="test" else None)
        lane_lock=Path(job["archive_root"]).parent/"ops"/"compute.lock"
        with exclusive_lock(directory / "worker.lock"), exclusive_lock(lane_lock,shared=job["dataset_role"]!="test",blocking=True):
            # Recheck every frozen input and source file before executing science.
            for ref in job["references"].values(): pin(ref)
            manifest = read_json(Path(job["source_root"]) / "source_manifest.json", {})
            if manifest.get("git_commit") != job["scientific_commit"]:
                raise ValueError("frozen scientific source manifest differs")
            for name, expected in manifest.get("files", {}).items():
                if digest(Path(job["source_root"]) / name) != expected:
                    raise ValueError("frozen scientific source file differs: " + name)
            from .runner import strict_cuda_preflight
            atomic_json(directory / "preflight.json", strict_cuda_preflight(job))
            update(stage="training" if job["kind"] == "train" else "running", error=None, started_at=status.get("started_at", now()))
            if job["kind"] == "train":
                return _train(directory,job,config,update,resume)
            if job['mode']=='teacher_merge':
                import shutil
                target=directory/'merged_teacher';target.mkdir(exist_ok=True)
                cases=[];samples=[]
                for source_index,reference in enumerate(job['teacher_sources']):
                    source=pin(reference);manifest=read_json(source,{})
                    if manifest.get('dataset_sha256')!=config['datasets']['train']['sha256']:
                        raise ValueError('teacher input provenance differs')
                    cases.extend(manifest['cases'])
                    for sample in manifest['samples']:
                        original=(source.parent/sample['path']).resolve()
                        if not original.is_relative_to(source.parent.resolve()):raise ValueError('teacher shard escape')
                        pin({'path':str(original),'sha256':sample['sha256']})
                        relative=Path('shards')/str(source_index)/sample['path']
                        copied=target/relative;copied.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,copied)
                        samples.append({**sample,'path':relative.as_posix()})
                if len(cases)!=config['datasets']['train']['count'] or len({c['case_id'] for c in cases})!=len(cases):
                    raise ValueError('canonical teacher cases incomplete or duplicated')
                samples.sort(key=lambda s:(s['mother_id'],s['profile'],s['path'].split('/')[-1]))
                cases.sort(key=lambda s:s['case_id'])
                merged=target/'teacher_manifest.json'
                atomic_json(merged,{'group':GROUP,'status':'completed','samples':samples,'cases':cases,
                    'count':len(cases),'scientific_commit':job['scientific_commit'],
                    'dataset_sha256':config['datasets']['train']['sha256'],'sources':job['teacher_sources']})
                atomic_json(directory/'artifacts.json',{'group':GROUP,'completed':True,
                    'teacher':{'path':str(merged),'sha256':digest(merged)},'count':len(cases)})
                update(stage='completed',completed_at=now());return
            if job["mode"] == "dataset":
                from satellite_rl.coverage.geoload.dataset import prepare_dataset
                record=prepare_dataset(job["dataset_spec"],directory/"prepared_dataset",job["source_root"],
                    progress=update,stopped=lambda:(directory/"stop.request").exists())
                atomic_json(directory/"artifacts.json",record)
                if record.get("status") == "data_unavailable":
                    update(stage="failed",error="official data unavailable; retained source evidence",scientific_failure=False)
                elif record.get("status") == "stopped":
                    update(stage="stopped",stopped_at=now())
                else:
                    update(stage="completed",completed_at=now())
                return
            groups = load_cases(config["datasets"][job["dataset_role"]])
            if job.get("mother_indices") is not None:
                groups=[groups[i] for i in job['mother_indices']]
            methods = job["methods"]
            output = directory / "cases"; output.mkdir(exist_ok=True)
            existing = {}
            journal = read_json(directory / "case_index.json", {})
            for key, record in journal.items():
                path = output / (key + ".json")
                if not path.is_file() or digest(path) != record["sha256"]:
                    raise ValueError("completed result SHA differs; cannot silently repeat case")
                existing[key] = record
            for raw in output.glob("*.json"):
                if raw.stem not in existing:
                    row=read_json(raw,{})
                    if result_key(row.get("case_id",""),row.get("method","")) != raw.stem:
                        raise ValueError("unindexed result identity differs")
                    existing[raw.stem]={"case_id":row["case_id"],"method":row["method"],
                        "sha256":digest(raw),"outcome":row["outcome"],"recovered_atomic_tail":True}
            unknown=[p.name for p in (directory/"started").glob("*.json") if p.stem not in existing]
            if unknown:
                atomic_json(directory/"unknown_budget.json",{"category":"interrupted_case_budget_unknown",
                    "case_records":unknown,"scientific_failure":False,"automatic_repeat_forbidden":True})
                raise ValueError("interrupted case budget requires audit; no silent scientific repetition")
            pending = []
            for group in groups:
                keys = [result_key(c["case_id"],m) for c in group for m in methods]
                if not all(k in existing for k in keys):
                    # Mother completion is atomically indexed; partial completed
                    # mothers are not rerun, preserving their budget accounting.
                    for c in group:
                        missing = [m for m in methods if result_key(c["case_id"],m) not in existing]
                        if missing: pending.append(([c],missing))
            search = dict(config["search"])
            if job["mode"] == "teacher": search.update(seconds=10., max_attempts=20480)
            if job["mode"] == "reference":
                search.update(seconds=30.,max_attempts=61440,max_geometry_calls=100000,
                    checkpoint_seconds=(2.,10.,30.),checkpoint_attempts=(4096,20480,61440))
            total = sum(len(g)*len(methods) for g in groups)
            update(total_cases=total, completed_cases=len(existing))
            # Newly started mothers stay together; resume only runs missing case
            # keys. Test uses one worker to retain strict interleaving/timing.
            if not existing:
                pending = [(group,methods) for group in groups]
            workers = 1 if job["dataset_role"] in ("test","validation") or job['mode']=='variable' else config["workers"]
            update(case_workers=workers,inference_device='cuda' if 'MLM' in methods else 'cpu',
                   worker_start_method='spawn',cold_model_load_account='separate_and_end_to_end')
            import multiprocessing
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                teacher_root=directory/"teacher";teacher_root.mkdir(exist_ok=True)
                futures = {pool.submit(_run_mother,g,m,search,job["mode"],str(teacher_root),job["references"].get("model"),str(output)): (g,m) for g,m in pending}
                for future in as_completed(futures):
                    rows = future.result()
                    for row in rows:
                        key = result_key(row["case_id"],row["method"])
                        atomic_json(output / (key+".json"), row)
                        existing[key] = {"case_id": row["case_id"], "method": row["method"],
                                         "sha256": digest(output / (key+".json")), "outcome": row["outcome"]}
                    atomic_json(directory / "case_index.json", existing)
                    update(completed_cases=len(existing))
                    if (directory / "stop.request").exists():
                        for pending_future in futures: pending_future.cancel()
                        update(stage="stopped"); return
            if len(existing) != total: raise ValueError("result matrix incomplete")
            if job["mode"] == "teacher":
                samples=[];teacher_cases=[]
                for key,item in existing.items():
                    row=read_json(output/(key+".json"),{})
                    samples.extend(row.get("samples",[]))
                    teacher_cases.append({"case_id":row["case_id"],"outcome":row["outcome"],
                        "budget":row.get("budget"),"teacher_shard":row.get("teacher_shard")})
                atomic_json(teacher_root/"teacher_manifest.json",{"group":GROUP,"status":"completed",
                    "samples":samples,"cases":teacher_cases,"count":total,"scientific_commit":job["scientific_commit"],
                    "dataset_sha256":config["datasets"][job["dataset_role"]]["sha256"]})
            atomic_json(directory / "artifacts.json", {"group": GROUP, "cases": str(output),
                "case_index": str(directory / "case_index.json"), "case_index_sha256": digest(directory / "case_index.json"),
                "completed": True, "count": total, "scientific_commit": job["scientific_commit"],
                "dataset_sha256": config["datasets"][job["dataset_role"]]["sha256"]})
            if job['mode']=='teacher':
                artifacts=read_json(directory/'artifacts.json',{})
                artifacts['teacher']={'path':str(teacher_root/'teacher_manifest.json'),'sha256':digest(teacher_root/'teacher_manifest.json')}
                atomic_json(directory/'artifacts.json',artifacts)
            update(stage="completed", completed_at=now())
    except InterruptedError:
        update(stage="stopped", stopped_at=now())
    except BaseException as error:
        atomic_json(directory / "failure.json", {"run_id": job.get("run_id"), "time": now(),
            "error": str(error), "traceback": traceback.format_exc(), "category": "execution_error",
            "scientific_failure": False, "last_status": status})
        update(stage="failed", error=str(error), attention_required=True)
        raise
    finally: quit_heartbeat.set()
