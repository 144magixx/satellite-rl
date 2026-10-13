"""Persistent bounded R7 worker; state is durable and failures stop successors."""
import os,threading,traceback
from pathlib import Path
from satellite_rl.training.run_monitor import atomic_json
from .storage import read_json,now,Store
from .geoload_runner import exclusive_lock
from . import frozen_round_tasks,coverage_ai_search_r7 as policy

def run_job(directory,resume=False):
    directory=Path(directory);job=read_json(directory/'job.json',{});config=read_json(directory/'config.json',{});status=read_json(directory/'status.json',{});done=threading.Event()
    if resume:raise ValueError('root same-ID reconciliation must be frozen before R7 recovery')
    def update(**kw):status.update(kw,pid=os.getpid(),heartbeat_at=now());atomic_json(directory/'status.json',status)
    def heartbeat():
        while not done.wait(10):
            progress=read_json(directory/'status_progress.json',{});parallel=read_json(directory/'parallel_ledger.json',{})
            update(env_step=progress.get('update',0),update_step=progress.get('update',0),active_seconds=progress.get('active_seconds'),completed_cases=progress.get('update',parallel.get('completed_cases',status.get('completed_cases',0))))
    try:
        with exclusive_lock(directory/'worker.lock'),exclusive_lock(Path(job['archive_root']).parent/'ops/compute.lock'):
            frozen_round_tasks.verify(Store(job['archive_root'],source_root=job['source_root']),job,policy);update(stage='running');threading.Thread(target=heartbeat,daemon=True).start()
            mode=job['mode'];refs=job['model_inputs']
            if mode in ('data_train_dev','labels'):
                from .r7_parallel import parallel_cases
                result=parallel_cases(job,config,directory);completed=len(result['cases'])
            else:
                from satellite_rl.coverage.ai_search_r6.runtime_v3 import configure_v3_runtime
                runtime=configure_v3_runtime();atomic_json(directory/'runtime_v3_identity.json',runtime)
                if mode=='train_setup':
                    from .r7_model_io import signed_json,verify_model_source_equivalence
                    initial=signed_json(refs['initialization_receipt']);verify_model_source_equivalence(refs['source_equivalence'],refs['initialization_receipt'],initial)
                    if runtime!=initial['training_runtime_identity']:raise ValueError('supply actual74 qualified runtime identity')
                    from scripts.stage1_ai_search_r7.supply import prepare
                    from .r7_model_train import MODEL_OWNER_ROOT
                    from satellite_rl.coverage.ai_search_r6.features import record_sha
                    MODEL_OWNER_ROOT.mkdir(parents=True,exist_ok=True);key=record_sha(job['model_identity']);owner_path=MODEL_OWNER_ROOT/(key+'.json')
                    with exclusive_lock(MODEL_OWNER_ROOT/(key+'.lock')):
                        owner=read_json(owner_path,{})
                        if owner.get('supply_run_id',job['run_id'])!=job['run_id']:raise ValueError('original supply one owner; no regeneration')
                        owner.update(model_identity=job['model_identity'],supply_run_id=job['run_id'],supply_executor_port=job['executor_port'],qualified_ref=refs['qualified']);atomic_json(owner_path,owner)
                        prepare(refs['qualified'],refs['labels_gate'],directory/'supply',initial['feature_graph_source_SHA'])
                    completed=len(job['case_indices'])
                elif mode=='train2000':
                    from .r7_model_train import train
                    result=train(job,config,directory);completed=result['updates']
                elif mode=='node2000':
                    from .r7_node import node
                    result=node(job,config,directory);completed=len(result['cases'])*4
                else:raise ValueError('later phase dispatch sealed')
            update(stage='completed',completed_cases=completed,completed_at=now())
    except BaseException as error:
        atomic_json(directory/'failure.json',dict(message=str(error),traceback=traceback.format_exc(),unknown_spend_preserved=True,automatic_retry=False));update(stage='failed',error=str(error));raise
    finally:done.set()
