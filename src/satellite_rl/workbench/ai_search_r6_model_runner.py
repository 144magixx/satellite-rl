"""Persistent source-admitted single learner or finite node24 worker."""
import os,threading,traceback
from pathlib import Path
from satellite_rl.training.run_monitor import atomic_json
from .storage import read_json,now,Store
from .geoload_runner import exclusive_lock
from . import frozen_round_tasks,coverage_ai_search_r6_model as policy

def run_job(directory,resume=False):
 directory=Path(directory);job=read_json(directory/'job.json',{});config=read_json(directory/'config.json',{});status=read_json(directory/'status.json',{});done=threading.Event()
 def update(**kw):status.update(kw,pid=os.getpid(),heartbeat_at=now());atomic_json(directory/'status.json',status)
 def heartbeat():
  while not done.wait(10):
   p=read_json(directory/'status_progress.json',{});update(env_step=p.get('update',0),update_step=p.get('update',0),completed_cases=p.get('update',status.get('completed_cases',0)),active_seconds=p.get('active_seconds'))
 try:
  with exclusive_lock(directory/'worker.lock'),exclusive_lock(Path(job['archive_root']).parent/'ops/compute.lock'):
   from satellite_rl.coverage.ai_search_r6.runtime_v3 import configure_v3_runtime
   atomic_json(directory/'runtime_v3_identity.json',configure_v3_runtime())
   frozen_round_tasks.verify(Store(job['archive_root'],source_root=job['source_root']),job,policy);update(stage='running');threading.Thread(target=heartbeat,daemon=True).start()
   if job['mode']=='train_setup':
    from scripts.stage1_ai_search_r6model.supply import prepare
    from .r6_model_io import signed_json,verify_model_source_equivalence
    refs=job['model_inputs'];initial=signed_json(refs['initialization_receipt'])
    if initial.get('passed')is not True:raise ValueError('source-bound model engineering admission required before supply')
    verify_model_source_equivalence(refs['source_equivalence'],refs['initialization_receipt'],initial)
    from .r6_model_train import MODEL_OWNER_ROOT
    from satellite_rl.coverage.ai_search_r6.features import record_sha
    MODEL_OWNER_ROOT.mkdir(parents=True,exist_ok=True);key=record_sha(job['model_identity']);owner_path=MODEL_OWNER_ROOT/(key+'.json')
    with exclusive_lock(MODEL_OWNER_ROOT/(key+'.lock')):
     owner=read_json(owner_path,{})
     if owner.get('supply_run_id',job['run_id'])!=job['run_id']:raise ValueError('single-model input supply already has another owner, no new-task regenerate')
     owner.update(model_identity=job['model_identity'],supply_run_id=job['run_id'],supply_executor_port=job['executor_port'],qualified_ref=refs['qualified']);atomic_json(owner_path,owner)
     result=prepare(refs['qualified'],refs['labels_gate'],directory/'supply',initial['feature_graph_source_SHA'])
    update(completed_cases=len(job['case_indices']))
   elif job['mode'].startswith('train'):
    from .r6_model_train import train
    result=train(job,config,directory);update(completed_cases=result['updates'],env_step=result['updates'],update_step=result['updates'])
   else:
    from .r6_model_node import node
    result=node(job,config,directory);update(completed_cases=len(result)*3)
   update(stage='completed',completed_at=now())
 except BaseException as error:atomic_json(directory/'failure.json',{'message':str(error),'traceback':traceback.format_exc(),'unknown_spend_preserved':True,'automatic_retry':False});update(stage='failed',error=str(error));raise
 finally:done.set()
