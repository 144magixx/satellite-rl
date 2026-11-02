"""One exact20 serial pilot, immutable completed slots and never replayed unknown D5."""
import argparse,os,sys,time,subprocess
from pathlib import Path
from .contracts import read,validate,atomic,immutable,pin,digest,identity,order
from .deadline import boundary,before_start,storage_boundary,Paused,bind_parent

def run(plan_ref,review_ref):
 began=time.monotonic();bind_parent('R8_SERVICE_PID');p=validate(plan_ref,review_ref);boundary(p,'worker_before_runtime')
 root=Path(p['archive_root']);recovery=p.get('operations_recovery');epoch=Path(recovery['epoch_root']) if recovery else root
 if recovery:
  from .recovery import validate_recovery
  validate_recovery(p)
 epoch.mkdir(parents=True,exist_ok=True);atomic(epoch/'source_input_validation_timer.json',dict(observed_seconds=time.monotonic()-began,scope='parent binding, full source/input/exact plan admission',plan_ref=plan_ref,timer_receipt_IO_additional=True));atomic(epoch/'worker_preflight.json',dict(actual_worker=identity(os.getpid()),boundary=boundary(p,'worker_before_imports'),plan_ref=plan_ref))
 source=Path(p['original_source_manifest_ref']['path']).parent;sys.path[:0]=[str(source/'src'),str(source)]
 from satellite_rl.workbench.r5_stage_guard import run_stage
 from satellite_rl.workbench.r6_io import read_asset,write_asset
 from satellite_rl.workbench.geoload_runner import exclusive_lock
 from scripts.stage1_ai_search_r4.audit import audit_layout
 from scripts.stage1_ai_search_r5.audit import audit_decoder_ledger
 with exclusive_lock(root/'worker.lock'),exclusive_lock(root/'compute.lock'):
  if recovery:
   from .recovery import validate_recovery
   validate_recovery(p);epoch.mkdir(parents=True,exist_ok=True)
   status_bytes=pin(recovery['original_failed_status_ref']).read_bytes()
   with (epoch/'original_failed_status.json').open('xb') as f:f.write(status_bytes);f.flush();os.fsync(f.fileno())
   index=read(recovery['original_slot_index_ref']).copy()
  else:index=read(dict(path=str(root/'slot_index.json'),sha256=digest(root/'slot_index.json'))) if (root/'slot_index.json').exists() else {}
  new_calls=dict(local=0,D5=0)
  atomic(root/'status.json',dict(stage='running',run_id=p['run_id'],actual_worker=identity(os.getpid()),completed_slots=len(index),total_slots=60))
  slot_started=None;audit_started=None;target=None
  try:
   for case in p['cases']:
    i=case['case_index']
    for arm in order(i):
     boundary(p,'queue_before_slot');storage_boundary(p,'queue_before_slot');key=f'{i}:{arm}';target=epoch/'slots'/str(i)/arm
     if key in index:
      pin(index[key]);continue
     target.mkdir(parents=True,exist_ok=True);local=target/'local';d5=target/'D5';slot_started=time.monotonic();audit_started=None
     def stop():
      try:boundary(p,'owned_child_monitor');return False
      except Paused:return True
     def spawn(command,**kwargs):
      before_start(p,'child_before_spawn');env=dict(os.environ);env['R8_PARENT_PID']=str(os.getpid());env['PYTHONDONTWRITEBYTECODE']='1';return subprocess.Popen(command,env=env,**kwargs)
     def guarded(stage,request):
      cap=recovery['remaining_actual_calls_max'] if recovery else dict(local=60,D5=60)
      if new_calls[stage]>=cap[stage]:raise ValueError('actual new epoch call reservation cap')
      before_start(p,'before_'+stage+'_intent');guard_started=time.monotonic();folder=target/stage;folder.mkdir(exist_ok=True);request_path=folder/'request.json'
      if request_path.exists():
       if read(dict(path=str(request_path),sha256=digest(request_path)))!=request:raise ValueError('immutable request differs')
      else:immutable(request_path,request)
      command=[sys.executable,'-B','-u','-m','scripts.stage1_ai_search_r8.child','--request',str(request_path),'--directory',str(folder)]
      new_calls[stage]+=1;atomic(epoch/'actual_new_epoch_reservations.json',dict(new_calls=new_calls,cap=cap,operations_recovery_ref=recovery.get('contract_ref') if recovery else None,actual_counts_are_outer_reservations_not_algorithm_entry=True));receipt=run_stage(command,folder,15,2147483648,stop=stop,popen=spawn)
      atomic(folder/'parent_guard_return_timer.json',dict(observed_seconds=time.monotonic()-guard_started,scope='request IO/native guard through return before output read/audit',guard_receipt_ref=dict(path=str(folder/'guard_receipt.json'),sha256=digest(folder/'guard_receipt.json')),timer_receipt_IO_additional=True))
      if receipt['status']!='completed':raise RuntimeError('unknown/hard unpublished '+stage+'; no automatic retry')
      asset_ref=read(receipt['output_ref'])['asset_ref'];read_started=time.monotonic();value=read_asset(asset_ref);atomic(folder/'output_read_timer.json',dict(observed_seconds=time.monotonic()-read_started,asset_ref=asset_ref,timer_receipt_IO_additional=True));return value,receipt,asset_ref
     base=dict(plan_ref=plan_ref,case_index=i,arm=arm)
     local_value,lg,local_ref=guarded('local',{**base,'stage':'local','state_path':str(target/'local_state/state.json'),'resume_admission':p.get('local_resume_admissions',{}).get(key)})
     if local_value['status']=='paused':raise Paused('durable local paused, no D5')
     boundary(p,'before_D5_queue')
     # Original RULE/T refs stay descriptive and never enter local/D5 request.
     decoded,dg,decoded_ref=guarded('D5',{**base,'stage':'D5','proposal_container_ref':local_ref});audit_started=time.monotonic()
     result=decoded['result'];refs=case['refs']
     if decoded['identity']!=case['identity'] or decoded['actual_arm']!=arm or any(result.get(k)!=v for k,v in case['identity'].items()):raise ValueError('actual terminal full identity/arm changed')
     if result.get('status') not in ('success','fallback','unavailable','unpublished'):raise ValueError('original D5 terminal union')
     import numpy as np
     with np.load(pin(refs['original_scene']),allow_pickle=False) as z:plain=dict(positions=z['positions'].tolist(),demands=z['demands'].tolist(),aircraft_ids=list(map(str,z['aircraft_ids'])),profile=str(z['profile']))
     prepared=read_asset(refs['original_prepared_Wenv'])['result']
     if not audit_decoder_ledger(result['cost'],len(plain['aircraft_ids']),'D5')['valid']:raise ValueError('original D5 ledger')
     published=result['status'] in ('success','fallback');check=audit_layout(plain,result['layout'],prepared['Ncap'],prepared['Omega_cap']) if published else None
     if published and not check['valid']:raise ValueError('independent terminal geometry/request/owner invalid')
     atomic(target/'parent_audit_timer.json',dict(observed_seconds=time.monotonic()-audit_started,scope='original input reread and terminal identity/ledger/geometry audit',timer_receipt_IO_additional=True))
     row=dict(case_index=i,actual_arm=arm,identity=case['identity'],original_input_refs=refs,local_asset_ref=local_ref,D5_asset_ref=decoded_ref,local_state_ref=local_value['state_ref'],local_static_ref=local_value['static_ref'],local_status=local_value['status'],D5_status=result['status'],terminal_metrics=check['metrics'] if check else None,independent_terminal_valid=check['valid'] if check else None,guard_refs={stage:dict(path=str(target/stage/'guard_receipt.json'),sha256=digest(target/stage/'guard_receipt.json')) for stage in ('local','D5')},status='completed',new_calls=dict(local=1,D5=1,W=0,T=0,forward=0,updates=0))
     publish_started=time.monotonic();ref=immutable(target/'slot_result.json',row);index[key]=ref;atomic(epoch/'slot_index.json',index);atomic(target/'parent_publish_timer.json',dict(observed_seconds=time.monotonic()-publish_started,scope='small slot row and updated slot index',timer_receipt_IO_additional=True));atomic(target/'slot_observed_timer.json',dict(observed_seconds=time.monotonic()-slot_started,scope='slot through publish timers, excludes this receipt IO',components_not_additive_with_enclosing_scope=True))
     atomic(root/'status.json',dict(stage='running',run_id=p['run_id'],actual_worker=identity(os.getpid()),completed_slots=len(index),total_slots=60,elapsed_seconds=time.monotonic()-began))
   if len(index)!=60:raise ValueError('full20x3 required')
   if recovery and new_calls!=dict(local=59,D5=59):raise ValueError('full recovered actual call accounting must equal original1 plus new59')
   immutable(epoch/'result_manifest.json',dict(group=p['group'],run_id=p['run_id'],plan_ref=plan_ref,slots=index,cases=20,arms=['G','A','R'],new_calls=dict(local=new_calls['local'],D5=new_calls['D5'],W=0,T=0,forward=0,updates=0),original_actual_calls=dict(local=1,D5=1) if recovery else dict(local=0,D5=0),whole_actual_calls=dict(local=60,D5=60),operations_recovery_ref=recovery.get('contract_ref') if recovery else None,future_dispatch_allowed=False))
   atomic(root/'status.json',dict(stage='completed',run_id=p['run_id'],completed_slots=60,total_slots=60,elapsed_seconds=time.monotonic()-began))
  except Exception as e:
   if target is not None and slot_started is not None:atomic(target/'failed_parent_observation_timer.json',dict(observed_slot_seconds=time.monotonic()-slot_started,observed_audit_seconds=time.monotonic()-audit_started if audit_started is not None else None,error=str(e),partial_scope_not_complete_method_cost=True,timer_receipt_IO_additional=True))
   atomic(root/'status.json',dict(stage='paused' if isinstance(e,Paused) else 'failed',run_id=p['run_id'],completed_slots=len(index),total_slots=60,error=str(e),observed_elapsed_seconds=time.monotonic()-began,automatic_retry=False));raise
  finally:
   atomic(epoch/'worker_epoch_finally.json',dict(actual_worker=identity(os.getpid()),observed_elapsed_seconds=time.monotonic()-began,provisional_until_finally_sample=True,complete_worker_exit_bound_closed=False,requires_external_sameboot_execution_ended_bound=True,includes_this_receipt_IO_and_interpreter_exit=False,plan_ref=plan_ref))
def main():
 began=time.monotonic();a=argparse.ArgumentParser();a.add_argument('--plan',required=True);a.add_argument('--plan-sha256',required=True);a.add_argument('--review',required=True);a.add_argument('--review-sha256',required=True);p=a.parse_args();plan_ref=dict(path=p.plan,sha256=p.plan_sha256);review_ref=dict(path=p.review,sha256=p.review_sha256)
 root=None
 try:
  early=read(plan_ref);review=read(review_ref)
  if review.get('plan_ref')==plan_ref and review.get('execution_authorized')is True and '/round8/' in early['archive_root']:root=Path(early.get('operations_recovery',{}).get('epoch_root',early['archive_root']))
  run(plan_ref,review_ref)
 finally:
  if root is not None:atomic(root/'worker_main_finally.json',dict(observed_seconds_since_main=time.monotonic()-began,scope='includes plan validation/import/lock failure if any',excludes_prior_module_imports=True,complete_worker_exit_bound_closed=False,provisional_until_finally_sample=True,excludes_this_receipt_IO_and_interpreter_exit=True))
if __name__=='__main__':main()
