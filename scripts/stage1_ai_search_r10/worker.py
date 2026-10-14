"""Six original mothers per host; one common and five serial bounded native methods."""
import argparse,os,sys,time
from pathlib import Path
from .contracts import validate,read,typed_asset,publish,immutable,digest,identity,io_snapshot,ARMS,check_identity
from .control import plan_boundary as boundary,storage_boundary,exclusive_lock,bind_parent,Paused
from .schedule import partition,Reservations
from .guard import run_stage

def check_output(case,stage,value,common_ref=None):
 check_identity(value['identity'],case['identity'])
 if value['original_index']!=case['original_index'] or value['actual_stage']!=stage or value.get('new_RULE_D5_W_T_model_calls')!=0:raise ValueError('actual method/identity/call source differs')
 result=value['common'] if stage=='common' else value['result']
 check_identity(result['identity'],case['identity'])
 if result['original_index']!=case['original_index']:raise ValueError('core original mother differs')
 if stage!='common' and (result.get('complete')is not True or result.get('method')!=stage or value.get('common_ref')!=common_ref):raise ValueError('native method incomplete or wrong common')
 ledger=result['common_ledger'] if stage=='common' else result['ledger'];n=case['aircraft_count']
 limits=dict(raypairs=8192*n,flow_queries=256,residual_edge_inspections=2**30,heap_pushes_plus_pops=2**31,heap_key_comparisons=2**37,graph_arc_initialization_writes=2**20,augmentation_capacity_writes=2**24,full_vector_node_visits=2**26,sort_key_comparisons=2000000)
 for key,limit in limits.items():
  if type(ledger.get(key))is not int or not 0<=ledger[key]<=limit:raise ValueError('actual primitive cap '+key)
 return dict(common_SHA=result.get('common_SHA'),ledger_counts={k:ledger[k] for k in limits},complete=stage=='common' or result['complete'])

def run(plan_ref,review_ref):
 began=time.monotonic();bind_parent('R10_SERVICE_PID');p=validate(plan_ref,review_ref);root=Path(p['archive_root']);boundary(p,'worker_admission');storage_boundary(p,'worker_admission')
 publish(root/'worker_setup_cost.json',dict(observed_seconds=time.monotonic()-began,scope='parent binding/full plan/source/input/pause validation outside stage guards',actual_process=identity(os.getpid())))
 with exclusive_lock(root/'worker.lock'),exclusive_lock(p['host_compute_lock']):
  immutable(root/'worker_epoch_intent.json',dict(plan_ref=plan_ref,actual_worker=identity(os.getpid()),automatic_retry=False));local=partition(p['cases'],p['mother_host_assignment'],p['executor_port']);completed={};reserved=dict(common=0,method=0)
  budget=Reservations(root/'budget_reservations',p['run_id'],p['execution_epoch_id'],{v['original_index']:v['identity'] for v in local});last_storage=time.monotonic()
  def monitor():
   nonlocal last_storage
   boundary(p,'owned_native_monitor')
   if time.monotonic()-last_storage>=5:storage_boundary(p,'owned_native_monitor');last_storage=time.monotonic()
  try:
   for case in local:
    index=case['original_index'];folder=root/'mothers'/str(index);folder.mkdir(parents=True,exist_ok=True);common_ref=None;stages={}
    for stage in ('common',*ARMS):
     before=time.monotonic();boundary(p,'before_'+stage+'_reservation');storage_boundary(p,'before_'+stage+'_reservation')
     reservation=budget.reserve(index,stage,dict(common_input_ref=case['common_input_ref']),common_ref)
     reserved['common' if stage=='common' else 'method']+=1
     if reserved['common']>6 or reserved['method']>30:raise ValueError('host finite6common30methods cap')
     target=folder/stage;target.mkdir(exist_ok=True);request=dict(plan_ref=plan_ref,root_review_ref=review_ref,original_index=index,stage=stage,budget_reservation_ref=reservation,common_ref=common_ref);request_ref=immutable(target/'request.json',request)
     publish(root/'reservations.json',dict(reserved_upper_bounds=reserved,actual_counts_from_original_entry_finish_records=True,unknown_never_reset=True));immutable(target/'parent_setup_cost.json',dict(observed_seconds=time.monotonic()-before,scope='parent pause/storage/budget/request IO outside native guard'))
     command=[sys.executable,'-B','-u','-m','scripts.stage1_ai_search_r10.child','--request',request_ref['path'],'--request-sha256',request_ref['sha256'],'--directory',str(target)]
     receipt=run_stage(command,target,stage,stop=monitor);audit_start=time.monotonic()
     try:
      pointer=read(receipt['output_ref']);value=typed_asset(pointer['asset_ref'])
      if value['run_id']!=p['run_id']:raise ValueError('native run differs')
      summary=check_output(case,stage,value,common_ref)
     finally:publish(target/'parent_audit_cost.json',dict(observed_seconds_to_finally=time.monotonic()-audit_start,scope='guard-return/pin/decode/identity/counters; outside native stage',receipt_IO_excluded=True))
     publication=time.monotonic();guard_ref=dict(path=str(target/'guard_receipt.json'),sha256=digest(target/'guard_receipt.json'));stages[stage]=dict(output_ref=pointer['asset_ref'],guard_ref=guard_ref,summary=summary)
     budget.finish(index,stage,dict(run_id=p['run_id'],identity=case['identity'],original_index=index,stage=stage,result_ref=pointer['asset_ref'],guard_ref=guard_ref,independent_scientific_audit_pending=True))
     if stage=='common':common_ref=pointer['asset_ref']
     publish(target/'parent_publish_cost.json',dict(observed_seconds=time.monotonic()-publication,scope='parent output refs/budget terminal',receipt_IO_excluded=True))
    completed[str(index)]=immutable(folder/'mother_result.json',dict(run_id=p['run_id'],original_index=index,identity=case['identity'],stages=stages,original_denominator=12,host_denominator=6,quality_gate_evaluated=False));publish(root/'mother_index.json',completed);publish(root/'status.json',dict(stage='running',run_id=p['run_id'],completed_mothers=len(completed),host_denominator=6,original_denominator=12,reservations=reserved))
   immutable(root/'result_manifest.json',dict(group=p['group'],plan_ref=plan_ref,run_id=p['run_id'],host=p['executor_port'],mothers=completed,host_denominator=6,global_original_denominator=12,methods=list(ARMS),reserved_upper_bounds=reserved,independent_gate_evaluated=False,automatic_next_stage=False))
   publish(root/'status.json',dict(stage='completed_pending_independent_audit',run_id=p['run_id'],completed_mothers=6,completed_method_stages=30,completed_common_stages=6,original_denominator=12,automatic_next_stage=False))
  except BaseException as e:
   publish(root/'status.json',dict(stage='paused' if isinstance(e,Paused) else 'partial_unknown',run_id=p['run_id'],completed_mothers=len(completed),host_denominator=6,original_denominator=12,error=type(e).__name__+': '+str(e),reserved_upper_bounds=reserved,unknown_work_exact=None,automatic_retry=False));raise
  finally:publish(root/'worker_finally_cost.json',dict(observed_seconds_to_finally=time.monotonic()-began,actual_process=identity(os.getpid()),process_IO=io_snapshot(),complete_worker_exit_cost_seconds=None,receipt_IO_and_process_exit_excluded=True))

def main():
 began=time.monotonic();a=argparse.ArgumentParser();a.add_argument('--plan',required=True);a.add_argument('--plan-sha256',required=True);a.add_argument('--review',required=True);a.add_argument('--review-sha256',required=True);v=a.parse_args();plan_ref=dict(path=v.plan,sha256=v.plan_sha256);root=Path(read(plan_ref)['archive_root'])
 try:run(plan_ref,dict(path=v.review,sha256=v.review_sha256))
 except BaseException as e:publish(root/'worker_main_failure.json',dict(error=type(e).__name__+': '+str(e),automatic_retry=False));raise
 finally:publish(root/'worker_main_finally_cost.json',dict(observed_seconds_to_finally=time.monotonic()-began,process_IO=io_snapshot(),complete_worker_exit_cost_seconds=None,receipt_IO_and_process_exit_excluded=True))
if __name__=='__main__':main()
