"""Reuse original completed rows; execute only44 remaining original logical stages."""
import argparse,os,time
from pathlib import Path
from scripts.stage1_ai_search_r10 import contracts as c
from scripts.stage1_ai_search_r10.control import bind_parent,exclusive_lock
from scripts.stage1_ai_search_r10.schedule import Reservations
from scripts.stage1_ai_search_r10.guard import run_stage
from scripts.stage1_ai_search_r10.worker import check_output
from .contracts import validate,check_pause,stage_key,limits_for,fref
from .storage import Observer
STAGE_FIELDS=('phase','output_ref','guard_ref','request_ref','reservation_ref','budget_terminal_ref','algorithm_entry_ref','failure_ref','runtime_stage_directory','runtime_budget_directory')
def view(directory,budget,*,reservation=None):
 result=dict(phase='not_started',output_ref=None,guard_ref=None,request_ref=None,reservation_ref=reservation,budget_terminal_ref=None,algorithm_entry_ref=None,failure_ref=None,runtime_stage_directory=str(directory),runtime_budget_directory=str(budget))
 for field,name in [('guard_ref','guard_receipt.json'),('request_ref','request.json'),('algorithm_entry_ref','algorithm_entry.json'),('failure_ref','child_failure.json')]:
  if (directory/name).exists():result[field]=fref(directory/name)
 if (directory/'stage_output.json').exists():result['output_ref']=c.read(fref(directory/'stage_output.json'))['asset_ref']
 if reservation is None and (budget/'stage.wal.json').exists():result['reservation_ref']=fref(budget/'stage.wal.json')
 if (budget/'terminal.json').exists():result['budget_terminal_ref']=fref(budget/'terminal.json')
 if result['guard_ref']:
  guard=c.read(result['guard_ref']);result['phase']='complete' if guard['status']=='completed' and all(result[k] is not None for k in ('output_ref','request_ref','reservation_ref','budget_terminal_ref','algorithm_entry_ref')) else 'partial_unknown'
 elif directory.exists() or budget.exists():result['phase']='partial_unknown'
 return result
def original_view(row):
 directory=Path(row['original_plan_archive_root'])/'mothers'/str(row['original_index'])/row['stage'];budget=Path(row['original_plan_archive_root'])/'budget_reservations/mothers'/str(row['original_index'])/row['stage'];return view(directory,budget)
def logical_row(p,row,selected,epoch,common,prior):return dict(executor_port=p['executor_port'],original_index=row['original_index'],stage=row['stage'],selected_epoch=epoch,selected_stage_view=selected,common_ref=None if row['stage']=='common' else common,prior_attempt_views=prior)
def run(plan_ref,review_ref):
 began=time.monotonic();bind_parent('R10_SERVICE_PID');p,x,rows=validate(plan_ref,review_ref);root=Path(p['archive_root']);root.mkdir(parents=True,exist_ok=True);bykey={stage_key(r):r for r in rows};selected={};native_used=0.0;newnative=0;distinct=0;bridges=0;observer=None
 def publish_table():return c.publish(root/'combined_stage_map.json',dict(schema='R10RecoveryCombinedStageMapV1',run_id=p['run_id'],executor_port=p['executor_port'],original_plan_ref=x['original_plan_ref'],original_root_review_ref=x['original_root_review_ref'],recovery_plan_ref=plan_ref,recovery_root_review_ref=review_ref,recovery_contract_ref=p['recovery_contract_ref'],original_denominator=12,methods_per_mother=5,rows=[selected[stage_key(r)] for r in rows if stage_key(r) in selected],all36_closed=len(selected)==36 and all(v['selected_stage_view']['phase']=='complete' for v in selected.values()),independent_quality_audit_performed=False))
 try:
  with exclusive_lock(root/'worker.lock'),exclusive_lock(p['host_compute_lock']):
   c.immutable(root/'worker_epoch_intent.json',dict(plan_ref=plan_ref,actual_worker=c.identity(os.getpid()),recovery_contract_ref=p['recovery_contract_ref'],automatic_retry=False));observer=Observer(p,x['worker_control_root'],storage_check=c.storage_check);observer.observe('worker_admission');last_storage=time.monotonic()
   budget=Reservations(root/'budget_reservations',p['run_id'],p['execution_epoch_id'],{v['original_index']:v['identity'] for v in p['cases']});completed={}
   for row in rows:
    index,stage=stage_key(row);old=dict(row,original_plan_archive_root=x['original_archive_root']);prior=[original_view(old)] if row['native_stage_wal_exists'] and not row['typed_output_exists'] else [];oldcommon=bykey[(index,'common')].get('output_ref')
    selected[(index,stage)]=logical_row(p,row,original_view(old) if row['typed_output_exists'] else view(root/'mothers'/str(index)/stage,root/'budget_reservations/mothers'/str(index)/stage),'original' if row['typed_output_exists'] else 'recovery',oldcommon,prior)
   publish_table();c.publish(root/'worker_setup_cost.json',dict(observed_seconds=time.monotonic()-began,scope='binding/full old reuse/source validation/storage/full36row initialization outside native'))
   def monitor():
    nonlocal last_storage
    check_pause(p,'recovery_worker_monitor')
    if time.monotonic()-last_storage>=5:observer.observe('recovery_worker_monitor');last_storage=time.monotonic()
   for case in [v for v in p['cases'] if str(v['original_index']) in p['mother_host_assignment'] and p['mother_host_assignment'][str(v['original_index'])]==p['executor_port']]:
    index=case['original_index'];common=None;stages={}
    for stage in ('common',*c.ARMS):
     row=bykey[(index,stage)];old=dict(row,original_plan_archive_root=x['original_archive_root']);prior=[]
     if row['typed_output_exists']:
      oldview=original_view(old);selected[(index,stage)]=logical_row(p,row,oldview,'original',common,[]);stages[stage]=dict(output_ref=row['output_ref'],guard_ref=row['guard_ref'],selected_epoch='original',summary=dict(complete=True,reused_original_bytes=True))
      if stage=='common':common=row['output_ref']
      publish_table();continue
     before=time.monotonic();check_pause(p,'before_recovery_'+stage);observer.observe('before_recovery_'+stage);limit=limits_for(row)
     if native_used+limit>x['native_ceiling_seconds']+1e-9:raise RuntimeError('conservative cumulative native envelope exhausted')
     target=root/'mothers'/str(index)/stage;target.mkdir(parents=True,exist_ok=False);bd=root/'budget_reservations/mothers'/str(index)/stage
     if row['native_stage_wal_exists']:
      if (p['executor_port'],index,stage)!=(20473,0,'S2'):raise ValueError('unknown old work cannot reenter')
      reservation=row['budget_reservation_ref'];bd.mkdir(parents=True,exist_ok=False);bridge=c.immutable(bd/'reservation_bridge.json',dict(schema='R10OriginalLogicalReservationNativeAttemptBridgeV1',run_id=p['run_id'],identity=case['identity'],original_index=index,stage=stage,original_reservation_ref=reservation,old_guard_ref=row['guard_ref'],recovery_contract_ref=p['recovery_contract_ref'],root_review_ref=review_ref,native_attempt=2,new_scientific_reservation=False,old_native_wall_seconds=row['actual_stage_wall_seconds'],remaining_native_hard_seconds=limit));bridges+=1;prior=[original_view(old)]
     else:reservation=budget.reserve(index,stage,dict(common_input_ref=case['common_input_ref']),common);distinct+=1
     newnative+=1
     if newnative>x['new_native_max'] or distinct>x['new_distinct_reservations'] or bridges>(1 if p['executor_port']==20473 else 0):raise ValueError('finite native/reservation caps')
     request=c.immutable(target/'request.json',dict(plan_ref=plan_ref,root_review_ref=review_ref,original_index=index,stage=stage,budget_reservation_ref=reservation,common_ref=common));c.immutable(target/'parent_setup_cost.json',dict(observed_seconds=time.monotonic()-before,scope='parent source/pause/storage/reservation/bridge/request IO outside native guard'))
     c.publish(root/'reservations.json',dict(new_native_starts=newnative,count_semantics='reserved_intent_upper_bound; actual_births_from_guard_and_entries',new_distinct_logical_reservations=distinct,original_S2_logical_bridges=bridges,unknown_never_reset=True));selected[(index,stage)]=logical_row(p,row,view(target,bd,reservation=reservation),'recovery',common,prior);publish_table()
     import sys
     command=[sys.executable,'-B','-u','-m','scripts.stage1_ai_search_r10.child','--request',request['path'],'--request-sha256',request['sha256'],'--directory',str(target)]
     try:receipt=run_stage(command,target,stage,stop=monitor,limits={stage:limit})
     finally:
      if (target/'guard_receipt.json').exists():native_used+=c.read(fref(target/'guard_receipt.json'))['actual_stage_wall_seconds']
      selected[(index,stage)]=logical_row(p,row,view(target,bd,reservation=reservation),'recovery',common,prior);publish_table()
     audit=time.monotonic()
     try:pointer=c.read(receipt['output_ref']);value=c.typed_asset(pointer['asset_ref']);summary=check_output(case,stage,value,common)
     finally:c.publish(target/'parent_audit_cost.json',dict(observed_seconds_to_finally=time.monotonic()-audit,scope='native guard return/typeddecode/identity/counters; no geometry; outside native'))
     publication=time.monotonic();guard=fref(target/'guard_receipt.json');terminal=dict(run_id=p['run_id'],identity=case['identity'],original_index=index,stage=stage,result_ref=pointer['asset_ref'],guard_ref=guard,independent_scientific_audit_pending=True)
     if row['native_stage_wal_exists']:c.immutable(bd/'terminal.json',dict(terminal,original_reservation_ref=reservation,native_attempt=2,reservation_bridge_ref=bridge))
     else:budget.finish(index,stage,terminal)
     stages[stage]=dict(output_ref=pointer['asset_ref'],guard_ref=guard,selected_epoch='recovery',summary=summary)
     if stage=='common':
      common=pointer['asset_ref']
      for key,pending in selected.items():
       if key[0]==index and key[1]!='common':pending['common_ref']=common
     c.publish(target/'parent_publish_cost.json',dict(observed_seconds=time.monotonic()-publication,scope='budget terminal/mixed refs outside native'));selected[(index,stage)]=logical_row(p,row,view(target,bd,reservation=reservation),'recovery',common,prior);publish_table()
    completed[str(index)]=c.immutable(root/'mothers'/str(index)/'mother_result.json',dict(run_id=p['run_id'],original_index=index,identity=case['identity'],stages=stages,original_denominator=12,host_denominator=6,quality_gate_evaluated=False));c.publish(root/'mother_index.json',completed);c.publish(root/'status.json',dict(stage='running',run_id=p['run_id'],completed_mothers=len(completed),host_denominator=6,original_denominator=12,new_native_starts=newnative))
   c.immutable(root/'result_manifest.json',dict(group=p['group'],plan_ref=plan_ref,run_id=p['run_id'],host=p['executor_port'],mothers=completed,host_denominator=6,global_original_denominator=12,methods=list(c.ARMS),combined_stage_map_ref=publish_table(),recovery_contract_ref=p['recovery_contract_ref'],new_native_starts=newnative,count_semantics='reserved_intent_upper_bound; actual_births_from_guard_and_entries',new_distinct_logical_reservations=distinct,original_S2_logical_bridges=bridges,independent_gate_evaluated=False,automatic_next_stage=False));c.publish(root/'status.json',dict(stage='completed_pending_independent_audit',run_id=p['run_id'],completed_mothers=6,completed_method_stages=30,completed_common_stages=6,original_denominator=12,new_native_starts=newnative,count_semantics='reserved_intent_upper_bound; actual_births_from_guard_and_entries',automatic_next_stage=False))
 except BaseException as e:
  c.publish(root/'status.json',dict(stage='paused' if type(e).__name__=='Paused' else 'partial_unknown',run_id=p['run_id'],error=type(e).__name__+': '+str(e),publication_failure=getattr(e,'publication_failure',None),new_native_starts=newnative,count_semantics='reserved_intent_upper_bound; actual_births_from_guard_and_entries',new_distinct_logical_reservations=distinct,original_S2_logical_bridges=bridges,unknown_work_exact=None,automatic_retry=False));raise
 finally:
  if observer:observer.close()
  c.publish(root/'worker_finally_cost.json',dict(actual_process=c.identity(os.getpid()),observed_seconds_to_finally=time.monotonic()-began,new_completed_guard_wall_seconds=native_used,old_outer_debit_seconds=x['old_outer_debit_seconds'],old_bound_not_added_to_nested_native_as_actualcost=True,complete_worker_exit_cost_seconds=None,receipt_IO_and_exit_excluded=True))
def main():
 a=argparse.ArgumentParser();a.add_argument('--plan',required=True);a.add_argument('--plan-sha256',required=True);a.add_argument('--review',required=True);a.add_argument('--review-sha256',required=True);v=a.parse_args();run(dict(path=v.plan,sha256=v.plan_sha256),dict(path=v.review,sha256=v.review_sha256))
if __name__=='__main__':main()
