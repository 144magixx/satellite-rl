"""Serial M12, immutable stage reservations, complete pools or explicit stop."""
import argparse,os,sys,time
from pathlib import Path
from .contracts import read,validate,atomic,immutable,digest,identity,CAPS,asset_read,pin,io_snapshot
from .control import boundary,storage_boundary,exclusive_lock,bind_parent,Paused
from .guard import run_stage

def run(plan_ref,review_ref):
 began=time.monotonic();bind_parent('R9_SERVICE_PID');p=validate(plan_ref,review_ref);root=Path(p['archive_root']);boundary(p,'worker_admission');storage_boundary(p,'worker_admission')
 atomic(root/'worker_setup_timer.json',dict(observed_seconds=time.monotonic()-began,scope='binding/complete source+input+root admission; outside per-mother guards',actual_process=identity(os.getpid())))
 with exclusive_lock(root/'worker.lock'),exclusive_lock(p['host_compute_lock']):
  immutable(root/'worker_epoch_intent.json',dict(plan_ref=plan_ref,actual_worker=identity(os.getpid()),automatic_retry=False));completed={};reserved=dict(RULE=0,D5=0,candidate=0)
  try:
   for case in p['cases']:
    i=case['original_index'];folder=root/'mothers'/str(i);folder.mkdir(parents=True,exist_ok=True);mother_start=time.monotonic();boundary(p,'before_mother');storage_boundary(p,'before_mother')
    def guarded(stage,extra=None):
     boundary(p,'before_'+stage+'_intent');storage_boundary(p,'before_'+stage+'_intent')
     if stage in ('RULE','D5'):
      if reserved[stage]>=CAPS[stage]:raise ValueError('M acquisition ceiling')
      reserved[stage]+=1
     elif stage=='pool':
      # Reserve complete fixed upper bound; actual extractions/entries remain separate.
      if reserved['candidate']+32>CAPS['candidate']:raise ValueError('M candidate ceiling')
      reserved['candidate']+=32
     target=folder/stage;target.mkdir(exist_ok=True);request=dict(plan_ref=plan_ref,root_review_ref=review_ref,original_index=i,stage=stage,**(extra or {}));request_ref=immutable(target/'request.json',request)
     atomic(root/'reservations.json',dict(reserved_upper_bounds=reserved,actual_algorithm_counts_from_entry_and_finish_records=True,unknown_counts_never_reset=True))
     command=[sys.executable,'-B','-u','-m','scripts.stage1_ai_search_r9.child','--request',request_ref['path'],'--request-sha256',request_ref['sha256'],'--directory',str(target)]
     receipt=run_stage(command,target,stage,stop=lambda:boundary(p,'owned_native_monitor'))
     pointer=read(receipt['output_ref']);value=asset_read(pointer['asset_ref'])
     if value.get('identity')!=case['identity'] or value.get('actual_stage')!=stage:raise ValueError('native output full identity/stage')
     if stage=='D5' and value['result'].get('status')!='success':raise ValueError('RULE unqualified; original D5 fallback/failure preserved, no retry or substitute')
     if stage=='common' and len(value['pool']['descriptors'])>32:raise ValueError('fixed descriptor ceiling')
     if stage=='pool':
      common=asset_read(extra['common_ref']);count=len(common['pool']['descriptors'])
      if value.get('complete')is not True or value['descriptor_extractions']!=count or set(value['candidate_refs'])!={str(j) for j in range(count)} or value['selected_position'] not in [-1,*range(count)]:raise ValueError('all actual fixed descriptors complete')
      for ref in value['candidate_refs'].values():pin(ref)
     return pointer['asset_ref'],dict(path=str(target/'guard_receipt.json'),sha256=digest(target/'guard_receipt.json'))
    stages={};rule_ref=case.get('reused_RULE_ref')
    if rule_ref is None:
     proposal_ref,stages['RULE']=guarded('RULE');rule_ref,stages['D5']=guarded('D5',dict(proposal_ref=proposal_ref))
    common_ref,stages['common']=guarded('common',dict(RULE_ref=rule_ref));pool_ref,stages['pool']=guarded('pool',dict(common_ref=common_ref))
    row=dict(original_index=i,identity=case['identity'],RULE_ref=rule_ref,common_ref=common_ref,pool_ref=pool_ref,guard_refs=stages,status='complete_pending_independent_audit',original_T_ref=case.get('original_T_terminal_ref'),original_T_cost_endpoint_ref=case.get('old_cost_endpoint_ref'),historical_full_T_cost_seconds=None,observed_mother_seconds=time.monotonic()-mother_start,nested_timers_not_additive=True)
    completed[str(i)]=immutable(folder/'mother_result.json',row);atomic(root/'mother_index.json',completed);atomic(root/'status.json',dict(stage='running',run_id=p['run_id'],completed_mothers=len(completed),original_denominator=12,reservations=reserved))
   if len(completed)!=12:raise ValueError('M fixed denominator12')
   immutable(root/'result_manifest.json',dict(group=p['group'],stage='M',plan_ref=plan_ref,mothers=completed,original_denominator=12,all_native_stages_completed=True,independent_scientific_gate_evaluated=False,reservations=reserved,automatic_L_dispatch=False,new_W_T_forward_updates=0))
   atomic(root/'status.json',dict(stage='completed_pending_independent_audit',run_id=p['run_id'],completed_mothers=12,original_denominator=12,automatic_next_stage=False))
  except BaseException as e:
   atomic(root/'status.json',dict(stage='paused' if isinstance(e,Paused) else 'partial_unknown',run_id=p['run_id'],completed_mothers=len(completed),original_denominator=12,error=type(e).__name__+': '+str(e),reserved_upper_bounds=reserved,actual_unknown_work=None,automatic_retry=False));raise
  finally:atomic(root/'worker_finally.json',dict(observed_seconds_since_run=time.monotonic()-began,actual_process=identity(os.getpid()),process_IO=io_snapshot(),complete_exit_cost_closed=False,receipt_IO_and_process_exit_excluded=True))
def main():
 a=argparse.ArgumentParser();a.add_argument('--plan',required=True);a.add_argument('--plan-sha256',required=True);a.add_argument('--review',required=True);a.add_argument('--review-sha256',required=True);v=a.parse_args();run(dict(path=v.plan,sha256=v.plan_sha256),dict(path=v.review,sha256=v.review_sha256))
if __name__=='__main__':main()
