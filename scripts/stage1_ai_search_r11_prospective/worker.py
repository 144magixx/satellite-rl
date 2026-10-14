"""Six serial Gh stages, each separately reserved; never repeat an unknown intent."""
import argparse,os,sys,time
from pathlib import Path
from scripts.stage1_ai_search_r10 import contracts as c
from scripts.stage1_ai_search_r10.control import bind_parent,exclusive_lock
from .reaper import run_stage,boottime
from .contracts import validate,check_pause,storage_check,PARTITION,fref,lifecycle
from .storage import Observer

def check_output(case,p,value):
 c.check_identity(value['identity'],case['identity']);r=value['result'];ledger=r['ledger']
 if value['schema']!='R11GhNativeOutputV1' or value['run_id']!=p['run_id'] or value['common_ref']!=case['common_ref'] or value['historical_common_run_id']!=case['historical_R10_run_id'] or value['original_index']!=case['original_index'] or r['common_SHA']!=case['historical_common_SHA'] or r.get('complete')is not True or r['method']!='Gh':raise ValueError('exact Gh output/common identity')
 if len(r['results'])!=3 or ledger['global_structure_slots']!=3 or ledger['flow_queries']!=case['available_global_solves']:raise ValueError('three structure slots, at most three solves')
 if any(type(ledger[k])is not int or ledger[k]<0 or ledger[k]>cap for k,cap in ledger['caps'].items()):raise ValueError('original primitive caps')
 return dict(complete=True,structural_slots=3,flow_queries=ledger['flow_queries'],quality_audit_pending=True)

def run(plan_ref,review_ref):
 began=time.monotonic();bind_parent('R11_SERVICE_PID');p=validate(plan_ref,review_ref);root=Path(p['archive_root']);life=lifecycle(p);observer=None;native=0.;intents=0;solves=0;completed={}
 try:
  with exclusive_lock(root/'worker.lock'),exclusive_lock(p['host_compute_lock']):
   c.immutable(root/'worker_epoch_intent.json',dict(actual_worker=c.identity(os.getpid()),plan_ref=plan_ref,automatic_retry=False));observer=Observer(p,p['worker_control_root'],storage_check=storage_check);observer.observe('worker_admission');last_storage=time.monotonic()
   def monitor():
    nonlocal last_storage
    check_pause(p,'worker_monitor')
    if time.monotonic()-last_storage>=5:observer.observe('worker_monitor');last_storage=time.monotonic()
   for case in [v for v in p['cases'] if v['original_index'] in PARTITION[p['executor_port']]]:
    index=case['original_index'];before=time.monotonic();
    if float(os.environ['R11_LANE_DEADLINE_BOOTTIME'])-boottime()<301:raise RuntimeError('insufficient existing lane deadline before new reservation')
    check_pause(p,'before_Gh');observer.observe('before_Gh');target=root/'mothers'/str(index)/'Gh';target.mkdir(parents=True,exist_ok=False);budget=root/'budget_reservations'/str(index)
    reservation=c.immutable(budget/'reservation.json',dict(schema='R11GhLogicalReservationV1',run_id=p['run_id'],execution_epoch_id=p['execution_epoch_id'],identity=case['identity'],original_index=index,stage='Gh',common_ref=case['common_ref'],unknown_never_replayed=True));intents+=1
    if intents>6:raise ValueError('six intents per host maximum')
    request=c.immutable(target/'request.json',dict(plan_ref=plan_ref,root_review_ref=review_ref,original_index=index,stage='Gh',budget_reservation_ref=reservation,common_ref=case['common_ref']));c.publish(target/'parent_setup_cost.json',dict(observed_seconds=time.monotonic()-before,scope='pause/storage/reservation/request outside native'))
    command=[sys.executable,'-B','-u','-m','scripts.stage1_ai_search_r11_prospective.child','--request',request['path'],'--request-sha256',request['sha256'],'--directory',str(target)]
    try:receipt=run_stage(command,target,'Gh',stop=monitor,life=life,limits={'Gh':300})
    finally:
     if (target/'guard_receipt.json').exists():native+=c.read(fref(target/'guard_receipt.json'))['actual_stage_wall_seconds']
    audit=time.monotonic()
    try:pointer=c.read(receipt['output_ref']);value=c.typed_asset(pointer['asset_ref']);summary=check_output(case,p,value)
    finally:c.publish(target/'parent_audit_cost.json',dict(observed_seconds_to_finally=time.monotonic()-audit,scope='typed decode/identity/counters only outside native'))
    solves+=summary['flow_queries']
    if solves>sum(v['available_global_solves'] for v in p['cases'] if v['original_index'] in PARTITION[p['executor_port']]):raise ValueError('host exact frozen solves ceiling')
    publication=time.monotonic();terminal=c.immutable(budget/'terminal.json',dict(run_id=p['run_id'],original_index=index,stage='Gh',identity=case['identity'],result_ref=pointer['asset_ref'],guard_ref=fref(target/'guard_receipt.json'),request_ref=request,reservation_ref=reservation,common_ref=case['common_ref'],summary=summary))
    completed[str(index)]=terminal;c.publish(root/'mother_index.json',completed);c.publish(root/'status.json',dict(stage='running',run_id=p['run_id'],completed_mothers=len(completed),host_denominator=6,original_denominator=12));c.publish(target/'parent_publish_cost.json',dict(observed_seconds=time.monotonic()-publication,scope='terminal/index/status outside native'))
   c.immutable(root/'result_manifest.json',dict(schema='R11GhHostResultManifestV1',plan_ref=plan_ref,root_review_ref=review_ref,run_id=p['run_id'],executor_port=p['executor_port'],mothers=completed,host_denominator=6,original_denominator=12,arms=['Gh'],common_recomputations=0,independent_gate_evaluated=False,automatic_next_stage=False));c.publish(root/'status.json',dict(stage='completed_pending_independent_audit',run_id=p['run_id'],completed_mothers=6,completed_method_stages=6,completed_common_stages=0,original_denominator=12,automatic_next_stage=False))
 except BaseException as e:
  c.publish(root/'status.json',dict(stage='paused' if type(e).__name__=='Paused' else 'partial_unknown',run_id=p['run_id'],error=type(e).__name__+': '+str(e),reserved_intent_upper_bound=intents,unknown_work_exact=None,automatic_retry=False));raise
 finally:
  if observer:observer.close()
  c.publish(root/'worker_finally_cost.json',dict(actual_process=c.identity(os.getpid()),observed_seconds_to_finally=time.monotonic()-began,completed_guard_wall_seconds=native,reserved_intent_upper_bound=intents,complete_worker_exit_cost_seconds=None,receipt_IO_and_exit_excluded=True))

def main():
 a=argparse.ArgumentParser();a.add_argument('--plan',required=True);a.add_argument('--plan-sha256',required=True);a.add_argument('--review',required=True);a.add_argument('--review-sha256',required=True);v=a.parse_args();run(dict(path=v.plan,sha256=v.plan_sha256),dict(path=v.review,sha256=v.review_sha256))
if __name__=='__main__':main()
