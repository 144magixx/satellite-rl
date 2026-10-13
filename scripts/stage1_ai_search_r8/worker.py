"""One exact20 serial pilot, immutable completed slots and never replayed unknown D5."""
import argparse,os,sys,time,subprocess
from pathlib import Path
from .contracts import read,validate,atomic,immutable,pin,digest,identity,order
from .deadline import boundary,Paused,bind_parent

def run(plan_ref,review_ref):
 bind_parent('R8_SERVICE_PID');p=validate(plan_ref,review_ref);boundary(p,'worker_before_runtime')
 root=Path(p['archive_root']);began=time.monotonic();atomic(root/'worker_preflight.json',dict(actual_worker=identity(os.getpid()),boundary=boundary(p,'worker_before_imports'),plan_ref=plan_ref))
 source=Path(p['original_source_manifest_ref']['path']).parent;sys.path[:0]=[str(source/'src'),str(source)]
 from satellite_rl.workbench.r5_stage_guard import run_stage
 from satellite_rl.workbench.r6_io import read_asset,write_asset
 from satellite_rl.workbench.geoload_runner import exclusive_lock
 from scripts.stage1_ai_search_r4.audit import audit_layout,audit_ledger
 with exclusive_lock(root/'worker.lock'),exclusive_lock(root/'compute.lock'):
  index=read(dict(path=str(root/'slot_index.json'),sha256=digest(root/'slot_index.json'))) if (root/'slot_index.json').exists() else {}
  atomic(root/'status.json',dict(stage='running',run_id=p['run_id'],actual_worker=identity(os.getpid()),completed_slots=len(index),total_slots=60))
  try:
   for case in p['cases']:
    i=case['case_index']
    for arm in order(i):
     boundary(p,'queue_before_slot');key=f'{i}:{arm}';target=root/'slots'/str(i)/arm
     if key in index:
      pin(index[key]);continue
     target.mkdir(parents=True,exist_ok=True);local=target/'local';d5=target/'D5'
     def stop():
      try:boundary(p,'owned_child_monitor');return False
      except Paused:return True
     def spawn(command,**kwargs):
      boundary(p,'child_before_spawn');env=dict(os.environ);env['R8_PARENT_PID']=str(os.getpid());env['PYTHONDONTWRITEBYTECODE']='1';return subprocess.Popen(command,env=env,**kwargs)
     def guarded(stage,request):
      folder=target/stage;folder.mkdir(exist_ok=True);request_path=folder/'request.json'
      if request_path.exists():
       if read(dict(path=str(request_path),sha256=digest(request_path)))!=request:raise ValueError('immutable request differs')
      else:immutable(request_path,request)
      command=[sys.executable,'-B','-u','-m','scripts.stage1_ai_search_r8.child','--request',str(request_path),'--directory',str(folder)]
      receipt=run_stage(command,folder,15,2147483648,stop=stop,popen=spawn)
      if receipt['status']!='completed':raise RuntimeError('unknown/hard unpublished '+stage+'; no automatic retry')
      return read_asset(read(receipt['output_ref'])['asset_ref']),receipt
     base=dict(plan_ref=plan_ref,case_index=i,arm=arm)
     local_value,lg=guarded('local',{**base,'stage':'local','state_path':str(target/'local_state/state.json'),'resume_admission':p.get('local_resume_admissions',{}).get(key)})
     if local_value['status']=='paused':raise Paused('durable local paused, no D5')
     boundary(p,'before_D5_queue')
     # Original RULE/T refs stay descriptive and never enter local/D5 request.
     proposal_ref=write_asset(target/'local_proposal.json.gz',local_value['proposal'])
     decoded,dg=guarded('D5',{**base,'stage':'D5','proposal_ref':proposal_ref})
     result=decoded['result'];refs=case['refs']
     if decoded['identity']!=case['identity'] or decoded['actual_arm']!=arm or any(result.get(k)!=v for k,v in case['identity'].items()):raise ValueError('actual terminal full identity/arm changed')
     if result.get('status') not in ('success','fallback','unavailable','unpublished'):raise ValueError('original D5 terminal union')
     import numpy as np
     with np.load(pin(refs['original_scene']),allow_pickle=False) as z:plain=dict(positions=z['positions'].tolist(),demands=z['demands'].tolist(),aircraft_ids=list(map(str,z['aircraft_ids'])),profile=str(z['profile']))
     prepared=read_asset(refs['original_prepared_Wenv'])['result']
     if not audit_ledger(result['cost'],len(plain['aircraft_ids']),'D5')['valid']:raise ValueError('original D5 ledger')
     published=result['status'] in ('success','fallback');check=audit_layout(plain,result['layout'],prepared['Ncap'],prepared['Omega_cap']) if published else None
     if published and not check['valid']:raise ValueError('independent terminal geometry/request/owner invalid')
     row=dict(case_index=i,actual_arm=arm,identity=case['identity'],original_input_refs=refs,local=local_value,D5=decoded,guards=dict(local=lg,D5=dg),independent_terminal=check,status='completed',new_calls=dict(local=1,D5=1,W=0,T=0,forward=0,updates=0))
     ref=immutable(target/'slot_result.json',row);index[key]=ref;atomic(root/'slot_index.json',index)
     atomic(root/'status.json',dict(stage='running',run_id=p['run_id'],actual_worker=identity(os.getpid()),completed_slots=len(index),total_slots=60,elapsed_seconds=time.monotonic()-began))
   if len(index)!=60:raise ValueError('full20x3 required')
   immutable(root/'result_manifest.json',dict(group=p['group'],run_id=p['run_id'],plan_ref=plan_ref,slots=index,cases=20,arms=['G','A','R'],new_calls=dict(local=60,D5=60,W=0,T=0,forward=0,updates=0),future_dispatch_allowed=False))
   atomic(root/'status.json',dict(stage='completed',run_id=p['run_id'],completed_slots=60,total_slots=60,elapsed_seconds=time.monotonic()-began))
  except Exception as e:
   atomic(root/'status.json',dict(stage='paused' if isinstance(e,Paused) else 'failed',run_id=p['run_id'],completed_slots=len(index),total_slots=60,error=str(e),automatic_retry=False));raise
def main():
 a=argparse.ArgumentParser();a.add_argument('--plan',required=True);a.add_argument('--plan-sha256',required=True);a.add_argument('--review',required=True);a.add_argument('--review-sha256',required=True);p=a.parse_args();run(dict(path=p.plan,sha256=p.plan_sha256),dict(path=p.review,sha256=p.review_sha256))
if __name__=='__main__':main()
