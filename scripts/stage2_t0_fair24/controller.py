"""Prepared persistent coordinator; local old-owner state precedes sparse remote reads.
Never infer lane release from calculation completion or GPUidle. No default start.
"""
from pathlib import Path
import argparse,json,hashlib,os,time,datetime
from scripts.stage2_t0_fair24 import ops
DEFAULT=Path('D:/paper/outputs/experiments/s2_t0_fair24_20261013')
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def state_snapshot(old_state):
 p=Path(old_state)
 if not p.exists():return {'exists':False,'release_authorized':False}
 b=p.read_bytes();j=json.loads(b);return {'exists':True,'raw_sha256':hashlib.sha256(b).hexdigest(),'phase':j.get('phase',j.get('stage')),'release_authorized':False,'meaning':'Computephase/labelscomplete never grants future-owner exclusion'}
def handoff_valid(value,port):
 return value.get('format')=='s2t24_owner_resource_handoff_v1' and port in value.get('released_ports',[]) and value.get('persistent_exclusion_verified') is True and value.get('active_tasks_by_port',{}).get(str(port))==[] and value.get('future_lane_reserved_by_port',{}).get(str(port)) is False and bool(value.get('coordinator_exclusion_evidence_ref'))
def loop(root,old_state,deadline):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);lock=root/'controller.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);state=read(root/'state.json') if (root/'state.json').exists() else {'tasks':{},'attempts':{},'scientific_submissions':0}
 try:
  while time.time()<deadline:
   old=state_snapshot(old_state);handoff_path=root.parent/'preflight/s1_ai17_resource_handoff.json';handoff=read(handoff_path) if handoff_path.exists() else {};ports=[p for p in ops.PORTS if handoff_valid(handoff,p)];state.update(observed_utc=now(),old_owner_state=old,eligible_handoff_ports=ports,phase='resource_waiting' if not ports else 'handoff_requires_root_live_source_engineering_admission',gpu_idle_is_release=False)
   # Root supplies actual frozen plan only after source/data/engineering qualification.
   plan_path=root/'approved_run_plan.json'
   if ports and plan_path.exists():
    plan=read(plan_path)
    if plan.get('admitted_by_root') is not True or plan.get('uniform_request_or_explicit_historical_reproduction_authority') is not True:raise ValueError('actual root admission missing')
    for key,task in plan['tasks'].items():
     if task['executor_port'] not in ports:continue
     if key not in state['tasks']:
      registered=ops.api(task['executor_port'],'/api/t0compare/submit',task['body']);state['tasks'][key]={'run_id':registered['job']['run_id'],'executor_port':task['executor_port'],'status':registered['status'],'request_id':task['body']['request_id']};state['scientific_submissions']+=1
    state['phase']='registered_resource_holds_waiting_fresh_admission'
   tmp=root/'state.tmp.json';tmp.write_text(json.dumps(state,indent=2),encoding='utf8');tmp.replace(root/'state.json');time.sleep(45)
 finally:os.close(fd);lock.unlink()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--root',type=Path,default=DEFAULT/'controller');p.add_argument('--old-state',type=Path,default=Path('D:/paper/outputs/experiments/first_stage_ai_search_20261013/round6/control/labels_runtime_deps_02/state.json'));p.add_argument('--deadline-unix',type=float);a=p.parse_args()
 if not a.execute:print(json.dumps({'prepared':True,'remote_calls':0,'SCI_calls':0,'handoff_required':True,'max_workers_plan':'2 perreleasedGPU; onecard3onlyactualengineeringqualification','admission_not_started':True}))
 else:
  if not a.deadline_unix:p.error('bounded actual plan deadline required')
  loop(a.root,a.old_state,a.deadline_unix)
