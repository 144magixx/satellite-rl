"""Persistent bounded DAG dispatcher. All scientific jobs enter workbench APIs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from . import ops
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.geoload_runner import exclusive_lock
from satellite_rl.workbench.storage import digest, now

def validate_plan(plan):
    if plan.get('group')!='S1-G16' or plan.get('authorized') is not True:raise ValueError('authorized G16 dispatch plan required')
    tasks=plan.get('tasks',{})
    if not tasks or len(tasks)>256:raise ValueError('bounded task DAG required')
    requests=set()
    for key,task in tasks.items():
        ops.instance(task['port'])
        request=task['body']['request_id']
        if request in requests:raise ValueError('duplicate request identity')
        requests.add(request)
        if set(task.get('depends',[]))-set(tasks):raise ValueError('unknown dependency')
    visited=set()
    while len(visited)<len(tasks):
        ready={k for k,v in tasks.items() if set(v.get('depends',[]))<=visited}-visited
        if not ready:raise ValueError('cyclic dispatch DAG')
        visited.update(ready)
    return plan

def tick(plan,state):
    entries=state.setdefault('tasks',{})
    active=[(key,entry) for key,entry in entries.items() if entry.get('run_id') and entry.get('stage') not in ('completed','skipped')]
    def observe(pair):
        key,entry=pair
        try:
            row=ops.api(entry['port'],'/api/runs/'+entry['run_id'])
            return key,{'stage':row['status']['stage'],'status':row['status'],'artifacts':row['artifacts'],'last_observed_at':now(),'transport_error':None}
        except Exception as error:return key,{'transport_error':str(error),'last_failed_observation_at':now()}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for key,observed in pool.map(observe,active):entries[key].update(observed)
    for key,task in plan['tasks'].items():
        if key in entries:continue
        if not all(entries.get(d,{}).get('stage')=='completed' for d in task.get('depends',[])):continue
        # Bodies are concrete frozen configurations. Dependency artifacts must
        # already be pinned in that body; no guessed dynamic science contract.
        try:
            job=ops.api(task['port'],'/api/geoload/jobs',task['body'])
            entries[key]={'port':task['port'],'run_id':job['run_id'],'request_id':task['body']['request_id'],
                          'stage':'queued','submitted_at':now(),'scientific_commit':job['scientific_commit']}
        except Exception as error:
            state.setdefault('submission_errors',{})[key]={'error':str(error),'at':now()}
    state.update(group='S1-G16',scientific_commit=plan['scientific_commit'],updated_at=now(),
                 complete=len(entries)==len(plan['tasks']) and all(v.get('stage')=='completed' for v in entries.values()))
    return state

def run(plan_path,state_path):
    plan_path=Path(plan_path);state_path=Path(state_path)
    plan=validate_plan(json.loads(plan_path.read_text(encoding='utf-8-sig')))
    sha=digest(plan_path)
    state=json.loads(state_path.read_text()) if state_path.exists() else {'group':'S1-G16','plan_sha256':sha,'tasks':{}}
    if state.get('plan_sha256')!=sha:raise ValueError('persistent dispatch plan changed')
    with exclusive_lock(state_path.with_suffix('.lock')):
        while not state.get('complete'):
            tick(plan,state);atomic_json(state_path,state)
            if state.get('complete') or state_path.with_suffix('.stop').exists():break
            time.sleep(20)
    return state

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--state',required=True)
    args=parser.parse_args();run(args.plan,args.state)

if __name__=='__main__':main()
