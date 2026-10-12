"""Persistent finite-contract dispatcher; science only through four workbenches.

The calling process survives chat turns. An immutable spec is one round/epoch;
later rounds use a new spec and explicit source epoch, never hot-edit a worker.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from copy import deepcopy
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.geoload_runner import exclusive_lock
from . import ops

TERMINAL = {'completed', 'failed', 'stopped', 'cancelled', 'interrupted'}


def refresh_route(port, commit, f20_return=None):
    info = ops.instance(port)
    if port == 20474 and not f20_return:
        return {'group': 'S1-AI17', 'ssh_port': port, 'ready': False,
                'reason': 'F20 final return proof absent; no 74 write/compute admission',
                'observed_utc': datetime.now(timezone.utc).isoformat()}
    code = f'''import sys,json,pathlib
sys.path.insert(0,{info['repo']!r}+'/src')
from satellite_rl.workbench.ai_search_resources import observe
from satellite_rl.training.run_monitor import atomic_json
observed=observe({info['archive']!r})
value={{'group':'S1-AI17','ssh_port':{port},'scientific_commit':{commit!r},'observed_utc':observed['observed_utc'],'gpu_uuid':observed['gpu'].split(',')[0].strip(),'ready':not observed['foreign_workers'],'observation':observed}}
release={f20_return!r}
if release:value['f20_return_receipt']=release
atomic_json(pathlib.Path({info['ops']!r})/'resource_route.json',value)
print(json.dumps(value))
'''
    result=ops.remote_python(code,port,timeout=45)
    if result['exit_code']:raise RuntimeError(result['stderr'][-800:])
    return json.loads(result['stdout'].splitlines()[-1])


def validate_spec(spec):
    if spec.get('group') != 'S1-AI17' or not isinstance(spec.get('tasks'), list) or not spec['tasks']:
        raise ValueError('explicit frozen group/task spec required')
    requests=[]
    for item in spec['tasks']:
        port=item['port'];ops.instance(port);body=item['body']
        if body.get('executor_port') != port or not body.get('request_id'):
            raise ValueError('request owner/ID missing')
        requests.append(body['request_id'])
    if len(set(requests)) != len(requests):raise ValueError('spec repeats a scientific request')
    return spec


def watch_training_nodes(spec,state):
    """Publish only pre-frozen templates, pinned to actual immutable node bytes."""
    requests=spec.get('watch_training_requests',[])
    state.setdefault('node_events',{});state.setdefault('dynamic_tasks',[])
    for request in requests:
        intent=state['task_intents'].get(request,{})
        job=intent.get('job')
        if not job or job.get('mode')!='train':continue
        port=intent['port'];root=job['model_dir']
        code=f'''import hashlib,json,pathlib
root=pathlib.Path({root!r});rows=[]
for node in (2500,5000,7500,10000):
 p=root/'diagnostic_nodes'/str(node)/'node_ready.json'
 if not p.is_file():continue
 data=p.read_bytes();ready=json.loads(data);source=pathlib.Path(ready['producer_plan']['path']);raw=source.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=ready['producer_plan']['sha256']:raise ValueError('original producer plan changed')
 rows.append({{'node':node,'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'ready':ready,'source_plan':json.loads(raw)}})
print(json.dumps(rows))
'''
        result=ops.remote_python(code,port,timeout=35)
        if result['exit_code']:raise RuntimeError(result['stderr'][-800:])
        rows=json.loads(result['stdout'].splitlines()[-1])
        for observed in rows:
            ready=observed['ready'];node=observed['node'];event_key=f'{job["run_id"]}:{node}'
            if (ready['producer_run_id']!=job['run_id'] or ready['scientific_commit']!=spec['scientific_commit'] or
                    ready['producer_task_key']!=job['task_key']):raise ValueError('model node belongs to another producer/epoch')
            if event_key in state['node_events']:continue
            original=observed['source_plan'];templates=original.get('node_templates',{}).get(str(node))
            if not templates:raise ValueError('node has no pre-frozen workbench template')
            derived={'group':'S1-AI17','authorized':True,'scientific_commit':spec['scientific_commit'],
                'round':original.get('round'),'model':ready['model'],'design':original.get('design'),
                'tasks':deepcopy(templates),'derived_from':{'source_plan':ready['producer_plan'],
                    'node_ready':{'path':observed['path'],'sha256':observed['sha256']},
                    'producer_run_id':job['run_id'],'producer_task_key':job['task_key'],'update_step':node}}
            # This is model binding of an existing template, not a new science budget.
            folder=ops.LOCAL/'controller/node_contracts';folder.mkdir(parents=True,exist_ok=True)
            local=folder/(job['run_id']+'_'+str(node)+'.json');atomic_json(local,derived)
            sha=hashlib.sha256(local.read_bytes()).hexdigest()
            target=ops.BASE+'/source/node_contracts/'+local.name
            receipt=ops.upload(local,target,port)
            if receipt['remote']!=sha:raise ValueError('published diagnostic contract bytes differ')
            additions=[]
            for task_key,task in templates.items():
                body={**task,'task_key':task_key,'request_id':f'ai17_{job["run_id"]}_{node}_{task_key}',
                      'plan':{'path':target,'sha256':sha}}
                additions.append({'port':task['executor_port'],'body':body})
            state['dynamic_tasks'].extend(additions)
            state['node_events'][event_key]={'phase':'templates_bound_requested','node_ready_sha256':observed['sha256'],
                'contract_sha256':sha,'requests':[item['body']['request_id'] for item in additions],
                'timing':ready['timing'],'actual_completed':False,'budget_newly_added':False}
            spec['tasks'].extend(additions)
    for event in state['node_events'].values():
        event['actual_completed']=all(state['task_intents'].get(r,{}).get('phase')=='completed' for r in event['requests'])


def run(spec_path, state_root, poll_seconds=30):
    spec_path=Path(spec_path).resolve();payload=spec_path.read_bytes()
    spec=validate_spec(json.loads(payload));expected=hashlib.sha256(payload).hexdigest()
    state_root=Path(state_root).resolve();state_root.mkdir(parents=True,exist_ok=True)
    state_path=state_root/'state.json'
    state=json.loads(state_path.read_text()) if state_path.is_file() else {
        'group':'S1-AI17','spec_sha256':expected,'task_intents':{},'routes':{},'phase':'running'}
    if state['spec_sha256'] != expected:raise ValueError('state belongs to another immutable spec')
    original_requests={item['body']['request_id'] for item in spec['tasks']}
    spec['tasks'].extend(item for item in state.get('dynamic_tasks',[]) if item['body']['request_id'] not in original_requests)
    source_path=Path(__file__).resolve();source_sha=hashlib.sha256(source_path.read_bytes()).hexdigest()
    with exclusive_lock(state_root/'controller.lock'):
        while True:
            if hashlib.sha256(spec_path.read_bytes()).hexdigest()!=expected or hashlib.sha256(source_path.read_bytes()).hexdigest()!=source_sha:
                raise ValueError('controller source/spec changed; refusing hot epoch mutation')
            state.update(observed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=source_sha)
            if (state_root/'stop.request').exists():
                state['phase']='dispatch_stopped_running_workers_preserved';atomic_json(state_path,state);return state
            try:watch_training_nodes(spec,state);state['node_watch_error']=None
            except Exception as error:state['node_watch_error']=type(error).__name__+': '+str(error)[:1200]
            ports=sorted(set(item['port'] for item in spec['tasks']))
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures={pool.submit(refresh_route,p,spec['scientific_commit'],spec.get('f20_return_receipt')):p for p in ports}
                for future in as_completed(futures):
                    p=futures[future]
                    try:state['routes'][str(p)]=future.result()
                    except Exception as error:state['routes'][str(p)]={'ready':False,'error':type(error).__name__+': '+str(error)[:1000]}
            atomic_json(state_path,state)
            # Each lane is independently advanced; API request IDs recover unknown outcomes.
            for item in spec['tasks']:
                port=item['port'];body=item['body'];request=body['request_id']
                if not state['routes'].get(str(port),{}).get('ready'):continue
                intent=state['task_intents'].setdefault(request,{'port':port,'body_sha256':hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'phase':'intent_saved'})
                atomic_json(state_path,state)
                try:
                    if not intent.get('run_id'):
                        job=ops.api(port,'/api/ai-search/tasks',body)
                        intent.update(run_id=job['run_id'],phase='bound')
                        atomic_json(state_path,state)
                    detail=ops.api(port,'/api/ai-search/tasks/'+intent['run_id'])
                    intent.update(job=detail['job'],status=detail['status'],phase=detail['status']['stage'],last_error=None)
                    if intent['phase']=='interrupted' and spec.get('resume_interrupted',True):
                        # Same frozen inputs/owner/run_id only, no new task or budget.
                        intent['phase']='resume_pending';atomic_json(state_path,state)
                        intent['resume']=ops.api(port,'/api/ai-search/tasks/'+intent['run_id']+'/resume',{})
                        intent['phase']='queued'
                except Exception as error:
                    intent['last_error']=type(error).__name__+': '+str(error)[:1400]
                atomic_json(state_path,state)
            for event in state.get('node_events',{}).values():
                event['actual_completed']=all(state['task_intents'].get(r,{}).get('phase')=='completed' for r in event['requests'])
            watched_ready=not spec.get('watch_training_requests') or (
                len(state.get('node_events',{}))==4*len(spec['watch_training_requests']) and not state.get('node_watch_error'))
            if watched_ready and len(state['task_intents'])==len(spec['tasks']) and all(i.get('phase') in TERMINAL for i in state['task_intents'].values()):
                state['phase']='completed' if all(i['phase']=='completed' for i in state['task_intents'].values()) else 'attention_required'
                atomic_json(state_path,state);return state
            time.sleep(max(5,min(60,poll_seconds)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',required=True)
    parser.add_argument('--state-root',required=True);parser.add_argument('--poll-seconds',type=int,default=30)
    args=parser.parse_args();run(args.spec,args.state_root,args.poll_seconds)
