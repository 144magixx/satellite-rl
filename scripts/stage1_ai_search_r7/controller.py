"""One immutable phase, finite persistent dispatch. Unknown POST is never replayed."""
import argparse,json,hashlib,time,urllib.request
from pathlib import Path
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.geoload_runner import exclusive_lock
from satellite_rl.workbench.coverage_ai_search import pin
from satellite_rl.workbench.storage import read_json,digest
from .contract import GROUP,FIRST_PHASES,PORTS,DESIGN_SHA
from .plan_builder import audit

def semantic(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf8')).hexdigest()

def validate_spec(spec):
    if spec.get('group')!=GROUP or spec.get('phase') not in FIRST_PHASES or spec.get('automatic_extension')is not False or spec.get('maximum_runtime_seconds')!=43200:raise ValueError('finite12h first-wave controller only')
    plan=read_json(pin(spec['plan_ref']),{});audit(plan)
    if spec['phase']!=plan['phase'] or spec['scientific_commit']!=plan['scientific_commit']:raise ValueError('exact phase/source spec')
    receipt=read_json(pin(spec['controller_root_admission_ref']),{})
    if receipt.get('passed')is not True or receipt.get('group')!=GROUP or receipt.get('plan_sha256')!=spec['plan_ref']['sha256'] or receipt.get('scientific_commit')!=spec['scientific_commit'] or receipt.get('design_sha256')!=DESIGN_SHA:raise ValueError('root exact controller phase admission')
    tasks=spec['tasks']
    if sorted(t['task_key'] for t in tasks)!=sorted(plan['tasks']) or len(tasks)!=len(plan['tasks']):raise ValueError('one original request per phase task')
    requests=[]
    for task in tasks:
        body=task['body'];key=task['task_key'];contract=plan['tasks'][key]
        if task['port'] not in PORTS or task['port']!=contract['executor_port'] or body['task_key']!=key or body['plan']!=spec['plan_ref'] or any(body.get(k)!=v for k,v in contract.items()):raise ValueError('exact lane/payload')
        requests.append(body['request_id'])
    if len(set(requests))!=len(requests):raise ValueError('duplicate requests')
    return plan

def tick(spec,state,api,observe,persist):
    """A bounded observation/dispatch pass, injectable for pure fault tests."""
    if state.get('phase') in ('completed','attention_required','unknown_post_outcome','dispatch_stopped_workers_preserved'):return state
    for task in spec['tasks']:
        port=task['port'];body=task['body'];request=body['request_id'];intent=state['intents'].get(request)
        if intent and intent.get('phase')=='post_pending':
            state.update(phase='unknown_post_outcome',failed_request=request,automatic_retry=False);persist(state);return state
        if intent and intent.get('phase')=='completed':continue
        if not intent or not intent.get('run_id'):
            try:
                route=observe(port,spec['scientific_commit'])
                if not route.get('ready') or not route.get('root_accepted') or route.get('ssh_port')!=port or route.get('scientific_commit')!=spec['scientific_commit']:raise ValueError('fresh root route required')
            except Exception as error:
                state.update(phase='attention_required',failure_operation='resource_route',error=str(error));persist(state);return state
        intent=state['intents'].setdefault(request,dict(port=port,payload_sha256=semantic(body),phase='intent_saved'))
        if intent['payload_sha256']!=semantic(body):raise ValueError('immutable original body changed')
        persist(state)
        operation='GET'
        try:
            if not intent.get('run_id'):
                operation='POST';intent['phase']='post_pending';persist(state)
                result=api(port,'/api/ai-search-r7/tasks',body);intent['run_id']=result['run_id'];intent['phase']='submitted';persist(state)
            operation='GET';detail=api(port,'/api/ai-search-r7/tasks/'+intent['run_id']);intent['phase']=detail['status']['stage'];persist(state)
            if intent['phase'] in ('failed','stopped','cancelled','interrupted'):
                state.update(phase='attention_required',failed_request=request,failure_operation='terminal_task');persist(state);return state
        except Exception as error:
            state.update(phase='unknown_post_outcome' if operation=='POST' else 'attention_required',failure_operation=operation,failed_request=request,error=str(error),automatic_retry=False);persist(state);return state
    if len(state['intents'])==len(spec['tasks']) and all(i.get('phase')=='completed' for i in state['intents'].values()):state['phase']='completed';persist(state)
    return state

def run(spec_path,state_root,api,observe):
    spec_path=Path(spec_path);spec=read_json(spec_path,{});validate_spec(spec);pinned=digest(spec_path)
    root=Path(state_root);root.mkdir(parents=True,exist_ok=True);path=root/'state.json'
    state=read_json(path,{}) or dict(group=GROUP,spec_sha256=pinned,phase='running',intents={},started_unix_seconds=time.time())
    if state['spec_sha256']!=pinned:raise ValueError('original state/spec changed')
    self_sha=digest(__file__)
    def persist(value):atomic_json(path,value)
    with exclusive_lock(root/'controller.lock'):
        while True:
            if digest(spec_path)!=pinned or digest(__file__)!=self_sha:state.update(phase='attention_required',failure_operation='frozen_source_changed');persist(state);return state
            if time.time()-state['started_unix_seconds']>=43200:state.update(phase='attention_required',failure_operation='finite_controller_wall_cap');persist(state);return state
            if (root/'stop.request').exists():state['phase']='dispatch_stopped_workers_preserved';persist(state);return state
            tick(spec,state,api,observe,persist)
            if state['phase']!='running':return state
            time.sleep(30)

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--state-root',required=True);args=p.parse_args();spec=read_json(args.spec,{})
    def api(port,path,body=None):
        base=spec['workbench_urls'][str(port)]
        req=urllib.request.Request(base.rstrip('/')+path,data=json.dumps(body).encode() if body else None,headers={'Content-Type':'application/json'},method='POST' if body else 'GET')
        with urllib.request.urlopen(req,timeout=45) as response:return json.load(response)
    def observe(port,commit):
        # A root-approved fresh receipt is still rechecked by workbench launch.
        receipt=read_json(pin(spec['resource_route_refs'][str(port)]),{})
        from datetime import datetime,timezone
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(receipt['observed_utc'])).total_seconds()
        if not -5<=age<=120:raise ValueError('route receipt expired; root must refresh actual resource facts')
        return receipt
    run(args.spec,args.state_root,api,observe)

if __name__=='__main__':main()
