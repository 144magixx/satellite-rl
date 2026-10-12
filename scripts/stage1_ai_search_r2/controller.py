"""Independent R2 finite spec dispatcher; no automatic pilot extension."""
from datetime import datetime,timezone
import argparse,hashlib,json,time
from pathlib import Path
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.geoload_runner import exclusive_lock
from . import ops


def route(port,commit):
    i=ops.instance(port)
    code=f'''import sys,json,pathlib,subprocess
registry=json.loads((pathlib.Path({i['ops']!r})/'service.json').read_text())
if registry['source_commit']!={commit!r}:raise ValueError('route source is not actual service epoch')
repo=pathlib.Path(registry['source_root']).resolve()
if not repo.is_relative_to(pathlib.Path({i['root']!r}).resolve()):raise ValueError('service repo outside isolated R2 root')
actual=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
if actual!={commit!r}:raise ValueError('actual route repo commit changed')
sys.path.insert(0,str(repo/'src'))
from satellite_rl.workbench.ai_search_resources import observe
from satellite_rl.training.run_monitor import atomic_json
raw=observe({i['archive']!r});record={{'group':{ops.GROUP!r},'ssh_port':{port},'scientific_commit':{commit!r},'observed_utc':raw['observed_utc'],'gpu_uuid':raw['gpu'].split(',')[0].strip(),'ready':not raw['foreign_workers'],'observation':raw}}
atomic_json(pathlib.Path({i['ops']!r})/'resource_route.json',record);print(json.dumps(record))
'''
    r=ops.remote_python(code,port,timeout=40)
    if r['exit_code']:raise RuntimeError(r['stderr'][-900:])
    return json.loads(r['stdout'].splitlines()[-1])


def run(spec_path,state_root):
    spec_path=Path(spec_path).resolve();raw=spec_path.read_bytes();spec=json.loads(raw);pinned=hashlib.sha256(raw).hexdigest()
    if spec['group']!=ops.GROUP or spec.get('automatic_extension')is not False:raise ValueError('finite R2 only; extension separately root-admitted')
    root=Path(state_root).resolve();root.mkdir(parents=True,exist_ok=True);statefile=root/'state.json'
    state=json.loads(statefile.read_text(encoding='utf-8')) if statefile.exists() else {'group':ops.GROUP,'spec_sha256':pinned,'intents':{},'phase':'running'}
    if state['spec_sha256']!=pinned:raise ValueError('state spec differs')
    self_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with exclusive_lock(root/'controller.lock'):
        while True:
            if hashlib.sha256(spec_path.read_bytes()).hexdigest()!=pinned or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=self_sha:
                raise ValueError('frozen source/spec changed')
            if (root/'stop.request').exists():state['phase']='dispatch_stopped_workers_preserved';atomic_json(statefile,state);return
            state['routes']={}
            for port in sorted({x['port'] for x in spec['tasks']}):
                try:state['routes'][str(port)]=route(port,spec['scientific_commit'])
                except Exception as e:state['routes'][str(port)]={'ready':False,'error':str(e)}
            for task in spec['tasks']:
                port=task['port'];body=task['body'];request=body['request_id']
                if not state['routes'][str(port)].get('ready'):continue
                intent=state['intents'].setdefault(request,{'port':port,'payload_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),'phase':'intent_saved'})
                atomic_json(statefile,state)
                try:
                    if not intent.get('run_id'):
                        job=ops.api(port,'/api/ai-search-r2/tasks',body);intent['run_id']=job['run_id'];atomic_json(statefile,state)
                    detail=ops.api(port,'/api/ai-search-r2/tasks/'+intent['run_id']);intent.update(status=detail['status'],phase=detail['status']['stage'],last_error=None)
                except Exception as e:intent['last_error']=str(e)
                state['observed_utc']=datetime.now(timezone.utc).isoformat();atomic_json(statefile,state)
            if len(state['intents'])==len(spec['tasks']) and all(i.get('phase')in ('completed','failed','stopped','cancelled','interrupted') for i in state['intents'].values()):
                state['phase']='completed' if all(i['phase']=='completed' for i in state['intents'].values()) else 'attention_required';atomic_json(statefile,state);return
            time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',required=True);parser.add_argument('--state-root',required=True);args=parser.parse_args();run(args.spec,args.state_root)
