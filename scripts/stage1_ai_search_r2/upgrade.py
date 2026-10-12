"""Review an idle three-lane service handoff; execute only after new freeze."""
import argparse,json,subprocess
from pathlib import Path
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.runtime import pid_alive
from . import ops,deploy

OLD='f02c0dcbf76561c325b5a70a1a5b9bf98882b35d'


def lane_code(port,execute=False,known_stopped=None):
    i=ops.instance(port)
    return f'''import pathlib,json,os,signal,time,urllib.request,hashlib
root=pathlib.Path({i['root']!r});registry=root/'ops/service.json';original=registry.read_bytes();service=json.loads(original)
if service['source_commit']!={OLD!r} or service['group']!={ops.GROUP!r}:raise ValueError('old registry is not the original idle f02 service')
def birth(pid):
 p=pathlib.Path('/proc')/str(pid)/'stat'
 if not p.exists():return None
 raw=p.read_text();return int(raw[raw.rfind(')')+2:].split()[19])
known={known_stopped!r}
if birth(service['pid']) is None and known and known['old_registry_sha256']==hashlib.sha256(original).hexdigest():
 print(json.dumps({{**known,'known_stop_reused':True,'old_service_stopped_birth_verified':True}}));raise SystemExit(0)
if birth(service['pid'])!=service['birth_ticks']:raise ValueError('old service birth differs; no signal allowed')
command=(pathlib.Path('/proc')/str(service['pid'])/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
if 'satellite_rl.workbench.server' not in command or str(root/'archive') not in command:raise ValueError('old service actual command differs')
with urllib.request.urlopen('http://127.0.0.1:{i['api_port']}/api/ai-search-r2/tasks',timeout=8) as response:tasks=json.load(response)['tasks']
if len(tasks)!=2 or any(r['job']['scientific_commit']!={OLD!r} or r['status']['stage']!='completed' for r in tasks):raise ValueError('both original pilot tasks must be completed; no queued/unknown job')
for record in tasks:
 path=root/'archive/outputs/workbench/jobs'/record['job']['run_id'];launch=json.loads((path/'launch.json').read_text());status=json.loads((path/'status.json').read_text())
 if birth(launch['pid']) is not None:raise ValueError('original science worker PID is still present; cannot stop service')
 if status['stage']!='completed' or status['run_id']!=record['job']['run_id']:raise ValueError('original status not closed')
foreign=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  c=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode('utf-8','replace')
  if 'satellite_rl.workbench.runner' in c:foreign.append(int(p.name))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
if foreign:raise ValueError('physical lane has an active science worker')
receipt={{'ssh_port':{port},'old_service':service,'actual_command':command,'old_registry_sha256':hashlib.sha256(original).hexdigest(),'jobs':[r['job']['run_id'] for r in tasks],'all_jobs_completed':True,'all_workers_exited':True,'old_repo_preserved':str(root/'repo'),'execute':{execute!r}}}
if {execute!r}:
 history=root/'ops/service_history';history.mkdir(exist_ok=True);target=history/({OLD!r}+'_'+str(service['pid'])+'_'+str(service['birth_ticks'])+'.json')
 if target.exists() and target.read_bytes()!=original:raise ValueError('original service history bytes changed')
 target.write_bytes(original)
 if birth(service['pid'])!=service['birth_ticks']:raise ValueError('birth changed before signal')
 os.kill(service['pid'],signal.SIGTERM)
 for _ in range(40):
  if birth(service['pid']) is None:break
  time.sleep(.25)
 else:raise ValueError('old service did not exit; retained without SIGKILL or new start')
 receipt['old_service_stopped_birth_verified']=True
print(json.dumps(receipt))
'''


def upgrade(commit,review_path,output,execute=False):
    if len(commit)!=40 or set(commit)-set('0123456789abcdef') or commit==OLD:raise ValueError('new exact frozen operations commit required')
    review=json.loads(Path(review_path).read_text(encoding='utf-8'))
    if review.get('passed')is not True or review.get('scientific_commit')!=commit:raise ValueError('new source freeze review required')
    if deploy.git('rev-parse','HEAD')!=commit or deploy.git('status','--porcelain','--','src','scripts','tests','pyproject.toml'):
        raise ValueError('clean newly frozen operations source required')
    for name in ('prepare_common','closed_pool'):
        launch=json.loads((ops.LOCAL/'control'/name/'launch.json').read_text(encoding='utf-8'))
        state=json.loads((ops.LOCAL/'control'/name/'state.json').read_text(encoding='utf-8'))
        if state.get('phase')!='completed' or pid_alive(launch['pid'],launch['process_start_time']):raise ValueError('original controller remains active or unknown')
    journal_path=Path(str(output)+'.journal.json')
    journal=json.loads(journal_path.read_text(encoding='utf-8')) if journal_path.exists() else {'new_commit':commit,'checks':{},'stop_intents':{},'stopped':{},'started':{}}
    if journal['new_commit']!=commit:raise ValueError('upgrade journal belongs to another epoch')
    # Every mutation has a durable original intent and an original-result slot.
    for port in ops.SSH_PORTS:
        key=str(port)
        if key in journal['started']:continue
        if key in journal['stopped'] or key in journal['stop_intents']:
            i=ops.instance(port)
            probe=f'''import pathlib,json,subprocess,urllib.request
root=pathlib.Path({i['root']!r});r=json.loads((root/'ops/service.json').read_text())
if r['source_commit']!={commit!r}:print(json.dumps({{'recovered_new_service':False}}))
else:
 raw=(pathlib.Path('/proc')/str(r['pid'])/'stat').read_text();birth=int(raw[raw.rfind(')')+2:].split()[19])
 command=(pathlib.Path('/proc')/str(r['pid'])/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 actual=subprocess.run(['git','-C',r['source_root'],'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
 if birth!=r['birth_ticks'] or str(root/'archive') not in command or actual!={commit!r}:raise ValueError('unknown new startup cannot be verified')
 with urllib.request.urlopen('http://127.0.0.1:{i['api_port']}/api/ai-search-r2/schema',timeout=4) as response:schema=json.load(response)
 if schema['group']!={ops.GROUP!r}:raise ValueError('unknown startup API belongs to another group')
 print(json.dumps({{'recovered_new_service':True,'service':{{'registry':r,'schema':schema,'actual_command':command}}}}))
'''
            recovered=deploy.remote(port,probe)
            if recovered.get('recovered_new_service'):
                journal['started'][key]=recovered;atomic_json(journal_path,journal);continue
        known=journal['stopped'].get(key) or journal['stop_intents'].get(key)
        journal['checks'][key]=deploy.remote(port,lane_code(port,False,known))
    if not execute:
        atomic_json(output,{'group':ops.GROUP,'new_commit':commit,'dry_run':True,'checks':journal['checks'],'science_jobs_submitted':0});return journal['checks']
    atomic_json(journal_path,journal)
    for port in ops.SSH_PORTS:
        key=str(port)
        if key in journal['stopped'] or key in journal['started']:continue
        intent=journal['checks'][key];journal['stop_intents'][key]=intent;atomic_json(journal_path,journal)
        journal['stopped'][key]=deploy.remote(port,lane_code(port,True,intent));atomic_json(journal_path,journal)
    folder=ops.LOCAL/'deployment';folder.mkdir(parents=True,exist_ok=True);bundle=folder/(commit+'.bundle')
    if not bundle.exists():subprocess.run(['git','-C',str(deploy.SOURCE),'bundle','create',str(bundle),'HEAD'],check=True)
    for port in ops.SSH_PORTS:
        key=str(port)
        if key in journal['started']:continue
        # deploy.lane recovers only a same-commit/same-birth registry after an
        # unknown startup outcome; it never creates another live service.
        journal['started'][key]=deploy.lane(port,commit,bundle);atomic_json(journal_path,journal)
    value={'group':ops.GROUP,'new_commit':commit,'old_services':journal['stopped'],'new_services':journal['started'],'science_jobs_submitted':0}
    atomic_json(output,value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('commit','source-review','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    print(json.dumps(upgrade(a.commit,a.source_review,a.output,a.execute)))
