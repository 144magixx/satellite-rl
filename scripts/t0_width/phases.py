"""Frozen-source and explicit workbench phase controls for W26 (no science itself)."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

from . import remote_ops as remote

REPO=Path(__file__).resolve().parents[2]
LOCAL=Path('D:/paper/outputs/experiments/s2_width26_20261014')


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def save_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    if path.exists() and path.read_bytes()!=raw:raise ValueError('immutable phase record already differs: '+str(path))
    if not path.exists():path.write_bytes(raw)
    return dict(path=str(path),sha256=digest(path))


def read_remote(port,path):
    client=remote.connect(port)
    try:
        with client.open_sftp() as sftp:
            with sftp.open(path,'rb') as stream:return stream.read()
    finally:client.close()


def publish(local_path,remote_path):
    local_path=Path(local_path);expected=digest(local_path)
    client=remote.connect(20418)
    try:
        with client.open_sftp() as sftp:
            try:
                with sftp.open(remote_path,'rb') as stream:existing=stream.read()
            except FileNotFoundError:existing=None
    finally:client.close()
    if existing is not None:
        if hashlib.sha256(existing).hexdigest()!=expected:raise ValueError('remote immutable record differs')
    else:remote.upload(20418,local_path,remote_path)
    return dict(path=remote_path,sha256=expected)


def package_source(commit,phase):
    folder=LOCAL/'deployment'/phase;folder.mkdir(parents=True,exist_ok=True)
    path=folder/('source_'+commit+'.tar.gz')
    paths=['src','scripts','pyproject.toml']
    paths += subprocess.check_output(['git','ls-tree','-r','--name-only',commit,'tests'],cwd=REPO).decode().splitlines()
    paths=[p for p in paths if not p.startswith('tests/') or p.startswith('tests/test_t0_width')]
    raw=subprocess.check_output(['git','archive','--format=tar',commit,*paths],cwd=REPO)
    files={}
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as incoming:
        members=[m for m in incoming if m.isfile()]
        with tarfile.open(path,'w:gz') as target:
            for member in members:
                data=incoming.extractfile(member).read();files[member.name]=hashlib.sha256(data).hexdigest()
                target.addfile(member,io.BytesIO(data))
    value=dict(source_commit=commit,files=files,python_manifest={p:s for p,s in files.items() if p.endswith('.py') and p.startswith(('src/','scripts/'))},archive_SHA=digest(path),archive_bytes=path.stat().st_size)
    save_new(folder/'source_members.json',value)
    return path,value


def install_source(commit,phase):
    archive,manifest=package_source(commit,phase)
    source=remote.BASE+'/source/'+commit
    reference=publish(archive,remote.BASE+'/deployment/'+phase+'/'+archive.name)
    code='''
import pathlib,hashlib,tarfile,json
root=pathlib.Path(SOURCE);manifest=FILES;archive=pathlib.Path(ARCHIVE)
if hashlib.sha256(archive.read_bytes()).hexdigest()!=ARCHIVE_SHA:raise ValueError('source transport SHA')
if root.exists() and any(root.iterdir()):
 for name,expected in manifest.items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError('existing source differs')
else:
 root.mkdir(parents=True,exist_ok=True)
 with tarfile.open(archive,'r:gz') as t:
  seen=set()
  for m in t:
   p=(root/m.name).resolve()
   if not m.isfile() or not p.is_relative_to(root.resolve()) or m.name not in manifest or m.name in seen:raise ValueError('source archive members')
   raw=t.extractfile(m).read()
   if hashlib.sha256(raw).hexdigest()!=manifest[m.name]:raise ValueError('source member SHA')
   p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);seen.add(m.name)
  if seen!=set(manifest):raise ValueError('source missing members')
print(json.dumps(dict(source_root=str(root),members=len(manifest),passed=True)))
'''.replace('ARCHIVE_SHA',repr(reference['sha256'])).replace('ARCHIVE',repr(reference['path'])).replace('SOURCE',repr(source)).replace('FILES',repr(manifest['files']))
    result=remote.remote_python(20418,code,interpreter=remote.DQN,timeout=60)
    save_new(LOCAL/'deployment'/phase/'source_install.json',result)
    if result['exit_code']:raise RuntimeError(result['stderr'][-1500:])
    verify='''
import pathlib,hashlib,json
root=pathlib.Path(SOURCE);manifest=FILES
actual={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
if actual!=manifest:raise ValueError('source complete independent readback differs')
print(json.dumps(dict(source_root=str(root),files=actual,passed=True)))
'''.replace('SOURCE',repr(source)).replace('FILES',repr(manifest['files']))
    with ThreadPoolExecutor(max_workers=2) as pool:
        values=list(pool.map(lambda port:(port,remote.remote_python(port,verify,interpreter=remote.DQN,timeout=60)),remote.PORTS))
    for port,value in values:
        save_new(LOCAL/'deployment'/phase/f'source_readback_{port}.json',value)
        if value['exit_code']:raise RuntimeError('source readback failed')
    return source,manifest


def resource_refs(phase):
    refs={}
    with ThreadPoolExecutor(max_workers=2) as pool:values=list(pool.map(remote.probe,remote.PORTS))
    for value in values:
        port=value['ssh_port'];save_new(LOCAL/'deployment'/phase/f'resources_observation_{port}.json',value)
        o=value.get('observation',{})
        if value['exit_code'] or o.get('gpu',{}).get('exit_code')!=0 or o.get('compute',{}).get('exit_code')!=0 or o['compute']['stdout'].strip():raise ValueError('GPU resource unavailable')
        foreign=[p for p in o['processes'] if remote.BASE not in p['command'] and 'raw_cdn_worker' not in p['command']]
        if foreign:raise ValueError('foreign active task owner prevents phase switch')
        receipt=dict(experiment_id='S2-W26',root_approved=True,passed=True,executor_port=port,
            checked_UTC=o['observed_utc'],boot_id=o['boot_id'],gpu_uuid=o['gpu']['stdout'].split(',')[0].strip(),max_age_seconds=300,
            own_processes=o['processes'],resource_scope='no GPU compute and no foreign project worker; own idle services/byte transport allowed')
        local=LOCAL/'deployment'/phase/f'resources_{port}.json';save_new(local,receipt)
        refs[port]=publish(local,remote.BASE+'/deployment/'+phase+'/'+local.name)
    return refs


def rotate_services(commit,phase,source,manifest,tasks,extra_admissions=None):
    """Rotate only idle owned W26 services, retaining every task's old immutable plan."""
    source_record=dict(experiment_id='S2-W26',root_approved=True,passed=True,source_commit=commit,
                       source_root=source,member_count=len(manifest['files']),allowed_scopes=sorted({t['mode'] for t in tasks.values()}),
                       actual_source_checks='Git blob archive and two-lane complete byte readback',
                       scientific_training_admission=False)
    path=LOCAL/'deployment'/phase/'source_admission.json';save_new(path,source_record)
    source_ref=publish(path,remote.BASE+'/deployment/'+phase+'/'+path.name)
    resources=resource_refs(phase);plans={}
    for port in remote.PORTS:
        plan=dict(schema='W26AuthorizedPlanV1',group='S2-W26',authorized=True,source_commit=commit,source_root=source,
                  source_manifest=manifest['python_manifest'],admission_refs=dict(source=source_ref,resources=resources[port],**(extra_admissions or {})),tasks=tasks)
        p=LOCAL/'deployment'/phase/f'plan_{port}.json';save_new(p,plan)
        plans[port]=publish(p,remote.BASE+'/deployment/'+phase+'/'+p.name)
        code='''
import pathlib,json,os,signal,subprocess,time,urllib.request
root=pathlib.Path(TASK_ROOT);root.mkdir(parents=True,exist_ok=True);owner=root/'server_owner.json'
def birth(pid):
 try:
  t=pathlib.Path('/proc',str(pid),'stat').read_text();f=t.rsplit(') ',1)[1].split();return int(f[19]),f[0]
 except ProcessLookupError:return None
 except FileNotFoundError:return None
def live(o):return o.get('pid')is not None and birth(o['pid'])== (o.get('process_birth'),'S')
if owner.exists():
 old=json.loads(owner.read_text());state=birth(old['pid'])
 if state is not None and state[0]==old['process_birth'] and state[1] not in ('Z','X'):
  for p in (root/'runs').glob('*/status.json'):
   row=json.loads(p.read_text());child=birth(row.get('pid',-1))
   if child is not None and child[0]==row.get('process_birth') and child[1]not in('Z','X'):raise ValueError('live worker blocks service rotation')
  argv=pathlib.Path('/proc',str(old['pid']),'cmdline').read_bytes().split(b'\\0')
  if b'satellite_rl.t0_width.workbench' not in argv or str(root).encode() not in argv or b'--worker' in argv:raise ValueError('server owner command differs')
  if birth(old['pid'])!=state:raise ValueError('server birth changed')
  os.kill(old['pid'],signal.SIGTERM)
  for _ in range(30):
   current=birth(old['pid'])
   if current is None or current[1] in ('Z','X'):break
   time.sleep(.1)
  else:raise RuntimeError('old owned server did not exit')
source=pathlib.Path(SOURCE);env=os.environ.copy();env['PYTHONPATH']=str(source/'src')+os.pathsep+str(source);env['PYTHONDONTWRITEBYTECODE']='1'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
env['SATELLITE_RL_OUTPUT_ROOT']=str(root/'archive')
command=[DQN,'-B','-m','satellite_rl.t0_width.workbench','--root',str(root),'--port',str(API),'--executor-port',str(PORT),'--plan',PLAN,'--max-workers','2']
with (root/'server.log').open('ab')as log:child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(json.dumps(dict(pid=child.pid,birth=birth(child.pid),command=command,source_root=str(source))))
'''.replace('SOURCE',repr(source)).replace('TASK_ROOT',repr(remote.BASE+'/'+str(port))).replace('DQN',repr(remote.DQN)).replace('API',str(remote.API_PORTS[port])).replace('PORT',str(port)).replace('PLAN',repr(plans[port]['path']))
        value=remote.remote_python(port,code,interpreter=remote.DQN,timeout=30)
        save_new(LOCAL/'deployment'/phase/f'service_rotation_{port}.json',value)
        if value['exit_code']:raise RuntimeError(value['stderr'][-1800:])
    for port in remote.PORTS:
        for _ in range(12):
            try:
                response=remote.api(port,'/api/status',timeout=10)
                if response['body']['source_commit']==commit and response['body']['plan_sha256']==plans[port]['sha256']:break
            except Exception:pass
            time.sleep(.5)
        else:raise RuntimeError('new source/plan not actually served')
        save_new(LOCAL/'deployment'/phase/f'service_actual_{port}.json',response)
    return plans


def submit(phase,plans,task_key,port,request_id):
    request=dict(plan_sha256=plans[port]['sha256'],task_key=task_key,request_id=request_id)
    response=remote.api(port,'/api/runs',request,timeout=30)
    save_new(LOCAL/'deployment'/phase/(task_key+'_POST.json'),dict(request=request,response=response))
    if response['status']!=200:raise RuntimeError(str(response))
    return response['body']
