"""Bounded W26 frozen-source deployment; only the official download task is admitted."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import time
import urllib.request

from scripts.t0_width import remote_ops as remote

COMMIT = '95646a929c7f5daedc211b5ebf493fc35aa040f6'
REPO = Path(__file__).resolve().parents[2]
CONTROL = Path('D:/paper/outputs/experiments/s2_width26_20261014/deployment/data_source_01')
SOURCE = remote.BASE + '/source/' + COMMIT
REMOTE_CONTROL = remote.BASE + '/deployment/data_source_01'
ASSETS = [dict(date='2024-02-23',release_asset_id=153286680,release_asset_bytes=1937121280),
          dict(date='2024-02-24',release_asset_id=153417613,release_asset_bytes=1811374080)]


def sha(raw):return hashlib.sha256(raw).hexdigest()


def receipt(name, value):
    CONTROL.mkdir(parents=True,exist_ok=True)
    path=CONTROL/name
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode('utf8')+b'\n'
    if path.exists():
        if path.read_bytes()!=raw:
            index=2;original=path
            while path.exists():
                path=original.with_name(original.stem+'_%02d'%index+original.suffix);index+=1
            path.write_bytes(raw)
    else:path.write_bytes(raw)
    return path


def run_remote(port, code, name, timeout=60, interpreter='python3'):
    result=remote.remote_python(port,code,timeout=timeout,interpreter=interpreter)
    if (CONTROL/name).exists():
        stem=Path(name).stem
        index=2
        while (CONTROL/(stem+'_%02d.json'%index)).exists():index+=1
        name=stem+'_%02d.json'%index
    receipt(name,result)
    if result['exit_code']:raise RuntimeError(name+' failed: '+result['stderr'][-1500:])
    return json.loads(result['stdout'].splitlines()[-1])


def package():
    members=subprocess.check_output(['git','ls-tree','-r','--name-only',COMMIT],cwd=REPO).decode().splitlines()
    selected=[p for p in members if p.startswith(('src/','scripts/')) or p in ('pyproject.toml','tests/test_t0_width_guard.py')]
    blobs={p:subprocess.check_output(['git','show',COMMIT+':'+p],cwd=REPO) for p in selected}
    manifest={p:sha(raw) for p,raw in blobs.items()}
    archive=CONTROL/'source_95646a9.tar.gz'
    if not archive.exists():
        with tarfile.open(archive,'w:gz') as tar:
            for path,raw in blobs.items():
                name=PurePosixPath(path)
                if name.is_absolute() or '..' in name.parts:raise ValueError('unsafe Git member')
                member=tarfile.TarInfo(path);member.size=len(raw);member.mode=0o644;member.mtime=0
                tar.addfile(member,io.BytesIO(raw))
    with tarfile.open(archive,'r:gz') as tar:
        if len(tar.getmembers())!=len(manifest):raise ValueError('source member count differs')
        for member in tar:
            if not member.isfile() or member.name not in manifest or sha(tar.extractfile(member).read())!=manifest[member.name]:
                raise ValueError('source tar member differs from exact Git blob')
    info=dict(source_commit=COMMIT,files=manifest,python_manifest={p:h for p,h in manifest.items() if p.endswith('.py')},
              archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,
              git_blob_bytes=True,newline_representation={p:dict(CRLF=raw.count(b'\r\n'),LF=raw.count(b'\n')-raw.count(b'\r\n')) for p,raw in blobs.items() if p.endswith('.py')})
    receipt('git_blob_source_manifest.json',info)
    return archive,info


def put_once(local, target):
    check=remote.remote_python(20418,"import pathlib,hashlib,json; p=pathlib.Path("+repr(target)+
        "); print(json.dumps(dict(exists=p.exists(),sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None)))")
    if check['exit_code']:raise RuntimeError('remote target precheck failed')
    state=json.loads(check['stdout'].splitlines()[-1]);expected=sha(Path(local).read_bytes())
    if state['exists']:
        if state['sha256']!=expected:raise ValueError('remote target contains different prior evidence')
    else:remote.upload(20418,local,target)
    return {'path':target,'sha256':expected}


def metadata():
    previous=CONTROL/'official_metadata.json'
    if previous.exists():
        saved=json.loads(previous.read_text(encoding='utf8'))['assets']
        for expected,row in zip(ASSETS,saved):
            if any(row[k]!=v for k,v in expected.items()):raise ValueError('original official metadata identity differs')
        return [{k:v for k,v in row.items() if k!='metadata'} for row in saved]
    values=[]
    for asset in ASSETS:
        req=urllib.request.Request('https://api.github.com/repos/adsblol/globe_history_2024/releases/assets/'+str(asset['release_asset_id']),headers={'User-Agent':'W26-root-source-audit'})
        with urllib.request.urlopen(req,timeout=30) as reply:value=json.load(reply)
        day=asset['date'].replace('-','.');url='https://github.com/adsblol/globe_history_2024/releases/download/v'+day+'-planes-readsb-prod-0/v'+day+'-planes-readsb-prod-0.tar'
        if value['id']!=asset['release_asset_id'] or value['size']!=asset['release_asset_bytes'] or value['browser_download_url']!=url:
            raise ValueError('official current API metadata differs from root prescribed asset')
        values.append({**asset,'source_url':url,'metadata':{k:value.get(k) for k in ('id','size','name','digest','browser_download_url')}})
    receipt('official_metadata.json',dict(checked_UTC=datetime.now(timezone.utc).isoformat(),assets=values))
    return [{k:v for k,v in row.items() if k!='metadata'} for row in values]


def source_install(archive,info):
    target=REMOTE_CONTROL+'/source_95646a9.tar.gz';ref=put_once(archive,target)
    code='''
import pathlib,tarfile,hashlib,json
root=pathlib.Path(SOURCE); archive=pathlib.Path(ARCHIVE); manifest=MANIFEST
if hashlib.sha256(archive.read_bytes()).hexdigest()!=ARCHIVE_SHA:raise ValueError('uploaded source archive SHA differs')
if root.exists() and any(root.iterdir()):
 for name,expected in manifest.items():
  p=root/name
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('existing frozen source differs')
else:
 root.mkdir(parents=True,exist_ok=True)
 with tarfile.open(archive,'r:gz') as tar:
  members=tar.getmembers()
  if len(members)!=len(manifest) or len({m.name for m in members})!=len(members):raise ValueError('source duplicate/count differs')
  for m in members:
   p=(root/m.name).resolve()
   if not m.isfile() or not p.is_relative_to(root.resolve()) or m.name not in manifest:raise ValueError('unsafe frozen member')
   raw=tar.extractfile(m).read()
   if hashlib.sha256(raw).hexdigest()!=manifest[m.name]:raise ValueError('member SHA differs')
   p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
print(json.dumps(dict(source=str(root),archive_SHA=hashlib.sha256(archive.read_bytes()).hexdigest(),member_count=len(manifest),installed=True)))
'''.replace('SOURCE',repr(SOURCE)).replace('ARCHIVE_SHA',repr(ref['sha256'])).replace('ARCHIVE',repr(target)).replace('MANIFEST',repr(info['files']))
    run_remote(20418,code,'source_extract_18.json',interpreter=remote.DQN)
    verify='''
import pathlib,hashlib,json
root=pathlib.Path(SOURCE); manifest=MANIFEST
actual={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
if actual!=manifest:raise ValueError('independent readback member/byte SHA differs')
print(json.dumps(dict(source=str(root),files=actual,verified=True)))
'''.replace('SOURCE',repr(SOURCE)).replace('MANIFEST',repr(info['files']))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda p:run_remote(p,verify,'source_readback_'+str(p)+'.json',interpreter=remote.DQN),remote.PORTS))


def guard_and_import():
    code='''
import pathlib,sys,os,json,unittest,torch
source=pathlib.Path(SOURCE);sys.path[:0]=[str(source/'src'),str(source)];os.environ['PYTHONPATH']=str(source/'src')+os.pathsep+str(source)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import satellite_rl, satellite_rl.t0_width.workbench, satellite_rl.t0_width.guard
for m in (satellite_rl,satellite_rl.t0_width.workbench,satellite_rl.t0_width.guard):
 if not pathlib.Path(m.__file__).resolve().is_relative_to(source):raise ValueError('actual DQN loaded old package')
suite=unittest.defaultTestLoader.discover(str(source/'tests'),pattern='test_t0_width_guard.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
value=dict(tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),skipped=len(result.skipped),passed=result.wasSuccessful(),python=sys.executable,torch=str(torch.__version__),cuda=torch.version.cuda,available=torch.cuda.is_available(),actual_package=satellite_rl.__file__,actual_server=satellite_rl.t0_width.workbench.__file__)
print(json.dumps(value))
if not value['passed'] or value['tests']<6 or value['skipped']:sys.exit(3)
'''.replace('SOURCE',repr(SOURCE))
    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(lambda p:run_remote(p,code,'guard_import_'+str(p)+'.json',timeout=60,interpreter=remote.DQN),remote.PORTS))


def fresh_resources():
    with ThreadPoolExecutor(max_workers=2) as pool:values=list(pool.map(remote.probe,remote.PORTS))
    refs={}
    for value in values:
        port=value['ssh_port'];receipt('fresh_probe_'+str(port)+'.json',value)
        if value.get('exit_code')!=0 or 'observation' not in value:raise ValueError('fresh resource probe failed')
        observation=value['observation']
        blockers=[p for p in observation['processes'] if not (p['command'].startswith('python3 - install '+remote.BASE+' ')
                    and remote.BASE+'/shared_data/transport/manifest.json' in p['command'])]
        if observation['compute']['exit_code'] or observation['compute']['stdout'].strip() or blockers:
            raise ValueError('occupied resource; no interruption/takeover')
        if observation['disk_free']<35*1024**3:raise ValueError('registered20GB peak+15GB safety margin unavailable')
        rows=observation['gpu']['stdout'].splitlines()
        if len(rows)!=1:raise ValueError('one explicit GPU required')
        fields=[x.strip() for x in rows[0].split(',')]
        if int(fields[3])<20000:raise ValueError('GPU margin unavailable')
        for api_port in (9060,9061):
            if 'body' in observation['apis'].get(str(api_port),observation['apis'].get(api_port,{})):
                raise ValueError('existing W26 service not automatically replaced')
        admission=dict(experiment_id='S2-W26',root_approved=True,passed=True,executor_port=port,
            checked_UTC=observation['observed_utc'],boot_id=observation['boot_id'],gpu_uuid=fields[0],max_age_seconds=300,
            admitted_mode='download',scope='source/resource delegated root factual qualification only')
        admission['coexisting_owned_W26_transport_processes']=observation['processes']
        path=receipt('resources_'+str(port)+'.json',admission)
        refs[port]=put_once(path,REMOTE_CONTROL+'/'+path.name)
    return refs


def main():
    CONTROL.mkdir(parents=True,exist_ok=True)
    assets=metadata();archive,info=package();source_install(archive,info);guards=guard_and_import()
    source_admission=dict(experiment_id='S2-W26',root_approved=True,passed=True,source_commit=COMMIT,
        source_root=SOURCE,allowed_scopes=['download','sourceprep','prepare'],readback_ports=list(remote.PORTS),actual_byte_manifest=info['files'],
        guard_tests_each=[g['tests'] for g in guards],engineering_or_scientific_admission=False)
    source_ref=put_once(receipt('source_admission.json',source_admission),REMOTE_CONTROL+'/source_admission.json')
    download=dict(schema='W26OfficialDownloadAdmissionV1',root_approved=True,remote_port=20418,
        canonical_directory=remote.BASE+'/shared_data/official',download_wall_seconds=7200,assets=assets)
    download_ref=put_once(receipt('download_admission.json',download),REMOTE_CONTROL+'/download_admission.json')
    resources=fresh_resources()
    plans={}
    for port in remote.PORTS:
        plan=dict(schema='W26AuthorizedPlanV1',group='S2-W26',authorized=True,source_commit=COMMIT,
            source_root=SOURCE,source_manifest=info['python_manifest'],admission_refs=dict(source=source_ref,resources=resources[port]),
            tasks={'download_official':dict(executor_port=20418,mode='download',config=dict(download_admission_ref=download_ref,
                request_id='w26-official-feb23-feb24-v1',executor_port=20418))})
        ref=put_once(receipt('plan_'+str(port)+'.json',plan),REMOTE_CONTROL+'/plan_'+str(port)+'.json');plans[port]=ref
        code='''
import pathlib,os,sys,json,subprocess,socket
source=pathlib.Path(SOURCE);root=pathlib.Path(ROOT);root.mkdir(parents=True,exist_ok=True)
for p in (9060,9061):
 s=socket.socket();busy=s.connect_ex(('127.0.0.1',p))==0;s.close()
 if busy:raise ValueError('existing server preserved; no takeover')
if (root/'server_owner.json').exists():raise ValueError('existing owner needs separate reconciliation')
env=os.environ.copy();env['PYTHONPATH']=str(source/'src')+os.pathsep+str(source);env['PYTHONDONTWRITEBYTECODE']='1'
command=[DQN,'-B','-m','satellite_rl.t0_width.workbench','--root',str(root),'--port',str(API),'--executor-port',str(PORT),'--plan',PLAN,'--max-workers','2']
with (root/'server.log').open('ab') as log:p=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=pathlib.Path('/proc',str(p.pid),'stat').read_text();birth=int(stat[stat.rfind(')')+2:].split()[19])
print(json.dumps(dict(pid=p.pid,birth_ticks=birth,command=command,background=True)))
'''.replace('SOURCE',repr(SOURCE)).replace('ROOT',repr(remote.BASE+'/'+str(port))).replace('DQN',repr(remote.DQN)).replace('API',str(remote.API_PORTS[port])).replace('PORT',str(port)).replace('PLAN',repr(ref['path']))
        run_remote(port,code,'server_spawn_'+str(port)+'.json',interpreter=remote.DQN)
    for port in remote.PORTS:
        response=None
        for attempt in range(6):
            try:response=remote.api(port,'/api/status');break
            except Exception:
                time.sleep(1)
        receipt('service_status_'+str(port)+'.json',response)
        if not response or response['status']!=200 or response['body']['source_commit']!=COMMIT or response['body']['plan_sha256']!=plans[port]['sha256']:
            raise ValueError('new workbench actual plan/source identity failed')
    request=dict(request_id='w26-official-feb23-feb24-v1',task_key='download_official',plan_sha256=plans[20418]['sha256'])
    receipt('download_request.json',request)
    response=remote.api(20418,'/api/runs',request);receipt('download_response.json',response)
    if response['status']!=200:raise ValueError('download submission rejected')
    run_id=response['body']['run_id']
    for attempt in range(12):
        result=remote.api(20418,'/api/runs/'+run_id);receipt('download_query_%02d.json'%attempt,result)
        row=result.get('body',{});progress=row.get('progress',{})
        if progress.get('bytes',0)>0 or row.get('status')=='failed':
            receipt('bounded_deployment_result.json',dict(source_commit=COMMIT,run_id=run_id,ssh_port=20418,
                bytes_observed=progress.get('bytes',0),status=row.get('status'),download_continues=row.get('status')=='running',
                trace_reads=0,engineering_submitted=0,training_submitted=0))
            print(json.dumps(dict(run_id=run_id,status=row.get('status'),progress=progress)));return
        time.sleep(2)
    raise RuntimeError('download submitted but bounded observation lacks bytes or explicit failure; preserve worker')


if __name__=='__main__':main()
