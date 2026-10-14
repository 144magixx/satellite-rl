"""Loopback-only W26 frozen-plan admission and persistent owned workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
from .logging import atomic_json,content_sha
from .guard import birth,proc_stat

def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()
def boot_id():
    if os.name!='nt':return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    import psutil
    return 'windows-'+str(int(psutil.boot_time()))

def gpu_uuids():
    result=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10)
    return result.stdout.split()

def alive(record):
    if record.get('pid') is None or record.get('process_birth') is None or record.get('boot_id')!=boot_id():return False
    if birth(record['pid'])!=record['process_birth']:return False
    if os.name!='nt':
        stat=proc_stat(record['pid'])
        return stat is not None and stat['state'] not in ('Z','X')
    import psutil
    try:return psutil.Process(record['pid']).status() not in (psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD)
    except psutil.NoSuchProcess:return False

S11_RECOVERY_RUN_ID='3b234d68cb1d39cdb076320d'
S11_RECOVERY_CURRENT_SHA='9222777e51c822177b9c5791d1da705d0cdf40d572c599023dda701a8a5253bb'

def validate_engineering_source_recovery(directory,receipt_ref,new_plan_ref,executor_port,*,allow_current_worker=False):
    """Exact S11/962 source-only repair; no replay or checkpoint fallback."""
    directory=Path(directory)
    if sha(receipt_ref['path'])!=receipt_ref['sha256']:raise ValueError('root source recovery receipt SHA')
    r=read(receipt_ref['path']);status=read(directory/'status.json');job=read(directory/'job.json');config=read(directory/'config.json')
    if (r.get('schema')!='W26EngineeringSourceRecoveryV1'or r.get('root_approved')is not True or r.get('passed')is not True
            or r.get('experiment_id')!='S2-W26'or r.get('run_id')!=S11_RECOVERY_RUN_ID
            or status['run_id']!=S11_RECOVERY_RUN_ID or r.get('executor_port')!=executor_port or executor_port!=20424
            or config.get('phase')!='engineering'or config.get('group')!='S11'
            or config['training']['total_steps']!=2048 or config['training']['seed']!=73026 or r.get('env_step')!=962):
        raise ValueError('only exact S11 engineering962 source recovery admitted')
    if allow_current_worker:
        if status.get('pid')!=os.getpid()or status.get('process_birth')!=birth(os.getpid())or status.get('boot_id')!=boot_id():
            raise ValueError('migration validation belongs only to actual owned new worker')
    elif alive(status):raise ValueError('original S11 worker remains alive')
    if (sha(directory/'job.json')!=r['original_job_SHA'] or sha(directory/'config.json')!=r['original_config_file_SHA']
            or content_sha(config)!=r['original_config_content_SHA'] or job['config_sha256']!=r['original_config_content_SHA']):
        raise ValueError('original immutable job/config differs')
    original_plan_ref=r['original_plan_ref']
    if sha(original_plan_ref['path'])!=original_plan_ref['sha256'] or original_plan_ref['sha256']!=job['status']['plan_sha256']:
        raise ValueError('original frozen plan differs')
    old_plan=read(original_plan_ref['path'])
    if job['source_manifest']!=old_plan['source_manifest']or job['source_root']!=old_plan['source_root']:
        raise ValueError('original source ancestry differs')
    for key in ('run_id','task_key','mode','request_sha256','plan_sha256','source_commit'):
        if status.get(key)!=job['status'].get(key):raise ValueError('original run/request identity differs')
    if r['original_source_commit']!=job['status']['source_commit']:raise ValueError('original source commit differs')
    for relative,expected in job['source_manifest'].items():
        p=(Path(job['source_root'])/relative).resolve()
        if not p.is_relative_to(Path(job['source_root']).resolve())or sha(p)!=expected:raise ValueError('original source bytes changed')
    if sha(new_plan_ref['path'])!=new_plan_ref['sha256']or r['new_plan_ref']!=new_plan_ref:
        raise ValueError('new root serving plan differs')
    new_plan=read(new_plan_ref['path']);source=Path(r['new_source_root']).resolve()
    if (new_plan.get('authorized')is not True or new_plan['source_commit']!=r['new_source_commit']
            or Path(new_plan['source_root']).resolve()!=source or new_plan['source_manifest']!=r['new_source_manifest']):
        raise ValueError('new frozen source/manifest/plan differs')
    required={p.relative_to(source).as_posix()for folder in ('src','scripts')for p in (source/folder).rglob('*.py')}
    if not required or not required.issubset(r['new_source_manifest']):raise ValueError('new complete Python source closure required')
    for relative,expected in r['new_source_manifest'].items():
        p=(source/relative).resolve()
        if not p.is_relative_to(source)or sha(p)!=expected:raise ValueError('new frozen source bytes differ')
    current=Path(config['model_dir'])/'current.pt'
    if r['current_ref']['path']!=str(current)or r['current_ref']['sha256']!=S11_RECOVERY_CURRENT_SHA or sha(current)!=S11_RECOVERY_CURRENT_SHA:
        raise ValueError('exact current962 checkpoint required; no fallback')
    import torch
    saved=torch.load(current,map_location='cpu',weights_only=False)
    needed={'version','config_sha','agent','progress','scene_rng','rng','current_case','environment','observation','log_offsets'}
    if not needed.issubset(saved)or saved['version']!='W26ResumeV1'or saved['config_sha']!=content_sha(config)or saved['progress']['env_step']!=962:
        raise ValueError('complete current962 state/config required')
    wal_ref=r['decision_wal_ref']
    if wal_ref['path']!=str(directory/'training_decision_wal.pt')or sha(wal_ref['path'])!=wal_ref['sha256']:
        raise ValueError('exact original committed decision WAL required')
    wal=torch.load(wal_ref['path'],map_location='cpu',weights_only=False)
    if wal.get('stage')!='committed'or wal.get('env_step')!=962 or wal.get('config_sha')!=content_sha(config)or wal.get('log_offsets')!=saved['log_offsets']:
        raise ValueError('only committed962/no new or unknown spend recovery allowed')
    if set(saved['log_offsets'])!={'steps','episodes','evaluations'}or set(r['log_refs'])!=set(saved['log_offsets']):
        raise ValueError('complete evidence offsets required')
    for name,offset in saved['log_offsets'].items():
        path=directory/'logs'/(name+'.jsonl.gz');ref=r['log_refs'][name]
        actual=path.stat().st_size if path.exists()else 0
        if actual!=offset or ref.get('bytes')!=offset or ref['path']!=str(path):raise ValueError('unique evidence suffix or offset mismatch; no replay')
        if offset and sha(path)!=ref['sha256']:raise ValueError('actual checkpoint evidence prefix differs')
        if not offset and ref['sha256']!=hashlib.sha256(b'').hexdigest():raise ValueError('empty evidence SHA differs')
    return r

HTML='''<!doctype html><meta charset="utf-8"><title>S2-W26 工作台</title>
<h1>S2-W26 工作台</h1><p>S00:207 · S10:208 · S01:407 · S11:408</p>
<div id="tasks"></div><pre id="status"></pre><div id="runs"></div>
<script>
async function send(path,body){let r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});alert(JSON.stringify(await r.json()));refresh()}
async function refresh(){let s=await(await fetch('/api/status')).json();document.querySelector('#status').textContent=JSON.stringify(s,null,2);let t=document.querySelector('#tasks');t.replaceChildren();for(let key of s.task_keys){let b=document.createElement('button');b.textContent='提交 '+key;b.onclick=()=>send('/api/runs',{request_id:key,task_key:key,plan_sha256:s.plan_sha256});t.append(b)}let runs=await(await fetch('/api/runs')).json();let root=document.querySelector('#runs');root.replaceChildren();for(let run of runs){let p=document.createElement('pre');p.textContent=JSON.stringify(run,null,2);p.title='Click for configuration, logs and results';p.onclick=async()=>{p.textContent=JSON.stringify(await(await fetch('/api/runs/'+run.run_id)).json(),null,2)};root.append(p);for(let action of ['stop','resume']){let b=document.createElement('button');b.textContent=action+' '+run.run_id;b.onclick=()=>send('/api/runs/'+run.run_id+'/'+action,{plan_sha256:s.plan_sha256});root.append(b)}}}refresh();setInterval(refresh,5000)
</script>'''

class Workbench:
    def __init__(self,root,plan_path,executor_port,max_workers=2):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.plan_path=Path(plan_path).resolve();self.plan_sha=sha(self.plan_path);self.plan=read(self.plan_path)
        self.executor_port=int(executor_port);self.max_workers=int(max_workers)
        if self.max_workers not in (1,2) or self.executor_port not in (20418,20424):raise ValueError('W26 lane or worker limit')
        self.lock=threading.RLock();self.children={}
        self.validate_plan()
        self.server_lock=self.root/'server_owner.json'
        owner=dict(pid=os.getpid(),process_birth=birth(os.getpid()),boot_id=boot_id(),created_UTC=now())
        try:
            fd=os.open(self.server_lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:
            if alive(read(self.server_lock)):raise RuntimeError('live workbench already owns lane')
            self.server_lock.unlink();fd=os.open(self.server_lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        with os.fdopen(fd,'w',encoding='utf8') as stream:json.dump(owner,stream)
        (self.root/'runs').mkdir(exist_ok=True);(self.root/'task_owners').mkdir(exist_ok=True)
        self.reconcile()

    def validate_plan(self):
        p=self.plan
        if sha(self.plan_path)!=self.plan_sha:raise ValueError('plan changed since service freeze')
        if p.get('schema')!='W26AuthorizedPlanV1' or p.get('group')!='S2-W26' or p.get('authorized') is not True:raise ValueError('root authorized W26 plan required')
        if not re.fullmatch('[0-9a-f]{40}',p.get('source_commit','')):raise ValueError('frozen source commit required')
        source=Path(p['source_root']).resolve()
        if not Path(__file__).resolve().is_relative_to(source):raise ValueError('server actual import is not frozen source root')
        manifest=p['source_manifest']
        required={x.relative_to(source).as_posix() for folder in ('src','scripts') for x in (source/folder).rglob('*.py')}
        if not required.issubset(manifest):raise ValueError('complete frozen Python source manifest required')
        for relative,expected in manifest.items():
            path=(source/relative).resolve()
            if not path.is_relative_to(source) or sha(path)!=expected:raise ValueError('source manifest changed: '+relative)
        if not isinstance(p.get('tasks'),dict) or not p['tasks']:raise ValueError('frozen tasks required')
        for key,task in p['tasks'].items():
            if not re.fullmatch('[A-Za-z0-9_-]{1,100}',key) or task['executor_port'] not in (20418,20424) or task['mode'] not in ('download','sourceprep','prepare','audit','engineering','train','evaluate') or not isinstance(task['config'],dict):raise ValueError('frozen task contract')
            if task['config'].get('executor_port',task['executor_port'])!=task['executor_port']:raise ValueError('frozen config executor mismatch')
            mode=task['mode'];config=task['config']
            if mode in ('download','sourceprep','prepare','audit'):
                if {'training','phase','task_kind'}&set(config):raise ValueError('data mode cannot carry runner selectors')
                if mode=='audit':self.validate_audit_task(task)
            elif mode in ('engineering','train'):
                training=config.get('training',{});engineering=mode=='engineering'
                expected_seed=73026 if engineering else 51001
                if config.get('phase')!=('engineering' if engineering else 'scientific') or config.get('task_kind') is not None:raise ValueError('mode and actual runner phase differ')
                if training.get('seed')!=expected_seed or training.get('scene_order_seed')!=expected_seed:raise ValueError('mode seed contract')
                steps=training.get('total_steps')
                if type(steps)is not int or not (2000<=steps<=2048 if engineering else steps==150000):raise ValueError('mode actual step budget')
            elif mode=='evaluate':
                if config.get('task_kind')!='confirmation' or 'training' in config or 'phase' in config:raise ValueError('evaluation must select confirmation only')
                refs=config.get('terminal_policies',{})
                if set(refs)!= {'S00','S10','S01','S11'} or any(set(ref)!= {'path','sha256'} for ref in refs.values()):raise ValueError('four frozen terminal policy refs required')
                unlock=config.get('confirmation_unlock',{})
                receipt=unlock.get('receipt_ref',{})
                if set(receipt)!= {'path','sha256'} or sha(receipt['path'])!=receipt['sha256']:raise ValueError('confirmation unlock receipt SHA')
                unlock_receipt=read(receipt['path'])
                if unlock_receipt.get('schema')!='W26ConfirmationUnlockV1' or unlock_receipt.get('root_approved') is not True or unlock_receipt.get('terminal_model_SHAs')!={g:ref['sha256'] for g,ref in refs.items()}:raise ValueError('confirmation unlock schema/model locks')
                manifest_ref=config['confirmation_manifest_ref']
                if sha(manifest_ref['path'])!=manifest_ref['sha256'] or unlock.get('manifest_SHA')!=manifest_ref['sha256'] or unlock_receipt.get('manifest_SHA')!=manifest_ref['sha256']:raise ValueError('confirmation manifest/unlock binding')
                confirmation=read(manifest_ref['path'])
                if confirmation.get('schema')!='W26PublishedDatasetV1' or confirmation.get('root_approved') is not True or confirmation.get('role')!='test' or len(confirmation.get('rows',[]))!=40:raise ValueError('complete confirmation manifest schema')
        prepare=[t for t in p['tasks'].values() if t['mode']=='prepare']
        if prepare:
            all_indices=[]
            manifests=set()
            for task in prepare:
                config=task['config'];indices=config['case_indices']
                ref=config['manifest_ref'];admission_ref=config['execution_admission_ref']
                if sha(ref['path'])!=ref['sha256'] or sha(admission_ref['path'])!=admission_ref['sha256']:raise ValueError('prepare root manifest/admission SHA')
                manifest=read(ref['path']);admission=read(admission_ref['path'])
                if admission.get('root_approved') is not True or admission.get('manifest_SHA')!=ref['sha256'] or admission.get('remote_port')!=task['executor_port'] or admission.get('case_indices')!=indices:raise ValueError('prepare admission root/shard binding')
                if config.get('mixed_allowed') is not True and config.get('role') is None:raise ValueError('mixed prepare requires explicit mixed_allowed')
                rows=manifest['rows']
                if len(rows)!=636 or any(rows[i]['role'] not in ('train','development','test') or config.get('role',rows[i]['role'])!=rows[i]['role'] for i in indices):raise ValueError('prepare frozen row roles')
                manifests.add(ref['sha256'])
                if not isinstance(indices,list) or any(type(i)is not int or not 0<=i<636 for i in indices):raise ValueError('prepare shard indices')
                all_indices.extend(indices)
            if len(manifests)!=1:raise ValueError('all prepare shards must share one root manifest SHA')
            if len(set(all_indices))!=len(all_indices) or sorted(all_indices)!=list(range(636)):raise ValueError('first prepare shards must disjointly cover all636 mothers')

    def validate_admission(self,mode):
        self.validate_plan()
        required=['source','resources']+([] if mode in ('download','sourceprep','prepare','audit') else ['data'])+(['engineering'] if mode in ('train','evaluate') else [])
        for name in required:
            ref=self.plan['admission_refs'][name]
            if sha(ref['path'])!=ref['sha256']:raise ValueError('admission SHA: '+name)
            admission=read(ref['path'])
            if admission.get('root_approved') is not True or admission.get('passed') is not True or admission.get('experiment_id')!='S2-W26':raise ValueError('root admission not passed: '+name)
            if admission.get('executor_port') not in (None,self.executor_port):raise ValueError('admission executor mismatch')
            if name=='source' and admission.get('source_commit')!=self.plan['source_commit']:raise ValueError('admission frozen source mismatch')
            if name=='resources':
                stamp=datetime.fromisoformat(admission['checked_UTC'].replace('Z','+00:00'))
                age=(datetime.now(timezone.utc)-stamp).total_seconds()
                max_age=admission.get('max_age_seconds',300)
                if not 0<max_age<=300 or not 0<=age<=max_age:raise ValueError('fresh resource receipt expired')
                if admission['boot_id']!=boot_id() or admission['gpu_uuid'] not in gpu_uuids():raise ValueError('resource boot/GPU identity mismatch')

    def validate_audit_task(self,task):
        config=task['config'];ref=config['audit_admission_ref']
        if sha(ref['path'])!=ref['sha256']:raise ValueError('audit admission SHA')
        receipt=read(ref['path']);port=task['executor_port']
        if (receipt.get('schema')!='W26DataAuditExecutionAdmissionV1' or receipt.get('root_approved') is not True):raise ValueError('root publication audit admission')
        ports=[receipt[k] for k in ('remote_port','executor_port') if k in receipt]
        if not ports or any(value!=port for value in ports) or config['executor_port']!=port:raise ValueError('publication audit executor mismatch')
        if receipt.get('source_commit')!=self.plan['source_commit'] or config['source_commit']!=self.plan['source_commit']:raise ValueError('publication audit source commit')
        manifest_ref=config['prepare_manifest_ref'];shards=config['shard_index_refs']
        if sha(manifest_ref['path'])!=manifest_ref['sha256'] or receipt.get('manifest_SHA')!=manifest_ref['sha256']:raise ValueError('publication audit manifest SHA binding')
        if len(shards)!=4 or len({ref['sha256'] for ref in shards})!=4 or receipt.get('shard_index_SHAs')!=[ref['sha256'] for ref in shards]:raise ValueError('four exact publication audit shard refs')
        for ref in shards:
            if sha(ref['path'])!=ref['sha256']:raise ValueError('publication audit shard SHA')
        if receipt.get('max_layout_checks')!=636 or receipt.get('expected_counts')!=dict(train=576,development=20,test=40) or receipt.get('expected_native_imports')!=594 or receipt.get('expected_new_prepare')!=42:raise ValueError('bounded publication audit636 contract')
        if not re.fullmatch('[0-9a-f]{64}',receipt.get('core_SHA','')):raise ValueError('publication audit frozen core SHA')
        if 'case_indices' in config:raise ValueError('publication audit consumes full frozen shard graph only')

    def status_path(self,run_id):
        if not re.fullmatch('[0-9a-f]{24}',run_id):raise ValueError('run id')
        return self.root/'runs'/run_id/'status.json'

    def reconcile(self):
        for run_id,child in list(self.children.items()):
            if child.poll() is not None:
                child.wait();del self.children[run_id]
        for path in (self.root/'runs').glob('*/status.json'):
            status=read(path)
            if status['status'] in ('starting','running','stop_requested') and not alive(status):
                status.update(status='interrupted',last_checked_UTC=now(),reason='worker not alive with recorded birth; no automatic restart')
                atomic_json(path,status)

    def runs(self):
        with self.lock:
            self.reconcile();return [read(p) for p in sorted((self.root/'runs').glob('*/status.json'))]

    def summary(self):
        return dict(group='S2-W26',executor_port=self.executor_port,plan_sha256=self.plan_sha,source_commit=self.plan['source_commit'],max_workers=self.max_workers,
            task_keys=[k for k,t in self.plan['tasks'].items() if t['executor_port']==self.executor_port],runs=self.runs())

    def detail(self,run_id):
        path=self.status_path(run_id);status=read(path)
        status['configuration']=read(path.parent/'config.json')
        status['log_files']=[dict(name=p.relative_to(path.parent).as_posix(),bytes=p.stat().st_size) for p in sorted(path.parent.rglob('*')) if p.is_file() and (p.suffix=='.log' or p.name.endswith('.jsonl.gz'))]
        log=path.parent/'worker.log'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0,log.stat().st_size-16384));status['worker_log_tail']=stream.read().decode('utf8',errors='replace')
        return status

    def submit(self,body):
        with self.lock:
            if set(body)!= {'request_id','task_key','plan_sha256'} or body['plan_sha256']!=self.plan_sha:raise ValueError('exact frozen submission fields/plan required')
            task=self.plan['tasks'][body['task_key']]
            if task['executor_port']!=self.executor_port:raise ValueError('task belongs to another executor')
            if not isinstance(body['request_id'],str) or not 1<=len(body['request_id'])<=200:raise ValueError('request identity')
            self.validate_admission(task['mode'])
            specific={'download':'download_admission_ref','sourceprep':'source_admission_ref','prepare':'execution_admission_ref'}.get(task['mode'])
            if specific:
                ref=task['config'][specific]
                if sha(ref['path'])!=ref['sha256']:raise ValueError('mode admission SHA mismatch')
                if read(ref['path']).get('remote_port')!=task['executor_port']:raise ValueError('mode admission remote executor mismatch')
            request_hash=content_sha(body);run_id=request_hash[:24];path=self.status_path(run_id)
            if path.exists():
                status=read(path)
                if status['request_sha256']!=request_hash:raise ValueError('request hash collision')
                return status
            owner=self.root/'task_owners'/(body['task_key']+'.json')
            if owner.exists():raise ValueError('task already has owner; resume original same-ID run')
            if sum(alive(r) for r in self.runs())>=self.max_workers:raise ValueError('worker capacity full')
            directory=path.parent;directory.mkdir()
            config=dict(task['config']);config['run_id']=run_id
            config['executor_port']=task['executor_port']
            model_dir=self.root/'models'/'runs'/run_id
            if 'model_dir' in config and Path(config['model_dir']).resolve()!=model_dir:raise ValueError('dedicated model directory required')
            config['model_dir']=str(model_dir)
            atomic_json(directory/'config.json',config)
            status=dict(run_id=run_id,task_key=body['task_key'],mode=task['mode'],status='created',request_sha256=request_hash,plan_sha256=self.plan_sha,source_commit=self.plan['source_commit'],created_UTC=now(),progress={},run_dir=str(directory),model_dir=config['model_dir'])
            atomic_json(directory/'job.json',dict(status=status,source_root=self.plan['source_root'],source_manifest=self.plan['source_manifest'],config_sha256=content_sha(config)))
            fd=os.open(owner,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            with os.fdopen(fd,'w',encoding='utf8') as stream:json.dump(dict(run_id=run_id,request_sha256=request_hash),stream)
            atomic_json(path,status)
            return self.launch(run_id)

    def launch(self,run_id,*,source_root=None,plan_path=None,control_source_commit=None):
        path=self.status_path(run_id);status=read(path);directory=path.parent
        environment=os.environ.copy();source=Path(source_root or self.plan['source_root'])
        environment['PYTHONPATH']=os.pathsep.join([str(source/'src'),str(source)])
        command=[sys.executable,'-m','satellite_rl.t0_width.workbench','--worker',str(directory),'--plan',str(plan_path or self.plan_path),'--executor-port',str(self.executor_port)]
        with (directory/'worker.log').open('ab') as log:
            proc=subprocess.Popen(command,cwd=source,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name!='nt')
        status.update(status='starting',pid=proc.pid,process_birth=birth(proc.pid),boot_id=boot_id(),worker_started_UTC=now())
        for obsolete in ('error','finished_UTC','reason'):
            status.pop(obsolete,None)
        if control_source_commit:status['control_source_commit']=control_source_commit
        atomic_json(path,status);self.children[run_id]=proc
        # Worker waits for this parent ownership record before taking updates.
        return status

    def recover_download(self,run_id,body):
        """Reuse two fully verified original bytes through the original worker.

        No generic budget recovery, URL substitution, or new download is allowed.
        """
        with self.lock:
            if set(body)!= {'plan_sha256','recovery_admission_ref'} or body['plan_sha256']!=self.plan_sha:raise ValueError('download recovery control identity')
            path=self.status_path(run_id);status=read(path);directory=path.parent
            if status['mode']!='download' or status['status'] not in ('failed','interrupted'):raise ValueError('only failed/interrupted original download recovery')
            if alive(status):raise ValueError('original download worker remains alive')
            self.validate_admission('download')
            ref=body['recovery_admission_ref']
            if set(ref)!= {'path','sha256'} or sha(ref['path'])!=ref['sha256']:raise ValueError('download recovery receipt SHA')
            receipt=read(ref['path'])
            if (receipt.get('schema')!='W26DownloadTransportRecoveryV1' or receipt.get('root_approved') is not True or receipt.get('passed') is not True or receipt.get('experiment_id')!='S2-W26' or receipt.get('executor_port')!=self.executor_port or receipt.get('run_id')!=run_id):raise ValueError('root download recovery scope')
            job_path=directory/'job.json';config_path=directory/'config.json'
            if sha(job_path)!=receipt['original_job_SHA']:raise ValueError('original job SHA mismatch')
            job=read(job_path);config=read(config_path)
            if content_sha(config)!=receipt['original_config_SHA'] or job['config_sha256']!=receipt['original_config_SHA']:raise ValueError('original config SHA mismatch')
            original_status=job['status']
            if any(original_status.get(k)!=status.get(k) for k in ('run_id','task_key','mode','request_sha256','plan_sha256')) or status['request_sha256'][:24]!=run_id:raise ValueError('original run/request identity mismatch')
            original_plan_ref=receipt['original_plan_ref']
            if sha(original_plan_ref['path'])!=original_plan_ref['sha256'] or original_plan_ref['sha256']!=status['plan_sha256']:raise ValueError('original frozen plan SHA mismatch')
            original_plan=read(original_plan_ref['path']);original_task=original_plan['tasks'][status['task_key']]
            if original_task['mode']!='download' or original_task['executor_port']!=self.executor_port or any(config.get(k)!=v for k,v in original_task['config'].items()):raise ValueError('original frozen download task differs')
            source=Path(job['source_root']).resolve()
            if str(source)!=str(Path(original_plan['source_root']).resolve()) or job['source_manifest']!=original_plan['source_manifest']:raise ValueError('original worker source scope mismatch')
            for relative,expected in job['source_manifest'].items():
                member=(source/relative).resolve()
                if not member.is_relative_to(source) or sha(member)!=expected:raise ValueError('original source manifest changed')
            original_admission_ref=config['download_admission_ref']
            if sha(original_admission_ref['path'])!=original_admission_ref['sha256']:raise ValueError('original source admission changed')
            original_admission=read(original_admission_ref['path'])
            if original_admission.get('schema')!='W26OfficialDownloadAdmissionV1' or original_admission.get('root_approved') is not True or original_admission.get('remote_port')!=self.executor_port:raise ValueError('original source provenance admission')
            assets=original_admission['assets'];canonical=receipt['canonical_assets']
            if len(assets)!=2 or len(canonical)!=2 or len({a['date'] for a in assets})!=2:raise ValueError('exact original two assets required')
            transport_ref=receipt['transport_receipt_ref']
            if sha(transport_ref['path'])!=transport_ref['sha256']:raise ValueError('transport receipt SHA')
            transport=read(transport_ref['path'])
            if transport.get('schema')!='W26OfficialTransportReceiptV1' or len(transport.get('assets',[]))!=2:raise ValueError('official transport provenance schema')
            canonical_directory=Path(original_admission['canonical_directory']).resolve()
            expected_paths={str(canonical_directory/(asset['date']+'.tar')) for asset in assets}
            if {str(Path(a['path']).resolve()) for a in canonical}!=expected_paths:raise ValueError('canonical assets differ from original paths')
            for asset in assets:
                final=canonical_directory/(asset['date']+'.tar')
                record=next(a for a in canonical if Path(a['path']).resolve()==final)
                if set(record)!= {'path','bytes','sha256'} or record['bytes']!=asset['release_asset_bytes'] or final.stat().st_size!=record['bytes'] or sha(final)!=record['sha256']:raise ValueError('complete canonical bytes/hash mismatch')
                provenance=next((a for a in transport['assets'] if Path(a['path']).resolve()==final),None)
                if provenance is None or any(provenance.get(k)!=v for k,v in dict(path=str(final),bytes=record['bytes'],sha256=record['sha256'],source_url=asset['source_url'],release_asset_id=asset['release_asset_id'],date=asset['date']).items()):raise ValueError('official transport asset identity mismatch')
                pinned=asset.get('release_asset_SHA')==record['sha256']
                owner=canonical_directory/(asset['date']+'.owner.json')
                expected_owner=dict(request_id=config['request_id'],source_admission_SHA=original_admission_ref['sha256'],source_url=asset['source_url'],release_asset_id=asset['release_asset_id'],release_asset_bytes=asset['release_asset_bytes'])
                if not pinned and (not owner.exists() or read(owner)!=expected_owner):raise ValueError('original canonical owner/source provenance mismatch')
                if asset.get('release_asset_SHA') and not pinned:raise ValueError('original root pinned asset SHA differs')
            if sum(alive(r) for r in self.runs())>=self.max_workers:raise ValueError('worker capacity full')
            log=directory/'worker.log'
            if log.exists() and log.stat().st_size>1024*1024:raise ValueError('large original log requires root bounded evidence archive')
            history=directory/'recovery_history'/ref['sha256'];history.mkdir(parents=True,exist_ok=False)
            atomic_json(history/'original_status.json',status)
            archived_log=history/'original_worker.log'
            if log.exists():archived_log.write_bytes(log.read_bytes())
            history_record=dict(recovery_admission_ref=ref,original_status_SHA=sha(history/'original_status.json'),worker_log_ref=dict(path=str(archived_log),bytes=archived_log.stat().st_size,sha256=sha(archived_log)) if archived_log.exists() else None,control_source_commit=self.plan['source_commit'],recorded_UTC=now())
            atomic_json(history/'recovery_admission.json',receipt)
            atomic_json(history/'index.json',history_record)
            return self.launch(run_id,source_root=str(source),plan_path=original_plan_ref['path'],control_source_commit=self.plan['source_commit'])

    def control(self,run_id,action,body):
        with self.lock:
            if body!={'plan_sha256':self.plan_sha}:raise ValueError('control plan identity')
            path=self.status_path(run_id);status=read(path)
            if action=='stop':
                (path.parent/'stop.request').write_text(now(),encoding='utf8')
                if alive(status):status.update(status='stop_requested')
                return status
            if action!='resume':raise ValueError('control action')
            self.reconcile();status=read(path)
            if alive(status):raise ValueError('owned worker still alive; refuse second owner')
            if status['status'] not in ('interrupted','stopped','stopped_during_evaluation'):raise ValueError('run not resumable')
            if status['mode'] in ('download','sourceprep','prepare','audit'):raise ValueError('data partial spend requires separate root recovery admission; no automatic replay')
            if not (Path(status['model_dir'])/'current.pt').exists() and not (Path(status['model_dir'])/'previous.pt').exists():raise ValueError('trustworthy checkpoint required')
            self.validate_admission(status['mode'])
            if sum(alive(r) for r in self.runs())>=self.max_workers:raise ValueError('worker capacity full')
            (path.parent/'stop.request').unlink(missing_ok=True)
            return self.launch(run_id)

    def close(self):
        if self.server_lock.exists() and read(self.server_lock).get('pid')==os.getpid():self.server_lock.unlink()

    def recover_engineering_source(self,run_id,body):
        with self.lock:
            if set(body)!={'plan_sha256','recovery_admission_ref'}or body['plan_sha256']!=self.plan_sha or run_id!=S11_RECOVERY_RUN_ID:
                raise ValueError('exact S11 source recovery control required')
            self.validate_admission('engineering')
            path=self.status_path(run_id);status=read(path)
            if status['status']not in ('needs_root_recovery_accounting','failed','interrupted')or alive(status):
                raise ValueError('only stopped failed/accounting S11 original worker allowed')
            if sum(alive(r)for r in self.runs())>=self.max_workers:raise ValueError('worker capacity full')
            new_plan_ref=dict(path=str(self.plan_path),sha256=self.plan_sha)
            receipt=validate_engineering_source_recovery(path.parent,body['recovery_admission_ref'],new_plan_ref,self.executor_port)
            sidecar=path.parent/'source_recoveries'/body['recovery_admission_ref']['sha256']/ 'migration.json'
            if sidecar.exists():raise ValueError('source recovery epoch already dispatched; no automatic duplicate')
            migration=dict(schema='W26EngineeringSourceMigrationV1',root_recovery_ref=body['recovery_admission_ref'],
                           new_plan_ref=new_plan_ref,run_id=run_id,env_step=962,old_job_SHA=receipt['original_job_SHA'])
            atomic_json(sidecar,migration)
            status['source_recovery_ref']=dict(path=str(sidecar),sha256=sha(sidecar));atomic_json(path,status)
            (path.parent/'stop.request').unlink(missing_ok=True)
            return self.launch(run_id,source_root=receipt['new_source_root'],plan_path=receipt['original_plan_ref']['path'],control_source_commit=receipt['new_source_commit'])

def worker(directory,plan_path,executor_port):
    import time
    directory=Path(directory);path=directory/'status.json'
    for _ in range(100):
        status=read(path)
        if status.get('pid')==os.getpid() and status.get('process_birth')==birth(os.getpid()) and status.get('boot_id')==boot_id():break
        time.sleep(.05)
    else:raise RuntimeError('worker parent birth ownership not recorded')
    plan=read(plan_path);job=read(directory/'job.json');config=read(directory/'config.json')
    if sha(plan_path)!=status['plan_sha256'] or content_sha(config)!=job['config_sha256']:raise ValueError('worker frozen identity')
    if status.get('source_recovery_ref'):
        sidecar_ref=status['source_recovery_ref']
        if sha(sidecar_ref['path'])!=sidecar_ref['sha256']:raise ValueError('migration sidecar bytes differ')
        migration=read(sidecar_ref['path'])
        # This worker itself is intentionally alive; external controller already
        # proved the old process gone. Revalidate every byte before first step.
        receipt_ref=migration['root_recovery_ref']
        # Skip only the current-worker liveness check via an explicit flag.
        receipt=validate_engineering_source_recovery(directory,receipt_ref,migration['new_plan_ref'],executor_port,allow_current_worker=True)
        if not Path(__file__).resolve().is_relative_to(Path(receipt['new_source_root']).resolve()):raise ValueError('actual migrated worker source differs')
        if status.get('control_source_commit')!=receipt['new_source_commit']:raise ValueError('migrated control source commit differs')
    else:
        if not Path(__file__).resolve().is_relative_to(Path(job['source_root']).resolve()):raise ValueError('worker actual source mismatch')
        for relative,expected in job['source_manifest'].items():
            if sha(Path(job['source_root'])/relative)!=expected:raise ValueError('worker source SHA changed')
    def progress(value):
        current=read(path);current.update(status='running',progress=value,last_updated_UTC=now());atomic_json(path,current)
    stop=lambda:(directory/'stop.request').exists()
    try:
        progress({})
        if status['mode']=='audit':
            from .publication import audit_prepared_dataset
            result=audit_prepared_dataset(config,directory,progress,stop)
        elif status['mode'] in ('download','sourceprep','prepare'):
            from . import data
            function={'download':'download_official_sources','sourceprep':'prepare_confirmation_sources','prepare':'prepare_dataset'}[status['mode']]
            result=getattr(data,function)(config,directory,progress,stop)
        else:
            from .runner import run_task
            result=run_task(config,directory,progress,stop)
        current=read(path);state=result.get('stage',result.get('status','completed')) if isinstance(result,dict) else 'completed'
        current.update(status=state,result=result,finished_UTC=now());atomic_json(path,current)
    except BaseException as exc:
        current=read(path);current.update(status='failed',error=repr(exc),finished_UTC=now());atomic_json(path,current)
        raise

def handler(workbench):
    class Handler(BaseHTTPRequestHandler):
        def respond(self,value,status=200,html=False):
            payload=(value if html else json.dumps(value,ensure_ascii=False)).encode('utf8')
            self.send_response(status);self.send_header('Content-Type','text/html; charset=utf-8' if html else 'application/json; charset=utf-8');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            try:
                path=urlparse(self.path).path
                if path=='/':return self.respond(HTML,html=True)
                if path=='/api/status':return self.respond(workbench.summary())
                if path=='/api/runs':return self.respond(workbench.runs())
                if path.startswith('/api/runs/'):return self.respond(workbench.detail(path.split('/')[3]))
                self.respond(dict(error='unknown endpoint'),404)
            except Exception as exc:self.respond(dict(error=str(exc)),400)
        def do_POST(self):
            try:
                # Loopback API is not writable by browser cross-origin forms.
                if self.headers.get('Content-Type')!='application/json' or self.headers.get('Origin') not in (None,'http://'+self.headers.get('Host','')):raise ValueError('same-origin JSON required')
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384:raise ValueError('small request body required')
                body=json.loads(self.rfile.read(length));path=urlparse(self.path).path
                if path=='/api/runs':return self.respond(workbench.submit(body))
                parts=path.split('/')
                if len(parts)==5 and parts[1:3]==['api','runs']:
                    if parts[4]=='recover-download':return self.respond(workbench.recover_download(parts[3],body))
                    if parts[4]=='recover-engineering-source':return self.respond(workbench.recover_engineering_source(parts[3],body))
                    return self.respond(workbench.control(parts[3],parts[4],body))
                self.respond(dict(error='unknown endpoint'),404)
            except Exception as exc:self.respond(dict(error=str(exc)),400)
        def log_message(self,*args):pass
    return Handler

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root');parser.add_argument('--port',type=int);parser.add_argument('--executor-port',type=int,required=True);parser.add_argument('--plan',required=True);parser.add_argument('--max-workers',type=int,default=2);parser.add_argument('--worker')
    args=parser.parse_args()
    if args.worker:return worker(args.worker,args.plan,args.executor_port)
    if args.port not in (9060,9061):raise ValueError('independent W26 loopback port required')
    workbench=Workbench(args.root,args.plan,args.executor_port,args.max_workers)
    try:ThreadingHTTPServer(('127.0.0.1',args.port),handler(workbench)).serve_forever()
    finally:workbench.close()

if __name__=='__main__':main()
