"""Persistent loopback W30 API; frozen tasks, exclusive GPU claim and owned workers."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from satellite_rl.t0_width.workbench import alive, birth, boot_id, now, handler, read, sha, gpu_uuids
from .logging import atomic_json, content_sha
from .control import read_ref
from .menus import MENUS


HTML = '<meta charset="utf-8"><h1>S2-W30 工作台</h1><p>冻结任务与进度：<a href="/api/status">状态</a> · <a href="/api/runs">运行</a></p>'


class Workbench:
    def __init__(self, root, plan_path, executor_port, max_workers=1):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.plan_path = Path(plan_path).resolve()
        self.plan_sha = sha(self.plan_path)
        self.plan = read(self.plan_path)
        self.executor_port = executor_port
        self.lock = threading.RLock()
        self.children = {}
        if executor_port not in (20655,20654,20653,20627) or max_workers != 1:
            raise ValueError('W30 only four authorized ports; one worker per unique GPU')
        self.validate_plan()
        for name in ('runs','task_owners'):
            (self.root/name).mkdir(exist_ok=True)
        self.server_lock = self.root/'server_owner.json'
        if self.server_lock.exists():
            if alive(read(self.server_lock)):
                raise RuntimeError('Existing live W30 workbench owner')
            self.server_lock.unlink()
        with self.server_lock.open('x', encoding='utf8') as stream:
            json.dump(dict(pid=os.getpid(),process_birth=birth(os.getpid()),boot_id=boot_id()),stream)
        self.reconcile()

    def validate_plan(self):
        p = self.plan
        if sha(self.plan_path) != self.plan_sha or p.get('schema') != 'W30AuthorizedPlanV1' or p.get('group') != 'S2-W30' or p.get('authorized') is not True:
            raise ValueError('Root frozen W30 plan required')
        source = Path(p['source_root']).resolve()
        if not Path(__file__).resolve().is_relative_to(source):
            raise ValueError('Actual source outside frozen source root')
        required = {x.relative_to(source).as_posix() for x in (source/'src').rglob('*.py')}
        if not required.issubset(p['source_manifest']):
            raise ValueError('Complete frozen source manifest required')
        for rel, expected in p['source_manifest'].items():
            path = (source/rel).resolve()
            if not path.is_relative_to(source) or sha(path) != expected:
                raise ValueError('Frozen source differs: '+rel)
        if not p.get('tasks'):
            raise ValueError('No frozen task map')
        if any(t['mode']=='fixtures' for t in p['tasks'].values()) and len(p['tasks'])!=1:
            raise ValueError('Fixtures require a separate single-task frozen plan')
        for key, task in p['tasks'].items():
            c = task['config']
            if not re.fullmatch('[A-Za-z0-9_-]{1,100}', key) or task['executor_port'] not in (20655,20654,20653,20627):
                raise ValueError('Task identity/port')
            if task['mode']=='fixtures':
                from .fixture_executor import validate_fixture_config
                validate_fixture_config(c)
                if c['source_sha']!=p['source_commit']:
                    raise ValueError('Fixture frozen source binding differs')
                continue
            if task['mode']=='evaluate':
                if c.get('phase')!='evaluation' or c.get('task_kind')!='confirmation':
                    raise ValueError('Only future sealed confirmation evaluate mode')
                from .evaluation import validate_confirmation
                validate_confirmation(c)
                continue
            if task['mode'] not in ('engineering','train') or c['phase'] != ('engineering' if task['mode']=='engineering' else 'scientific'):
                raise ValueError('Only qualified W30 engineering/training entry')
            if c['menu'] not in MENUS or c['group'] not in ('M0','M1'):
                raise ValueError('Frozen eight-cell matrix')
            t = c['training']
            allowed_seeds=(73030,) if task['mode']=='engineering' else ((51001,) if c.get('first_round',True) else (51002,51003))
            if t['total_steps'] != (2048 if task['mode']=='engineering' else 150000) or t['seed'] not in allowed_seeds:
                raise ValueError('First-round step/seed budget')
            if task['mode']=='train' and not c.get('first_round',True):
                extension=read_ref(c['conditional_training_unlock_ref'])
                if (extension.get('experiment_id')!='S2-W30' or extension.get('root_approved') is not True
                        or extension.get('first_round_complete') is not True or extension.get('development_gate_passed') is not True
                        or extension.get('new40_sealed') is not True):
                    raise ValueError('Conditional full-matrix training unlock required')
            if c['source_sha'] != p['source_commit']:
                raise ValueError('Task frozen source binding')

    def validate_admission(self, mode, ops_ref=None):
        self.validate_plan()
        required = ['source','resources','data','execution'] + (['engineering'] if mode in ('train','evaluate') else [])
        for name in required:
            receipt = read_ref(ops_ref if name=='resources' and ops_ref else self.plan['admission_refs'][name])
            if receipt.get('experiment_id')!='S2-W30' or receipt.get('root_approved') is not True or receipt.get('passed') is not True:
                raise ValueError('Root admission missing: '+name)
            if receipt.get('executor_port') not in (None,self.executor_port):
                raise ValueError('Receipt executor differs')
            if name=='resources':
                if ops_ref and ops_ref!=self.plan['admission_refs']['resources'] and (receipt.get('plan_sha256')!=self.plan_sha or receipt.get('source_commit')!=self.plan['source_commit']):
                    raise ValueError('Fresh dispatch ops receipt must bind unchanged plan and source')
                if os.name!='nt':
                    import pwd
                    if pwd.getpwuid(os.getuid()).pw_name!='u2025140002':
                        raise ValueError('Actual remote account is not authorized W30 owner')
                from datetime import datetime, timezone
                age=(datetime.now(timezone.utc)-datetime.fromisoformat(receipt['checked_UTC'].replace('Z','+00:00'))).total_seconds()
                if not 0 <= age <= min(300,receipt.get('max_age_seconds',300)):
                    raise ValueError('Fresh resource evidence expired')
                if receipt['boot_id'] != boot_id() or receipt['gpu_uuid'] not in gpu_uuids() or receipt.get('username')!='u2025140002':
                    raise ValueError('Resource boot/GPU/account identity differs')
                if receipt.get('exclusive_claim_verified') is not True or receipt.get('storage_peak_verified') is not True:
                    raise ValueError('Unique GPU claim and measured storage admission required')
                claim=read(receipt['claim_path'])
                if (claim.get('workbench_pid')!=os.getpid() or claim.get('boot_id')!=boot_id()
                        or claim.get('gpu_uuid')!=receipt['gpu_uuid'] or claim.get('process_birth')!=birth(os.getpid())):
                    raise ValueError('Live unique GPU claim birth/boot ownership differs')

    def status_path(self,run_id):
        if not re.fullmatch('[0-9a-f]{24}',run_id):
            raise ValueError('Invalid run ID')
        return self.root/'runs'/run_id/'status.json'

    def runs(self):
        return [read(p) for p in sorted((self.root/'runs').glob('*/status.json'))]

    def reconcile(self):
        for status in self.runs():
            if status['status'] in ('starting','running','stop_requested') and not alive(status):
                status.update(status='interrupted',reason='owned birth/boot exited; no automatic replay')
                atomic_json(self.status_path(status['run_id']),status)

    def summary(self):
        self.reconcile()
        return dict(group='S2-W30',plan_sha256=self.plan_sha,executor_port=self.executor_port,task_keys=[k for k,v in self.plan['tasks'].items() if v['executor_port']==self.executor_port],runs=self.runs())

    def detail(self,run_id):
        p=self.status_path(run_id)
        return dict(status=read(p),config=read(p.parent/'config.json'))

    def submit(self,body):
        with self.lock:
            if set(body) not in ({'request_id','task_key','plan_sha256'},{'request_id','task_key','plan_sha256','ops_admission_ref'}) or body['plan_sha256']!=self.plan_sha:
                raise ValueError('Exact frozen request contract')
            task=self.plan['tasks'][body['task_key']]
            if task['executor_port']!=self.executor_port:
                raise ValueError('Wrong executor lane')
            ops_ref=body.get('ops_admission_ref',self.plan['admission_refs']['resources'])
            self.validate_admission(task['mode'],ops_ref)
            request_identity={k:body[k] for k in ('request_id','task_key','plan_sha256')}
            digest=content_sha(request_identity);run_id=digest[:24];path=self.status_path(run_id)
            if path.exists():
                return read(path) # Idempotent; never reexecute a known request.
            owner=self.root/'task_owners'/(body['task_key']+'.json')
            if owner.exists() or any(alive(s) for s in self.runs()):
                raise ValueError('Task already claimed or unique GPU busy')
            # An exclusive shared GPU claim must be held by this service before
            # submit; ops receipt pins its canonical filesystem path and UUID.
            resource=read_ref(ops_ref)
            claim=read(resource['claim_path'])
            if claim.get('workbench_pid')!=os.getpid() or claim.get('boot_id')!=boot_id() or claim.get('gpu_uuid')!=resource['gpu_uuid']:
                raise ValueError('Actual unique GPU claim does not belong to service')
            path.parent.mkdir()
            config=dict(task['config']);config['run_id']=run_id
            config['model_dir']=str(self.root/'models'/run_id)
            atomic_json(path.parent/'config.json',config)
            status=dict(run_id=run_id,task_key=body['task_key'],mode=task['mode'],status='created',request_sha256=digest,plan_sha256=self.plan_sha,source_commit=self.plan['source_commit'],created_UTC=now(),progress={},run_dir=str(path.parent),dispatch_ops_ref=ops_ref,dispatch_ops_history=[ops_ref])
            atomic_json(path.parent/'job.json',dict(config_sha256=content_sha(config),status=status))
            with owner.open('x',encoding='utf8') as f:
                json.dump(dict(run_id=run_id,request_sha256=digest),f)
            atomic_json(path,status)
            return self.launch(run_id)

    def launch(self,run_id):
        path=self.status_path(run_id);status=read(path)
        source=Path(self.plan['source_root'])
        env=os.environ.copy();env['PYTHONPATH']=str(source/'src')
        command=[sys.executable,'-m','satellite_rl.power30.workbench','--worker',str(path.parent),'--plan',str(self.plan_path),'--executor-port',str(self.executor_port)]
        with (path.parent/'worker.log').open('ab') as log:
            child=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name!='nt')
        status.update(status='starting',pid=child.pid,process_birth=birth(child.pid),boot_id=boot_id(),worker_started_UTC=now())
        atomic_json(path,status);self.children[run_id]=child
        return status

    def control(self,run_id,action,body):
        with self.lock:
            if set(body) not in ({'plan_sha256'},{'plan_sha256','ops_admission_ref'}) or body['plan_sha256']!=self.plan_sha:
                raise ValueError('Frozen plan control')
            p=self.status_path(run_id);s=read(p)
            if action=='stop':
                (p.parent/'stop.request').write_text(now(),encoding='utf8')
                return s
            if action!='resume' or alive(s) or s['status'] not in ('stopped','stopped_during_evaluation'):
                raise ValueError('Only clean same-ID saved run may resume; failed/unknown requires root reconciliation')
            ops_ref=body.get('ops_admission_ref',self.plan['admission_refs']['resources'])
            self.validate_admission(s['mode'],ops_ref)
            if any(alive(r) for r in self.runs()):
                raise ValueError('Unique GPU busy')
            s['dispatch_ops_ref']=ops_ref;s.setdefault('dispatch_ops_history',[]).append(ops_ref)
            atomic_json(p,s)
            (p.parent/'stop.request').unlink(missing_ok=True)
            return self.launch(run_id)

    def close(self):
        if self.server_lock.exists() and read(self.server_lock).get('pid')==os.getpid():
            self.server_lock.unlink()


def worker(directory,plan_path,executor_port):
    directory=Path(directory);path=directory/'status.json'
    for _ in range(100):
        s=read(path)
        if s.get('pid')==os.getpid() and s.get('process_birth')==birth(os.getpid()) and s.get('boot_id')==boot_id():
            break
        time.sleep(.05)
    else:
        raise ValueError('No parent recorded birth identity')
    plan=read(plan_path);config=read(directory/'config.json');job=read(directory/'job.json')
    if sha(plan_path)!=s['plan_sha256'] or content_sha(config)!=job['config_sha256']:
        raise ValueError('Worker frozen config/plan changed')
    source=Path(plan['source_root']).resolve()
    if not Path(__file__).resolve().is_relative_to(source):
        raise ValueError('Worker actual source root differs')
    for rel,expected in plan['source_manifest'].items():
        member=(source/rel).resolve()
        if not member.is_relative_to(source) or sha(member)!=expected:
            raise ValueError('Worker frozen source differs')
    def progress(value):
        s=read(path);s.update(status=value.get('stage','running'),progress=value,last_updated_UTC=now());atomic_json(path,s)
    try:
        if s['mode']=='fixtures':
            from .fixture_executor import run_owned_fixture_task as run_task
        elif s['mode']=='evaluate':
            from .evaluation import run_confirmation_task as run_task
        else:
            from .runner import run_task
        result=run_task(config,directory,progress,lambda:(directory/'stop.request').exists())
        s=read(path);s.update(status=result['stage'],result=result,finished_UTC=now());atomic_json(path,s)
    except BaseException as exc:
        s=read(path);s.update(status='failed',error=repr(exc),finished_UTC=now());atomic_json(path,s)
        raise


def main():
    parser=argparse.ArgumentParser()
    for name in ('root','plan','worker'):
        parser.add_argument('--'+name,required=name=='plan')
    parser.add_argument('--port',type=int);parser.add_argument('--executor-port',type=int,required=True)
    args=parser.parse_args()
    if args.worker:
        return worker(args.worker,args.plan,args.executor_port)
    if not 1024<=args.port<=65535:
        raise ValueError('Fresh ops-selected loopback port required')
    bench=Workbench(args.root,args.plan,args.executor_port)
    # Existing handler uses only shared API methods; its old static root page
    # is overridden here so no W26 task text appears in the independent scope.
    base=handler(bench)
    class Handler(base):
        def do_GET(self):
            if self.path=='/':
                return self.respond(HTML,html=True)
            return super().do_GET()
    try:
        ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
    finally:
        bench.close()


if __name__=='__main__':
    main()
