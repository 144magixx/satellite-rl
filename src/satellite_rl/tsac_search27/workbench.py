"""Loopback W27 frozen tasks. One worker per card; no automatic continuation."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime, timezone
from satellite_rl.t0_width.data import file_sha, read_json_ref
from satellite_rl.t0_width.logging import atomic_json, content_sha
from .pause import Gate, file_lock, proc_identity, same_identity
from .control import validate_continuation


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def validate_task(task):
    c = task['config']; t = c.get('training', {})
    if task['executor_port'] not in (20418, 20424, 20627):
        raise ValueError('only authorized three ports')
    if task['mode'] not in ('engineering', 'science', 'known_evaluation'):
        raise ValueError('unknown mode')
    if task['mode'] == 'known_evaluation':
        from .known_evaluation import validate_lock
        validate_lock(c)
        return
    engineering = task['mode'] == 'engineering'
    if c.get('variant') not in ('B','E','LP') or c.get('group') != 'S10':
        raise ValueError('variant/schema mismatch')
    seed = 73027 if engineering else 51001
    if (c.get('phase') != ('engineering' if engineering else 'scientific')
            or t.get('seed') != seed or t.get('scene_order_seed') != seed
            or t.get('total_steps') != (2048 if engineering else 150000)
            or (not engineering and t.get('pilot_stop_node') != 50000)
            or c.get('task_kind') is not None):
        raise ValueError('frozen mode/budget/seed mismatch')


class Workbench:
    def __init__(self, root, plan_path, executor_port, control_dir):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.plan_path = Path(plan_path).resolve(); self.plan_sha = file_sha(self.plan_path)
        self.plan = read(self.plan_path); self.port = int(executor_port)
        self.gate = Gate(control_dir); self.lock = threading.RLock()
        self.validate_plan(); (self.root/'runs').mkdir(exist_ok=True)

    def validate_plan(self):
        p = self.plan
        if (file_sha(self.plan_path) != self.plan_sha or p.get('schema') != 'W27AuthorizedPlanV1'
                or p.get('experiment_id') != 'S2-W27' or p.get('authorized') is not True
                or self.port not in (20418,20424,20627) or p.get('max_workers') != 1):
            raise ValueError('root frozen W27 single-worker plan required')
        source = Path(p['source_root']).resolve()
        if not Path(__file__).resolve().is_relative_to(source):
            raise ValueError('actual service import outside frozen source')
        required = {x.relative_to(source).as_posix() for folder in ('src','scripts') for x in (source/folder).rglob('*.py')}
        if not required or not required.issubset(p['source_manifest']):
            raise ValueError('complete Python source closure required')
        for rel, digest in p['source_manifest'].items():
            path = (source/rel).resolve()
            if not path.is_relative_to(source) or file_sha(path) != digest:
                raise ValueError('frozen source changed')
        for key, task in p['tasks'].items():
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', key):
                raise ValueError('invalid task key')
            validate_task(task)
            if task['mode']=='known_evaluation' and task['config'].get('evaluation_source_commit')!=p['source_commit']:
                raise ValueError('known evaluation actual frozen Git source mismatch')

    def admission(self, task, dispatch_epoch=None, resource_ref=None):
        self.validate_plan(); c=task['config']
        self.gate.validate(dispatch_epoch or c['rootpauseepoch'])
        names = ['source','resources','data'] + (['engineering'] if task['mode'] in ('science','known_evaluation') else [])
        for name in names:
            r = read_json_ref(resource_ref if name=='resources' and resource_ref is not None else self.plan['admission_refs'][name])
            if r.get('root_approved') is not True or r.get('passed') is not True or r.get('experiment_id') != 'S2-W27':
                raise ValueError('root admission not passed: '+name)
            if r.get('executor_port') not in (None,self.port):
                raise ValueError('other resource owner')
            if name == 'source' and r.get('source_SHA') != c['source_sha']:
                raise ValueError('source admission mismatch')
            if name == 'resources':
                age=(datetime.now(timezone.utc)-datetime.fromisoformat(r['checked_UTC'].replace('Z','+00:00'))).total_seconds()
                if not 0 <= age <= min(300,r.get('max_age_seconds',300)):
                    raise ValueError('resource admission expired')
                identity=proc_identity(os.getpid())
                if not identity or identity['boot_id'] != r['boot_id']:
                    raise ValueError('resource boot identity mismatch')
                uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()
                if r['gpu_uuid'] not in uuids:
                    raise ValueError('resource GPU mismatch')

    def detail(self, key):
        if key not in self.plan['tasks']: raise ValueError('unknown frozen task')
        d=self.root/'runs'/key
        status=read(d/'status.json') if (d/'status.json').exists() else {'stage':'planned','task_key':key}
        status=dict(status,config=self.plan['tasks'][key]['config'],run_dir=str(d),
                    owner=read(d/'owner.json') if (d/'owner.json').exists() else None,
                    models=str(Path(self.plan['tasks'][key]['config'].get('model_dir',d/'models'))))
        log=d/'worker.log'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0,log.stat().st_size-16384)); status['log_tail']=stream.read(16384).decode('utf8','replace')
        else: status['log_tail']=''
        return status

    def status(self):
        return dict(experiment_id='S2-W27',executor_port=self.port,plan_path=str(self.plan_path),
                    plan_SHA=self.plan_sha,source_root=self.plan['source_root'],source_manifest=self.plan['source_manifest'],
                    gate=self.gate.snapshot(),tasks={k:self.detail(k) for k in self.plan['tasks']})

    def start(self, key, continuation_ref=None, *, dispatch_epoch=None, resource_ref=None):
        with self.lock:
            task=self.plan['tasks'][key]
            if task['executor_port'] != self.port: raise ValueError('task owned by another card')
            epoch=dispatch_epoch or task['config']['rootpauseepoch']
            self.admission(task,epoch,resource_ref)
            for d in (self.root/'runs').iterdir():
                if (d/'owner.json').exists():
                    owner=read(d/'owner.json')
                    if same_identity(proc_identity(owner['pid']),owner): raise ValueError('card already has owned live worker')
            d=self.root/'runs'/key; d.mkdir(exist_ok=True)
            if (d/'config.json').exists():
                if content_sha(read(d/'config.json')) != content_sha(task['config']): raise ValueError('immutable config changed')
                if continuation_ref is not None: validate_continuation(d,task['config'],continuation_ref)
                elif epoch == task['config']['rootpauseepoch'] or self.detail(key).get('stage') not in ('stopped','stopped_during_evaluation'):
                    raise ValueError('existing task requires accounted pilot continuation or explicit human epoch safe-stop recovery')
            elif continuation_ref is not None: raise ValueError('cannot continue nonexistent pilot')
            intent=d/'dispatch_intent.json'
            if intent.exists() and read(intent).get('stage') == 'starting': raise ValueError('pending dispatch must be independently reconciled')
            if not (d/'job.json').exists():
                atomic_json(d/'job.json',dict(task_key=key,plan_sha=self.plan_sha,config_SHA=content_sha(task['config'])))
                atomic_json(d/'config.json',task['config'])
            generation=len(list(d.glob('dispatch_*.json')))
            runtime=dict(task_key=key,plan_sha=self.plan_sha,continuation_ref=continuation_ref,
                         dispatch_epoch=epoch,resource_ref=resource_ref,generation=generation,stage='starting')
            atomic_json(d/('dispatch_%06d.json'%generation),runtime)
            atomic_json(intent,runtime)
            if (d/'stop.json').exists(): (d/'stop.json').unlink()
            command=[sys.executable,'-B','-m','satellite_rl.tsac_search27.workbench','worker','--root',str(self.root),'--plan',str(self.plan_path),
                     '--port',str(self.port),'--control',str(self.gate.directory),'--task',key,'--generation',str(generation)]
            with (d/'worker.log').open('ab') as log:
                process=subprocess.Popen(command,cwd=self.plan['source_root'],stdout=log,stderr=subprocess.STDOUT)
            return {'stage':'starting','pid':process.pid,'task_key':key}

    def stop(self,key):
        self.detail(key); atomic_json(self.root/'runs'/key/'stop.json',{'requested':True})
        return {'stage':'stop_requested','task_key':key}


def worker(args):
    w=Workbench(args.root,args.plan,args.port,args.control); task=w.plan['tasks'][args.task]
    d=w.root/'runs'/args.task; job=read(d/('dispatch_%06d.json'%args.generation)); c=task['config']; owner=proc_identity(os.getpid()); epoch=job['dispatch_epoch']
    owner.update(experiment_id='S2-W27',task_id=c['run_id'],rootpauseepoch=epoch,
                 stop_url=w.plan['stop_url_base']+'/api/runs/'+c['run_id']+'/stop')
    with file_lock(w.root/'card.lock',nonblocking=True):
        w.admission(task,epoch,job.get('resource_ref'))
        # Gate registration must additionally validate accounted same-epoch replacement.
        w.gate.register_worker(epoch,owner, continuation=(dict(run_dir=str(d), config=c, receipt_ref=job['continuation_ref']) if job.get('continuation_ref') else None))
        atomic_json(d/'owner.json',owner)
        atomic_json(d/'dispatch_intent.json',dict(job,stage='registered',owner=owner))
        if task['mode']=='known_evaluation':
            from .known_evaluation import run_known_task as run_task
        else:
            from .runner import run_task
        def progress(value): atomic_json(d/'status.json',value)
        try:
            stop=lambda: (d/'stop.json').exists() or w.gate.stop_requested(epoch)
            if task['mode']=='known_evaluation': run_task(c,d,progress,stop)
            else: run_task(c,d,progress,stop,continuation_ref=job.get('continuation_ref'))
        except BaseException as e:
            progress({'stage':'failed','error':repr(e)}); raise


PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>S2-W27 Workbench</title>
<style>body{font:16px system-ui;margin:2em}pre{white-space:pre-wrap;max-height:40em;overflow:auto}button{margin:.3em}</style>
<h1>S2-W27 Workbench</h1><p>Frozen configuration only. Resume and continuation require receipt-backed API requests.</p><div id="tasks"></div><pre id="detail"></pre>
<script>
const box=document.getElementById('tasks'),detail=document.getElementById('detail');
async function act(key,action){const r=await fetch('/'+encodeURIComponent(key)+'/'+action,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});detail.textContent=JSON.stringify(await r.json(),null,2);await refresh();}
async function refresh(){const r=await fetch('/status');const s=await r.json();box.replaceChildren();
const p=document.createElement('p');p.textContent='Port '+s.executor_port+' | Source '+s.source_root+' | Hold '+s.gate.hold+' | epoch '+s.gate.epoch;box.append(p);
for(const [key,t] of Object.entries(s.tasks)){const row=document.createElement('div');const label=document.createElement('span');label.textContent=key+' | '+t.stage+' | '+(t.env_step||0)+' ';row.append(label);
for(const [text,action] of [['Start','start'],['Stop','stop']]){const b=document.createElement('button');b.textContent=text;b.onclick=()=>act(key,action);row.append(b);}const view=document.createElement('button');view.textContent='Config / Log / Models';view.onclick=async()=>{detail.textContent=JSON.stringify(await (await fetch('/'+encodeURIComponent(key))).json(),null,2)};row.append(view);box.append(row);}}
refresh();setInterval(refresh,10000);
</script></html>"""


def handler(w):
    class Handler(BaseHTTPRequestHandler):
        def reply(self,value,code=200):
            raw=json.dumps(value,ensure_ascii=False).encode('utf8'); self.send_response(code)
            self.send_header('Content-Type','application/json; charset=utf-8'); self.end_headers(); self.wfile.write(raw)
        def do_GET(self):
            try:
                path=self.path.strip('/').split('/')
                if self.path=='/':
                    raw=PAGE.encode('utf8'); self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8')
                    self.end_headers(); self.wfile.write(raw); return
                self.reply(w.status() if path==['status'] else w.detail(path[-1]))
            except Exception as e: self.reply({'error':str(e)},400)
        def do_POST(self):
            try:
                if self.headers.get('Content-Type','').split(';')[0].strip().lower()!='application/json':
                    raise ValueError('application/json required')
                origin=self.headers.get('Origin')
                if origin is not None and origin != 'http://'+self.headers.get('Host',''):
                    raise ValueError('same-origin request required')
                size=int(self.headers.get('Content-Length','0'))
                if not 0<=size<=65536: raise ValueError('request too large')
                raw=self.rfile.read(size)
                body=json.loads(raw or b'{}'); path=self.path.strip('/').split('/')
                if path==['resume']:
                    self.reply({'rootpauseepoch':w.gate.resume(body['receipt'])}); return
                if len(path)==5 and path[:2]==['api','runs'] and path[-1]=='stop':
                    raise ValueError('invalid stop path')
                if len(path)==4 and path[:2]==['api','runs'] and path[-1]=='stop':
                    keys=[k for k,t in w.plan['tasks'].items() if t['config']['run_id']==path[2]]
                    if len(keys)!=1: raise ValueError('stop run ID not uniquely owned')
                    self.reply(w.stop(keys[0])); return
                if len(path)!=2: raise ValueError('task/action required')
                key,action=path
                if action=='stop': result=w.stop(key)
                elif action=='start' and not body: result=w.start(key)
                elif action in ('continue','resume') and set(body).issubset({'receipt_ref','dispatch_epoch','resource_ref'}): result=w.start(key,body.get('receipt_ref'),dispatch_epoch=body.get('dispatch_epoch'),resource_ref=body.get('resource_ref'))
                else: raise ValueError('only frozen starts or explicit receipt continuation')
                self.reply(result)
            except Exception as e: self.reply({'error':str(e)},400)
    return Handler


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['serve','worker'])
    for name in ('root','plan','control'): p.add_argument('--'+name,required=True)
    p.add_argument('--port',type=int,required=True); p.add_argument('--http-port',type=int,default=9070)
    p.add_argument('--task'); p.add_argument('--generation',type=int); a=p.parse_args()
    if a.action=='worker': worker(a)
    else:
        w=Workbench(a.root,a.plan,a.port,a.control)
        with file_lock(w.root/'server.lock',nonblocking=True):
            ThreadingHTTPServer(('127.0.0.1',a.http_port),handler(w)).serve_forever()

if __name__=='__main__': main()
