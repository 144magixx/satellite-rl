"""Independent loopback W28 backend. Frozen tasks; no automatic retry or resume."""
import argparse
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import threading
from time import perf_counter
from satellite_rl.t0_width.data import read_json_ref,file_sha
from satellite_rl.t0_width.logging import atomic_json as _atomic_json,content_sha
from satellite_rl.tsac_search27.pause import Gate,file_lock,proc_identity,same_identity,validate_scope
from .workbench import validate_plan
from .transaction import fsync_directory

def atomic_json(path,value):
    _atomic_json(path,value);fsync_directory(Path(path).parent)

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))

class Workbench:
    def __init__(self,root,plan_path,control):
        self.root=Path(root).resolve();self.plan_path=Path(plan_path).resolve();self.plan_SHA=file_sha(plan_path)
        self.plan=read(plan_path);self.gate=Gate(control);self.lock=threading.RLock()
        if (self.plan.get('schema')!='W28WorkbenchPlanV1' or self.plan.get('experiment_id')!='S2-W28'
                or self.plan.get('max_workers')!=1 or self.plan.get('executor_port')!=20418
                or self.gate.spec['experiment_id']!='S2-W28'):raise ValueError('independent W28 single-card workbench required')
        self.root.mkdir(parents=True,exist_ok=True);(self.root/'runs').mkdir(exist_ok=True)
        if sum(task.get('schema')=='W28EngineeringPlanV1' for task in self.plan['tasks'].values())>1:raise ValueError('one cumulative engineering task only; no budget reset by new task IDs')
    def detail(self,key):
        if key not in self.plan['tasks']:raise ValueError('unknown immutable task')
        folder=self.root/'runs'/key
        value=read(folder/'status.json') if (folder/'status.json').exists() else dict(stage='planned')
        value.update(task_key=key,config=self.plan['tasks'][key],owner=read(folder/'owner.json') if (folder/'owner.json').exists() else None)
        cursor=folder/'matrix_cursor.pkl'
        if not cursor.exists() and (folder/'engineering_cursor.pkl').exists():
            c=pickle.loads((folder/'engineering_cursor.pkl').read_bytes())
            value['matrix']=dict(cursor=c.get('cursor'),unknown=c.get('unknown',['initialization_incomplete']),elapsed_active_seconds=c.get('elapsed_active_seconds',0.))
        if cursor.exists():
            c=pickle.loads(cursor.read_bytes());value['matrix']=dict(cursor=c['cursor'],costs_completed_episodes=c['costs'],unknown=c['unknown'],elapsed_active_seconds=c['elapsed_active_seconds'])
            if c['active'] is not None:
                j=folder/'episodes'/('%04d'%c['cursor'])/'current.pkl'
                if j.exists():
                    active=pickle.loads(j.read_bytes());value['matrix']['active_attempts']=active['attempts'];value['matrix']['active_completed']=active['completed'];value['matrix']['pending']=active['transaction'].get('pending') if active['transaction'] else None
        return value
    def status(self):return dict(experiment_id='S2-W28',plan_SHA=self.plan_SHA,gate=self.gate.snapshot(),tasks={key:self.detail(key) for key in self.plan['tasks']})
    def admission(self,key,resource_ref=None,dispatch_epoch=None):
        if file_sha(self.plan_path)!=self.plan_SHA or self.plan.get('runtime_approved') is not True:raise ValueError('draft/unfrozen workbench cannot run')
        task=self.plan['tasks'][key];self.gate.validate(dispatch_epoch or task['rootpauseepoch'])
        if task.get('runtime_approved') is not True:raise ValueError('draft task cannot run')
        rows=read_json_ref(task['metadata_ref'])['rows']
        if task.get('schema')=='W28EngineeringPlanV1':
            from .engineering import validate_engineering
            validate_engineering(task,self.gate,dispatch_epoch)
        else:validate_plan(task,rows,self.gate,runtime=True,dispatch_epoch=dispatch_epoch)
        for name in ('source','data','resources','cost'):
            value=read_json_ref(resource_ref if name=='resources' and resource_ref else self.plan['admission_refs'][name])
            if value.get('root_approved') is not True or value.get('passed') is not True or value.get('experiment_id')!='S2-W28':raise ValueError('W28 admission required: '+name)
            if name=='source' and value.get('source_commit')!=task['source_commit']:raise ValueError('source admission differs')
            if name=='data' and task['phase'] not in value.get('allowed_phases',[]):raise ValueError('data admission does not cover this phase')
            if name=='cost' and (value.get('engineering_caps')!=dict(physical_calls_attempted=320,real_commit_attempts=64,candidate_restore_attempts=256,model_forward_attempts=64)
                    or value.get('science_capacity_reward_cap')!=179722 or value.get('unknown_cost_not_zero') is not True):raise ValueError('cumulative finite cost admission differs')
            if name=='resources':
                age=(datetime.now(timezone.utc)-datetime.fromisoformat(value['checked_UTC'])).total_seconds()
                if not 0<=age<=min(300,value.get('max_age_seconds',300)):raise ValueError('fresh resource receipt required')
                current=proc_identity(os.getpid())
                if current['boot_id']!=value['boot_id']:raise ValueError('boot identity differs')
                uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()
                if value['gpu_uuid'] not in uuids:raise ValueError('GPU UUID differs')
                compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                if compute:raise ValueError('actual GPU compute owner already present')
                watchdog=value.get('watchdog_identity')
                if not watchdog or not same_identity(proc_identity(watchdog['pid']),watchdog):raise ValueError('current W28 watchdog not live')
                if str(self.gate.directory) not in watchdog['argv']:raise ValueError('watchdog outside this independent Gate')
        return task
    def stop(self,key):
        self.detail(key);folder=self.root/'runs'/key;folder.mkdir(exist_ok=True);atomic_json(folder/'stop.json',dict(requested=True))
        return dict(stage='stop_requested')
    def start(self,key,*,recovery_ref=None,resource_ref=None,dispatch_epoch=None):
        if not re.fullmatch('[A-Za-z0-9_-]{1,100}',key):raise ValueError('unsafe task key')
        with self.lock:
            task=self.admission(key,resource_ref,dispatch_epoch);folder=self.root/'runs'/key;folder.mkdir(exist_ok=True)
            for other in (self.root/'runs').iterdir():
                if (other/'owner.json').exists():
                    owner=read(other/'owner.json')
                    if same_identity(proc_identity(owner['pid']),owner):raise ValueError('W28 card already has a live worker')
            intents=list(folder.glob('dispatch_*.json'))
            if intents:
                if recovery_ref is None:raise ValueError('existing dispatch is observe-only without exact root recovery receipt')
                receipt=read_json_ref(recovery_ref)
                if (receipt.get('schema')!='W28AccountedRecoveryV1' or receipt.get('root_approved') is not True
                        or receipt.get('task_key')!=key or receipt.get('plan_SHA')!=self.plan_SHA
                        or receipt.get('previous_dispatch_SHA')!=file_sha(sorted(intents)[-1])
                        or receipt.get('unknown_calls_replayed')!=0 or receipt.get('same_ID') is not True):raise ValueError('accounted same-ID recovery required')
                wall=receipt.get('unrecorded_wall_upper_bound_seconds')
                if (isinstance(wall,bool) or not isinstance(wall,(int,float)) or not math.isfinite(wall) or wall<0
                        or (wall==0 and receipt.get('zero_unrecorded_wall_proven') is not True)):
                    raise ValueError('recovery needs explicit debit for unrecorded wall, never default zero')
                cursor=folder/('engineering_cursor.pkl' if task.get('schema')=='W28EngineeringPlanV1' else 'matrix_cursor.pkl')
                if receipt.get('cursor_SHA')!=(file_sha(cursor) if cursor.exists() else 'ABSENT_NO_RUNTIME'):
                    raise ValueError('recovery does not bind exact last durable cursor')
            generation=len(intents);epoch=dispatch_epoch or task['rootpauseepoch']
            runtime=dict(task_key=key,generation=generation,dispatch_epoch=epoch,resource_ref=resource_ref,recovery_ref=recovery_ref,plan_SHA=self.plan_SHA)
            atomic_json(folder/('dispatch_%06d.json'%generation),runtime)
            if (folder/'stop.json').exists():(folder/'stop.json').unlink()
            command=[sys.executable,'-B','-m','satellite_rl.tsac_screen28.backend','worker','--root',str(self.root),'--plan',str(self.plan_path),'--control',str(self.gate.directory),'--task',key,'--generation',str(generation)]
            environment=os.environ.copy();environment['PYTHONPATH']=self.plan['source_root']+'/src'+os.pathsep+self.plan['source_root'];environment['PYTHONDONTWRITEBYTECODE']='1'
            with (folder/'worker.log').open('ab') as log:
                child=subprocess.Popen(command,cwd=self.plan['source_root'],env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            return dict(stage='starting',pid=child.pid,generation=generation)

def register(w,owner,epoch,recovery_ref):
    # W27's same-epoch branch is pilot-training specific. W28 validates its own
    # root-accounted same-ID receipt and uses the same durable Gate transaction.
    w.gate.refresh();validate_scope(owner,w.gate.spec)
    if not same_identity(proc_identity(owner['pid']),owner):raise ValueError('actual worker identity differs')
    with w.gate.transaction() as state:
        w.gate._cutoff(state,w.gate.clock())
        if state['hold'] or state['epoch']!=epoch:raise ValueError('current W28 hold/epoch prevents worker')
        for previous in state['workers']:
            if previous['task_id']==owner['task_id']:
                if same_identity(proc_identity(previous['pid']),previous):raise ValueError('previous task worker remains live')
                receipt=read_json_ref(recovery_ref) if recovery_ref else {}
                if receipt.get('root_approved') is not True or receipt.get('previous_worker')!=previous:raise ValueError('previous exact worker not accounted')
        state['workers'].append(dict(owner,rootpauseepoch=epoch))

def worker(args):
    w=Workbench(args.root,args.plan,args.control);folder=w.root/'runs'/args.task
    dispatch=read(folder/('dispatch_%06d.json'%args.generation));task=w.admission(args.task,dispatch.get('resource_ref'),dispatch['dispatch_epoch'])
    if dispatch['plan_SHA']!=w.plan_SHA:raise ValueError('dispatch source plan changed')
    owner=proc_identity(os.getpid());owner.update(experiment_id='S2-W28',task_id=args.task,rootpauseepoch=dispatch['dispatch_epoch'],stop_url=w.plan['stop_url_base']+'/api/runs/'+args.task+'/stop')
    with file_lock(w.root/'card.lock',nonblocking=True):
        register(w,owner,dispatch['dispatch_epoch'],dispatch.get('recovery_ref'));atomic_json(folder/'owner.json',owner)
        started=perf_counter()
        engineering=task.get('schema')=='W28EngineeringPlanV1'
        phase_keys=[key for key,value in w.plan['tasks'].items() if (value.get('schema')=='W28EngineeringPlanV1')==engineering]
        prior_seconds=sum(w.detail(key).get('matrix',{}).get('elapsed_active_seconds',0.) for key in phase_keys)
        if engineering and task.get('source_migration_ref'):
            from .migration import validate_source_requalification
            inheritance=validate_source_requalification(task);existing=folder/'engineering_cursor.pkl'
            if not existing.exists():prior_seconds+=inheritance['prior_wall_seconds']
            else:
                inherited=pickle.loads(existing.read_bytes())
                if (inherited.get('prior_actual_costs')!=inheritance['prior_actual'] or inherited.get('source_migration_ref')!=task['source_migration_ref']
                        or inherited.get('elapsed_active_seconds',-1)<inheritance['prior_wall_seconds']):raise ValueError('inherited source wall/cost metadata incomplete; no zero fallback')
        for key in phase_keys:
            for prior in (w.root/'runs'/key).glob('dispatch_*.json'):
                receipt_ref=read(prior).get('recovery_ref')
                if receipt_ref:prior_seconds+=read_json_ref(receipt_ref)['unrecorded_wall_upper_bound_seconds']
        stop=lambda:(folder/'stop.json').exists() or w.gate.stop_requested(dispatch['dispatch_epoch']) or prior_seconds+perf_counter()-started>=w.plan['engineering_wall_budget_seconds' if engineering else 'science_wall_budget_seconds']
        def progress(value):atomic_json(folder/'status.json',value)
        try:
            if task.get('schema')=='W28EngineeringPlanV1':
                from .engineering import run_engineering
                result=run_engineering(task,folder,stop,progress)
            else:
                from .runtime import run_matrix
                result=run_matrix(task,folder,stop,progress)
            progress(result)
        except BaseException as exc:progress(dict(stage='failed',error=type(exc).__name__,message=str(exc),automatic_retry=False));raise

def handler(w):
    class Handler(BaseHTTPRequestHandler):
        def reply(self,value,status=200):
            raw=json.dumps(value,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            try:self.reply(w.status() if self.path=='/status' else w.detail(self.path.strip('/')))
            except Exception as exc:self.reply(dict(error=str(exc)),400)
        def do_POST(self):
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<=size<=65536 or self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('bounded JSON required')
                if self.headers.get('Origin') not in (None,'http://'+self.headers.get('Host','')):raise ValueError('same origin required')
                body=json.loads(self.rfile.read(size) or b'{}');parts=self.path.strip('/').split('/')
                if len(parts)==4 and parts[:2]==['api','runs'] and parts[3]=='stop':key,action=parts[2],'stop'
                else:key,action=parts
                if action=='stop':result=w.stop(key)
                elif action=='start' and not body:result=w.start(key)
                elif action=='resume' and set(body)<= {'recovery_ref','resource_ref','dispatch_epoch'}:result=w.start(key,**body)
                else:raise ValueError('only frozen start, cooperative stop and accounted resume')
                self.reply(result)
            except Exception as exc:self.reply(dict(error=str(exc)),400)
    return Handler

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('serve','worker'))
    for name in ('root','plan','control'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--task');parser.add_argument('--generation',type=int);parser.add_argument('--http-port',type=int,default=9080)
    args=parser.parse_args()
    if args.action=='worker':worker(args)
    else:
        w=Workbench(args.root,args.plan,args.control)
        with file_lock(w.root/'server.lock',nonblocking=True):ThreadingHTTPServer(('127.0.0.1',args.http_port),handler(w)).serve_forever()

if __name__=='__main__':main()
