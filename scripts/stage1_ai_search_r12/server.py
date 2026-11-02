"""Single-owner main-thread HTTP service for one exact R12 batch. No queue.

Only explicit root POST starts the already admitted frozen worker. There are no
resume/retry/training APIs. The main thread also bounds and reaps its one child.
"""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .admission import check, STOP_NEW, DEADLINE
from .runtime_epoch import validate as validate_epoch,stopped,absolute_deadline,prior_dispatch_absent
from .persistence import Journal, identity
from .worker import pinned, source_check, bound_parent, _markers, _storage

def process_identity(pid):
    import psutil
    p=psutil.Process(pid)
    return dict(pid=pid,birth_unix=p.create_time(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())

class Endpoint:
    def __init__(self,plan_ref,review_ref):
        self.plan_ref,self.review_ref=plan_ref,review_ref
        self.plan=json.loads(pinned(plan_ref));self.review=json.loads(pinned(review_ref))
        source_check(self.plan)
        if 'runtime_epoch' in self.plan:
            validate_epoch(self.plan)
            prior_dispatch_absent(self.plan)
        self.root=Path(self.plan['control_root'])
        if not self.root.resolve().is_relative_to(Path(self.plan['storage_root']).resolve()):raise ValueError('service shared owned storage root')
        self.journal=Journal(self.root)
        self.worker=None;self.registry={};self.peak_RSS=0;self.started=time.monotonic();self.finished=False;self.pending_exit_since=None
        self.path='/api/ai-search-r12/tasks/'+self.plan['run_id']
        self.owner=process_identity(os.getpid())
        self.journal.write('service_identity',dict(actual_service=self.owner,plan_content_SHA=self.plan['content_SHA'],
                                                 execution_start_requires_explicit_root_POST=True))

    def get(self,path):
        if path=='/health':return dict(group='S1-AI17-R12',actual_service=self.owner,plan_content_SHA=self.plan['content_SHA'],
                                     main_thread_HTTP=True,automatic_retry=False)
        if path!=self.path:raise ValueError('sole exact R12 task')
        result=dict(run_id=self.plan['run_id'],started=self.worker is not None,finished=self.finished)
        if self.journal.has('worker_exit'):result['exit']=self.journal.read('worker_exit')
        complete=Path(self.plan['archive_root'])/'batch'/'complete.json'
        if complete.exists():result['batch']=json.loads(complete.read_text(encoding='utf8'))
        return result

    def post(self,path,body,*,popen=subprocess.Popen):
        if body.get('request_id')!=self.plan['request_id'] or body.get('plan_content_SHA')!=self.plan['content_SHA']:
            raise ValueError('exact single root request and content binding')
        if set(body)!= {'request_id','plan_content_SHA'}:raise ValueError('bounded exact request')
        if path==self.path+'/stop':
            marker=Path(self.plan['own_stop_marker'])
            if str(marker) not in self.plan['watched_pause_paths']:raise ValueError('owned watched stop marker')
            if not marker.exists():
                with marker.open('x',encoding='utf8') as f:
                    json.dump(dict(paused=True,run_id=self.plan['run_id']),f);f.flush();os.fsync(f.fileno())
            return dict(stop_requested=True)
        if path!=self.path+'/start':raise ValueError('single start/stop only')
        if self.journal.has('submission_intent'):raise ValueError('submitted or unknown intent; never replay')
        # Re-read exact pinned approval immediately before the sole start.
        self.review=json.loads(pinned(self.review_ref));pinned(self.plan_ref)
        if time.monotonic()-self.started>=120:raise ValueError('service idle expired; never renew lease')
        if 'runtime_epoch' in self.plan:prior_dispatch_absent(self.plan)
        check(self.plan,self.review,now=datetime.now(timezone.utc),marker_values=_markers(self.plan))
        self.journal.write('submission_intent',dict(request=body,actual_service=self.owner,
                           plan_content_identity=identity(self.plan_ref),review_content_identity=identity(self.review_ref)))
        command=[sys.executable,'-B','-u','-m','scripts.stage1_ai_search_r12.worker',
            '--plan',self.plan_ref['path'],'--plan-sha256',self.plan_ref['sha256'],
            '--review',self.review_ref['path'],'--review-sha256',self.review_ref['sha256'],
            '--archive',self.plan['archive_root']]
        with (self.root/'worker.stdout.log').open('xb') as out,(self.root/'worker.stderr.log').open('xb') as err:
            self.worker=popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,
                              preexec_fn=lambda:bound_parent(self.owner['pid']),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        owner=process_identity(self.worker.pid);self.registry[(owner['pid'],owner['birth_unix'])]=owner
        self.journal.write('worker_launch',dict(command=command,actual_worker=owner))
        return dict(started=True,actual_worker=owner,POST_replayed=False)

    def monitor(self):
        import psutil
        if stopped(self.plan):
            self.stop_owned('service_cooperative_pause_marker');return
        if self.worker is None:
            if time.monotonic()-self.started>=120:
                self.journal.write('idle_expired',dict(submitted=False,automatic_restart=False));self.finished=True
            return
        for p in psutil.Process().children(recursive=True):
            try:self.registry[(p.pid,p.create_time())]=dict(pid=p.pid,birth_unix=p.create_time(),boot_id=self.owner['boot_id'])
            except psutil.NoSuchProcess:pass
        live=[];rss=psutil.Process().memory_info().rss
        for key,owner in self.registry.items():
            try:
                p=psutil.Process(owner['pid'])
                if p.create_time()!=owner['birth_unix']:continue
                if p.status()!=psutil.STATUS_ZOMBIE:live.append(p);rss+=p.memory_info().rss
            except psutil.NoSuchProcess:pass
        self.peak_RSS=max(self.peak_RSS,rss)
        _storage(self.plan['storage_root'])
        if time.monotonic()-self.started>=2430 or stopped(self.plan) or self.peak_RSS>2147483648:
            self.stop_owned('service_time_pause_or_RSS')
            return
        code=self.worker.poll()
        if code is not None:
            leftovers=[p.pid for p in live if p.pid!=self.worker.pid]
            if leftovers:
                if self.pending_exit_since is None:self.pending_exit_since=time.monotonic()
                if time.monotonic()-self.pending_exit_since<3:return
                self.stop_owned('worker_exit_with_registered_live_descendants');return
            complete=Path(self.plan['archive_root'])/'batch'/'complete.json'
            scientific_complete=code==0 and complete.exists() and json.loads(complete.read_text(encoding='utf8')).get('complete') is True
            self.journal.write('worker_exit',dict(exit_code=code,registered_descendants_exit_verified=True,scientific_complete=scientific_complete,
                registered_identities=list(self.registry.values()),peak_parent_children_RSS_bytes=self.peak_RSS,
                actual_worker_observed=True))
            self.finished=True

    def stop_owned(self,reason):
        import psutil
        for p in psutil.Process().children(recursive=True):
            try:self.registry[(p.pid,p.create_time())]=dict(pid=p.pid,birth_unix=p.create_time(),boot_id=self.owner['boot_id'])
            except psutil.NoSuchProcess:pass
        deadline=min(time.monotonic()+15,absolute_deadline(self.plan))
        live=[]
        for owner in self.registry.values():
            try:
                p=psutil.Process(owner['pid'])
                if p.create_time()==owner['birth_unix'] and p.status()!=psutil.STATUS_ZOMBIE:live.append(p);p.terminate()
            except psutil.NoSuchProcess:pass
        _,alive=psutil.wait_procs(live,timeout=max(0,min(10,deadline-time.monotonic())))
        for p in alive:
            try:p.kill()
            except psutil.NoSuchProcess:pass
        _,alive=psutil.wait_procs(alive,timeout=max(0,deadline-time.monotonic()))
        if self.worker is not None:self.worker.poll()
        self.journal.write('worker_exit',dict(reason=reason,registered_descendants_exit_verified=not alive,
            still_live=[p.pid for p in alive],registered_identities=list(self.registry.values()),
            peak_parent_children_RSS_bytes=self.peak_RSS,scientific_complete=False))
        self.finished=True

    def close(self):
        try:
            if self.worker is not None and not self.finished:self.stop_owned('service_finally')
            self.journal.write('service_retirement',dict(actual_service=self.owner,
                retirement_expected=True,external_process_exit_observation_required=True,
                worker_finished=self.finished,automatic_restart=False))
        finally:self.journal.close()

def main():
    parser=argparse.ArgumentParser()
    for key in ('plan','plan-sha256','review','review-sha256'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--port',required=True,type=int);args=parser.parse_args()
    def ref(path,sha):return dict(path=path,sha256=sha,bytes=Path(path).stat().st_size)
    endpoint=Endpoint(ref(args.plan,args.plan_sha256),ref(args.review,args.review_sha256))
    if args.port!=endpoint.plan['workbench_port'] or not 1024<=args.port<=65535:raise ValueError('exact root-qualified workbench port')
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup();self.connection.settimeout(2.)
        def dispatch(self):
            try:
                if self.command=='GET':value=endpoint.get(self.path)
                else:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<=8192:raise ValueError('bounded exact request')
                    value=endpoint.post(self.path,json.loads(self.rfile.read(length)))
                raw=json.dumps(value).encode();self.send_response(200)
            except Exception as error:raw=json.dumps(dict(error=str(error),automatic_retry=False)).encode();self.send_response(409)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        do_GET=dispatch;do_POST=dispatch
        def log_message(self,*args):pass
    server=HTTPServer(('127.0.0.1',args.port),Handler);server.timeout=.25
    try:
        while not endpoint.finished and not stopped(endpoint.plan):
            server.handle_request();endpoint.monitor()
    finally:server.server_close();endpoint.close()

if __name__=='__main__':main()
