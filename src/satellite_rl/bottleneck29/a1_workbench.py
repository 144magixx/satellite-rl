"""A1 loopback workbench and phase-only durable dispatch, A0 remains read-only."""
import argparse
import hashlib
import html
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
from .ledger import Halt,digest,atomic_pickle
from .workbench import identity,process_birth,read_record


def validate_storage_allocation(config):
    allocation=config['storage_allocation']
    cap=allocation['total_working_cap_bytes']
    a0=allocation['existing_working_bytes'];shared=allocation['other_reserved_bytes']
    science=allocation['lane_science_working_bytes'];engineering=allocation['lane_engineering_working_bytes']
    if (len(science)!=4 or len(engineering)!=4 or
            any(type(value)is not int or value<0 for value in [cap,a0,shared]+science+engineering)):
        raise Halt('storage allocation requires exact measured bytes and four lane caps')
    if cap>2*1024**3 or a0<=0 or sum(science)+sum(engineering)+shared+a0>cap:
        raise Halt('A0 plus A1 working sets exceed shared2GiB allowance')
    lane=config['lane']
    if config['science_storage_bytes']!=science[lane] or config['engineering_storage_bytes']!=engineering[lane]:
        raise Halt('phase storage limit differs from root shared allocation')
    return allocation


def admit(config,mode,actual=None):
    if mode not in ('engineering','science'):raise Halt('unknown A1 scope')
    if type(config['lane'])is not int or not 0<=config['lane']<4:raise Halt('invalid lane')
    receipt=json.loads(Path(config[mode+'_admission']).read_bytes())
    if receipt.get('root_approved') is not True or receipt.get('phase')!='A1' or receipt.get('mode')!=mode:raise Halt('root A1 scope admission absent')
    if receipt.get('binding')!=config['binding'] or receipt.get('config_SHA')!=config['_config_sha'] or receipt.get('lane')!=config['lane']:raise Halt('binding/config/lane changed')
    if receipt.get('storage_allocation')!=validate_storage_allocation(config):raise Halt('root storage allocation receipt absent or differs')
    if not 0<=time.time()-receipt['checked_unix']<=config['receipt_max_age_seconds']:raise Halt('stale resource receipt')
    current=identity() if actual is None else actual
    if current!=receipt['identity']:raise Halt('actual UID/boot/GPU/interpreter/threads differs')
    if actual is None and any(value!='1' for value in current['threads'].values()):raise Halt('thread scope differs')
    if any(Path(p).exists() for p in config['stop_paths']):raise Halt('new pause or task stop')
    required={'source_SHA','data_SHA','contract_SHA','authority_SHA','A0_source_SHA','A0_summary_SHA','actor_SHA','Q_status'}
    if not required<=config['binding'].keys() or config['binding']['Q_status']!='unavailable_same_node_150000':raise Halt('A1 exact provenance or missing-Q boundary absent')
    for path,sha in config['pinned_files'].items():
        if digest(path)!=sha:raise Halt('pinned source/input/contract/actor differs')
    limits=config[mode+'_limits']
    if receipt.get('limits')!=limits:raise Halt('allocation receipt differs')
    if mode=='science':
        caps=dict(phy=158925,enumeration=128925,suffix=30000,forward=30015,kernel=0,prefix=0,seconds=3600)
        if any(limits.get(key,0)>value or limits.get(key,0)<0 for key,value in caps.items()):raise Halt('science allocation exceeds fixed missing-Q lane cap')
        if not 0<config['science_storage_bytes']<=512*1024**2:raise Halt('four-lane storage cap exceeds2GiB')
        if receipt.get('engineering_accepted') is not True or receipt.get('full_loop_seconds_upper_bound',float('inf'))>receipt.get('remaining_science_seconds',0):raise Halt('full-loop engineering throughput not accepted')
    else:
        phy_cap=114 if config['lane']==0 else 113
        if limits.get('phy',0)>phy_cap or limits.get('kernel',0)>3 or limits.get('forward',0)>16 or limits.get('seconds',0)>894.7310753585:raise Halt('engineering lane allocation exceeds shared remainder split')
    return receipt


def _claim(root,name):
    # Atomic mkdir works across actual shared NFS clients; never auto-remove stale claims.
    path=Path(root)/(name+'.claim')
    try:path.mkdir()
    except FileExistsError:raise Halt('existing owner claim requires actual owner audit')
    atomic_pickle(path/'identity.pkl',dict(pid=os.getpid(),birth=process_birth(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    return path


def _release(path):
    (path/'identity.pkl').unlink();path.rmdir()


def serve(config_path,port):
    config_path=Path(config_path).resolve();config=json.loads(config_path.read_bytes());config['_config_sha']=digest(config_path)
    root=Path(config['root']);root.mkdir(parents=True,exist_ok=True);owner=_claim(root,'service')
    children={};views={}
    class Handler(BaseHTTPRequestHandler):
        def reply(self,value,code=200,content='application/json'):
            raw=value.encode() if isinstance(value,str) else json.dumps(value,allow_nan=False).encode()
            self.send_response(code);self.send_header('Content-Type',content+'; charset=utf-8');self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            status=dict(phase='A1',lane=config['lane'],binding=config['binding'],dispatch={},progress={})
            for mode in ('engineering','science'):
                status['dispatch'][mode]=read_record(root/(mode+'_dispatch.pkl'))
                where=root/mode
                try:
                    from .a1_stream import StreamStore
                    store=views.get(mode)
                    if store is None:
                        store=StreamStore(where,config['binding'],config[mode+'_limits'],storage_limit=config[mode+'_storage_bytes'],read_only=True);views[mode]=store
                    else:store.refresh()
                    status['progress'][mode]=store.summary()
                except Exception as error:status['progress'][mode]=dict(status='unavailable_or_in_progress',reason=str(error))
            if self.path=='/config':self.reply(config)
            elif self.path=='/status':self.reply(status)
            elif self.path=='/':self.reply('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="10"><h1>W29 A1</h1><pre>'+html.escape(json.dumps(status,indent=2))+'</pre><a href="/config">Configuration</a> | <a href="/logs">Logs</a> | <a href="/status">Progress and results</a>',content='text/html')
            elif self.path=='/logs':
                path=root/'worker.log'
                if path.exists():
                    with path.open('rb') as stream:stream.seek(max(0,path.stat().st_size-65536));raw=stream.read()
                    self.reply(raw.decode(errors='replace'),content='text/plain')
                else:self.reply('',content='text/plain')
            else:self.reply(dict(error='not found'),404)
        def do_POST(self):
            if self.path=='/stop':
                path=Path(config['stop_paths'][0]);atomic_pickle(path,dict(time_unix=time.time(),reason='A1 workbench stop'))
                self.reply(dict(stopping=True));return
            mode=self.path.removeprefix('/start/')
            try:
                if digest(config_path)!=config['_config_sha']:raise Halt('config changed since service start')
                receipt=admit(config,mode)
                if (root/'worker.claim').exists() or any(child.poll() is None for child in children.values()):raise Halt('owned worker exists')
                path=root/(mode+'_dispatch.pkl');prior=read_record(path)
                if prior:
                    if prior['status']=='completed':raise Halt('completed same ID cannot restart')
                    if receipt.get('recovery_approved') is not True or receipt.get('recovery_of_dispatch_SHA')!=digest(path):raise Halt('root audited recovery receipt absent')
                    if receipt['generation']<=prior['generation']:raise Halt('recovery generation must increase')
                    if receipt.get('unledgered_time_debit_seconds',0)<prior.get('unledgered_seconds',0):raise Halt('unledgered preparation needs explicit time debit before dispatch')
                    atomic_pickle(root/'dispatch_history'/(digest(path)+'.pkl'),prior)
                token=secrets.token_hex(24)
                atomic_pickle(path,dict(phase='A1',mode=mode,task_id=config['task_id']+'/'+mode,status='launching',generation=receipt['generation'],config_SHA=config['_config_sha'],token=token,created_unix=time.time()))
                with (root/'worker.log').open('ab') as log:
                    children[mode]=subprocess.Popen([sys.executable,'-m','satellite_rl.bottleneck29.a1_workbench','--config',str(config_path),'--execute',mode,'--dispatch-token',token],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                self.reply(dict(started=mode,pid=children[mode].pid),202)
            except Exception as error:self.reply(dict(error=str(error)),409)
    try:HTTPServer(('127.0.0.1',port),Handler).serve_forever()
    finally:_release(owner)


def main():
    started=time.monotonic();unix=time.time()
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--port',type=int,default=9100)
    parser.add_argument('--execute',choices=['engineering','science']);parser.add_argument('--dispatch-token');args=parser.parse_args()
    if not args.execute:return serve(args.config,args.port)
    config=json.loads(Path(args.config).read_bytes());config['_config_sha']=digest(args.config);root=Path(config['root'])
    path=root/(args.execute+'_dispatch.pkl');dispatch=read_record(path)
    if not dispatch or dispatch.get('status')!='launching' or dispatch.get('token')!=args.dispatch_token or dispatch.get('config_SHA')!=config['_config_sha'] or dispatch.get('mode')!=args.execute:raise Halt('CLI cannot bypass admitted durable workbench dispatch')
    claim=None;store=None
    dispatch.update(status='running',pid=os.getpid(),birth=process_birth(),started_unix=unix,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip());atomic_pickle(path,dispatch)
    try:
        claim=_claim(root,'worker');receipt=admit(config,args.execute)
        from .a1_stream import StreamStore
        store=StreamStore(root/args.execute,config['binding'],config[args.execute+'_limits'],
            storage_limit=config[args.execute+'_storage_bytes'],stop_paths=config['stop_paths'],minimum_free_bytes=config['minimum_free_bytes'],active_started=started)
        if receipt.get('unledgered_time_debit_seconds',0)>0:
            store.debit_time(receipt['recovery_of_dispatch_SHA'],receipt['unledgered_time_debit_seconds'])
        store.begin()
        if args.execute=='science' and receipt['full_loop_seconds_upper_bound']>store.remaining-(time.monotonic()-started):raise Halt('throughput upper bound exceeds actual retained time')
        if args.execute=='engineering':from .a1_engineering import execute
        else:from .a1_worker import execute
        execute(config,store,mode=args.execute)
        store.record_metadata('status','completed');dispatch['status']='completed'
    except BaseException as error:
        dispatch.update(status='stopped',error=repr(error));raise
    finally:
        recorded=store is not None and store.session is not None
        if store is not None:store.finish()
        elapsed=time.monotonic()-started
        dispatch.update(elapsed_seconds=elapsed,finished_unix=time.time(),unledgered_seconds=0 if recorded else elapsed)
        atomic_pickle(path,dispatch)
        if claim is not None:_release(claim)


if __name__=='__main__':main()
