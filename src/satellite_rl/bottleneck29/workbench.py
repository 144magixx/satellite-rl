"""Loopback-only admitted dispatch and human-readable progress; no SSH."""
import argparse
import html
import json
import os
import subprocess
import sys
import time
import secrets
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .ledger import Halt, digest, atomic_pickle

def read_record(path):
    import pickle
    if not path.exists():return None
    with path.open('rb') as stream:return pickle.load(stream)

def process_birth():
    return Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19] if Path('/proc/self/stat').exists() else None


def identity():
    return dict(uid=os.getuid() if hasattr(os,'getuid') else None,
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if Path('/proc/sys/kernel/random/boot_id').exists() else None,
        gpu_uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip(),
        interpreter=str(Path(sys.executable).resolve()),
        threads={key:os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS')})


def admit(config, mode, actual=None):
    if mode not in ('science','engineering'):
        raise Halt('unknown scope')
    if actual is None and any(os.environ.get(key)!='1' for key in
            ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS')):
        raise Halt('four numerical thread environment values must equal 1')
    receipt = json.loads(Path(config[mode+'_admission']).read_text())
    if receipt.get('root_approved') is not True or receipt.get('mode') != mode:
        raise Halt('root admission absent')
    if not 0 <= time.time()-receipt['checked_unix'] <= config['receipt_max_age_seconds']:
        raise Halt('resource receipt stale')
    if receipt['binding'] != config['binding'] or receipt['lane'] != config['lane']:
        raise Halt('admission binding differs')
    if receipt.get('config_SHA') != config.get('_config_sha'):
        raise Halt('configuration SHA differs')
    if receipt['identity'] != (identity() if actual is None else actual):
        raise Halt('UID/boot/GPU/interpreter changed')
    for path, sha in config['pinned_files'].items():
        if digest(path) != sha:
            raise Halt('source/input/contract bytes changed: '+path)
    if not {'source_SHA','data_SHA','contract_SHA','authority_SHA'} <= config['binding'].keys():
        raise Halt('missing immutable binding')
    if any(Path(p).exists() for p in config['stop_paths']):
        raise Halt('pause/stop exists')
    if mode == 'science' and (receipt.get('engineering_accepted') is not True
            or receipt.get('full_loop_seconds_upper_bound', float('inf')) > receipt.get('remaining_science_seconds',0)):
        raise Halt('engineering or conservative full-loop throughput qualification absent')
    return receipt


def serve(config_path, port):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    config['_config_sha']=digest(config_path)
    root = Path(config['root']); root.mkdir(parents=True,exist_ok=True)
    # Kernel-held lock survives sessions and releases only on actual process exit.
    lock = (root/'owner.lock').open('a+b')
    if os.name == 'nt':
        import msvcrt
        lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    children = {}
    class Handler(BaseHTTPRequestHandler):
        def reply(self, value, status=200, content='application/json'):
            raw = value.encode() if isinstance(value,str) else json.dumps(value).encode()
            self.send_response(status); self.send_header('Content-Type',content+'; charset=utf-8')
            self.end_headers(); self.wfile.write(raw)
        def do_GET(self):
            from .ledger import Ledger
            progress={}
            for scope,where in [('science',root),('engineering',root/'engineering')]:
                if (where/'ledger.jsonl').exists():
                    from .worker import SCIENCE_LIMITS,ENGINEERING_LIMITS
                    try:
                        ledger=Ledger(where,SCIENCE_LIMITS if scope=='science' else ENGINEERING_LIMITS,config['binding'],read_only=True)
                        progress[scope]=dict(counts=ledger.counts(),unknown=ledger.unknown(),storage_bytes=ledger.storage_bytes(),
                            cursor=read_record(where/'cursor.pkl'),evidence=read_record(where/'evidence.pkl'),
                            completed=read_record(where/'completed.pkl'),stopped=read_record(where/'stopped.pkl'))
                    except Exception as error:progress[scope]=dict(status='unavailable_or_in_progress',reason=str(error))
                else:progress[scope]=dict(status='ledger_not_created_or_missing')
            state = dict(lane=config['lane'], binding=config['binding'],progress=progress,
                dispatch={m:read_record(root/(m+'_dispatch.pkl')) for m in ('engineering','science')},
                tasks={k:dict(pid=v.pid,exit=v.poll()) for k,v in children.items()},
                outputs={name:(root/name).exists() for name in ('completed.pkl','stopped.pkl','cursor.pkl')})
            if self.path == '/':
                self.reply('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="5">'
                    '<h1>W29 A0 lane '+str(config['lane'])+'</h1><pre>'+html.escape(json.dumps(state,indent=2))+
                    '</pre><p><a href="/config">Configuration</a> · <a href="/logs">Logs</a> · <a href="/status">Results / status</a></p>',content='text/html')
            elif self.path == '/config': self.reply(config)
            elif self.path == '/status': self.reply(state)
            elif self.path == '/logs':
                path=root/'worker.log'
                with path.open('rb') if path.exists() else open(os.devnull,'rb') as stream:
                    stream.seek(0,2); size=stream.tell(); stream.seek(max(0,size-65536)); raw=stream.read()
                self.reply(raw.decode('utf-8',errors='replace'),content='text/plain')
            else: self.reply(dict(error='not found'),404)
        def do_POST(self):
            if self.path=='/stop':
                stop=Path(config['stop_paths'][0]);stop.parent.mkdir(parents=True,exist_ok=True)
                atomic_pickle(stop,dict(reason='workbench stop',time_unix=time.time()))
                self.reply(dict(stopping=True));return
            mode = self.path.removeprefix('/start/')
            try:
                if any(p.poll() is None for p in children.values()):
                    raise Halt('owner already active')
                if digest(config_path)!=config['_config_sha']:raise Halt('configuration changed since service start')
                receipt=admit(config,mode)
                dispatch_path=root/(mode+'_dispatch.pkl');previous=read_record(dispatch_path)
                if previous is not None:
                    # No implicit replay after server restart, completed or crashed dispatch.
                    if receipt.get('recovery_of_dispatch_SHA')!=digest(dispatch_path) or receipt.get('recovery_approved') is not True:
                        raise Halt('existing durable dispatch requires explicit root recovery receipt')
                    if previous['status']=='completed':raise Halt('completed task cannot be restarted')
                    if previous.get('pid') and Path('/proc/'+str(previous['pid'])).exists():raise Halt('previous worker still exists')
                token=secrets.token_hex(24)
                atomic_pickle(dispatch_path,dict(status='launching',mode=mode,task_id=config['task_id']+'/'+mode,
                    generation=receipt['generation'],config_SHA=config['_config_sha'],token=token,created_unix=time.time()))
                # Exclusive worker lock prevents restart of workbench duplicating orphan worker.
                log=(root/'worker.log').open('ab')
                children[mode]=subprocess.Popen([sys.executable,'-m','satellite_rl.bottleneck29.workbench',
                    '--config',str(config_path),'--execute',mode,'--dispatch-token',token],stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                self.reply(dict(started=mode,pid=children[mode].pid),202)
            except Exception as error: self.reply(dict(error=str(error)),409)
    HTTPServer(('127.0.0.1',port),Handler).serve_forever()


def main():
    active_started=time.monotonic();active_started_unix=time.time()
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    parser.add_argument('--port',type=int,default=9090); parser.add_argument('--execute',choices=['science','engineering'])
    parser.add_argument('--dispatch-token')
    args=parser.parse_args()
    if args.execute:
        config=json.loads(Path(args.config).read_text());config['_config_sha']=digest(args.config)
        import fcntl
        lock=(Path(config['root'])/'worker.lock').open('a+b')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dispatch_path=Path(config['root'])/(args.execute+'_dispatch.pkl');dispatch=read_record(dispatch_path)
        if not dispatch or dispatch['status']!='launching' or dispatch['mode']!=args.execute or dispatch['token']!=args.dispatch_token or dispatch['config_SHA']!=config['_config_sha']:
            raise Halt('worker requires valid workbench dispatch')
        dispatch.update(status='running',pid=os.getpid(),birth=process_birth(),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            active_started_unix=active_started_unix,active_started_monotonic=active_started)
        atomic_pickle(dispatch_path,dispatch)
        try:
            admit(config,args.execute)
            if args.execute == 'science':
                from .inputs import input_loader
                from .worker import run_lane
                run_lane(config,input_loader(config),active_started=active_started)
            else:
                from .engineering import run_engineering
                run_engineering(config,active_started=active_started)
            dispatch['status']='completed'
        except BaseException as error:
            dispatch.update(status='stopped',error=repr(error));raise
        finally:
            dispatch.update(active_elapsed_seconds=time.monotonic()-active_started,active_finished_unix=time.time())
            scope=Path(config['root'])/('engineering' if args.execute=='engineering' else '')
            dispatch['ledger_exists']=(scope/'ledger.jsonl').exists()
            session_recorded=False
            if dispatch['ledger_exists']:
                try:
                    with (scope/'ledger.jsonl').open('rb') as stream:
                        for line in stream:
                            event=json.loads(line)['payload']
                            if event.get('type')=='session' and event.get('started_unix',0)>=active_started_unix:session_recorded=True
                except Exception:pass
            dispatch['ledger_session_recorded']=session_recorded
            dispatch['unledgered_preparation_requires_root_charge']=not session_recorded
            atomic_pickle(dispatch_path,dispatch)
    else: serve(args.config,args.port)


if __name__ == '__main__': main()
