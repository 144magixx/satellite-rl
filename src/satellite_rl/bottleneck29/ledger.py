"""Single-owner fsync append journal, safe from SQLite WAL assumptions on NFS."""
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

class Halt(RuntimeError):
    pass

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_pickle(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as stream:
        pickle.dump(value,stream,protocol=5); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp,path)
    if os.name!='nt':
        fd=os.open(path.parent,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)

class Ledger:
    def __init__(self,root,limits,binding,read_only=False,active_started=None):
        self.created=time.monotonic() if active_started is None else active_started; self.root=Path(root);self.read_only=read_only
        if not read_only:self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'ledger.jsonl'; self.limits=limits; self.operations={}; self.artifacts={}; self.sessions={}; self.session=None;self._counts={};self.file_sizes={};self.journal_bytes=0;self.storage_limit=512*1024**2
        self.object_bytes=0
        expected=dict(limits=limits,binding=binding); bound=None
        sentinel=self.root/'ledger_identity.json'
        if read_only and (not self.path.exists() or not sentinel.exists()):raise Halt('ledger unavailable; read-only status never initializes')
        if not self.path.exists():
            if sentinel.exists() or any((self.root/name).exists() for name in
                    ('operations','rows','snapshots','cursor.pkl','completed.pkl','stopped.pkl','evidence.pkl')):
                raise Halt('missing journal with existing task identity/assets')
            with sentinel.open('x',encoding='utf-8') as stream:
                json.dump(expected,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
        elif not sentinel.exists():raise Halt('journal identity sentinel missing')
        if json.loads(sentinel.read_text(encoding='utf-8'))!=expected:raise Halt('identity sentinel differs')
        if self.path.exists():
            with self.path.open('rb') as stream:
                for line in stream:
                    self.journal_bytes+=len(line)
                    if not line.endswith(b'\n'): raise Halt('torn journal requires root audit')
                    try:
                        event=json.loads(line); payload=event['payload']
                        encoded=json.dumps(payload,sort_keys=True,separators=(',',':')).encode()
                        if hashlib.sha256(encoded).hexdigest()!=event['sha256']: raise Halt('journal checksum mismatch')
                    except (ValueError,KeyError) as error: raise Halt('invalid journal') from error
                    if payload['type']=='binding': bound=payload['value']
                    self._apply(payload)
        if bound is not None and bound!=expected: raise Halt('ledger binding or allocation changed')
        if bound is None:
            if read_only:raise Halt('binding incomplete')
            self._append(dict(type='binding',value=expected))
        self.sentinel_bytes=sentinel.stat().st_size

    def _apply(self,event):
        kind=event['type']
        if kind=='operation':
            if event['id'] not in self.operations:self._counts[event['kind']]=self._counts.get(event['kind'],0)+1
            self.operations[event['id']]=event
        elif kind=='artifact': self.artifacts[event['id']]=event
        elif kind=='session': self.sessions[event['id']]=event
        if 'bytes' in event:
            self.object_bytes+=event['bytes']-self.file_sizes.get(event['path'],0)
            self.file_sizes[event['path']]=event['bytes']

    def storage_bytes(self):return self.journal_bytes+self.object_bytes+getattr(self,'sentinel_bytes',0)

    def _save(self,path,value):
        name=str(Path(path).relative_to(self.root));size=len(pickle.dumps(value,protocol=5))
        if size>1024**2:raise Halt('single serialized object exceeds frozen 1MiB bound')
        if self.storage_bytes()+size+4096>self.storage_limit:raise Halt('registered lane storage allowance including atomic temporary peak')
        atomic_pickle(path,value)
        return size

    def _append(self,payload):
        if self.read_only:raise Halt('read-only journal')
        encoded=json.dumps(payload,sort_keys=True,separators=(',',':')).encode()
        raw=json.dumps(dict(payload=payload,sha256=hashlib.sha256(encoded).hexdigest()),separators=(',',':')).encode()+b'\n'
        delta=payload.get('bytes',0)-self.file_sizes.get(payload.get('path'),0) if 'bytes' in payload else 0
        if self.storage_bytes()+len(raw)+delta>self.storage_limit:raise Halt('registered lane journal storage allowance')
        with self.path.open('ab',buffering=0) as stream:
            if stream.write(raw)!=len(raw): raise Halt('short journal write')
            os.fsync(stream.fileno())
        self.journal_bytes+=len(raw)
        self._apply(payload)

    def counts(self):
        return dict(self._counts)

    def operation_count(self): return len(self.operations)

    def unknown(self):
        return [(key,e['kind'],e['status']) for key,e in self.operations.items() if e['status']!='done']

    def begin(self):
        # One bounded startup inventory; never repeated per candidate or watchdog.
        observed={}
        for name in ('cursor.pkl','completed.pkl','stopped.pkl','evidence.pkl'):
            if (self.root/(name+'.tmp')).exists():raise Halt('atomic root temporary requires root audit')
            if (self.root/name).exists() and name not in self.file_sizes:raise Halt('unregistered root artifact requires root audit')
        for name in ('operations','rows','snapshots'):
            directory=self.root/name
            if directory.exists():
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if not entry.is_file(follow_symlinks=False):raise Halt('unexpected storage member')
                        observed[name+'/'+entry.name]=entry.stat().st_size
                        if len(observed)>60000:raise Halt('startup inventory bound')
        for name in self.file_sizes:
            if '/' not in name and '\\' not in name:
                observed[name]=(self.root/name).stat().st_size
        if observed!=self.file_sizes:raise Halt('unregistered/orphan or changed-size task artifact; root audit required')
        if self.unknown(): raise Halt('unresolved intent; root audit required, no automatic replay')
        prior=sum(e.get('elapsed',e['reserved']) for e in self.sessions.values())
        self.remaining=self.limits['seconds']-prior
        if self.remaining<=0: raise Halt('active time exhausted or uncertain prior session reserved remainder')
        self.started=self.created; self.session=len(self.sessions)
        self._append(dict(type='session',id=self.session,reserved=self.remaining,started_unix=time.time()))
        self.boundary()

    def boundary(self):
        if self.session is None or time.monotonic()-self.started>=self.remaining: raise Halt('active time allowance exhausted')

    def finish(self):
        if self.session is not None:
            event=dict(self.sessions[self.session]);event['elapsed']=time.monotonic()-self.started
            self._append(event);self.session=None

    def artifact(self,artifact_id,path,value=None,replace=False):
        event=self.artifacts.get(artifact_id)
        if event and not replace:
            if digest(self.root/event['path'])!=event['sha']: raise Halt('artifact checksum mismatch')
            with (self.root/event['path']).open('rb') as stream:return pickle.load(stream)
        if value is None:return None
        size=self._save(path,value)
        self._append(dict(type='artifact',id=artifact_id,path=str(Path(path).relative_to(self.root)),sha=digest(path),bytes=size))
        return value

    def meter(self,kind,call_id,operation):
        self.boundary();event=self.operations.get(call_id)
        if event:
            if event['kind']!=kind or event['status']!='done' or digest(self.root/event['path'])!=event['sha']: raise Halt('unknown or mismatched operation '+call_id)
            with (self.root/event['path']).open('rb') as stream:return pickle.load(stream)
        counts=self.counts()
        if kind not in self.limits or counts.get(kind,0)>=self.limits[kind]:raise Halt('budget exhausted: '+kind)
        if kind.startswith('phy_') and sum(v for k,v in counts.items() if k.startswith('phy_'))>=self.limits['phy']:raise Halt('total PHY budget exhausted')
        name='operations/'+hashlib.sha256(call_id.encode()).hexdigest()+'.pkl'
        event=dict(type='operation',id=call_id,kind=kind,status='intent',path=name)
        self._append(event)
        value=operation();size=self._save(self.root/name,value)
        event=dict(event,status='done',sha=digest(self.root/name),bytes=size);self._append(event)
        return value
