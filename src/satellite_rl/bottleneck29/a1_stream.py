"""A1 single-writer framed append store: durable intent, value, completion.

A0 files are never written. Pickle is only read from this SHA-bound local store.
"""
import hashlib
import base64
import json
import os
import pickle
import struct
import time
import zlib
from pathlib import Path
from .ledger import Halt

HEADER=struct.Struct('>Q32s')
MAX_RECORD_BYTES=2*1024**2
MAX_CHUNK_ROWS=256


def _frame(raw):
    if len(raw)>MAX_RECORD_BYTES:raise Halt('record exceeds frozen bound')
    return HEADER.pack(len(raw),hashlib.sha256(raw).digest())+raw


def _frames(stream):
    while True:
        offset=stream.tell();head=stream.read(HEADER.size)
        if not head:return
        if len(head)!=HEADER.size:raise Halt('partial frame header; preserve for root audit')
        size,sha=HEADER.unpack(head)
        if size>MAX_RECORD_BYTES:raise Halt('frame bound')
        raw=stream.read(size)
        if len(raw)!=size or hashlib.sha256(raw).digest()!=sha:raise Halt('partial or corrupt frame; preserve for root audit')
        yield offset,HEADER.size+size,sha.hex(),raw


def _syncdir(path):
    if os.name!='nt':
        fd=os.open(path,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)


class StreamStore:
    def __init__(self,root,binding,limits,*,storage_limit=512*1024**2,
                 stop_paths=(),minimum_free_bytes=2*1024**3,active_started=None,read_only=False):
        self.started=time.monotonic() if active_started is None else active_started
        self.root=Path(root);self.binding=binding;self.limits=limits;self.storage_limit=storage_limit
        self.stop_paths=tuple(Path(p) for p in stop_paths);self.minimum_free_bytes=minimum_free_bytes
        self.read_only=read_only;self.operations={};self.chunks={};self.sessions={};self.counts={}
        self.bytes=0;self.sequence=0;self.session=None;self.closed_states=set();self.metadata={};self.debits={}
        self.journal_path=self.root/'a1.journal';self.identity_path=self.root/'a1.identity'
        expected=dict(schema='W29A1AppendStoreV1',binding=binding,limits=limits,storage_limit=storage_limit)
        if not read_only:self.root.mkdir(parents=True,exist_ok=True)
        if not self.journal_path.exists():
            if read_only or self.identity_path.exists() or (self.root/'chunks').exists():raise Halt('missing journal cannot reset existing task')
            with self.identity_path.open('xb') as stream:
                stream.write(json.dumps(expected,sort_keys=True,separators=(',',':')).encode());os.fsync(stream.fileno())
            self.journal_path.touch(exist_ok=False);_syncdir(self.root)
        if not self.identity_path.exists() or json.loads(self.identity_path.read_bytes())!=expected:raise Halt('store identity/limits mismatch')
        self.bytes=self.identity_path.stat().st_size+self.journal_path.stat().st_size
        with self.journal_path.open('rb') as stream:
            for _,_,_,raw in _frames(stream):self._apply(json.loads(raw))
            self.journal_offset=stream.tell()
        self.bytes+=sum(chunk['bytes'] for chunk in self.chunks.values())
        self.journal=None if read_only else self.journal_path.open('ab',buffering=0)
        self.open_files={}
        if not read_only:(self.root/'chunks').mkdir(exist_ok=True)

    def _apply(self,event):
        self.sequence+=1;kind=event['type']
        if kind=='intent':
            if event['id'] in self.operations:raise Halt('duplicate intent')
            self.operations[event['id']]=event
            for key,value in event['charges'].items():self.counts[key]=self.counts.get(key,0)+value
        elif kind=='done':
            op=self.operations.get(event['id'])
            if not op or op.get('done'):raise Halt('completion without unique intent')
            op['done']=event['ref']
            chunk=self.chunks.setdefault(event['ref']['chunk'],dict(rows=0,bytes=0,sealed=False))
            chunk['rows']+=1;chunk['bytes']+=event['ref']['length']
            if chunk['rows']>MAX_CHUNK_ROWS:raise Halt('chunk row count bound')
        elif kind=='seal':
            chunk=self.chunks.get(event['chunk'])
            if chunk is None or chunk['sealed']:raise Halt('invalid seal')
            chunk.update(sealed=True,sha=event['sha'])
        elif kind=='alias':
            if event['id'] in self.operations:raise Halt('duplicate alias')
            self.operations[event['id']]=dict(alias=event['source'],charges={})
        elif kind=='session':self.sessions[event['id']]=event
        elif kind=='state_closed':self.closed_states.add(event['state'])
        elif kind=='metadata':self.metadata[event['key']]=event['value']
        elif kind=='debit':self.debits[event['id']]=event
        else:raise Halt('unknown journal event')

    def _event(self,event):
        if self.read_only:raise Halt('read-only store')
        raw=_frame(json.dumps(event,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
        self._room(len(raw))
        if self.journal.write(raw)!=len(raw):raise Halt('short journal write')
        os.fsync(self.journal.fileno());self.bytes+=len(raw);self._apply(event)

    def _room(self,extra):
        if self.bytes+extra>self.storage_limit:raise Halt('registered storage limit')

    def _chunk_path(self,name):
        return self.root/'chunks'/(name+('.sealed' if self.chunks.get(name,{}).get('sealed') else '.open'))

    def _inventory(self):
        # Once per actual writer start; fixed one-level registered chunk directory.
        known={self._chunk_path(name).name:name for name in self.chunks}
        with os.scandir(self.root/'chunks') as entries:
            for entry in entries:
                if entry.name not in known or not entry.is_file(follow_symlinks=False):raise Halt('orphan chunk requires root audit')
        total=0
        for name,info in self.chunks.items():
            path=self._chunk_path(name)
            if path.stat().st_size!=info['bytes']:raise Halt('partial/unindexed chunk; preserve unknown result')
            if info['sealed'] and hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha']:raise Halt('sealed chunk checksum mismatch')
            with path.open('rb') as stream:
                rows=0
                for _,_,_,_ in _frames(stream):rows+=1
            if rows!=info['rows']:raise Halt('chunk row index mismatch')
            total+=info['bytes']

    def begin(self):
        if self.read_only:raise Halt('read-only store')
        self._inventory()
        if self.unknown():raise Halt('unknown charged operation requires root repair, no automatic replay')
        prior=sum(s.get('elapsed',s['reserved']) for s in self.sessions.values())+sum(d['seconds'] for d in self.debits.values())
        self.remaining=self.limits['seconds']-prior
        if self.remaining<=0:raise Halt('active time exhausted or uncertain session conservatively charged')
        self.session=len(self.sessions)
        self._event(dict(type='session',id=self.session,reserved=self.remaining,started_unix=time.time()))
        self.boundary()

    def boundary(self):
        if self.session is None or time.monotonic()-self.started>=self.remaining:raise Halt('active allowance exhausted')
        if any(path.exists() for path in self.stop_paths):raise Halt('pause/stop; no new query')
        stat=os.statvfs(self.root) if hasattr(os,'statvfs') else None
        if stat is not None and stat.f_bavail*stat.f_frsize<self.minimum_free_bytes:raise Halt('free recovery margin')

    def unknown(self):
        return [key for key,op in self.operations.items() if 'alias' not in op and 'done' not in op]

    def _next_chunk(self,state):
        key=hashlib.sha256(state.encode()).hexdigest()[:24]
        number=0
        while True:
            name=key+'.'+str(number).zfill(5);info=self.chunks.get(name)
            if info is None or not info['sealed'] and info['rows']<MAX_CHUNK_ROWS:return name
            number+=1

    def get(self,op_id):
        op=self.operations.get(op_id)
        if op is None:return None
        if 'alias' in op:return dict(status='alias',source=op['alias'])
        if 'done' not in op:raise Halt('unknown charged operation '+op_id)
        ref=op['done'];path=self._chunk_path(ref['chunk'])
        with path.open('rb') as stream:
            stream.seek(ref['offset']);head=stream.read(HEADER.size)
            if len(head)!=HEADER.size:raise Halt('result header missing')
            size,sha=HEADER.unpack(head)
            if size>MAX_RECORD_BYTES:raise Halt('result record bound')
            raw=stream.read(size)
        if HEADER.size+size!=ref['length'] or sha.hex()!=ref['sha'] or hashlib.sha256(raw).digest()!=sha:raise Halt('result digest mismatch')
        record=pickle.loads(zlib.decompress(raw))
        if record['id']!=op_id:raise Halt('record identity mismatch')
        return record['value']

    def transact(self,op_id,charges,operation,*,state):
        self.boundary()
        if op_id in self.operations:
            if self.operations[op_id]['charges']!=charges:raise Halt('cost identity changed')
            return self.get(op_id)
        if state in self.closed_states:raise Halt('state stream already immutable/closed')
        for key,value in charges.items():
            if type(value)is not int or value<0 or key not in self.limits or self.counts.get(key,0)+value>self.limits[key]:raise Halt('budget/charge violation '+key)
        name=self._next_chunk(state)
        # Reserve worst frame plus bounded intent/done metadata before the call.
        self._room(MAX_RECORD_BYTES+16384)
        self._event(dict(type='intent',id=op_id,charges=charges,state=state,chunk=name))
        value=operation()
        raw=zlib.compress(pickle.dumps(dict(id=op_id,value=value),protocol=5),level=1)
        framed=_frame(raw);self._room(len(framed)+8192)
        stream=self.open_files.get(name)
        if stream is None:
            path=self.root/'chunks'/(name+'.open');existed=path.exists()
            stream=path.open('ab',buffering=0);self.open_files[name]=stream
            if not existed:_syncdir(path.parent)
        offset=stream.tell()
        if stream.write(framed)!=len(framed):raise Halt('short result write')
        os.fsync(stream.fileno());self.bytes+=len(framed)
        ref=dict(chunk=name,offset=offset,length=len(framed),sha=hashlib.sha256(raw).hexdigest())
        self._event(dict(type='done',id=op_id,ref=ref))
        if self.chunks[name]['rows']==MAX_CHUNK_ROWS:self._seal(name)
        return value

    def alias(self,op_id,source,*,state):
        self.boundary()
        if op_id in self.operations:
            if self.operations[op_id].get('alias')!=source:raise Halt('alias source changed')
            return
        if state in self.closed_states:raise Halt('closed state')
        if not isinstance(source,dict) or not {'source_SHA','snapshot_SHA','row_id','record_SHA'}<=source.keys():raise Halt('alias requires exact immutable A0 provenance')
        self._event(dict(type='alias',id=op_id,state=state,source=source))

    def _seal(self,name):
        stream=self.open_files.pop(name,None)
        if stream is not None:stream.close()
        path=self.root/'chunks'/(name+'.open')
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        # journal seal precedes rename: interrupted rename is fail-closed on restart.
        self._event(dict(type='seal',chunk=name,sha=sha))
        os.replace(path,self.root/'chunks'/(name+'.sealed'));_syncdir(path.parent)

    def close_state(self,state,expected_ids=None):
        self.boundary()
        completion={}
        if expected_ids is not None:
            if len(set(expected_ids))!=len(expected_ids):raise Halt('duplicate expected state row ID')
            bitmap=bytearray((len(expected_ids)+7)//8)
            for index,op_id in enumerate(expected_ids):
                op=self.operations.get(op_id)
                if op is None or 'done' not in op and 'alias' not in op:raise Halt('state missing expected row')
                bitmap[index//8]|=1<<(index%8)
            completion=dict(expected_rows=len(expected_ids),completed_bitmap_base64=base64.b64encode(bitmap).decode(),
                ordered_ids_SHA=hashlib.sha256(json.dumps(expected_ids,separators=(',',':')).encode()).hexdigest())
        prefix=hashlib.sha256(state.encode()).hexdigest()[:24]+'.'
        for name,info in list(self.chunks.items()):
            if name.startswith(prefix) and not info['sealed']:self._seal(name)
        self._event(dict(type='state_closed',state=state,**completion))

    def iter_values(self):
        """Lossless sequential reader, one open per chunk rather than per query."""
        for name in sorted(self.chunks):
            with self._chunk_path(name).open('rb') as stream:
                for offset,length,sha,raw in _frames(stream):
                    record=pickle.loads(zlib.decompress(raw));op_id=record['id']
                    op=self.operations.get(op_id)
                    if not op or op.get('done')!=dict(chunk=name,offset=offset,length=length,sha=sha):raise Halt('stream/index mismatch')
                    yield op_id,record['value']

    def record_metadata(self,key,value):
        self._event(dict(type='metadata',key=key,value=value))

    def debit_time(self,debit_id,seconds):
        import math
        if not math.isfinite(seconds) or seconds<0:raise Halt('invalid audited time debit')
        if debit_id in self.debits:
            if self.debits[debit_id]['seconds']!=seconds:raise Halt('time debit changed')
            return
        self._event(dict(type='debit',id=debit_id,seconds=seconds))

    def finish(self):
        if self.session is not None:
            event=dict(self.sessions[self.session]);event['elapsed']=time.monotonic()-self.started
            self._event(event);self.session=None
        for stream in self.open_files.values():stream.close()
        self.open_files.clear()
        if self.journal is not None:self.journal.close()

    def summary(self):
        return dict(counts=dict(self.counts),unknown=self.unknown(),stored_bytes=self.bytes,
            completed_operations=sum('done' in op for op in self.operations.values()),
            aliases=sum('alias' in op for op in self.operations.values()),closed_states=sorted(self.closed_states),metadata=self.metadata)

    def refresh(self):
        """Incremental status tail: no directory scans, no writes, partial tail waits."""
        if not self.read_only:raise Halt('refresh belongs to read-only status')
        with self.journal_path.open('rb') as stream:
            stream.seek(self.journal_offset)
            while True:
                start=stream.tell();head=stream.read(HEADER.size)
                if not head:return
                if len(head)!=HEADER.size:return
                size,sha=HEADER.unpack(head)
                if size>MAX_RECORD_BYTES:raise Halt('status frame bound')
                raw=stream.read(size)
                if len(raw)!=size:return
                if hashlib.sha256(raw).digest()!=sha:raise Halt('status checksum mismatch')
                event=json.loads(raw);self._apply(event)
                self.bytes+=HEADER.size+size
                if event['type']=='done':self.bytes+=event['ref']['length']
                self.journal_offset=stream.tell()
