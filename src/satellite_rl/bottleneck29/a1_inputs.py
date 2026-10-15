"""SHA-bound read-only A0 facts for A1. No model/PHY/kernel construction."""
import gzip,hashlib,json,pickle
from pathlib import Path
from .inputs import input_loader
from . import a1_core


def digest(raw):return hashlib.sha256(raw).hexdigest()


class A0Facts:
    def __init__(self,config):
        self.config=config;self.loader=input_loader(config);self.bundle=None
        if 'a0_bundle_manifest' in config:
            self.root=Path(config['a0_bundle_root']).resolve();ref=config['a0_bundle_manifest'];raw=Path(ref['path']).read_bytes()
            if digest(raw)!=ref['sha256']:raise ValueError('A0 cache manifest SHA')
            self.bundle=json.loads(raw)
            if self.bundle['schema']!='W29A1A0CacheManifestV1':raise ValueError('A0 cache manifest schema')
            self.binding=self.bundle['A0_binding']
            if self.binding['source_SHA']!=config['binding']['A0_source_SHA'] or self.bundle['A0_summary_SHA']!=config['binding']['A0_summary_SHA']:raise ValueError('A0 accepted bundle source/closure')
            if config.get('a0_binding',self.binding)!=self.binding:raise ValueError('A0 bundle original binding')
            return
        self.root=Path(config['a0_root']).resolve()
        raw=(self.root/'ledger.jsonl').read_bytes()
        if digest(raw)!=config['a0_ledger_SHA']:raise ValueError('accepted A0 ledger SHA')
        self.operations={};self.artifacts={};sessions={};binding=None
        for line in raw.splitlines(keepends=True):
            if not line.endswith(b'\n'):raise ValueError('A0 torn ledger')
            event=json.loads(line);value=event['payload']
            if digest(json.dumps(value,sort_keys=True,separators=(',',':')).encode())!=event['sha256']:raise ValueError('A0 ledger event SHA')
            if value['type']=='binding':binding=value['value']['binding']
            elif value['type']=='operation':self.operations[value['id']]=value
            elif value['type']=='artifact':self.artifacts[value['id']]=value
            elif value['type']=='session':sessions[value['id']]=value
        if binding!=config['a0_binding'] or binding['source_SHA']!=config['binding']['A0_source_SHA']:raise ValueError('A0 original scientific binding')
        if any(v['status']!='done' for v in self.operations.values()) or any('elapsed' not in v for v in sessions.values()):raise ValueError('unresolved A0 work')
        self.binding=binding
        completed,_=self._artifact('completed')
        if completed['status']!='completed':raise ValueError('A0 not complete')

    def _read(self,event):
        path=(self.root/event['path']).resolve()
        if self.root not in path.parents:raise ValueError('A0 ref outside scope')
        raw=path.read_bytes()
        if digest(raw)!=event['sha']:raise ValueError('A0 immutable result SHA')
        return pickle.loads(raw),dict(path=str(path),record_SHA=event['sha'])

    def _artifact(self,key):
        if key not in self.artifacts:raise ValueError('A0 artifact absent: '+key)
        return self._read(self.artifacts[key])

    def _bundle_value(self,ref):
        path=(self.root/ref['member']).resolve()
        if self.root not in path.parents:raise ValueError('cache member outside bundle')
        raw=path.read_bytes()
        if digest(raw)!=ref['sha256']:raise ValueError('A0 cache member SHA')
        if ref['codec']=='gzip_pickle':raw=gzip.decompress(raw)
        elif ref['codec']!='pickle':raise ValueError('A0 cache codec')
        return pickle.loads(raw)

    def mother(self,mother):
        stage_input,actions=self.loader(mother);key=mother['case_id']+'/kernel'
        if self.bundle is not None:kernel=self._bundle_value(self.bundle['kernels'][mother['case_id']])
        else:
            event=self.operations[key]
            if event['kind']!='kernel':raise ValueError('A0 static kernel provenance')
            kernel,_=self._read(event)
        return a1_core.reuse_context(stage_input,kernel),actions

    def state(self,context,mother,prefix):
        sid=mother['case_id']+'/'+str(prefix)
        package=None
        if self.bundle is not None:
            package=self._bundle_value(self.bundle['states'][sid])
            if package['schema']!='W29A1A0StateCacheV1' or package['state_id']!=sid:raise ValueError('A0 bundled state identity')
            snapshot_ref=package['snapshot_source_ref'];raw=package['snapshot_raw']
            if digest(raw)!=snapshot_ref['record_SHA']:raise ValueError('A0 bundled raw snapshot SHA')
            snapshot=pickle.loads(raw)
        else:snapshot,snapshot_ref=self._artifact(sid+'/snapshot')
        if snapshot['beam_idx']!=prefix:raise ValueError('A0 snapshot cursor')
        state=a1_core.state_from_a0(context,snapshot);snapshot_sha=snapshot_ref['record_SHA']
        for key,value in dict(source_SHA=self.binding['source_SHA'],snapshot_SHA=snapshot_sha,row_id=sid+'/snapshot').items():
            if key in snapshot_ref and snapshot_ref[key]!=value:raise ValueError('A0 snapshot source metadata mismatch')
        source_snapshot=dict(snapshot_ref,source_SHA=self.binding['source_SHA'],snapshot_SHA=snapshot_sha,row_id=sid+'/snapshot')
        loaded={}
        def resolve(key,stack=()):
            if key in stack or not key.startswith(sid+'/'):raise ValueError('A0 cache cycle/cross state')
            if key in loaded:return loaded[key]
            if package is not None:
                source=dict(package['row_sources'][key]);raw=package['rows'][key]
                if digest(raw)!=source['record_SHA']:raise ValueError('A0 bundled raw row SHA')
                row=pickle.loads(raw)
                source['A0_archive_SHA']=self.bundle['A0_archive_SHA']
            else:row,source=self._artifact(key)
            if row['row_id']!=key or row['snapshot_sha']!=snapshot_sha or row['binding']!=self.binding:raise ValueError('A0 row identity')
            result=row['result']
            if result['status']=='cache':
                target=resolve(result['reference'],stack+(key,))
                if tuple(row['spec'][k] for k in ('start','count','power_w'))!=tuple(target['spec'][k] for k in ('start','count','power_w')):raise ValueError('A0 cache point mismatch')
                result=target['result']
            elif result['status'] not in ('completed','invalid'):raise ValueError('A0 unresolved row')
            for field,value in dict(source_SHA=self.binding['source_SHA'],snapshot_SHA=snapshot_sha,row_id=key).items():
                if field in source and source[field]!=value:raise ValueError('A0 row source metadata mismatch')
            source.update(source_SHA=self.binding['source_SHA'],snapshot_SHA=snapshot_sha,row_id=key)
            loaded[key]=dict(spec=row['spec'],result=result,source_ref=source)
            return loaded[key]
        cache={}
        for table,sequences in [('minimum5',range(955)),('fine_power',range(9,100,10))]:
            for sequence in sequences:
                point=resolve(sid+'/'+table+'/'+str(sequence));spec=point['spec']
                if spec['power_w'] not in range(5,51,5):raise ValueError('A1 cannot consume relaxed A0 power point')
                cache.setdefault((spec['start'],spec['count'],float(spec['power_w'])),point)
        return state,source_snapshot,cache
