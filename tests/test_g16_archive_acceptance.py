import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile
from scripts.stage1_geoload.archive_acceptance import (audit,pinned_refs,safe_member,stream_pinned_refs,
    HashingReader,JSON_LIMIT,absolute_reference,parser_metadata)


class ArchiveAcceptanceTests(unittest.TestCase):
    def fixture(self, root, mutate=None):
        commit='a'*40; run='prepare_g16'; base='/remote/archive'; jobroot=base+'/outputs/workbench/jobs/'+run
        job={'run_id':run,'experiment_id':'S1-G16','scientific_commit':commit,
             'archive_root':base,'model_dir':'/remote/model','source_root':'/remote/source',
             'references':{'plan':{'path':jobroot+'/plan.json','sha256':hashlib.sha256(b'{}').hexdigest()}}}
        members={'job/job.json':json.dumps(job).encode(),'job/status.json':b'{"stage":"completed"}', 'job/plan.json':b'{}'}
        manifest={'run_id':run,'scientific_commit':commit,'members':{name:{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)} for name,data in members.items()}}
        if mutate: mutate(members,manifest)
        path=root/'job.zip'
        with zipfile.ZipFile(path,'w') as z:
            for name,data in members.items(): z.writestr(name,data)
            z.writestr('archive_manifest.json',json.dumps(manifest))
        state={'group':'S1-G16','all_computation_completed':True,'compute_and_archives_complete':True,
               'tasks':{'prepare':{'run_id':run,'stage':'completed'}},
               'archives':{'prepare':{'local_path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}}
        return {'group':'S1-G16','scientific_commit':commit},state

    def test_verified_zip_still_requires_source_and_upstream(self):
        with tempfile.TemporaryDirectory() as temp:
            plan,state=self.fixture(Path(temp)); result=audit(plan,state)
            self.assertEqual(result['status'],'not_ready')
            codes={x['code'] for x in result['issues']}
            self.assertTrue({'source_snapshot_missing','scientific_git_bundle_missing','upstream_raw_closure_not_verified'}<=codes)
            self.assertTrue(any(e['sourceRef']=='prepare:job/plan.json' for e in result['edges']))
            self.assertEqual(result['model_validation']['status'],'not_ready')

    def test_corrupt_member_rejected_even_with_matching_zip_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            plan,state=self.fixture(Path(temp),lambda members,manifest:members.update({'job/plan.json':b'{"wrong":true}'}))
            self.assertTrue(any(x['code']=='archive_invalid' for x in audit(plan,state)['issues']))

    def test_inventory_omission_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            plan,state=self.fixture(Path(temp),lambda members,manifest:members.update({'job/unlisted':b'x'}))
            self.assertTrue(any(x['code']=='archive_invalid' for x in audit(plan,state)['issues']))

    def test_incomplete_state_cannot_accept(self):
        result=audit({'group':'S1-G16'},{'group':'S1-G16','tasks':{}})
        self.assertEqual(result['status'],'not_ready')
        self.assertIn('computation_or_export_incomplete',{x['code'] for x in result['issues']})

    def test_paths_and_derived_archive_pin(self):
        self.assertFalse(safe_member('../escape')); self.assertFalse(safe_member('C:\\escape'))
        self.assertEqual(list(pinned_refs({'archive_path':'/raw/a.tar','archive_sha256':'a'*64}))[0][1],'/raw/a.tar')

    def test_sha_match_does_not_invent_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            plan,state=self.fixture(Path(temp)); path=Path(temp)/'external'; path.write_bytes(b'{}')
            result=audit(plan,state,{'resources':[{'id':'other','path':str(path),'sha256':hashlib.sha256(b'{}').hexdigest(),'source_paths':['/unrelated'] }],
                'requirements':[{'source_path':'/missing','sha256':hashlib.sha256(b'{}').hexdigest()}]})
            self.assertTrue(any(x['code']=='dependency_missing' and x['source_path']=='/missing' for x in result['issues']))

    def test_complete_external_source_closure_accepts_without_model_load_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);plan,state=self.fixture(root);resources=[]
            payloads={'code.py':b'actual scientific source','bundle':b'synthetic bundle',
                      'source_manifest.json':json.dumps({'git_commit':'a'*40,'files':{'code.py':hashlib.sha256(b'actual scientific source').hexdigest()}}).encode()}
            for name,data in payloads.items():
                p=root/name;p.write_bytes(data)
                resources.append({'id':name,'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),
                    'kind':'git_bundle' if name=='bundle' else 'file','commit':'a'*40,
                    'source_paths':['/remote/source/'+name]})
            pin={'path':'/remote/source/code.py','sha256':hashlib.sha256(payloads['code.py']).hexdigest()}
            upstream=[{'role':role,**{key:[pin] for key in ('raw_archives','parser','registry','inputs')}} for role in ('train','development')]
            with patch('scripts.stage1_geoload.archive_acceptance.subprocess.run',return_value=SimpleNamespace(stdout='a'*40+' HEAD\n')):
                result=audit(plan,state,{'resources':resources,'upstream_sources':upstream})
            self.assertEqual(result['issues'],[])
            self.assertEqual(result['status'],'accepted')
            self.assertEqual(result['model_validation']['status'],'not_ready')

    def test_external_zip_inventory_and_every_member_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);plan,state=self.fixture(root);p=root/'shared.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('raw/a.tar',b'actual data')
            registry={'resources':[{'id':'shared','kind':'zip','path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                         'members':{'raw/a.tar':{'sha256':'0'*64,'bytes':11}}}]}
            result=audit(plan,state,registry)
            self.assertTrue(any(i['code']=='external_invalid' for i in result['issues']))

    def test_stream_matches_reference_with_arbitrary_key_order_nested_arrays_unicode(self):
        obj={'nested':[0,False,None,['numeric',3],{'sha256':'a'*64,'path':'/one','source_path':'/original'}],
             'unicode':{'artifact_sha256':'b'*64,'artifact':'/位置/é','ignored':[1,2,3]},'other':{'archive_sha256':'c'*64,'archive_path':'/raw/a.tar'}}
        raw=json.dumps(obj,ensure_ascii=False).encode()
        self.assertCountEqual(list(stream_pinned_refs(io.BytesIO(raw))),list(pinned_refs(obj)))
        self.assertEqual(parser_metadata()['version'],'3.3.0')

    def test_hash_and_pin_parser_consume_same_stream_including_bom(self):
        raw=b'\xef\xbb\xbf'+json.dumps({'sha256':'a'*64,'path':'/pin','matrix':[[1,2],[3,4]]}).encode()
        reader=HashingReader(io.BytesIO(raw));pins=list(stream_pinned_refs(reader))
        self.assertEqual(pins,[('$','/pin','a'*64)])
        self.assertEqual(reader.count,len(raw));self.assertEqual(reader.hash.hexdigest(),hashlib.sha256(raw).hexdigest())
        with self.assertRaisesRegex(ValueError,'unbounded'):reader.read()

    def test_teacher_relative_path_and_source_manifest_safety(self):
        obj={'samples':[{'sha256':'a'*64,'path':'shards/one.npz'}]}
        pins=list(stream_pinned_refs(io.BytesIO(json.dumps(obj).encode()),'/job/teacher/manifest.json'))
        self.assertIn(('$.samples','/job/teacher/shards/one.npz','a'*64),pins)
        obj['samples'][0]['path']='../escape.npz'
        with self.assertRaisesRegex(ValueError,'unsafe_teacher_sample'):list(stream_pinned_refs(io.BytesIO(json.dumps(obj).encode()),'/job/teacher/manifest.json'))
        summary={};source={'git_commit':'a'*40,'files':{'src/a.py':'b'*64}}
        refs=list(stream_pinned_refs(io.BytesIO(json.dumps(source).encode()),'/source/source_manifest.json',summary))
        self.assertIn(('$.files.src/a.py','/source/src/a.py','b'*64),refs);self.assertEqual(summary['source_files_count'],1)

    def test_duplicate_keys_and_truncated_json_rejected(self):
        with self.assertRaisesRegex(ValueError,'duplicate JSON key'):list(stream_pinned_refs(io.BytesIO(b'{"path":"/first","path":"/second","sha256":"x"}')))
        with self.assertRaisesRegex(ValueError,'invalid streaming JSON'):list(stream_pinned_refs(io.BytesIO(b'{"path":')))

    def test_large_json_pins_past_old_limit_without_unbounded_read(self):
        class GeneratedStream:
            def __init__(self):
                self.head=b'{"numbers":[';self.remaining=JSON_LIMIT+1024;self.tail=b'0],"ref":{"sha256":"'+b'a'*64+b'","path":"/tail/source"}}';self.max_read=0
            def read(self,size):
                if size<0:raise AssertionError('full read forbidden')
                self.max_read=max(self.max_read,size);result=b''
                if self.head:
                    result=self.head[:size];self.head=self.head[len(result):];size-=len(result)
                if size and self.remaining:
                    count=min(size,self.remaining);count-=count%2
                    result+=b'0,'*(count//2);self.remaining-=count;size-=count
                if size and not self.remaining:result+=self.tail[:size];self.tail=self.tail[size:]
                return result
        source=GeneratedStream();reader=HashingReader(source)
        pins=list(stream_pinned_refs(reader))
        self.assertEqual(pins,[('$.ref','/tail/source','a'*64)])
        self.assertGreater(reader.count,JSON_LIMIT);self.assertLessEqual(source.max_read,64*1024)

    def test_urls_and_windows_are_exact_absolute_sources(self):
        self.assertTrue(absolute_reference('https://api.github.com/releases/tag'))
        self.assertTrue(absolute_reference('D:\\paper\\source.json'))
        self.assertFalse(absolute_reference('shards/sample.npz'))

    def test_case_member_never_uses_zip_read_and_missing_tail_pin_is_audited(self):
        with tempfile.TemporaryDirectory() as temp:
            def extra(members,manifest):
                data=json.dumps({'journal':[{'candidate':[1,2,3]}]*200,'ref':{'path':'/missing/tail','sha256':'d'*64}}).encode()
                members['job/cases/example.json']=data
                manifest['members']['job/cases/example.json']={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            plan,state=self.fixture(Path(temp),extra);original=zipfile.ZipFile.read
            def bounded(z,name,*args,**kwargs):
                if '/cases/' in name:raise AssertionError('case must be streamed')
                return original(z,name,*args,**kwargs)
            with patch.object(zipfile.ZipFile,'read',bounded):result=audit(plan,state)
            self.assertTrue(any(i['code']=='dependency_missing' and i['source_path']=='/missing/tail' for i in result['issues']))
            self.assertTrue(any(n['member']=='job/cases/example.json' for n in result['nodes'] if 'member' in n))

    def test_invalid_case_json_still_gets_complete_member_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            def extra(members,manifest):
                data=b'{"bad":'+b'x'*200000
                members['job/cases/invalid.json']=data
                manifest['members']['job/cases/invalid.json']={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            plan,state=self.fixture(Path(temp),extra);result=audit(plan,state)
            self.assertTrue(any(i['code']=='invalid_json' for i in result['issues']))
            node=next(n for n in result['nodes'] if n.get('member')=='job/cases/invalid.json')
            self.assertEqual(node['bytes'],200007)

    def test_progress_is_compact_and_default_interface_stays_compatible(self):
        with tempfile.TemporaryDirectory() as temp:
            plan,state=self.fixture(Path(temp));events=[]
            result=audit(plan,state,progress=events.append)
            self.assertEqual(events[0]['phase'],'starting');self.assertEqual(events[-1]['phase'],'complete')
            self.assertGreater(events[-1]['bytes_read'],0);self.assertEqual(events[-1]['json_documents_scanned'],3)
            self.assertTrue(all(set(e)<=set(events[-1]) for e in events))
            self.assertEqual(result['status'],'not_ready')

    def test_exact_url_alias_resolves_and_tampered_source_remains_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            url='https://api.github.com/repos/example/releases/tag'
            def extra(members,manifest):
                data=json.dumps({'release_api_sha256':hashlib.sha256(b'{}').hexdigest(),'release_api':url}).encode()
                members['job/preflight.json']=data
                manifest['members']['job/preflight.json']={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            root=Path(temp);plan,state=self.fixture(root,extra);p=root/'release.json';p.write_bytes(b'{}')
            registry={'resources':[{'id':'release','path':str(p),'sha256':hashlib.sha256(b'{}').hexdigest(),'source_paths':[url]}]}
            result=audit(plan,state,registry)
            self.assertTrue(any(e['source_path']==url and e['sourceRef']=='release' for e in result['edges']))
            registry['resources'][0]['source_paths']=[url+'/wrong']
            result=audit(plan,state,registry)
            self.assertTrue(any(i['code']=='dependency_missing' and i['source_path']==url for i in result['issues']))


if __name__=='__main__': unittest.main()
