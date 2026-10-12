import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile
from scripts.stage1_geoload.archive_acceptance import audit, pinned_refs, safe_member


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


if __name__=='__main__': unittest.main()
