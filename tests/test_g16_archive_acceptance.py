import hashlib
import json
from pathlib import Path
import tempfile
import unittest
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


if __name__=='__main__': unittest.main()
