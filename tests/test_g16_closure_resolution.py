import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.stage1_geoload.closure_resolution import (SCIENCE,resolve_teacher,resolve_gate,
    require_sha,rows,signature,node_check)


class ClosureResolutionTests(unittest.TestCase):
    def fixture(self):
        owner='teacher_20424';case_id='train:time:base';h=hashlib.sha256((case_id+'\0JOINT').encode()).hexdigest()
        job={'mode':'teacher','kind':'prepare','methods':['JOINT'],'archive_root':'/group/20424/archive','run_id':'job1'}
        root=job['archive_root']+'/outputs/workbench/jobs/job1';case=owner+':job/cases/'+h+'.json';manifest=owner+':job/teacher/'+h+'/manifest.json';target=owner+':job/teacher/'+h+'/sample_00000.npz'
        nodes={case:('c'*64,owner,'job/cases/'+h+'.json',10),manifest:('m'*64,owner,'job/teacher/'+h+'/manifest.json',20),target:('s'*64,owner,'job/teacher/'+h+'/sample_00000.npz',30)}
        anchor={'from':manifest,'location':'$.samples[0]','source_path':root+'/teacher/'+h+'/sample_00000.npz','expected_sha256':'s'*64,'sourceRef':target}
        shard={'from':case,'location':'$.teacher_shard','source_path':root+'/teacher/'+h+'/manifest.json','expected_sha256':'m'*64,'sourceRef':manifest}
        issue={'code':'dependency_missing','from':case,'location':'$.samples[0]','source_path':root+'/cases/'+h+'/sample_00000.npz','expected_sha256':'s'*64,'sourceRef':None}
        samples={(manifest,'$.samples[0]'):anchor,(manifest,anchor['source_path'],'s'*64):anchor}
        indexes={owner:{h:{'case_id':case_id,'method':'JOINT','sha256':'c'*64,'outcome':'completed'}}}
        return issue,{owner:job},nodes,{case:shard},samples,indexes,target

    def test_same_case_index_and_manifest_proof_resolve_only_exact_owner(self):
        issue,jobs,nodes,shards,samples,indexes,target=self.fixture()
        result,proof=resolve_teacher(issue,jobs,nodes,shards,samples,indexes)
        self.assertEqual(result,target);self.assertEqual(proof['original_issue'],issue)
        issue['location']='$.samples';result,proof=resolve_teacher(issue,jobs,nodes,shards,samples,indexes)
        self.assertEqual(result,target);self.assertIn('exact same-manifest',proof['condition'])

    def test_wrong_index_or_shard_sha_cannot_use_global_matching_sample(self):
        data=self.fixture();data[0]['location']='$.samples[1]'
        with self.assertRaisesRegex(ValueError,'manifest sample'):resolve_teacher(*data[:6])
        data=self.fixture();data[3][data[0]['from']]['expected_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'canonical node'):resolve_teacher(*data[:6])
        data=self.fixture();data[0]['source_path']=data[0]['source_path'].replace('/cases/','/other/')
        with self.assertRaisesRegex(ValueError,'outside the same case'):resolve_teacher(*data[:6])

    def test_unsafe_relative_path_and_different_job_mode_rejected(self):
        data=self.fixture();data[0]['source_path']=data[0]['source_path'].replace('sample_00000.npz','../sample_00000.npz')
        with self.assertRaisesRegex(ValueError,'relative path'):resolve_teacher(*data[:6])
        data=self.fixture();next(iter(data[1].values()))['mode']='evaluate'
        with self.assertRaisesRegex(ValueError,'teacher producer'):resolve_teacher(*data[:6])

    def test_gate_original_is_bound_to_same_consumer_copy(self):
        owner='teacher_merge';root='/group/20473/archive/outputs/workbench/jobs/merge';original='/group/20473/ops/development_gate.json';copied=root+'/input_sources/gate/development_gate.json';target=owner+':job/input_sources/gate/development_gate.json'
        gate={'source_path':original,'path':copied,'sha256':'a'*64};job={'archive_root':'/group/20473/archive','run_id':'merge','references':{'gate':gate}}
        issue={'code':'dependency_missing','from':owner+':job/job.json','location':'$.references.gate.source_path','source_path':original,'expected_sha256':'a'*64,'sourceRef':None}
        node=('a'*64,owner,'job/input_sources/gate/development_gate.json',12);anchor={'sourceRef':target,'source_path':copied,'expected_sha256':'a'*64}
        resolved,proof=resolve_gate(issue,{owner:job},{target:node},{owner:anchor});self.assertEqual(resolved,target)
        self.assertEqual(proof['original_issue'],issue)
        changed=copy.deepcopy(issue);changed['source_path']='/unrelated/development_gate.json'
        with self.assertRaisesRegex(ValueError,'provenance'):resolve_gate(changed,{owner:job},{target:node},{owner:anchor})

    def test_cache_sha_and_node_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'cache.json';p.write_bytes(b'{"nodes":[]}');expected=hashlib.sha256(p.read_bytes()).hexdigest()
            require_sha(p,expected);p.write_bytes(b'{"nodes":[{}]}')
            with self.assertRaisesRegex(ValueError,'immutable SHA'):require_sha(p,expected)
        with self.assertRaisesRegex(ValueError,'canonical node'):node_check({'node':('a'*64,'owner','member',1)},'node','b'*64)

    def test_cache_arrays_are_streamed_without_whole_json_load(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'cache.json';p.write_text(json.dumps({'issues':[{'code':'dependency_missing'} for _ in range(1000)]}),encoding='utf-8')
            generator=rows(p,'issues');self.assertEqual(next(generator),{'code':'dependency_missing'});generator.close()
        self.assertEqual(signature({'from':'a','location':'b','source_path':'c','expected_sha256':'d'}),('a','b','c','d'))


if __name__=='__main__':unittest.main()
