"""Meaningful contract/ownership/recovery checks; no formal/local experiment."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from satellite_rl.coverage.ai_search.config import GeometryConfig, SearchConfig
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench import coverage_ai_search as service
from satellite_rl.workbench.ai_search_runner import load_cases, recover_rows, result_key,random_seed,train_job,cached_orderer
from satellite_rl.workbench.storage import Store, digest


class AiSearchWorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.store=Store(self.root/'20418/archive',source_root=self.root)
        self.source=self.store.root/'source_snapshots'/('a'*40);self.source.mkdir()
        atomic_json(self.source/'source_manifest.json',{'git_commit':'a'*40,'files':{'x.py':'not_used_for_science'}})
        self.store.freeze_source=lambda:(self.source,'a'*40)
        self.dataset=self.root/'inputs.npz'
        np.savez_compressed(self.dataset,positions=np.array([[40.,20.],[41.,21.]]),demands=np.array([0.,4.]),
            aircraft_ids=np.array(['zero','positive']),offsets=np.array([0,2]),case_ids=np.array(['case1']),
            mother_ids=np.array(['mother1']),demand_profiles=np.array(['base']),metadata_json=np.array('{"teacher_answer":123}'))
        self.ref={'path':str(self.dataset),'sha256':digest(self.dataset),'role':'development','count':1}
        self.config={'schema_version':9,'research_stage':'coverage','algorithm':'ai_search','name':'bounded check',
            'geometry':asdict(GeometryConfig()),'search':asdict(SearchConfig()),'datasets':{'development':self.ref},
            'workers':1,'resources':{'min_free_disk_bytes':1,'max_rss_bytes':1<<30}}
        # JSON round-tripping mirrors the actual wire contract (tuple -> list).
        self.config=json.loads(json.dumps(self.config))
        self.contract={'config':self.config,'mode':'development','dataset_role':'development',
            'case_indices':[0],'methods':['RULE'],'executor_port':20418}
        self.plan=self.root/'plan.json'
        atomic_json(self.plan,{'group':service.GROUP,'authorized':True,'scientific_commit':'a'*40,'tasks':{'dev':self.contract}})
        self.body={**self.contract,'request_id':'one-scientific-task','task_key':'dev',
                   'plan':{'path':str(self.plan),'sha256':digest(self.plan)}}
        self.app=SimpleNamespace(store=self.store)
        self.dep=patch.object(service,'deployment',return_value='20418');self.dep.start();self.addCleanup(self.dep.stop)

    def test_case_supply_keeps_zero_requests_and_excludes_answer_metadata(self):
        rows=load_cases(self.ref,[0]);self.assertEqual(rows[0]['aircraft_ids'],('zero','positive'))
        self.assertEqual(rows[0]['demands'].tolist(),[0.,4.]);self.assertNotIn('metadata_json',rows[0])
        self.assertNotIn('teacher_answer',rows[0])

    def test_random_seed_explicitly_contains_51001_and_case_identity(self):
        import hashlib
        seed,record=random_seed('x')
        self.assertEqual(record['basis_utf8_json'],'[51001,"x","random"]')
        self.assertEqual(seed,int(hashlib.sha256(b'[51001,"x","random"]').hexdigest()[:16],16))
        self.assertEqual(random_seed('x')[0],seed);self.assertNotEqual(random_seed('y')[0],seed)

    def test_async_node_callback_pins_checkpoint_without_running_search(self):
        model_dir=self.root/'model';model_dir.mkdir();(model_dir/'checkpoint.pt').write_bytes(b'complete_state_fixture')
        cfg=deepcopy(self.config);cfg['datasets']={'train':{**self.ref,'role':'train'},'validation':{**self.ref,'role':'validation'}}
        job={'model_dir':str(model_dir),'run_id':'one_model','task_key':'train','scientific_commit':'a'*40,
            'training_spec':{'validation_dispatch':'workbench_async','validation_case_indices':[0],'protocol':{'updates':10000}},
            'references':{'teacher_manifest':{'path':'not_opened_fixture'},'plan':{'path':str(self.plan),'sha256':digest(self.plan)}}}
        observed=[]
        def train_stub(manifest,output,protocol,**kwargs):
            result=kwargs['validation'](None,None,2500,model_dir/'unused_validation')
            self.assertEqual(result['status'],'requested');observed.append(result)
            return {'status':'stopped','update':2500}
        with patch('satellite_rl.coverage.ai_search.learning.train',side_effect=train_stub),patch('satellite_rl.coverage.ai_search.search.run_search') as search:
            train_job(self.root,job,cfg,lambda **kw:None,False);search.assert_not_called()
        ready=json.loads(Path(observed[0]['path']).read_text())
        self.assertEqual(ready['update_step'],2500);self.assertEqual(ready['status'],'requested')
        self.assertEqual(Path(ready['model']['path']).read_bytes(),b'complete_state_fixture')
        self.assertFalse(ready['additional_AI_10000_diagnostic_allowed'])

    def test_model_cache_counts_only_first_cold_load_and_separates_diagnostic_admission(self):
        import satellite_rl.workbench.ai_search_runner as runner
        runner._MODEL_CACHE.clear();ref={'path':'immutable-model','sha256':'b'*64}
        callback=lambda *args:None
        with patch('satellite_rl.coverage.ai_search.learning.load_orderer',return_value=(callback,2.5)) as loader:
            self.assertEqual(cached_orderer(ref,self.config)[1],2.5)
            self.assertEqual(cached_orderer(ref,self.config)[1],0.)
            self.assertEqual(loader.call_count,1)
            cached_orderer(ref,self.config,diagnostic=True)
            self.assertEqual(loader.call_count,2);self.assertTrue(loader.call_args.kwargs['allow_intermediate'])
        runner._MODEL_CACHE.clear()

    def test_exact_scientific_budget_is_bound_to_plan(self):
        changed=deepcopy(self.body);changed['config']['search']['seconds']+=1
        with self.assertRaisesRegex(ValueError,'frozen round contract'):service.create_job(self.app,changed)
        self.assertEqual(self.store.jobs(),[])

    def test_unknown_submission_retry_reuses_original_owner_and_run(self):
        first=service.create_job(self.app,self.body);second=service.create_job(self.app,deepcopy(self.body))
        self.assertEqual(first['run_id'],second['run_id']);self.assertEqual(len(self.store.jobs()),1)
        service.verify_job(self.store,first)

    def test_new_request_cannot_duplicate_same_frozen_scientific_task(self):
        service.create_job(self.app,self.body);another=deepcopy(self.body);another['request_id']='duplicate-new-request'
        with self.assertRaisesRegex(ValueError,'another owner/configuration'):service.create_job(self.app,another)
        self.assertEqual(len(self.store.jobs()),1)

    def test_same_request_cannot_move_across_lanes(self):
        service.create_job(self.app,self.body)
        another=Store(self.root/'20424/archive',source_root=self.root)
        another.freeze_source=self.store.freeze_source
        with self.assertRaisesRegex(ValueError,'different physical workbench|frozen round contract'):
            with patch.object(service,'deployment',return_value='20424'):
                service.create_job(SimpleNamespace(store=another),self.body)
        self.assertEqual(another.jobs(),[])

    def test_job_semantics_tampering_is_rejected_before_resume(self):
        job=service.create_job(self.app,self.body);job['methods']=['RANDOM']
        atomic_json(self.store.job_dir(job['run_id'])/'job.json',job)
        with self.assertRaisesRegex(ValueError,'original unique owner'):service.verify_job(self.store,job)

    def test_source_path_tampering_rejected_even_with_rebound_job_hash(self):
        job=service.create_job(self.app,self.body);job['source_root']=str(self.root)
        path=self.store.job_dir(job['run_id'])/'job.json';atomic_json(path,job)
        owner=json.loads(Path(job['owner_path']).read_text());owner['job_sha256']=digest(path)
        atomic_json(Path(job['owner_path']),owner)
        with self.assertRaisesRegex(ValueError,'source path/manifest'):service.verify_job(self.store,job)

    def test_pinned_input_change_is_rejected(self):
        job=service.create_job(self.app,self.body)
        Path(job['references']['development']['path']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'source SHA differs'):service.verify_job(self.store,job)

    def test_atomic_case_tail_recovers_but_unknown_spent_budget_does_not_repeat(self):
        directory=self.root/'worker';(directory/'cases').mkdir(parents=True);(directory/'started').mkdir()
        key=result_key('case1','RULE');atomic_json(directory/'cases'/(key+'.json'),{'case_id':'case1','method':'RULE','outcome':'completed'})
        index=recover_rows(directory,{key});self.assertTrue(index[key]['recovered_atomic_tail'])
        unknown=result_key('case2','RULE');atomic_json(directory/'started'/(unknown+'.json'),{'case_id':'case2'})
        with self.assertRaisesRegex(ValueError,'unknown spent budget'):recover_rows(directory,{key,unknown})
        self.assertTrue((directory/'unknown_budget.json').is_file())

    def test_independent_test_is_not_implicitly_development(self):
        contract=deepcopy(self.contract);contract['dataset_role']='test';contract['config']['datasets']={'test':{**self.ref,'role':'test'}}
        plan={'group':service.GROUP,'authorized':True,'tasks':{'test':contract}}
        with self.assertRaisesRegex(ValueError,'cannot become development'):
            service.validate_task({**contract,'task_key':'test'},plan,'20418')

    def test_test_preparation_is_allowed_without_test_quality_access(self):
        contract=deepcopy(self.contract);contract.update(mode='dataset',dataset_role='test',case_indices=[],methods=['INIT'])
        contract['config']['datasets']={}
        parser=self.root/'parser.py';parser.write_text('# pinned parser')
        metadata=self.root/'metadata.json';atomic_json(metadata,{'official_metadata':True})
        contract['dataset_spec']={'kind':'official_day','role':'test','date':'2024-02-19','count':20,
            'raw_shared_root':str(self.root/'source/raw/2024-02-19'),
            'parser':{'path':str(parser),'sha256':digest(parser)},
            'metadata_pins':{'release_api':{'path':str(metadata),'sha256':digest(metadata)}}}
        plan={'group':service.GROUP,'authorized':True,'scientific_commit':'a'*40,'tasks':{'testprep':contract}}
        atomic_json(self.plan,plan)
        body={**contract,'task_key':'testprep','request_id':'test-metadata-only',
              'plan':{'path':str(self.plan),'sha256':digest(self.plan)}}
        with patch.object(service,'PARENT',self.root):
            job=service.create_job(self.app,body)
        self.assertEqual(job['kind'],'prepare');self.assertEqual(job['case_indices'],[])
        service.verify_job(self.store,job)

    def test_formal_common_timing_rejects_parallel_cases(self):
        contract=deepcopy(self.contract);contract['mode']='validation';contract['dataset_role']='validation'
        contract['config']['datasets']={'validation':{**self.ref,'role':'validation'}};contract['config']['workers']=8
        plan={'group':service.GROUP,'authorized':True,'tasks':{'validation':contract}}
        with self.assertRaisesRegex(ValueError,'one case worker'):
            service.validate_task({**contract,'task_key':'validation'},plan,'20418')


if __name__=='__main__':unittest.main()
