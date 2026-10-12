from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from satellite_rl.coverage.ai_search.config import GeometryConfig
from satellite_rl.coverage.ai_search_r2.config import TaskConfig
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.storage import Store,digest
from satellite_rl.workbench import coverage_ai_search_r2 as policy,frozen_round_tasks
from satellite_rl.workbench import ai_search_r2_runner as runner,ai_search_r2_extension as extension
from satellite_rl.coverage.ai_search.core import Scene
from scripts.stage1_ai_search_r2 import plan_builder
from scripts.stage1_ai_search_r2 import controller,deploy,ops


class RoundContract(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.store=Store(self.root/'20418/archive',source_root=self.root)
        self.source=self.store.root/'source_snapshots'/(extension.PILOT_COMMIT);self.source.mkdir()
        atomic_json(self.source/'source_manifest.json',{'git_commit':extension.PILOT_COMMIT,'files':{'fixture':'not_science'}})
        self.store.freeze_source=lambda:(self.source,extension.PILOT_COMMIT)
        data=self.root/'input.npz';data.write_bytes(b'raw input fixture signature')
        common=self.root/'common.json';atomic_json(common,{'fixture':'no solver'})
        design=Path(__file__).resolve().parents[1]/'scripts/stage1_ai_search_r2/round2_design.json'
        config={'schema_version':10,'research_stage':'coverage','algorithm':'ai_search_r2','name':'paired probe',
            'geometry':asdict(GeometryConfig()),'task':asdict(TaskConfig()),
            'datasets':{'development':{'path':str(data),'sha256':digest(data),'role':'development','count':80}},
            'workers':1,'resources':{'minimum_free_bytes':1,'max_rss_bytes':1<<30}}
        config=json.loads(json.dumps(config))
        self.contract={'config':config,'mode':'evaluate_closed_pool','dataset_role':'development',
            'case_indices':[0,1,2,3,44,45,46,47],'executor_port':20418,'variants':['D0','D1'],'source_refs':{},
            'common_ref':{'path':str(common),'sha256':digest(common)}}
        self.plan_path=self.root/'plan.json';self.plan={'group':policy.GROUP,'authorized':True,'scientific_commit':extension.PILOT_COMMIT,
            'training_allowed':False,'test_allowed':False,'design':{'path':str(design),'sha256':digest(design)},'tasks':{'probe':self.contract},
            'phase':'pilot','mother_order':[f'm{i}' for i in range(20)],'selected_mothers':['m0','m4','m8','m11','m15','m19'],
            'selected_case_indices':[0,1,2,3,16,17,18,19,32,33,34,35,44,45,46,47,60,61,62,63,76,77,78,79]}
        for port,indices in [(20424,[16,17,18,19,60,61,62,63]),(20473,[32,33,34,35,76,77,78,79])]:
            task=deepcopy(self.contract);task.update(executor_port=port,case_indices=indices);self.plan['tasks']['probe_'+str(port)]=task
        qualification=self.root/'qualification.json';atomic_json(qualification,{'passed':True,'scientific_commit':extension.PILOT_COMMIT,
            'design_sha256':digest(design),'core_git_blob_sha256':{'fixture':'b'*64}})
        self.plan['qualification_refs']={k:{'path':str(qualification),'sha256':digest(qualification)}
            for k in ('validator_equivalence','charge_before_budget')}
        atomic_json(self.plan_path,self.plan)
        review=self.root/'root_review.json';atomic_json(review,{'group':policy.GROUP,'passed':True,'scientific_commit':extension.PILOT_COMMIT,
            'plan_sha256':digest(self.plan_path),'design_sha256':digest(design)})
        self.body={**self.contract,'task_key':'probe','request_id':'one-pilot-probe','plan':{'path':str(self.plan_path),'sha256':digest(self.plan_path)},
            'root_review':{'path':str(review),'sha256':digest(review)}}
        self.patch=patch.object(policy,'deployment',return_value='20418');self.patch.start();self.addCleanup(self.patch.stop)
        self.app=SimpleNamespace(store=self.store)

    def test_exact_source_owner_and_unknown_request_reuse(self):
        first=frozen_round_tasks.create(self.app,self.body,policy)
        again=frozen_round_tasks.create(self.app,deepcopy(self.body),policy)
        self.assertEqual(first['run_id'],again['run_id']);self.assertEqual(len(self.store.jobs()),1)
        frozen_round_tasks.verify(self.store,first,policy)

    def test_new_request_cannot_duplicate_semantic_pilot(self):
        frozen_round_tasks.create(self.app,self.body,policy);changed=deepcopy(self.body);changed['request_id']='a-second-request'
        with self.assertRaisesRegex(ValueError,'another owner/request'):frozen_round_tasks.create(self.app,changed,policy)

    def test_parallel_case_timing_and_budget_change_rejected(self):
        changed=deepcopy(self.contract);changed['config']['workers']=8
        with self.assertRaisesRegex(ValueError,'one case worker'):policy.validate_task(changed,self.plan,'20418')
        changed=deepcopy(self.contract);changed['config']['task']['ray_pair_limit']+=1
        with self.assertRaisesRegex(ValueError,'pinned design'):policy.validate_task(changed,self.plan,'20418')

    def test_validation_test_or_training_cannot_be_submitted(self):
        changed=deepcopy(self.contract);changed['dataset_role']='test'
        with self.assertRaises(ValueError):policy.validate_task(changed,self.plan,'20418')
        changed=deepcopy(self.plan);changed['training_allowed']=True
        with self.assertRaisesRegex(ValueError,'exclude training'):policy.validate_task(self.contract,changed,'20418')

    def test_geometry_feasible_set_cannot_change_under_same_design(self):
        for field,value in [('max_beams',199),('width_menu_deg',[.3,.5]),('tolerance',1e-10),('width_min_deg',.2)]:
            changed=deepcopy(self.contract);changed['config']['geometry'][field]=value
            with self.assertRaisesRegex(ValueError,'feasible set'):policy.validate_task(changed,self.plan,'20418')

    def test_job_semantics_and_external_common_change_fail_closed(self):
        job=frozen_round_tasks.create(self.app,self.body,policy)
        Path(job['references']['common_manifest']['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'source SHA differs'):frozen_round_tasks.verify(self.store,job,policy)

    def test_completed_original_pilot_stays_verifiable_and_cannot_resume(self):
        job=frozen_round_tasks.create(self.app,self.body,policy)
        frozen_round_tasks.verify(self.store,job,policy)
        directory=self.store.job_dir(job['run_id']);atomic_json(directory/'status.json',{'stage':'completed','pid':None})
        with self.assertRaisesRegex(ValueError,'completed tasks cannot resume'):
            frozen_round_tasks.resume(SimpleNamespace(store=self.store),job['run_id'],policy)

    def test_direct_api_scope_and_admission_cannot_bypass_publisher(self):
        changed=deepcopy(self.plan);changed['phase']='extension_without_gate'
        with self.assertRaisesRegex(ValueError,'pilot'):policy.validate_task(self.contract,changed,'20418')
        changed=deepcopy(self.contract);changed['case_indices']=[4,5,6,7]
        with self.assertRaisesRegex(ValueError,'unselected'):policy.validate_task(changed,self.plan,'20418')
        with self.assertRaises((ValueError,TypeError,KeyError)):policy.validate_admission(self.plan,self.body['plan'],None)
        changed=deepcopy(self.plan);changed['qualification_refs']={}
        with self.assertRaisesRegex(ValueError,'qualifications'):policy.validate_admission(changed,self.body['plan'],self.body['root_review'])

    def test_root_admission_bytes_or_commit_cannot_change_before_resume(self):
        job=frozen_round_tasks.create(self.app,self.body,policy)
        Path(job['references']['publication_root_review']['path']).write_bytes(b'changed proof')
        with self.assertRaisesRegex(ValueError,'source SHA differs'):frozen_round_tasks.verify(self.store,job,policy)


class PairedDurability(unittest.TestCase):
    """No optimization: mocks verify dispatch/budget durability at crash boundaries."""
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.output=self.root/'cases';self.output.mkdir()
        self.case={'case_id':'mother:base','mother_id':'mother','demand_profile':'base','input_case_index':0,
            'positions':[[50,10]],'demands':[1.0],'aircraft_ids':['a'],'profile':'europe_v1','input_dataset_sha256':'d'*64}
        scene=Scene(self.case['positions'],self.case['demands'],self.case['aircraft_ids'],self.case['profile'])
        self.common=SimpleNamespace(input_sha=scene.input_sha256,archive_sha='a'*64,parent_sha='b'*64,frozen_bytes=b'common')
        common_path=self.root/'common.json';atomic_json(common_path,{'engineering_fixture':True})
        self.manifest=self.root/'common_manifest.json'
        atomic_json(self.manifest,{'cases':{'0':{'case_id':self.case['case_id'],'common':{'path':str(common_path),'sha256':digest(common_path)}}}})
        plan=self.root/'plan.json';atomic_json(plan,{'phase':'pilot'})
        self.job={'scientific_commit':extension.PILOT_COMMIT,'references':{'plan':{'path':str(plan),'sha256':digest(plan)},'common_manifest':{'path':str(self.manifest),'sha256':digest(self.manifest)}},
            'variants':['D0','D1'],'mode':'evaluate_closed_pool'}
        self.config={'task':asdict(TaskConfig()),'resources':{'minimum_free_bytes':1,'max_rss_bytes':4<<30}}
        self.prepared={'status':'prepared','pool_sha':'p'*64,'input_sha':scene.input_sha256,
            'archive_sha':self.common.archive_sha,'parent_sha':self.common.parent_sha,'tasks':[],
            'preparation_cost':{'seconds':.25,'ray_pair_checks':5,'geometry_calls':0,'attempts':0},'config':self.config['task']}
        self.from_record=patch('satellite_rl.coverage.ai_search_r2.initials.CommonStart.from_record',return_value=self.common)
        self.from_record.start();self.addCleanup(self.from_record.stop)
        self.order=[]

    def prepare(self,*args,**kwargs):
        starts=list((self.root/'pool_started').glob('*.json'));self.assertEqual(len(starts),1)
        self.assertEqual(json.loads(starts[0].read_text())['status'],'started_before_any_shared_preparation')
        return deepcopy(self.prepared)

    def evaluate(self,scene,common,variant,**kwargs):
        self.order.append(variant);prepared=kwargs['prepared_pool']
        return {'pool_sha':prepared['pool_sha'],'input_sha':common.input_sha,'parent_sha':common.parent_sha,
            'closed':prepared['status']=='prepared','status':prepared['status'],'results':[],
            'preparation_cost':prepared['preparation_cost'],'training_updates':0}

    def test_prepare_durable_once_and_both_variant_parity(self):
        self.case['input_case_index']=1
        manifest=json.loads(self.manifest.read_text());manifest['cases']['1']=manifest['cases'].pop('0');atomic_json(self.manifest,manifest)
        self.job['references']['common_manifest']['sha256']=digest(self.manifest)
        with patch('satellite_rl.coverage.ai_search_r2.probe.prepare_pool',side_effect=self.prepare) as prep,patch(
                'satellite_rl.coverage.ai_search_r2.probe.evaluate_closed_pool',side_effect=self.evaluate):
            rows=runner.paired(self.case,self.job,self.config,str(self.output),set())
            self.assertEqual(self.order,['D1','D0']);self.assertEqual(prep.call_count,1);self.assertEqual(len(rows),2)
            runner.paired(self.case,self.job,self.config,str(self.output),{key for key,_ in rows})
            self.assertEqual(prep.call_count,1)

    def test_typed_preparation_guard_retains_measured_cost_for_both(self):
        self.prepared.update(status='incomplete_preparation',pool_sha=None,reason='wall_guard')
        with patch('satellite_rl.coverage.ai_search_r2.probe.prepare_pool',side_effect=self.prepare),patch(
                'satellite_rl.coverage.ai_search_r2.probe.evaluate_closed_pool',side_effect=self.evaluate):
            rows=runner.paired(self.case,self.job,self.config,str(self.output),set())
        self.assertEqual(len(rows),2)
        for _,ref in rows:
            row=json.loads(Path(ref['path']).read_text());self.assertFalse(row['result']['closed'])
            self.assertEqual(row['result']['preparation_cost'],self.prepared['preparation_cost'])

    def test_unknown_prepare_crash_cannot_repeat(self):
        with patch('satellite_rl.coverage.ai_search_r2.probe.prepare_pool',side_effect=RuntimeError('unknown crash')):
            with self.assertRaisesRegex(RuntimeError,'unknown crash'):runner.paired(self.case,self.job,self.config,str(self.output),set())
        with patch('satellite_rl.coverage.ai_search_r2.probe.prepare_pool') as prep:
            with self.assertRaisesRegex(ValueError,'unknown spent work'):runner.paired(self.case,self.job,self.config,str(self.output),set())
            prep.assert_not_called()
        self.assertTrue(json.loads((self.root/'unknown_budget_pool.json').read_text())['automatic_reprepare_forbidden'])

    def test_bound_pool_source_or_bytes_cannot_change(self):
        with patch('satellite_rl.coverage.ai_search_r2.probe.prepare_pool',side_effect=self.prepare),patch(
                'satellite_rl.coverage.ai_search_r2.probe.evaluate_closed_pool',side_effect=self.evaluate):
            runner.paired(self.case,self.job,self.config,str(self.output),set())
        pool=next((self.root/'pools').glob('*.json'));pool.write_bytes(pool.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'artifact changed'):runner.paired(self.case,self.job,self.config,str(self.output),set())

    def test_preexisting_abort_does_not_start_queued_case(self):
        (self.root/'stop.request').write_text('hard error')
        with patch('satellite_rl.coverage.ai_search_r2.probe.prepare_pool') as prep:
            self.assertEqual(runner.paired(self.case,self.job,self.config,str(self.output),set()),[])
            self.assertEqual(runner.one(self.case,'COMMON',self.job,self.config,str(self.output)),[])
            prep.assert_not_called()

    def test_hard_failure_is_durable_before_queued_futures_cancel(self):
        seen=[]
        def cancelled():
            self.assertTrue((self.root/'stop.request').is_file())
            evidence=json.loads((self.root/'abort.json').read_text())
            self.assertEqual(evidence['completed_rows'],7);self.assertTrue(evidence['unknown_active_work_preserved'])
            seen.append(True)
        runner.abort_pending(self.root,[SimpleNamespace(cancel=cancelled) for _ in range(3)],RuntimeError('hard'),7)
        self.assertEqual(len(seen),3)


class PlanExpansion(unittest.TestCase):
    def test_exact_pilot_whole_mothers_and_384_task_slots(self):
        design=Path(__file__).resolve().parents[1]/'scripts/stage1_ai_search_r2/round2_design.json'
        inventory={'selected_mothers':[f'm{i}' for i in range(6)],'selected_case_indices':list(range(24)),
            'development_identities':{'mother_ids':[f'm{i//4}' for i in range(24)]},
            'source_refs':{str(i):[{'method':m} for m in ('07-CV','07-LEX','JOINT-CV','JOINT-LEX','POST_SHRINK')] for i in range(24)}}
        prepared=plan_builder.prepare_plan('a'*40,design,inventory,{'path':'/signed/packed.npz','sha256':'d'*64,'role':'development','count':80})
        self.assertEqual(plan_builder.audit(prepared,inventory)['result_rows'],24)
        evaluated=plan_builder.evaluation_plan(prepared,{str(p):{'path':f'/signed/common_{p}.json','sha256':'e'*64} for p in (20418,20424,20473)})
        self.assertEqual(plan_builder.audit(evaluated,inventory)['decoder_task_calls'],384)
        for task in evaluated['tasks'].values():self.assertEqual(len(task['case_indices']),8)
        evaluated['tasks'][next(iter(evaluated['tasks']))]['case_indices'][0]=24
        with self.assertRaises((ValueError,IndexError)):plan_builder.audit(evaluated,inventory)


class ControllerBoundaries(unittest.TestCase):
    def test_unknown_post_retries_exact_request_and_finite_controller_exits(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);body={'request_id':'same-stable-request','task_key':'pilot'}
            spec={'group':ops.GROUP,'automatic_extension':False,'scientific_commit':'a'*40,'tasks':[{'port':20418,'body':body}]}
            path=root/'spec.json';atomic_json(path,spec);posts=[]
            def api(port,path,payload=None):
                if payload is not None:
                    posts.append(deepcopy(payload))
                    if len(posts)==1:raise TimeoutError('unknown POST outcome')
                    return {'run_id':'original-run'}
                return {'status':{'stage':'completed','completed_cases':16}}
            with patch.object(controller,'route',return_value={'ready':True}),patch.object(controller.ops,'api',side_effect=api),patch.object(controller.time,'sleep'):
                controller.run(path,root/'state')
            self.assertEqual(posts,[body,body]);state=json.loads((root/'state/state.json').read_text())
            self.assertEqual(state['phase'],'completed');self.assertEqual(state['intents']['same-stable-request']['run_id'],'original-run')

    def test_remote_deploy_and_route_code_compiles_without_execution(self):
        def compile_only(port,code):compile(code,'remote_deploy','exec');return {'engineering_syntax_only':True}
        with patch.object(deploy.ops,'upload',return_value={'local_sha256':'s','remote':'s'}),patch.object(deploy,'remote',side_effect=compile_only):
            self.assertTrue(deploy.lane(20418,'a'*40,Path('unused.bundle'))['service']['engineering_syntax_only'])
        def route_only(code,port,timeout):compile(code,'remote_route','exec');return {'exit_code':0,'stdout':'{}','stderr':''}
        with patch.object(controller.ops,'remote_python',side_effect=route_only):controller.route(20418,'a'*40)

    def test_fourth_lane_and_other_group_api_are_rejected(self):
        with self.assertRaises(ValueError):ops.instance(20474)
        with self.assertRaisesRegex(ValueError,'cannot operate R1'):ops.api(20418,'/api/ai-search/tasks')


if __name__=='__main__':unittest.main()
