"""Pure rejection fixtures and actual create/source-freeze chain; no science."""
import copy,json,tempfile,unittest,os,shutil,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.storage import digest,Store
from satellite_rl.workbench.r7_case_binding import validate_binding
from satellite_rl.workbench.r7_label_guard_audit import audit_stage
from satellite_rl.workbench.r6_io import write_asset

def ref(path):return dict(path=str(path),sha256=digest(path))
def case(role='train',index=0):
    source='train' if role=='train' else 'validation';utc='2024-01-08T00:02:00Z';mother=source+':'+utc
    return dict(actual_role=role,source_role=source,source={'original_case_index':index},UTC=utc,original_mother_id=mother,
                case_id=f'r7:{role}:{mother}:uniform21600000',content_sha256='a'*64,aircraft_ids=['a','b'],total_request_bps_int64=43200000)

class RoleBindingRejects(unittest.TestCase):
    def test_development_cannot_be_wrapped_as_training(self):
        with self.assertRaises(ValueError):validate_binding(case('development'),'train',0)
    def test_index_cannot_change_job_interpretation(self):
        with self.assertRaises(ValueError):validate_binding(case(index=1),'train',0)
    def test_producer_partition_swap_rejected(self):
        record=case();row=validate_binding(record,'train',0);row['role']='development'
        with self.assertRaises(ValueError):validate_binding(record,'train',0,row)
    def test_count_and_total_from_case_not_producer(self):
        record=case();row=validate_binding(record,'train',0);row['aircraft_count']=1
        with self.assertRaises(ValueError):validate_binding(record,'train',0,row)

class GuardClosureRejects(unittest.TestCase):
    def fixture(self,root):
        ident=dict(role='train',case_id='r7_mock',identity_sha256='a'*64);scene_ref=dict(path='/synthetic/scene',sha256='b'*64)
        request=dict(stage='preparation',identity=ident,scene_ref=scene_ref,source_sha256='c'*64,producer_sha256='d'*64,
                     phase_manifest=dict(stage='labels',role='train',admitted=True,identity_sha256='a'*64,dataset_sha256='e'*64,cases={'r7_mock':ident}))
        request_path=root/'request.json';atomic_json(request_path,request);started=root/'started.json';atomic_json(started,dict(request_sha256=digest(request_path),science_call_reserved=True))
        asset=write_asset(root/'asset.json.gz',dict(stage='preparation',identity=ident,source_sha256='c'*64,python='/envs/DQN/bin/python',result={'status':'unavailable'}))
        output=root/'stage_output.json';atomic_json(output,dict(asset_ref=asset,stage='preparation',identity=ident))
        command=['/envs/DQN/bin/python','-B','-u','-m','satellite_rl.workbench.r7_stage','--request',str(request_path),'--directory',str(root)]
        wal=root/'stage.wal.json';atomic_json(wal,dict(hard_seconds=20,owned_parent_children_rss_bytes=2147483648,status='child_started',pid=100,birth=20,command=command))
        guard=dict(status='completed',reason=None,exit_code=0,unknown_algorithm_spend=False,hard_interface_error=False,monitor_error=None,actual_stage_wall_seconds=1.,peak_parent_children_rss_bytes=1024,pid=100,birth=20,output_ref=ref(output))
        gp=root/'guard.json';original=root/'guard_receipt.json';atomic_json(gp,guard);atomic_json(original,guard)
        args=dict(guard_ref=ref(gp),request_ref=ref(request_path),started_ref=ref(started),wal_ref=ref(wal),guard_receipt_ref=ref(original),stage='preparation',identity=ident,scene_ref=scene_ref,source_sha='c'*64,producer_sha='d'*64,dataset_sha='e'*64,asset_ref=asset)
        return args,guard,gp,original,wal
    def test_original_guards_close_and_unknown_never_gets_credit(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,guard,gp,original,_=self.fixture(Path(tmp));self.assertTrue(audit_stage(**args)['closed'])
            guard['unknown_algorithm_spend']=True;atomic_json(gp,guard);atomic_json(original,guard);args.update(guard_ref=ref(gp),guard_receipt_ref=ref(original));self.assertFalse(audit_stage(**args)['closed'])
    def test_wrong_original_hard_cap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,_,_,_,wal=self.fixture(Path(tmp));value=json.loads(wal.read_text());value['hard_seconds']=45;atomic_json(wal,value);args['wal_ref']=ref(wal)
            with self.assertRaises(ValueError):audit_stage(**args)
    def test_bad_wall_memory_and_exit_cannot_close(self):
        for field,value in (('actual_stage_wall_seconds',21.),('peak_parent_children_rss_bytes',2147483649),('exit_code',1)):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                args,guard,gp,original,_=self.fixture(Path(tmp));guard[field]=value;atomic_json(gp,guard);atomic_json(original,guard);args.update(guard_ref=ref(gp),guard_receipt_ref=ref(original));self.assertFalse(audit_stage(**args)['closed'])

class CompleteSourceRejects(unittest.TestCase):
    def test_subset_source_bridge_is_rejected(self):
        import satellite_rl.workbench.r7_source_bridge as subject
        record=dict(schema='R7CompleteSourceInterfaceBridgeV1',passed=True,group='S1-AI17-R7',scientific_commit='a'*40,design_sha256='b'*64,admitted_phases=['data_train_dev'],source_file_sha256={'src/a.py':'x'})
        with patch.object(subject,'source_members',return_value={'src/a.py':'x','scripts/required.py':'y'}):
            with self.assertRaisesRegex(ValueError,'every actual'):subject.validate_bridge(record,'.','a'*40,'b'*64,'data_train_dev')

class TypedUnavailableAndRawSceneTests(unittest.TestCase):
    def test_unavailable_preparation_never_starts_teacher_or_decoder(self):
        import numpy as np
        import satellite_rl.workbench.r7_labels as subject
        job=dict(dataset_role='train',case_indices=[0],scientific_commit='a'*40,source_manifest_sha256='b'*64,experiment_id='S1-AI17-R7')
        config=dict(datasets={'train':dict(sha256='c'*64)})
        scene=SimpleNamespace(positions=np.array([[50.,8.]]),demands=np.array([21600000.]),aircraft_ids=('a',),profile='europe_v1')
        ident=dict(mother_id='train:UTC');request=dict(identity=ident,scene_ref=dict(path='/mock/scene',sha256='d'*64));source=dict(case_ref=dict(path='/mock/case',sha256='e'*64));calls=[]
        def guarded(target,req,limit,stop,**kwargs):
            calls.append(req['stage']);target.mkdir(parents=True,exist_ok=True)
            for name in ('guard.json','request.json','started.json','stage.wal.json','guard_receipt.json'):atomic_json(target/name,{})
            return dict(actual_stage_wall_seconds=1.),dict(result=dict(status='unavailable',ledger={})),dict(path=str(target/'unavailable.json'),sha256='f'*64)
        with tempfile.TemporaryDirectory() as tmp,patch.object(subject,'supply_case',return_value=({},scene,request,source)),patch.object(subject,'guarded',side_effect=guarded),patch('satellite_rl.workbench.r7_label_guard_audit.audit_chain',return_value=dict(closed=True,actual_calls=dict(prepare=1,T=0,D5=0))),patch('scripts.stage1_ai_search_r4.audit.audit_ledger',return_value={'valid':True}):
            result=subject.labels(job,config,Path(tmp));row=json.loads(Path(result['cases']['0']['path']).read_text())
        self.assertEqual(calls,['preparation']);self.assertIsNone(row['original_T_terminal_ref']);self.assertIsNone(row['D5_qualification_ref']);self.assertFalse(row['label_audit']['qualified'])
    def test_existing_wrong_scene_rejected_before_science(self):
        import numpy as np
        import satellite_rl.workbench.r7_labels as subject
        record=case();row=validate_binding(record,'train',0);row['case_ref']=dict(path='/mock/case',sha256='a'*64)
        scene=SimpleNamespace(positions=np.array([[50.,8.],[51.,9.]]),demands=np.array([21600000.,21600000.]),aircraft_ids=('a','b'),profile='europe_v1')
        job=dict(dataset_role='train',model_inputs={'authority':dict(path='/mock/authority',sha256='a'*64)})
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp);dataset=target/'dataset.json';atomic_json(dataset,{'cases':[row]});np.savez_compressed(target/'scene.npz',positions=scene.positions,demands=[1.,1.],aircraft_ids=scene.aircraft_ids,profile='europe_v1')
            config=dict(datasets={'train':ref(dataset)})
            with patch.object(subject,'load_algorithm_case',return_value=(record,scene)):
                with self.assertRaisesRegex(ValueError,'existing raw scene'):subject.supply_case(job,config,0,target)

class ActualCreateSourceChain(unittest.TestCase):
    def test_actual_create_uses_format2_hook_and_foreign_worker_import(self):
        from satellite_rl.workbench import coverage_ai_search_r7 as policy,frozen_round_tasks
        from scripts.stage1_ai_search_r7.plan_builder import build
        from scripts.stage1_ai_search_r7.contract import DESIGN_SHA
        repository=Path(__file__).resolve().parents[1]
        design=Path('D:/paper/outputs/experiments/first_stage_ai_search_20261013/design/round7_design.json')
        if not design.is_file():self.skipTest('frozen root v2 design artifact required for actual create fixture')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repo=root/'repo';repo.mkdir()
            for name in ('src','scripts'):shutil.copytree(repository/name,repo/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            for name in ('pyproject.toml','.gitattributes'):shutil.copy2(repository/name,repo/name)
            def git(*args):return subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True).stdout
            git('init','-q');git('add','.');git('-c','user.name=PureMock','-c','user.email=mock@example.invalid','commit','-qm','pure actual create full source closure')
            commit=git('rev-parse','HEAD').decode().strip();mock=root/'root_mock.json';atomic_json(mock,dict(pure_mock=True));mr=ref(mock)
            p=build(commit,'data_train_dev',dict(design_ref=dict(path=str(design),sha256=DESIGN_SHA),identity_ref=mr,prerequisite_gate_ref=mr,qualification_refs={'complete_source_interface_bridge':mr},model_inputs={'recipe':mr},datasets={}))
            plan=root/'plan.json';atomic_json(plan,p);key='R7_data_train_dev_20474';contract=p['tasks'][key]
            store=Store(root/'lane'/'round7'/'archive',source_root=repo);app=SimpleNamespace(store=store)
            body={**contract,'request_id':'pure_mock_r7_actual_create','task_key':key,'plan':ref(plan),'root_review':mr}
            # Only remote admission checks are mocks. The actual Store,
            # create, policy.freeze_source, snapshot and job publication run.
            with patch.object(policy,'deployment',return_value='20474'),patch.object(policy,'validate_admission'),patch.object(policy,'validate_task'),patch.object(store,'freeze_source',side_effect=AssertionError('legacy format1 must not run')):
                job=frozen_round_tasks.create(app,body,policy)
            source=Path(job['source_root']);manifest=json.loads((source/'source_manifest.json').read_text())
            self.assertEqual(manifest['format_version'],2);self.assertEqual(manifest['group'],'S1-AI17-R7');self.assertTrue((source/'scripts/stage1_ai_search_r7/plan_builder.py').is_file());self.assertEqual(job['scientific_commit'],commit)
            foreign=root/'foreign';foreign.mkdir();env=os.environ.copy();env.update(PYTHONPATH=str(source/'src')+os.pathsep+str(source),PYTHONDONTWRITEBYTECODE='1')
            code='import satellite_rl.workbench.r7_runner;import satellite_rl.workbench.coverage_ai_search_r7;import satellite_rl.workbench.r7_case_worker;import scripts.stage1_ai_search_r7.controller;print("actual job source closure passed")'
            result=subprocess.run([sys.executable,'-B','-c',code],cwd=foreign,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);self.assertIn('actual job source closure passed',result.stdout)

if __name__=='__main__':unittest.main()
