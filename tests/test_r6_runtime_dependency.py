import copy,json,tempfile,unittest,hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from satellite_rl.workbench.r6_runtime_dependency import SCIENCE,TARGETS,stable_job_sha,validate_contract,prepare

class RuntimeDependencyTests(unittest.TestCase):
    def job(self):return dict(experiment_id='S1-AI17-R6',scientific_commit=SCIENCE,git_commit=SCIENCE,executor_port=20418,run_id=TARGETS[20418],request_id='original',references={'train':{'sha256':'input'}})
    def contract(self,job):return dict(schema='R6RuntimeScriptDependencyAdmissionV1',passed=True,group='S1-AI17-R6',science_commit=SCIENCE,tasks={run:dict(stable_job_sha256=stable_job_sha(job),config_sha256='config',snapshot_manifest_sha256='snapshot',executor_port=20418) for run in TARGETS.values()})
    def test_original_same_ID_resume_only_transient_fields(self):
        job=self.job();contract=self.contract(job);validate_contract(contract,job,'config','snapshot')
        job.update(resume=True,resume_attempt=1);validate_contract(contract,job,'config','snapshot')
        for field,value in [('request_id','new'),('scientific_commit','b'*40),('run_id','new')]:
            wrong=copy.deepcopy(job);wrong[field]=value
            with self.assertRaises(ValueError):validate_contract(contract,wrong,'config','snapshot')
        with self.assertRaises(ValueError):validate_contract(contract,job,'changed','snapshot')

    def test_exact_pinned_tree_first_snapshot_and_script_imports(self):
        def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf8');return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);archive=base/'20418/round6/archive';archive.mkdir(parents=True);snapshot=archive/'snapshots'/SCIENCE;snapshot.mkdir(parents=True)
            repo=base/'20418/round6/source_epochs'/SCIENCE/'repo';script=repo/'scripts/x.py';script.parent.mkdir(parents=True);script.write_bytes(b'# immutable\n');sha=hashlib.sha256(script.read_bytes()).hexdigest()
            directory=archive/'jobs'/TARGETS[20418];config=write(directory/'config.json',{})
            manifest=write(snapshot/'source_manifest.json',dict(git_commit=SCIENCE,source_root=str(repo)))
            job=dict(self.job(),source_root=str(snapshot),source_manifest_sha256=manifest['sha256'])
            review=write(base/'review.json',dict(passed=True,scientific_commit=SCIENCE,source_file_sha256={'scripts/x.py':sha}))
            contract=self.contract(job);contract['tasks'][job['run_id']].update(config_sha256=config['sha256'],snapshot_manifest_sha256=manifest['sha256'])
            contract.update(operational_service_commit='ops',original_source_review_ref=review,script_file_sha256={'scripts/x.py':sha})
            admission=write(base/'admission.json',contract);write(archive.parent/'ops/runtime_dependency_admission_ref.json',admission)
            service=base/'service';store=SimpleNamespace(archive_root=archive,source_root=service)
            def git(root,*args):
                if args==('rev-parse','HEAD'):return 'ops' if root==service else SCIENCE
                if args[0]=='ls-tree':return 'scripts/x.py'
                return ''
            identities=dict(package=str(snapshot/'src/satellite_rl/__init__.py'),policy=str(snapshot/'src/satellite_rl/workbench/coverage_ai_search_r6.py'),stage=str(snapshot/'src/satellite_rl/workbench/ai_search_r6_stage.py'),plan_builder=str(repo/'scripts/stage1_ai_search_r6/plan_builder.py'),label_audit=str(repo/'scripts/stage1_ai_search_r6/label_audit.py'))
            def probe(*args,**kwargs):
                self.assertEqual(kwargs['env']['PYTHONPATH'].split(__import__('os').pathsep),[str(snapshot/'src'),str(repo)])
                return SimpleNamespace(returncode=0,stdout=json.dumps(identities),stderr='')
            with patch('satellite_rl.workbench.r6_runtime_dependency.BASE',base),patch('satellite_rl.workbench.r6_runtime_dependency.git',side_effect=git),patch('subprocess.check_output',return_value=b'# immutable\n'),patch('subprocess.run',side_effect=probe):
                result=prepare(store,job,directory,'python');self.assertEqual(result['new_algorithm_calls'],0)
                self.assertEqual(result['actual_import_identities'],identities)
                script.write_bytes(b'# changed\n')
                with self.assertRaises(ValueError):prepare(store,job,directory,'python')

    def test_route_distinguishes_operation_and_science_epoch(self):
        from scripts.stage1_ai_search_r6.controller import route
        captured=[]
        def remote(code,*args,**kwargs):captured.append(code);return dict(exit_code=0,stdout='{"scientific_commit":"science","operational_service_commit":"operations"}',stderr='')
        with patch('scripts.stage1_ai_search_r6.controller.ops.remote_python',side_effect=remote):result=route(20418,'science','operations')
        self.assertEqual(result['operational_service_commit'],'operations');self.assertIn("registry['source_commit']!='operations'",captured[0]);self.assertIn("'scientific_commit':'science'",captured[0])

    def test_controller_launch_exports_operational_epoch_only_mock(self):
        import zipfile
        from scripts.stage1_ai_search_r6.launch import launch
        for expected,extra in [('operations',dict(operational_service_commit='operations')),('science',{})]:
            with self.subTest(expected=expected),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);spec=root/'spec.json';spec.write_text(json.dumps(dict(scientific_commit='science',**extra)),encoding='utf8')
                def fake_archive(args,**kwargs):
                    target=next(a.split('=',1)[1] for a in args if a.startswith('--output='))
                    self.assertEqual(args[-1],expected)
                    with zipfile.ZipFile(target,'w') as z:z.writestr('stub.txt','not executable')
                def fake_child(*args,**kwargs):kwargs['stdout'].close();return SimpleNamespace(pid=42)
                with patch('scripts.stage1_ai_search_r6.launch.git',side_effect=lambda *args:expected if args[0]=='rev-parse' else ''),patch('scripts.stage1_ai_search_r6.launch.subprocess.run',side_effect=fake_archive),patch('scripts.stage1_ai_search_r6.launch.subprocess.Popen',side_effect=fake_child),patch('scripts.stage1_ai_search_r6.launch.process_start_time',return_value=123):
                    result=launch(spec,root/'controller')
                self.assertEqual(result['source_commit'],expected);self.assertEqual(result['scientific_commit'],'science')

if __name__=='__main__':unittest.main()
