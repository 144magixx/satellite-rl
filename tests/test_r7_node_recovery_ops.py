"""Pure additive recovery fixtures. No model, geometric kernel, remote/API or optimizer."""
import contextlib,copy,importlib.util,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from scripts.stage1_ai_search_r7_node_recovery.compat import baseline_view,audit_baseline,FULL_FIELDS,preserve_unknown_cost
from scripts.stage1_ai_search_r7_node_recovery import contracts,server
from scripts.stage1_ai_search_r7_node_recovery.qualification import original_gate_and_recovery_facts
from scripts.stage1_ai_search_r7_node_recovery.bootstrap import install_node_hooks

def fixture(method='RULE'):
    identity={key:'value:'+key for key in FULL_FIELDS};identity.update(role='development',profile='uniform21600000')
    reference={'path':'original/prepared.gz','sha256':'1'*64}
    prepared=dict(identity=copy.deepcopy(identity),input_sha256=identity['input_sha256'],source_sha256=contracts.CORE_SHA)
    provenance=dict(case_identity=copy.deepcopy(identity),input_sha=identity['input_sha256'],role_sha=identity['identity_sha256'],source_method=method,producer_sha256='producer',prepared_source_sha256=contracts.CORE_SHA,evidence_refs=[reference])
    proposal=dict(case_identity=copy.deepcopy(identity),input_sha=identity['input_sha256'],role_sha=identity['identity_sha256'],provenance=provenance,scores_n_by_K=[[1.]],mask_n_by_K=[[True]])
    return identity,reference,prepared,proposal

class CompatibilityTests(unittest.TestCase):
    def test_rule_random_use_only_memory_projection(self):
        for method in ('RULE','RANDOM'):
            identity,reference,prepared,proposal=fixture(method);original=copy.deepcopy(proposal)
            legacy=Mock(return_value={'valid':True})
            result=audit_baseline(legacy,{},proposal,identity,prepared,method,'producer',reference)
            self.assertTrue(result['valid']);self.assertEqual(len(legacy.call_args.args[1]['case_identity']),6)
            self.assertEqual(proposal,original);self.assertEqual(legacy.call_args.args[1]['scores_n_by_K'],proposal['scores_n_by_K'])
    def test_extension_tamper_rejected_before_legacy(self):
        for key in ('shared_case_content_sha256','source_asset_SHA'):
            for target in ('case_identity','provenance'):
                identity,ref,prepared,proposal=fixture();legacy=Mock()
                (proposal['case_identity'] if target=='case_identity' else proposal['provenance']['case_identity'])[key]='tampered'
                with self.assertRaises(ValueError):audit_baseline(legacy,{},proposal,identity,prepared,'RULE','producer',ref)
                legacy.assert_not_called()
    def test_producer_input_role_prepared_ref_rejected(self):
        for key in ('producer_sha256','input_sha','role_sha','prepared_source_sha256','evidence_refs'):
            identity,ref,prepared,proposal=fixture();proposal['provenance'][key]='tampered'
            with self.assertRaises(ValueError):baseline_view(proposal,identity,prepared,'RULE','producer',ref)
    def test_no_extra_or_missing_identity_fields(self):
        identity,ref,prepared,proposal=fixture();identity['surprise']='extra'
        with self.assertRaises(ValueError):baseline_view(proposal,identity,prepared,'RULE','producer',ref)
    def test_unknown_cost_never_recovered_from_finite_observations(self):
        row=dict(case_index=0,cost_components={'input':1.,'preparation_parent':2.,'AI_parent':8.},costs={m:10. for m in ('AI','RULE','RANDOM','T')})
        preserve_unknown_cost(row,{'sha256':'c'}, {'sha256':'o'}, [])
        self.assertTrue(all(x is None for x in row['costs'].values()));self.assertEqual(row['recovery_observed_timing_components']['AI_parent'],8.)
        self.assertEqual(row['recovery_accounting']['new_epoch_calls'],dict(prepare=0,T=1,D5=1))
    def test_named_facts_never_auto_unlock_original_or_future_gate(self):
        cases=[dict(case_index=i,costs={m:None for m in ('AI','RULE','RANDOM','T')}) for i in range(20)]
        old=Mock(return_value=dict(passed=False,hard_identity_budget_cost_closed=False,nonfallback=16,quality_successes=7))
        result=original_gate_and_recovery_facts(cases,old,amendment_ref={'sha256':'a'},bound_ref={'sha256':'b'})
        self.assertTrue(result['unchanged_quality_thresholds_met']);self.assertFalse(result['recovery_qualification_passed']);self.assertNotIn('passed',result)
        self.assertFalse(result['future_dispatch_allowed'])
    def test_original_caller_alias_publishes_null_cost_endpoint(self):
        caller_costs={method:20. for method in ('AI','RULE','RANDOM','T')}
        row=dict(case_index=0,costs=caller_costs,cost_components={'input':1.,'AI_parent':8.})
        preserve_unknown_cost(row,{'sha256':'c'},{'sha256':'o'},[])
        endpoint=dict(costs_before_publication=caller_costs)
        self.assertIs(row['costs'],caller_costs)
        self.assertTrue(all(value is None for value in endpoint['costs_before_publication'].values()))
        self.assertIn('NOT additive',row['recovery_observed_timing_components_scope'])

class RestrictedEndpointTests(unittest.TestCase):
    def test_outer_allowlist_no_create_or_foreign_resume(self):
        self.assertTrue(server.allowed('POST',server.RESUME_PATH));self.assertTrue(server.allowed('GET',server.TASK_PATH))
        for method,path in [('POST','/api/ai-search-r7/tasks'),('POST','/runs'),('POST',server.RESUME_PATH+'x'),('DELETE',server.TASK_PATH),('POST','/health')]:self.assertFalse(server.allowed(method,path))
    def test_unknown_prior_intent_cannot_spawn_again(self):
        with tempfile.TemporaryDirectory() as folder:
            endpoint=server.RecoveryEndpoint.__new__(server.RecoveryEndpoint);endpoint.operations=Path(folder);endpoint.contract_ref={'sha256':'c'};endpoint.review_ref={'sha256':'r'}
            contracts.atomic(Path(folder)/'resume_intent.json',dict(phase='spawn_pending'))
            fake=types.ModuleType('satellite_rl.workbench.geoload_runner');fake.exclusive_lock=lambda *a,**k:contextlib.nullcontext()
            spawn=Mock()
            with patch.dict(sys.modules,{'satellite_rl.workbench.geoload_runner':fake}),patch.object(server,'validate',side_effect=AssertionError('must not replay')):
                result=endpoint.resume(server.RESUME_PATH,dict(contract_ref=endpoint.contract_ref,root_review_ref=endpoint.review_ref),popen=spawn)
            spawn.assert_not_called();self.assertFalse(result['POST_replayed'])
    def test_PID_birth_units_and_live_owner_fail_closed(self):
        observation=dict(pid=12,birth_ticks=34,birth_kind='proc_start_ticks',boot_id='boot')
        with patch.object(contracts,'process_identity',return_value={**observation,'state':'S'}):
            with self.assertRaises(ValueError):contracts.require_gone(observation)
        with patch.object(contracts,'process_identity',return_value={**observation,'state':'Z'}):contracts.require_gone(observation)
        with self.assertRaises(ValueError):contracts.require_gone(dict(pid=12,birth=34))
    def test_logical_metadata_transport_reads_identical_actual_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            actual=Path(directory)/'original_root_bytes.json';actual.write_text('{"unchanged":true}')
            actual_ref=dict(path=str(actual),sha256=contracts.digest(actual));logical={**actual_ref,'path':'D:/paper/original_root_bytes.json'}
            mapping=Path(directory)/'map.json';contracts.atomic(mapping,dict(schema='R7NodeRecoveryLogicalRefTransportMapV1',remote_ref_by_original_path={logical['path']:actual_ref}))
            contract=dict(logical_reference_transport_map_ref=dict(path=str(mapping),sha256=contracts.digest(mapping)))
            contracts.require_logical_copy(logical,actual_ref,contract)
            with self.assertRaises(ValueError):contracts.require_logical_copy({**logical,'sha256':'0'*64},actual_ref,contract)
    def test_direct_spawn_handoff_is_after_locks_and_no_queued_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ops=root/'ops';ops.mkdir();jobdir=root/'job';jobdir.mkdir();(jobdir/'job.json').write_text('original bytes')
            endpoint=server.RecoveryEndpoint.__new__(server.RecoveryEndpoint);endpoint.operations=ops;endpoint.directory=jobdir;endpoint.contract_ref={'sha256':'c','path':'contract'};endpoint.review_ref={'sha256':'r','path':'review'}
            job=dict(source_root='original523',archive_root=str(root/'archive'))
            contract=dict(operations_source_files={},original_source_manifest_ref={'sha256':'science'})
            active=[]
            @contextlib.contextmanager
            def lock(path):
                active.append(Path(path).name)
                try:yield
                finally:active.remove(Path(path).name)
            def spawn(*args,**kwargs):
                self.assertIn('worker.lock',active);self.assertIn('compute.lock',active)
                self.assertEqual(json.loads((ops/'resume_intent.json').read_bytes())['phase'],'spawn_pending')
                self.assertFalse((ops/'parent_locks_released.json').exists());return types.SimpleNamespace(pid=999)
            guards=types.ModuleType('satellite_rl.workbench.r7_deadline');guards.stop_boundary=lambda *args:{'fixture_only':True}
            locks=types.ModuleType('satellite_rl.workbench.geoload_runner');locks.exclusive_lock=lock
            identity=dict(pid=999,state='R',birth_ticks=100,birth_kind='proc_start_ticks',boot_id='fixture')
            with patch.dict(sys.modules,{'satellite_rl.workbench.r7_deadline':guards,'satellite_rl.workbench.geoload_runner':locks}),patch.object(server,'validate',return_value=(contract,job,{})),patch.object(server,'fresh_resource'),patch.object(server,'process_identity',return_value=identity):
                result=endpoint.resume(server.RESUME_PATH,dict(contract_ref=endpoint.contract_ref,root_review_ref=endpoint.review_ref),popen=spawn)
            self.assertTrue(result['same_ID']);self.assertEqual(active,[])
            self.assertTrue(json.loads((ops/'parent_locks_released.json').read_bytes())['locks_released'])
            self.assertEqual((jobdir/'job.json').read_text(),'original bytes');self.assertFalse((jobdir/'status.json').exists())

class ActualWiringMockTests(unittest.TestCase):
    def test_actual_row_callback_preserves_AI_and_projects_checked_baselines(self):
        identity,ref,prepared,proposal=fixture()
        legacy=Mock(return_value=dict(valid=True))
        audit_module=types.SimpleNamespace(audit_proposal=legacy)
        def original_row(scene,w,value,request):
            if request['method']=='AI':return {'AI_original_path':True}
            return audit_module.audit_proposal(scene,value,request['identity'],w,None,request['method'],request['producer_sha256'])
        node=types.SimpleNamespace(audit_row=original_row,audit_costs=Mock(),guarded=Mock(),supply_case=Mock(),node=Mock())
        workbench=types.ModuleType('satellite_rl.workbench');workbench.r7_node=node;workbench.r7_node_audit=audit_module
        storage=types.ModuleType('satellite_rl.workbench.storage');storage.digest=lambda path:'mock'
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,{'satellite_rl.workbench':workbench,'satellite_rl.workbench.storage':storage}):
            receipt=install_node_hooks({'original_evidence_refs':[],'original_source_manifest_ref':{'sha256':'producer'}},{'sha256':'contract'},{'sha256':'ops'},Path(directory))
            request=dict(method='RULE',identity=identity,producer_sha256='producer',prepared_ref=ref,b7_decoder_source_SHA=contracts.CORE_SHA)
            self.assertTrue(node.audit_row({},prepared,proposal,request)['valid'])
            self.assertEqual(len(legacy.call_args.args[1]['case_identity']),6)
            self.assertEqual(node.audit_row({},prepared,{},dict(method='AI')),{'AI_original_path':True})
            self.assertTrue(receipt['AI_audit_path_unchanged'])
    def test_completed_guard_reuse_records_zero_new_call_before_audit(self):
        original_guard=Mock(return_value=({'reused_completed_call':True},{},{'sha256':'result'}))
        node=types.SimpleNamespace(audit_row=Mock(),audit_costs=Mock(),guarded=original_guard,supply_case=Mock(),node=Mock())
        workbench=types.ModuleType('satellite_rl.workbench');workbench.r7_node=node;workbench.r7_node_audit=types.SimpleNamespace(audit_proposal=Mock())
        storage=types.ModuleType('satellite_rl.workbench.storage');storage.digest=lambda path:'mock'
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,{'satellite_rl.workbench':workbench,'satellite_rl.workbench.storage':storage}):
            target=Path(directory);(target/'request.json').write_text('{}')
            install_node_hooks({'original_evidence_refs':[]},{'sha256':'contract'},{'sha256':'ops'},target)
            node.guarded(target,{},15,lambda:False)
            timer=json.loads((target/'recovery_parent_timer.json').read_bytes())
            self.assertEqual(timer['new_scientific_calls'],0);self.assertTrue(timer['reused_original_call'])
    def test_original_guard_sees_actual_composite_worker_from_status_without_old_launch(self):
        # Exact root guard loaded for its pure tempfile API; ALL write paths supplied below.
        guard_path=Path('D:/paper/outputs/experiments/first_stage_ai_search_20261013/round7/operations/deadline_0700/deadline_guard.py')
        spec=importlib.util.spec_from_file_location('test_fixture_only_deadline_guard',guard_path);guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=root/'archive';jobdir=archive/'outputs/workbench/jobs'/contracts.RUN_ID;jobdir.mkdir(parents=True)
            contracts.atomic(jobdir/'job.json',dict(experiment_id='S1-AI17-R7',group='S1-AI17-R7',executor_port=20474,archive_root=str(archive),run_id=contracts.RUN_ID,mode='node2000',scientific_commit=contracts.SCIENCE))
            contracts.atomic(jobdir/'status.json',dict(stage='running',pid=123,command=['old original argv']))
            actual=dict(pid=123,birth_ticks=456,alive=True,command='python -m scripts.stage1_ai_search_r7_node_recovery.bootstrap')
            fixture_spec=dict(guard_root=str(root/'guard'),global_pause_path=str(root/'global_pause.json'),ssh_port=20474)
            with patch.object(guard,'validate'),patch.object(guard,'registrations',return_value=[dict(archive_root=str(archive),experiment_id='S1-AI17-R7',modes=['node2000'])]):
                guard.apply_due(fixture_spec,wall=guard.DEADLINE_UNIX+1,identity=lambda pid:actual)
            intents=json.loads((root/'guard/stop_intents.json').read_bytes());self.assertEqual(intents[str(jobdir)]['original_worker_identity'],actual)
            self.assertTrue((jobdir/'stop.request').exists());self.assertTrue((root/'global_pause.json').exists())

if __name__=='__main__':unittest.main()
