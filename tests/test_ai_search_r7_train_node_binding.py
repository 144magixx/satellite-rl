"""Future phase pure bindings/typed-unavailable checks; no numeric or science calls."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.storage import digest
from satellite_rl.workbench.r7_train_binding import TRAIN_KEYS,validate_setup,validate_owner

def ref(name):return dict(path='/pure_mock/'+name,sha256='a'*64)
def setup_fixture():
    refs={k:ref(k) for k in TRAIN_KEYS};refs['train_setup_endpoint']=ref('setup')
    endpoint=dict(schema='R7TrainSetupEndpointV1',group='S1-AI17-R7',phase='train_setup',passed=True,optimizer_updates=0,
                  model_identity='one_r7',executor_port=20474,case_indices=[0],dataset_ref=ref('dataset'),training_refs={k:refs[k] for k in TRAIN_KEYS},
                  run_id='setup_original',runtime_identity={'fixed':'runtime'},charged_train_setup_seconds=3.,
                  activity_origin=dict(scope='entire_original_train_setup_only_once',setup_process_pid=1,setup_process_birth=2,supply_started_ref=ref('started')))
    endpoint['dataset_source_binding']={'pure_mock':'matching_original_source'}
    started=dict(qualified_ref=refs['qualified'],gate_ref=refs['labels_gate'],charged_to_training_activity=True)
    supply=dict(qualified_ref=refs['qualified'],normalization_ref=refs['normalization'],cache_ref=refs['feature_cache'],charged_supply_seconds_including_publication=2.)
    return refs,endpoint,started,supply

class SetupBindings(unittest.TestCase):
    def test_every_setup_ref_same_endpoint_and_clock(self):
        refs,endpoint,started,supply=setup_fixture()
        with patch('satellite_rl.workbench.r7_training_dataset.validate_training_dataset',return_value=endpoint['dataset_source_binding']),patch('satellite_rl.workbench.r7_train_binding.pin',side_effect=lambda r:Path(r['path'])),patch('satellite_rl.workbench.r7_train_binding.read_json',side_effect=lambda p,*a:started if str(p).endswith('started') else supply):
            self.assertEqual(validate_setup(endpoint,refs['train_setup_endpoint'],refs,'one_r7',20474,[0],ref('dataset')),3.)
            for key in TRAIN_KEYS:
                changed=copy.deepcopy(refs);changed[key]=ref('different_'+key)
                with self.subTest(key=key),self.assertRaises(ValueError):validate_setup(endpoint,refs['train_setup_endpoint'],changed,'one_r7',20474,[0],ref('dataset'))
    def test_supply_clock_and_original_process_cannot_reset(self):
        refs,endpoint,started,supply=setup_fixture()
        with patch('satellite_rl.workbench.r7_training_dataset.validate_training_dataset',return_value=endpoint['dataset_source_binding']),patch('satellite_rl.workbench.r7_train_binding.pin',side_effect=lambda r:Path(r['path'])),patch('satellite_rl.workbench.r7_train_binding.read_json',side_effect=lambda p,*a:started if str(p).endswith('started') else supply):
            endpoint['charged_train_setup_seconds']=1.
            with self.assertRaises(ValueError):validate_setup(endpoint,refs['train_setup_endpoint'],refs,'one_r7',20474,[0],ref('dataset'))
    def test_original_global_owner_required_and_all_refs_bound(self):
        refs,endpoint,_,_=setup_fixture()
        owner=dict(model_identity='one_r7',train_setup_endpoint_ref=refs['train_setup_endpoint'],supply_run_id='setup_original',supply_executor_port=20474,
                   setup_training_refs=endpoint['training_refs'],qualified_ref=refs['qualified'],setup_activity_origin=endpoint['activity_origin'],setup_runtime_identity=endpoint['runtime_identity'],setup_dataset_ref=endpoint['dataset_ref'],setup_case_indices=[0])
        self.assertTrue(validate_owner(owner,endpoint,refs['train_setup_endpoint']))
        with self.assertRaises(ValueError):validate_owner({},endpoint,refs['train_setup_endpoint'])
        for key in ('qualified_ref','setup_activity_origin','setup_training_refs','setup_runtime_identity','setup_dataset_ref','setup_case_indices','supply_run_id'):
            changed=copy.deepcopy(owner);changed[key]='changed'
            with self.subTest(key=key),self.assertRaises(ValueError):validate_owner(changed,endpoint,refs['train_setup_endpoint'])

class NodeUnavailableBindings(unittest.TestCase):
    def test_twenty_unavailable_cases_no_teacher_no_decode_and_gate_fails(self):
        import numpy as np
        import satellite_rl.workbench.r7_node as subject
        from satellite_rl.workbench.r7_node_audit import node_gate
        from satellite_rl.workbench.r7_node_guard_audit import audit_costs
        job=dict(mode='node2000',dataset_role='development',case_indices=list(range(20)),model_inputs={k:ref(k) for k in ('initialization_receipt','source_equivalence','checkpoint','normalization','training_endpoint','train_setup_endpoint')},
                 model_identity='one_r7',scientific_commit='a'*40,source_manifest_sha256='b'*64,experiment_id='S1-AI17-R7')
        cfg=dict(datasets={'development':dict(sha256='c'*64)})
        scene=SimpleNamespace(positions=np.array([[50.,8.]]),demands=np.array([21600000.]),aircraft_ids=('a',),profile='europe_v1')
        source=dict(case_ref=ref('case'));calls=[]
        def supply(job,cfg,i,target):return dict(total_request_bps_int64=21600000),scene,dict(identity={'role':'development','case_id':str(i)},scene_ref=ref('scene')),source
        def guarded(target,request,*a,**k):
            calls.append(request['stage']);return dict(actual_stage_wall_seconds=0.),dict(result=dict(status='unavailable',ledger={})),ref('prepared')
        sources={k:'sha' for k in ('model_graph_source_SHA','feature_graph_source_SHA','adapter_source_SHA','b7_decoder_source_SHA')}
        with tempfile.TemporaryDirectory() as tmp,patch('satellite_rl.workbench.r7_deadline.stop_boundary',return_value=True),patch('satellite_rl.workbench.r7_successor_binding.evaluation_binding',return_value=2000),patch.object(subject,'signed_json',return_value={'model_initial_state_SHA':'init'}),patch.object(subject,'verify_model_source_equivalence',return_value=sources),patch.object(subject,'supply_case',side_effect=supply),patch.object(subject,'guarded',side_effect=guarded),patch.object(subject,'stage_proof',return_value={'pure_mock':True}),patch.object(subject,'audit_preparation_teacher',return_value={'preparation':dict(closed=True,wall_seconds=0.)}):
            result=subject.node(job,cfg,Path(tmp));rows=[json.loads(Path(result['cases'][str(i)]['path']).read_text()) for i in range(20)]
        self.assertEqual(calls,['preparation']*20);self.assertEqual(result['new_calls'],dict(prepare=20,T=0,D5=0));self.assertFalse(node_gate(rows)['passed']);self.assertEqual(node_gate(rows)['denominator'],20)
        for row in rows:
            self.assertEqual(set(row['result_refs']),set(('AI','RULE','RANDOM','T')));self.assertTrue(all(v is None for v in row['result_refs'].values()));self.assertEqual(row['new_calls'],dict(prepare=1,T=0,D5=0))
    def test_unknown_teacher_guard_cannot_close(self):
        import satellite_rl.workbench.r7_node_guard_audit as subject
        row=dict(status='complete',stage_proofs={'preparation':{},'teacher':{}},identity={'role':'development'},scene_ref=ref('scene'),core_sha256='c'*64,producer_sha256='d'*64,source_dataset_sha256='e'*64,prepared_ref=ref('prepared'),result_refs={'T':ref('teacher')})
        with patch.object(subject,'audit_stage',side_effect=[dict(closed=True,wall_seconds=1.),dict(closed=False,wall_seconds=1.)]) as audit:
            proof=subject.audit_preparation_teacher(row)
        self.assertEqual(audit.call_count,2);self.assertFalse(proof['teacher']['closed']);self.assertEqual(audit.call_args.kwargs['phase_role'],'development')

if __name__=='__main__':unittest.main()
