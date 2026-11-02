"""Necessary08 deadline/stage/unknown-kernel tests; no real process or geometry."""
import tempfile
from pathlib import Path
import unittest
from .runtime import ObjectStore,control_scope,exclusive,BudgetLedger,json_bytes
from .lane import run_child
from .test_lane import FakeGuard,FakeChild
from .pipeline import stage_plan
from .stage_admission import verify_stage
from .dispatcher import make_request
from .bindings import read_json
from .kernel_persistence import call_saved

class EngineeringGateTests(unittest.TestCase):
    def test_expired_preparation_never_popen(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TimeoutError):run_child(['NO_REAL_PROCESS'],{'outer_seconds':120},Path(d)/'task',FakeGuard(),
                started=0.,clock=lambda:108.,identify=lambda p:{'pid':p},
                popen=lambda *a,**k:self.fail('expired preparation launched child'))

    def test_parent_control_interrupts_lock_wait_preserves_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'lock';p.write_text('unknown');ticks=[107.]
            def check():
                if ticks[0]>=108:raise TimeoutError('own stop-new boundary')
            def sleep(_):ticks[0]+=.5
            with self.assertRaises(TimeoutError),control_scope(check):
                with exclusive(p,clock=lambda:ticks[0],sleeper=sleep):pass
            self.assertEqual(p.read_text(),'unknown')

    def test_stable_stage_no_reference_cycle_and_only_original8(self):
        with tempfile.TemporaryDirectory() as d:
            store=ObjectStore(d);asset=store.put_json({'fake_scalar':1});tasks=stage_plan()['engineering_tasks']
            stage={'schema':'R14StableStageAdmissionV1','accepted':True,'group':'S1-AI17-R14','engineering_only':True,
                'permitted_ports':[20473],'allowed_tasks':tasks,'source_manifest_ref':asset,'runtime_epoch_ref':asset,
                'input_bindings_by_task':{t['task_id']:{'fixture_definition_ref':asset} for t in tasks}}
            stage_ref=store.put_json(stage)
            fields=('scientific_execution_authorized','root_exact_source_input_admitted','source_specific_pure_checks_accepted',
              'actual_runtime_owner_stop_space_admitted','runtime_epoch_ref','port','retry','source_root','source_refs','source_manifest_ref',
              'private_dependency_paths','private_dependency_manifest_ref','shared_object_root','shared_audit_ledger',
              'shared_input_ledger_root','shared_fit_intent_root','own_stop_marker','global_stop_marker')
            plan={k:False for k in fields};plan.update(port=20473,retry=0,source_root='/fake/source',source_refs=[],
                source_manifest_ref=asset,runtime_epoch_ref=asset,child_stage_admission_ref=stage_ref,
                output_root='/fake/output',control_root='/fake/control',root_admission_ref={'fake_final_review':1})
            request=make_request(plan,tasks[0],inputs={'fixture_definition_ref':asset},dependencies={})
            self.assertTrue(verify_stage(request));plan['root_admission_ref']={'fake_final_review':2}
            self.assertEqual(json_bytes(request),json_bytes(make_request(plan,tasks[0],inputs={'fixture_definition_ref':asset},dependencies={})))
            with self.assertRaises(ValueError):verify_stage({**request,'kind':'data'})
            with self.assertRaises(ValueError):verify_stage({**request,'task_id':'engineering:8'})

    def test_existing_partial_kernel_is_saved_on_fake_throw(self):
        class Failed(RuntimeError):pass
        error=Failed('fake');error.partial_kernel={'raw':[1,2],'complete':False};error.partial_ledger={'flows':1}
        def fail(*args):raise error
        with tempfile.TemporaryDirectory() as d:
            store=ObjectStore(Path(d)/'objects');out=Path(d)/'output';out.mkdir();refs=[]
            with self.assertRaises(Failed):call_saved(fail,(),store,out,refs)
            evidence=read_json(refs[0]);self.assertEqual(evidence['partial_kernel'],error.partial_kernel)
            self.assertEqual(evidence['partial_ledger'],error.partial_ledger)

    def test_nested_audit_cannot_shrink_hard_outer_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):BudgetLedger(d).reserve('fake','trajectory',120,{'audit':7})

    def test_group_deadline_caps_child_and_cleanup(self):
        ticks=[95.];child=FakeChild(ticks,999);env=[]
        class GroupGuard:
            started=0.;seconds=110.
            def check(self):
                if ticks[0]>=99:raise RuntimeError('group stop-new')
        def popen(*a,**kw):env.append(kw['env']);return child
        def kill(*a,**kw):child.killed=True;return 0
        with tempfile.TemporaryDirectory() as d:
            facts=run_child(['FAKE'],{'outer_seconds':120},Path(d)/'task',GroupGuard(),started=95.,
                clock=lambda:ticks[0],sleeper=lambda _:ticks.__setitem__(0,ticks[0]+1),popen=popen,
                identify=lambda pid:{'pid':pid,'process_start_identity':'FAKE'},terminate=kill,sample=lambda *a:None)
            self.assertLessEqual(ticks[0],110);self.assertEqual(facts['effective_hard_deadline_monotonic'],110)
            self.assertEqual(float(env[0]['R14_REQUEST_DEADLINE_MONOTONIC']),110)
            self.assertEqual(float(env[0]['R14_REQUEST_STOP_NEW_MONOTONIC']),99)

if __name__=='__main__':unittest.main()
