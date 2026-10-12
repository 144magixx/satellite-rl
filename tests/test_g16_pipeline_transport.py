"""Bridge tests use synthetic modules/contracts; never install into live pipeline."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from scripts.stage1_geoload import pipeline_transport as bridge


class BridgeTests(unittest.TestCase):
    def test_only_export_assignment_other_functions_keep_identity(self):
        def export(*args):pass
        def submit(*args):pass
        def tick(*args):pass
        def run(*args):pass
        module=SimpleNamespace(export_job=export,submit=submit,tick=tick,run=run,ops=object())
        before=dict(vars(module))
        with tempfile.TemporaryDirectory() as temp:
            old=bridge.install(module,SimpleNamespace(),temp)
            self.assertIs(old,export)
            for name,value in before.items():
                if name!='export_job':self.assertIs(getattr(module,name),value)
            self.assertIsNot(module.export_job,export)
            with self.assertRaises(ValueError):bridge.install(module,SimpleNamespace(),temp)

    def test_adapter_pins_existing_receipts_and_4mib(self):
        calls=[]
        def one_export(*args,**kwargs):calls.append((args,kwargs));return {'verified':True}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dest=root/'canonical.zip';pin=dest.with_suffix('.remote.receipt.json');pin.write_text('{}')
            adapter=bridge.adapter(SimpleNamespace(one_export=one_export),root/'journal')
            adapter(20424,'fixture',dest)
            self.assertEqual(calls[0][1]['window'],4 << 20)
            self.assertEqual(calls[0][1]['existing_receipt'],pin)
            self.assertTrue(calls[0][1]['ownership_confirmed'])

    def test_unknown_failure_sticks_without_20sec_remote_retry(self):
        calls=[]
        def one_export(*args,**kwargs):calls.append(args);raise ValueError('unknown packer')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);adapter=bridge.adapter(SimpleNamespace(one_export=one_export),root/'journal')
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError,'needs_attention'):adapter(20473,'fixture',root/'canonical.zip')
            self.assertEqual(len(calls),1)
            self.assertFalse(bridge.read(root/'journal'/'20473_fixture.json')['remote_packer_retry_allowed'])

    def test_existing_zip_without_pin_never_repacks(self):
        calls=[]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dest=root/'canonical.zip';dest.write_bytes(b'unknown')
            adapter=bridge.adapter(SimpleNamespace(one_export=lambda *a,**k:calls.append(a)),root/'journal')
            with self.assertRaisesRegex(RuntimeError,'no exact receipt'):adapter(20424,'fixture',dest)
            self.assertEqual(calls,[])

    def test_guard_source_and_plan_failure_before_process_or_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in bridge.ORIGINAL:(root/name).write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError,'source SHA'):bridge.guard(root/'plan',root/'state',root/'lease',directory=root,old_alive=lambda *a:False)
            with patch.object(bridge,'ORIGINAL',{}),patch.object(bridge,'TRANSPORT_SHA',bridge.sha(root/'pipeline.py')):
                (root/'archive_transport.py').write_bytes(b'wrong');(root/'plan').write_bytes(b'wrong')
                with self.assertRaisesRegex(ValueError,'plan SHA'):bridge.guard(root/'plan',root/'state',root/'lease',directory=root,old_alive=lambda *a:False)

    def test_full_synthetic_lease_and_old_alive_or_archival_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'source').mkdir()
            (root/'archive_transport.py').write_bytes(b'synthetic transport')
            (root/'source'/'science_plan.json').write_bytes(b'synthetic science')
            science_sha=bridge.sha(root/'source'/'science_plan.json')
            plan=root/'plan.json';bridge.save(plan,dict(group='S1-G16',authorized=True,scientific_commit=bridge.SCIENCE,
                science_plan_refs={str(p):dict(sha256=science_sha) for p in (20424,20473)}))
            state=root/'state.json';bridge.save(state,dict(group='S1-G16',scientific_commit=bridge.SCIENCE,
                plan_sha256=bridge.sha(plan),tasks={'train_mlm':{'stage':'training','kind':'train',
                    'port':20473,'run_id':'train_geoload_fixture','request_id':'g16_fixture_train_mlm',
                    'status':{'stage':'training','run_id':'train_geoload_fixture','pid':12345,
                        'update_step':2500,'budget_updates':10000}}}))
            lease=root/'lease.json';bridge.save(lease,dict(schema='g16_transport_handoff_v1',status='handoff_verified',group='S1-G16',
                old_controller_pid=24220,old_controller_birth=1791790345.992857,old_controller_exit_verified=True,
                state_lock_release_verified=True,legacy_export_excluded=True,controller_commit=bridge.CONTROLLER,
                scientific_commit=bridge.SCIENCE,plan_sha256=bridge.sha(plan),state_sha256=bridge.sha(state),state_path=str(state),
                bridge_sha256=bridge.sha(bridge.__file__),transport_sha256=bridge.sha(root/'archive_transport.py')))
            with patch.object(bridge,'ORIGINAL',{}),patch.object(bridge,'TRANSPORT_SHA',bridge.sha(root/'archive_transport.py')),patch.object(bridge,'PLAN_SHA',bridge.sha(plan)),patch.object(bridge,'SCIENCE_PLAN_SHA',science_sha):
                value=bridge.guard(plan,state,lease,directory=root,old_alive=lambda *a:False)
                self.assertEqual(value['only_replacement'],'pipeline.export_job')
                with self.assertRaisesRegex(ValueError,'still alive'):bridge.guard(plan,state,lease,directory=root,old_alive=lambda *a:True)
                good=bridge.read(state)
                for bad_stage in ('queued','failed','starting','unknown'):
                    bad=json.loads(json.dumps(good));bad['tasks']['train_mlm']['stage']=bad_stage
                    bad['tasks']['train_mlm']['status']['stage']=bad_stage;bridge.save(state,bad)
                    with self.assertRaisesRegex(ValueError,'stage must'):bridge.guard(plan,state,lease,directory=root,old_alive=lambda *a:False)
                bad=json.loads(json.dumps(good));bad['tasks']['train_mlm']['status'].pop('pid');bridge.save(state,bad)
                with self.assertRaisesRegex(ValueError,'PID evidence'):bridge.guard(plan,state,lease,directory=root,old_alive=lambda *a:False)
                bridge.save(state,good)
                data=bridge.read(state);data['archives']={'done':{}};bridge.save(state,data)
                with self.assertRaisesRegex(ValueError,'archival phase'):bridge.guard(plan,state,lease,directory=root,old_alive=lambda *a:False)


if __name__=='__main__':unittest.main()
