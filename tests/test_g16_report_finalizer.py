import json
import tempfile
import unittest
from pathlib import Path
from scripts.stage1_geoload.finalizer import run,save,read,identity,exclusive,wait_until_terminal
from scripts.stage1_geoload.strict_audit import digest,AuditError


class FinalizerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.inputs={}
        values=dict(plan=dict(group='S1-G16',scientific_commit='a'*40),
            state=dict(group='S1-G16',compute_and_archives_complete=True,gate=dict(passed=False),
                tasks=dict(common_test_20424=dict(stage='completed',run_id='r',port=20424)),archives=dict(common_test_20424=dict(dummy=True))),
            registry=dict(resources=[]),controller=dict(pid=999999,process_start_time=1))
        for name,value in values.items():
            path=self.root/(name+'.json');save(path,value);self.inputs[name]=path
        value=read(self.inputs['state']);value['plan_sha256']=digest(self.inputs['plan']);save(self.inputs['state'],value)
        self.hooks=dict(controller=lambda _:dict(status='original_controller_exited'),
            closure=lambda *args:dict(status='accepted',issues=[]),report=self.report,publication=self.publication)

    def tearDown(self):self.temp.cleanup()
    def report(self,plan,state,out,closure):
        Path(out).mkdir();value=dict(group='S1-G16',status='formal_ready',formal_ready=True,issues=[])
        save(Path(out)/'summary.json',value);return value
    def publication(self,report,out):
        Path(out).mkdir();save(Path(out)/'manifest.json',dict(synthetic=False));return dict(synthetic=False)

    def test_gatefail_finishes_once_and_real_identity(self):
        value=run(self.inputs,self.root/'post',hooks=self.hooks)
        self.assertEqual(value['status'],'complete');self.assertEqual(len(value['steps']),5)
        self.assertEqual(value['learning_branch'],'science_stopped_teacher_gate_failed')
        self.assertTrue(value['analysis_verified']);self.assertFalse(value['delivery_verified'])
        self.assertEqual(read(self.root/'post'/'lease.json')['lock'],'released')
        self.assertEqual(value['process']['pid'],identity()['pid'])
        self.assertEqual(run(self.inputs,self.root/'post',hooks=self.hooks)['status'],'complete')

    def test_rejected_closure_no_report_and_no_retry(self):
        self.hooks['closure']=lambda *args:dict(status='not_ready',issues=['missing'])
        first=run(self.inputs,self.root/'post',hooks=self.hooks)
        self.assertEqual(first['status'],'needs_attention');self.assertFalse((self.root/'post'/'report').exists())
        second=run(self.inputs,self.root/'post',hooks=self.hooks)
        self.assertEqual(second['history'][-1]['event'],'automatic_retry_refused')

    def test_changed_source_invalidates_committed_snapshot(self):
        run(self.inputs,self.root/'post',hooks=self.hooks)
        save(self.inputs['registry'],dict(resources=['changed']))
        value=run(self.inputs,self.root/'post',hooks=self.hooks)
        self.assertEqual(value['status'],'needs_attention');self.assertIn('input changed',value['error'])

    def test_gatepass_actual_receipts_required(self):
        state=read(self.inputs['state']);state['gate']['passed']=True;save(self.inputs['state'],state)
        value=run(self.inputs,self.root/'post',hooks=self.hooks)
        self.assertEqual(value['status'],'needs_attention');self.assertIn('receipts missing',value['error'])

    def test_interrupted_stage_not_repeated(self):
        out=self.root/'post';out.mkdir();save(out/'postfinalizer_state.json',dict(status='running',steps=dict(report=dict(status='running')),history=[]))
        value=run(self.inputs,out,hooks=self.hooks)
        self.assertEqual(value['status'],'needs_attention');self.assertIn('interrupted',value['error'])

    def test_kernel_lock_prevents_second_owner(self):
        path=self.root/'lock'
        with exclusive(path):
            with self.assertRaises(AuditError):
                with exclusive(path):pass

    def test_live_controller_blocks_snapshot(self):
        self.hooks['controller']=lambda _:dict(status='original_controller_alive')
        value=run(self.inputs,self.root/'post',hooks=self.hooks)
        self.assertEqual(value['status'],'needs_attention');self.assertIn('remains alive',value['error'])

    def test_bounded_wait_ready_and_unknown_timeout(self):
        self.assertTrue(wait_until_terminal(self.inputs,self.root/'wait',timeout=.01,poll_seconds=1,
            probe=lambda _:dict(status='original_controller_exited')))
        science=read(self.inputs['state']);science['compute_and_archives_complete']=False;save(self.inputs['state'],science)
        self.assertFalse(wait_until_terminal(self.inputs,self.root/'unknown',timeout=.001,poll_seconds=1,sleeper=lambda _:None))
        self.assertEqual(read(self.root/'unknown'/'wait_state.json')['status'],'needs_attention')


if __name__=='__main__':unittest.main()
