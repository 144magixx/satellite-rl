"""Pure control tests: no PHY, neural forward, update or remote operations."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from satellite_rl.power30.ledger import CallLedger, initialize, reconcile_unknown


class PermanentLedgerTest(unittest.TestCase):
    def test_unknown_reconcile_resume_close_never_refunds_burned_calls_or_time(self):
        import hashlib
        with tempfile.TemporaryDirectory() as root:
            initialize(root,{})
            old=CallLedger(root,'original','engineering',block=4)
            old.charge('phy');old.flush()
            path=Path(root)/'workers'/'original.json'
            original=json.loads(path.read_text(encoding='utf8'))
            reserved_time=original['reserved_seconds']['engineering']
            receipt=dict(schema='W30UnknownSpendReconciliationV1',experiment_id='S2-W30',
                         run_id='original',root_approved=True,independent_review_passed=True,
                         old_process_exit_verified=True,worker_ledger_SHA=hashlib.sha256(path.read_bytes()).hexdigest(),
                         outer_process_seconds_by_phase={'engineering':reserved_time+10})
            rpath=Path(root)/'root_receipt.json';rpath.write_text(json.dumps(receipt),encoding='utf8')
            reconcile_unknown(root,'original',{'path':str(rpath),'sha256':hashlib.sha256(rpath.read_bytes()).hexdigest()})
            resumed=CallLedger(root,'original','engineering',block=4)
            self.assertEqual(resumed.snapshot()['burned_unknown']['phy'],3)
            resumed.charge('phy');resumed.close()
            budget=json.loads((Path(root)/'budget.json').read_text(encoding='utf8'))
            value=resumed.snapshot()
            self.assertEqual(budget['charged']['phy'],5)
            self.assertGreaterEqual(budget['charged_seconds']['engineering'],reserved_time+10)
            self.assertAlmostEqual(budget['charged_seconds']['engineering'],value['seconds']['engineering']+value['burned_unknown_seconds']['engineering'])
            from satellite_rl.power30.control import RuntimeControl
            value['seconds']['engineering']=0.
            value['burned_unknown_seconds']['engineering']=8*3600
            path.write_text(json.dumps(value),encoding='utf8')
            control=RuntimeControl(dict(run_id='original',phase='engineering',
                ledger_root=root,pause_markers=[str(Path(root)/'absent_pause')],
                deadline_utc='2099-01-01T00:00:00+00:00',reserve_free_bytes=0),root)
            self.assertTrue(control.stop_requested(force=True))
            self.assertEqual(control.reason,'engineering_group_time_limit')
    def test_extra_evidence_suffix_is_never_truncated(self):
        from satellite_rl.power30.logging import EvidenceLogs
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'steps.jsonl.gz';p.write_bytes(b'preserve-unique-tail')
            with self.assertRaises(ValueError):
                EvidenceLogs(root,{'steps':3})
            self.assertEqual(p.read_bytes(),b'preserve-unique-tail')
    def test_unknown_exit_keeps_reservations_and_blocks_free_resume(self):
        with tempfile.TemporaryDirectory() as root:
            initialize(root, {'path':'authority','sha256':'test'})
            ledger=CallLedger(root,'run-a','engineering',block=4)
            ledger.charge('phy')
            ledger.flush()
            budget=json.loads((Path(root)/'budget.json').read_text())
            self.assertEqual(budget['charged']['phy'],4)
            with self.assertRaises(RuntimeError):
                CallLedger(root,'run-a','engineering')
            with self.assertRaises(FileExistsError):
                initialize(root,{})

    def test_clean_close_releases_only_proven_unentered_calls(self):
        with tempfile.TemporaryDirectory() as root:
            initialize(root,{})
            ledger=CallLedger(root,'run-a','engineering',block=4)
            ledger.charge('netforward',network='actor',batch_size=256)
            ledger.close()
            budget=json.loads((Path(root)/'budget.json').read_text())
            self.assertEqual(budget['charged']['netforward'],1)
            resumed=CallLedger(root,'run-a','engineering',block=4)
            self.assertEqual(resumed.snapshot()['actual']['netforward'],1)
            resumed.charge('netforward',network='critic1',batch_size=1)
            resumed.close()
            budget=json.loads((Path(root)/'budget.json').read_text())
            self.assertEqual(budget['charged']['netforward'],2)

    def test_independent_workers_share_one_limit_and_phase_time(self):
        with tempfile.TemporaryDirectory() as root:
            initialize(root,{})
            a=CallLedger(root,'a','engineering',block=4)
            b=CallLedger(root,'b','engineering',block=4)
            a.charge('phy'); b.charge('phy')
            a.set_phase('evaluation');a.charge('evaluation_episode')
            a.close();b.close()
            budget=json.loads((Path(root)/'budget.json').read_text())
            self.assertEqual(budget['charged']['phy'],2)
            self.assertEqual(budget['charged']['evaluation_episode'],1)
            self.assertIn('evaluation',a.snapshot()['seconds'])

    def test_mock_runner_clean_stop_resumes_same_id_without_replayed_step(self):
        from satellite_rl.power30 import runner
        class Agent:
            def __init__(self,*a,**kw):
                self.device=__import__('torch').device('cpu');self.update_step=0
                self.schema={};self.extension_rng={}
                self.actor=self.critic1=self.critic2=SimpleNamespace(parameters=lambda:[])
            def state_dict(self):return {'mock':True}
            def load_state_dict(self,s):pass
            def parameter_counts(self):return {}
            def act(self,*a,**k):return (0,0,0)
            def observe(self,*a,**k):pass
            def ready(self,*a):return False
        class Env:
            actual_steps=0
            def __init__(self,*a):self.stage_input=object();self.i=0
            def reset(self):return self._observation()
            def _observation(self):return np.array([self.i],dtype=np.float32)
            def action_mask(self):return {}
            def state_dict(self):return {'i':self.i}
            def load_state_dict(self,s):self.i=s['i']
            def step(self,a):
                Env.actual_steps+=1;self.i+=1
                return self._observation(),0.,self.i==3,{'actual_action':a}
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root)/'run';ledger_root=Path(root)/'ledger'
            initialize(ledger_root,{})
            c=dict(phase='correctness',group='M0',menu='P10',run_id='mock',
                   train_cases=[{}],dev_cases=[],source_sha='mock',ledger_root=str(ledger_root),
                   training=dict(seed=51001,scene_order_seed=51001,total_steps=4,
                                 warmup_steps=2000,checkpoint_interval=2,log_interval=1,device='cpu'))
            with patch('satellite_rl.power30.runner.Power30Agent',Agent),patch('satellite_rl.power30.runner.RuntimeControl') as control,patch('satellite_rl.power30.runner._terminal',lambda *a,**k:k):
                control.return_value.stop_requested.return_value=False
                a=runner.run_task(c,directory,stop_requested=lambda:Env.actual_steps>=2,
                                  loader=lambda x:object(),env_factory=Env)
                self.assertEqual(a['stage'],'stopped')
                self.assertEqual(Env.actual_steps,2)
                b=runner.run_task(c,directory,loader=lambda x:object(),env_factory=Env)
                self.assertEqual(b['stage'],'completed')
                self.assertEqual(Env.actual_steps,4)
                self.assertEqual(b['calls']['actual']['scientific_interaction'],4)


if __name__=='__main__':
    unittest.main()
