import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from satellite_rl.shared_interface.fixtures import object_recipe
from satellite_rl.shared_interface.contracts import Authority
from satellite_rl.shared_interface.stages import UniformStage2Input
from satellite_rl.shared_interface.spherical_gaussian import PHYSICS
from satellite_rl.t0_width.agent import WidthAgent,build_config
from satellite_rl.t0_width.environment import T0WidthEnv
from satellite_rl.t0_width.runner import run_task,evaluate_cases,rng_state,restore_rng,seed_all,RecoveryAccountingRequired
from satellite_rl.t0_width.logging import EvidenceLogs


def fixture(_):
    obj=object_recipe(dict(id='runner-fixture',N=5,expected={},widths_rule='[.3,.5,1.0][i%3]'))
    return UniformStage2Input(obj['case'],obj['layout'],
                             Authority(obj['authority_registry'],obj['authority_registry_sha']),PHYSICS)


def config():
    value=build_config('S11','cpu','correctness')
    value['train_cases']=[{'case_id':'fixture','role':'train'}]
    value['training'].update(total_steps=6,batch_size=2,warmup_steps=2,checkpoint_interval=3,log_interval=1)
    return value


def read_rows(path):
    with gzip.open(path,'rt',encoding='utf8') as stream:return [json.loads(x) for x in stream]


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_mid_episode_resume_identical_next_action_update_and_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            uninterrupted=Path(td)/'whole';split=Path(td)/'split'
            a=run_task(config(),uninterrupted,loader=fixture,env_factory=T0WidthEnv)
            tracker={'step':0}
            def progress(value):tracker['step']=value['env_step']
            b=run_task(config(),split,progress,lambda:tracker['step']>=3,loader=fixture,env_factory=T0WidthEnv)
            self.assertEqual(b['stage'],'stopped');self.assertEqual(b['env_step'],3)
            b=run_task(config(),split,loader=fixture,env_factory=T0WidthEnv)
            self.assertEqual(a['update_step'],b['update_step'])
            x=torch.load(uninterrupted/'models/current.pt',weights_only=False)
            y=torch.load(split/'models/current.pt',weights_only=False)
            self.assertEqual(x['progress']['env_step'],6)
            for name in x['agent']['networks']:
                for key,value in x['agent']['networks'][name].items():
                    torch.testing.assert_close(value,y['agent']['networks'][name][key],rtol=0,atol=0)
            xr=read_rows(uninterrupted/'logs/steps.jsonl.gz');yr=read_rows(split/'logs/steps.jsonl.gz')
            self.assertEqual([r['requested_action'] for r in xr],[r['requested_action'] for r in yr])
            self.assertEqual([r['reward'] for r in xr],[r['reward'] for r in yr])
            self.assertNotIn('beam_results',xr[-1]);self.assertNotIn('layout',xr[-1])
            terminals=read_rows(split/'logs/episodes.jsonl.gz')
            self.assertEqual(len(terminals),2)
            self.assertEqual(len(terminals[-1]['beams']),5)
            self.assertTrue(terminals[-1]['budget_truncated'])
            self.assertEqual(len(list((split/'models').glob('*.pt'))),2)

    def test_post_checkpoint_evidence_never_deleted_or_free_replayed(self):
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td)
            run_task(config(),directory,loader=fixture,env_factory=T0WidthEnv)
            logs=EvidenceLogs(directory/'logs');logs.append('steps',{'unique_unknown_spend':True});logs.close()
            before=(directory/'logs/steps.jsonl.gz').read_bytes()
            result=run_task(config(),directory,loader=fixture,env_factory=T0WidthEnv)
            self.assertEqual(result['stage'],'needs_root_recovery_accounting')
            self.assertEqual(before,(directory/'logs/steps.jsonl.gz').read_bytes())
            self.assertTrue(result['unknown_spend_not_zero'])

    def test_evaluation_rng_isolated_and_same_repeat_reproducible(self):
        seed_all(51001);agent=WidthAgent(build_config('S00'))
        state=rng_state()
        rows,complete=evaluate_cases(agent,[{}],'S00',loader=fixture,env_factory=T0WidthEnv,repeats=1)
        after=rng_state()
        self.assertTrue(complete)
        self.assertEqual(state['python'],after['python'])
        np.testing.assert_array_equal(state['numpy'][1],after['numpy'][1])
        torch.testing.assert_close(state['torch'],after['torch'],rtol=0,atol=0)
        again,_=evaluate_cases(agent,[{}],'S00',loader=fixture,env_factory=T0WidthEnv,repeats=1)
        self.assertEqual(rows[0]['beams'],again[0]['beams'])
        self.assertEqual(agent.update_step,0)

    def test_failed_identity_and_short_engineering_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            run_task(config(),td,loader=fixture,env_factory=T0WidthEnv)
            changed=config();changed['group']='S00'
            with self.assertRaisesRegex(ValueError,'identity'):
                run_task(changed,td,loader=fixture,env_factory=T0WidthEnv)
            changed=build_config('S00','cpu','engineering');changed['training']['total_steps']=8;changed['train_cases']=[{}]
            with self.assertRaisesRegex(ValueError,'real T0 update'):
                run_task(changed,Path(td)/'invalid',loader=fixture,env_factory=T0WidthEnv)

    def test_evaluation_partial_wal_resumes_without_repeating_physical_steps(self):
        from satellite_rl.t0_width.runner import _atomic_torch
        def power_cut(path,value):
            _atomic_torch(path,value)
            if value.get('stage')=='committed' and value['environment']['beam_idx']==2:
                raise RuntimeError('simulated power interruption after durable WAL')
        seed_all(51001);agent=WidthAgent(build_config('S11'))
        with tempfile.TemporaryDirectory() as td:
            wal=Path(td)/'evaluation.pt'
            with patch('satellite_rl.t0_width.runner._atomic_torch',side_effect=power_cut):
                with self.assertRaisesRegex(RuntimeError,'simulated'):
                    evaluate_cases(agent,[{}],'S11',loader=fixture,env_factory=T0WidthEnv,repeats=1,wal_path=wal)
            self.assertEqual(torch.load(wal,weights_only=False)['environment']['beam_idx'],2)
            resumed,_=evaluate_cases(agent,[{}],'S11',loader=fixture,env_factory=T0WidthEnv,repeats=1,wal_path=wal)
            expected,_=evaluate_cases(agent,[{}],'S11',loader=fixture,env_factory=T0WidthEnv,repeats=1)
            self.assertEqual(resumed[0]['beams'],expected[0]['beams'])
            self.assertEqual(resumed[0]['allocation_SHA'],expected[0]['allocation_SHA'])
            self.assertFalse(wal.exists())

    def test_evaluation_unknown_intent_never_replayed(self):
        class CrashEnvironment(T0WidthEnv):
            def step(self,action):raise RuntimeError('unknown physical execution')
        seed_all(51001);agent=WidthAgent(build_config('S00'))
        with tempfile.TemporaryDirectory() as td:
            wal=Path(td)/'evaluation.pt'
            with self.assertRaisesRegex(RuntimeError,'unknown physical'):
                evaluate_cases(agent,[{}],'S00',loader=fixture,env_factory=CrashEnvironment,repeats=1,wal_path=wal)
            before=wal.read_bytes()
            with self.assertRaises(RecoveryAccountingRequired):
                evaluate_cases(agent,[{}],'S00',loader=fixture,env_factory=T0WidthEnv,repeats=1,wal_path=wal)
            self.assertEqual(before,wal.read_bytes())


if __name__=='__main__':unittest.main()
