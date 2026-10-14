import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from satellite_rl.tsac_search27.agent import build_config
from satellite_rl.tsac_search27.runner import run_task
from satellite_rl.tsac_search27.workbench import validate_task
from test_t0_width_runner import fixture
from satellite_rl.t0_width.environment import T0WidthEnv

class SearchRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)
    def config(self):
        c=build_config('B','cpu','correctness'); c['train_cases']=[{'role':'train'}]
        c['training'].update(total_steps=6,batch_size=2,warmup_steps=2,checkpoint_interval=3)
        return c
    def test_exact_mid_episode_resume_no_replay(self):
        c=self.config()
        with tempfile.TemporaryDirectory() as temp:
            whole=Path(temp)/'whole'; split=Path(temp)/'split'; n={'step':0}
            a=run_task(c,whole,loader=fixture,env_factory=T0WidthEnv)
            run_task(c,split,lambda v:n.update(step=v['env_step']),lambda: (split/'models/resume_index.json').exists() and __import__('json').loads((split/'models/resume_index.json').read_text())['env_step']>=3,loader=fixture,env_factory=T0WidthEnv)
            b=run_task(c,split,loader=fixture,env_factory=T0WidthEnv)
            self.assertEqual(a['env_step'],6); self.assertEqual(b['env_step'],6)
            x=torch.load(whole/'models/current.pt',weights_only=False); y=torch.load(split/'models/current.pt',weights_only=False)
            self.assertEqual(x['progress']['position'],y['progress']['position'])
            for k,v in x['agent']['networks']['actor'].items(): torch.testing.assert_close(v,y['agent']['networks']['actor'][k],rtol=0,atol=0)
    def test_pilot_node_develops_before_next_training_step(self):
        c=self.config(); c['phase']='scientific'; c['training'].update(pilot_stop_node=3)
        c['dev_cases']=[]
        from satellite_rl.tsac_search27.agent import SearchAgent
        def tiny_agent(config):
            return SearchAgent(self.config())
        with tempfile.TemporaryDirectory() as temp, patch('satellite_rl.tsac_search27.runner.SearchAgent',tiny_agent), patch('satellite_rl.tsac_search27.runner._validate_task'), patch('satellite_rl.tsac_search27.runner.DEVELOPMENT_NODES',(0,3,6)):
            result=run_task(c,temp,loader=fixture,env_factory=T0WidthEnv)
            checkpoint=torch.load(Path(temp)/'models/current.pt',weights_only=False)
            self.assertEqual(result['stage'],'pilot_complete'); self.assertEqual(result['env_step'],3)
            self.assertIn(3,checkpoint['progress']['completed_nodes']); self.assertFalse(checkpoint['progress'].get('completed',False))
    def test_dispatch_intent_prevents_double_start_and_human_epoch_keeps_config(self):
        import json, threading
        from satellite_rl.tsac_search27.workbench import Workbench
        from satellite_rl.t0_width.logging import atomic_json,content_sha
        from types import SimpleNamespace
        c=build_config('B','cpu','scientific'); c.update(rootpauseepoch='original',run_id='runB')
        with tempfile.TemporaryDirectory() as temp:
            w=Workbench.__new__(Workbench); w.root=Path(temp); (w.root/'runs').mkdir()
            w.plan={'source_root':temp,'tasks':{'B':{'config':c,'executor_port':20418,'mode':'science'}}}
            w.port=20418; w.plan_sha='plan'; w.plan_path=Path(temp)/'plan.json'
            w.gate=SimpleNamespace(directory=Path(temp)/'control'); w.lock=threading.RLock()
            observed=[]; w.admission=lambda task,epoch,ref: observed.append((epoch,ref))
            with patch('satellite_rl.tsac_search27.workbench.subprocess.Popen',return_value=SimpleNamespace(pid=321)) as launch:
                w.start('B')
                self.assertEqual(launch.call_args.args[0][3],'satellite_rl.tsac_search27.workbench')
                with self.assertRaises(ValueError): w.start('B')
                d=w.root/'runs/B'; before=content_sha(json.loads((d/'config.json').read_text()))
                atomic_json(d/'dispatch_intent.json',{'stage':'registered'})
                atomic_json(d/'status.json',{'stage':'stopped','env_step':3})
                w.start('B',dispatch_epoch='human-new',resource_ref={'path':'fresh','sha256':'sha'})
                self.assertEqual(before,content_sha(json.loads((d/'config.json').read_text())))
                self.assertEqual(observed[-1],('human-new',{'path':'fresh','sha256':'sha'}))
                self.assertEqual(json.loads((d/'job.json').read_text())['config_SHA'],before)

    def test_mode_budget_and_selector_rejected(self):
        c=build_config('E','cpu','scientific')
        validate_task(dict(config=c,executor_port=20627,mode='science'))
        for change in [dict(total_steps=50000),dict(seed=73027),dict(pilot_stop_node=50001)]:
            bad=dict(c,training=dict(c['training'],**change))
            with self.assertRaises(ValueError): validate_task(dict(config=bad,executor_port=20627,mode='science'))
        with self.assertRaises(ValueError): validate_task(dict(config=c,executor_port=20473,mode='science'))

if __name__=='__main__': unittest.main()
