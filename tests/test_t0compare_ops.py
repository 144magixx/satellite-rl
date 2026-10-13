import json,unittest,tempfile,random
from pathlib import Path
import numpy as np,torch
from satellite_rl.workbench.t0compare_contract import training_config,validate_t0_config,reject_third_stage,NODES
from satellite_rl.workbench import t0compare_checkpoints as checkpoints
from satellite_rl.t0_comparison.algorithms import build_agent
REFERENCE=Path(__file__).parent/'fixtures/t0_original_config.json'
class OpsContractTests(unittest.TestCase):
 def test_three_actual_masked_configs_and_shapes(self):
  reference=json.loads(REFERENCE.read_text(encoding='utf8'))
  for algorithm in ('sac','ppo','dqn'):
   config=training_config(reference,algorithm);self.assertEqual(validate_t0_config(config,reference)['algorithm'],algorithm);self.assertEqual(checkpoints._dimensions(config),(207,[100,10,10]))
  self.assertEqual(len(NODES),18)
 def test_third_phase_rejected_everywhere(self):
  for payload in ({'mode':'refine'},{'refinement':{}},{'matrix':{'full_B':True}},{'solver':'sca_si_lp'}):
   with self.assertRaises(ValueError):reject_third_stage(payload)
 def test_namespace_checkpoint_global_rng_and_empty_agent_roundtrip(self):
  reference=json.loads(REFERENCE.read_text(encoding='utf8'));config=training_config(reference,'ppo');agent=build_agent(config,207,(100,10,10),'cpu');random.seed(53);np.random.seed(53);torch.manual_seed(53)
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'checkpoint.pt';checkpoints.save_checkpoint(p,agent,{'env_step':0,'episode':0},config,{'scene_order':{'fixture':True}});expected=(random.random(),float(np.random.rand()),torch.rand(3));other=build_agent(config,207,(100,10,10),'cpu');progress,env=checkpoints.load_checkpoint(p,other,config);actual=(random.random(),float(np.random.rand()),torch.rand(3));self.assertEqual(expected[:2],actual[:2]);self.assertTrue(torch.equal(expected[2],actual[2]));self.assertEqual(progress['env_step'],0);self.assertTrue(env['scene_order']['fixture'])
if __name__=='__main__':unittest.main()
class MatrixIntegrationTests(unittest.TestCase):
 def test_full_560_mock_direct_load_wrapper_rng_isolation_resume(self):
  from unittest.mock import patch
  from types import SimpleNamespace
  from satellite_rl.workbench import t0compare_runner as runner
  reference=json.loads(REFERENCE.read_text(encoding='utf8'));config=training_config(reference,'sac');calls=[];policies={name:SimpleNamespace(agent=SimpleNamespace(rng=random.Random(100)),metadata={'provenance':{'env_step':150000}}) for name in ('sac','ppo','dqn')};before={k:v.agent.rng.getstate() for k,v in policies.items()}
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);job={'run_id':'fixture_matrix','t0_mode':'matrix','request_body':{'mode':'matrix'},'scene_sets':{'test':{}},'endpoint_models':{k:{'model_path':k,'model_files':[]} for k in policies}};(d/'job.json').write_text(json.dumps(job));(d/'config.json').write_text(json.dumps(config));rng_before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
   def episode(env,policy,scene,identity,seed,action_mode,**kw):
    calls.append((identity,seed,action_mode));random.random();np.random.rand();torch.rand(1)
    if hasattr(policy,'agent'):policy.agent.rng.random()
    return {'metrics':{'fixture':0.},'beams':[],'trace':[]}
   with patch.object(runner,'scene_set_files',return_value=[(Path('fixture'+str(i)),str(i)) for i in range(40)]),patch.object(runner,'_load_scene',return_value={}),patch.object(runner,'make_env',return_value=SimpleNamespace()),patch.object(runner,'load_policy',side_effect=lambda path,device:policies[path]),patch.object(runner,'_evaluation_episode',side_effect=episode):
    runner.run_matrix(d);self.assertEqual(len(calls),560);self.assertEqual(json.loads((d/'status.json').read_text())['stage'],'completed');runner.run_matrix(d);self.assertEqual(len(calls),560)
   self.assertEqual(random.getstate(),rng_before[0]);self.assertTrue(torch.equal(torch.get_rng_state(),rng_before[2]));self.assertTrue(np.array_equal(np.random.get_state()[1],rng_before[1][1]))
   for k,v in policies.items():self.assertEqual(v.agent.rng.getstate(),before[k])
class DataHoldTests(unittest.TestCase):
 def test_no_task_publication_without_uniform_or_historical_authority(self):
  from satellite_rl.workbench.t0compare_service import Workbench
  app=Workbench.__new__(Workbench)
  with self.assertRaisesRegex(ValueError,'source/data hold'):app.submit({'mode':'train','algorithm':'sac','request_id':'s2t24-sac','scene_sets':{}})
 def test_policy_metadata_actual_algorithm_and_modules(self):
  reference=json.loads(REFERENCE.read_text(encoding='utf8'));config=training_config(reference,'ppo');agent=build_agent(config,207,(100,10,10),'cpu')
  with tempfile.TemporaryDirectory() as temp:
   checkpoints.export_policy(temp,agent,config,{'env_step':0,'fixture':'no training'});metadata=json.loads((Path(temp)/'model.json').read_text(encoding='utf8'));loaded=checkpoints.load_policy(temp,device='cpu');self.assertEqual(metadata['algorithm'],'ppo');self.assertEqual(loaded.agent.algorithm,'ppo');self.assertEqual(metadata['parameter_count'],sum(p.numel() for p in agent.policy_network().parameters()));self.assertEqual(metadata['parameter_count_scope'],'actual_exported_inference_policy')
