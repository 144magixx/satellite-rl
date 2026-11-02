import json,unittest,tempfile,random,hashlib
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
   d=Path(tmp);job={'run_id':'fixture_matrix','t0_mode':'matrix','request_body':{'mode':'matrix'},'design_contract':{'baselines':{}},'scene_sets':{'test':{}},'endpoint_models':{k:{'model_path':k,'model_files':[],'model_run_id':'fixture_'+k} for k in policies}};(d/'job.json').write_text(json.dumps(job));(d/'config.json').write_text(json.dumps(config));rng_before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
   def episode(env,policy,scene,identity,seed,action_mode,**kw):
    calls.append((identity,seed,action_mode));random.random();np.random.rand();torch.rand(1)
    if hasattr(policy,'agent'):policy.agent.rng.random()
    from satellite_rl.workbench.environment import _sgm
    rates=np.zeros(100);demand=np.ones(100)*100.;sgm=_sgm(rates,demand,100.)
    beams=[{'beam_id':i,'original_index':i,'lat':45.,'lon':10.,'required_bps':100.,'rate_bps':0.,'group':i%8,'freq':i//8,'slots':1,'power_w':5.,'si':0.,'sgm':float(sgm[i])} for i in range(100)]
    trace=[{'requested_action':{'freq':i//8,'slots':1,'power_w':5.},'actual_action':{'freq':i//8,'slots':1,'power_w':5.}} for i in range(100)]
    return {'metrics':{'complete':True,'processed_count':100,'final_si_mean':0.,'final_sgm_mean':float(sgm.mean())},'beams':beams,'trace':trace}
   with patch.object(runner,'scene_set_files',return_value=[(Path('fixture'+str(i)),str(i)) for i in range(40)]),patch.object(runner,'_load_scene',return_value={'beam_location':np.tile([45.,10.],(100,1)),'rate':np.full((100,1),100.),'original_indices':np.arange(100)}),patch.object(runner,'make_env',return_value=SimpleNamespace()),patch.object(runner,'load_policy',side_effect=lambda path,device:policies[path]),patch.object(runner,'_evaluation_episode',side_effect=episode):
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
class ServiceLifecycleMockTests(unittest.TestCase):
 def test_resourceheld_submit_idempotent_queuedstop_and_worker_escape(self):
  from unittest.mock import patch
  from satellite_rl.workbench import t0compare_service as svc
  from satellite_rl.workbench import t0compare_runner as runner
  from satellite_rl.workbench.storage import Store
  from satellite_rl.training.run_monitor import atomic_json
  reference=json.loads(REFERENCE.read_text(encoding='utf8'))
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);authority=root/'authority.json';authority.write_text(json.dumps({'format':'s2t24_data_purpose_admission_v1','uniform_request_or_explicit_historical_reproduction_authority':True,'purpose':'explicit_historical_reproduction','final_variable_layout_pipeline_complete':False,'actual_source_or_human_authority_ref':{'fixture_only':True}}));body={'mode':'train','algorithm':'sac','request_id':'s2t24-sac','uniform_request_or_explicit_historical_reproduction_authority_ref':{'path':str(authority),'sha256':hashlib.sha256(authority.read_bytes()).hexdigest()},'scene_sets':{'train':{'count':576},'validation':{'count':20}}};snapshot=root/'snapshot';snapshot.mkdir()
   with patch.object(svc,'BASE',root),patch.object(Store,'freeze_source',return_value=(snapshot,'fixturecommit')) as freeze,patch.object(svc,'freeze_scene_set',side_effect=lambda record,target,profile:record):
    app=svc.Workbench(20418,root,reference,{},max_workers=2);a=app.submit(body);b=app.submit(body);self.assertEqual(a['job']['run_id'],b['job']['run_id']);self.assertEqual(freeze.call_count,1);self.assertEqual(a['status']['stage'],'queued_resources');self.assertTrue(a['job']['from_scratch']);self.assertEqual(app.stop(a['job']['run_id'])['status']['stage'],'cancelled')
    with self.assertRaises(ValueError):app.submit({**body,'mode':'refine'})
   wrong=root/'foreign';wrong.mkdir();(wrong/'job.json').write_text(json.dumps(a['job']))
   with patch.object(svc,'BASE',root):
    with self.assertRaises(ValueError):runner.validate_worker_identity(wrong)
class PrepareRPCMockTests(unittest.TestCase):
 def test_prepare_rpc_committed40_without_download_model_or_phy(self):
  from unittest.mock import patch
  from satellite_rl.workbench import t0compare_runner as runner
  from satellite_rl.workbench import t0compare_data as data
  with tempfile.TemporaryDirectory() as temp:
   d=Path(temp);(d/'job.json').write_text(json.dumps({'run_id':'fixture_prepare','data_config':{'experiment_id':'S2-T24','owner_run_id':'fixture_prepare'}}));(d/'data_preparation_admission.json').write_text(json.dumps({'approved':True,'fixture_only':True}));result={'status':'completed','scene_sets':{'test':{'id':'fixture','count':40,'path':str(d/'fixture'),'files':[]}}}
   with patch.object(data,'prepare_test_data',return_value=result) as prepare:runner.run_prepare(d)
   self.assertEqual(prepare.call_count,1);self.assertEqual(json.loads((d/'status.json').read_text())['data_scene_count'],40);self.assertIn('admission_reference',prepare.call_args.args[0]);self.assertEqual(prepare.call_args.args[0]['owner_run_id'],'fixture_prepare')
class FinalGatePureTests(unittest.TestCase):
 def test_training_has_no_terminal_test_and_passes_physical_action(self):
  from satellite_rl.workbench import t0compare_training as t
  import inspect
  source=inspect.getsource(t._training)
  self.assertNotIn('test_count',source);self.assertIn('no_internal_test',source);self.assertIn('info["actual_action"]',source)
 def test_bare_source_booleans_cannot_authorize(self):
  from satellite_rl.workbench.t0compare_qualification import qualify
  with self.assertRaisesRegex(ValueError,'nested actual'):qualify({}, {'source_qualified':True},'train')
 def test_controller_dependencies_and_actual_fresh_proof(self):
  from scripts.stage2_t0_fair24.controller import tick,resolve
  state={'tasks':{},'scientific_submissions':0};calls=[]
  plan={'tasks':{'prepare':{'executor_port':20418,'body':{'request_id':'prepare'},'admission_template_ref':{'path':'actual','sha256':'pin'}},'matrix':{'executor_port':20418,'depends_on':['prepare'],'body':{'request_id':'matrix'}}}}
  def api(port,path,body=None):
   calls.append(path)
   if path.endswith('submit'):return {'job':{'run_id':'prepare'},'status':{'stage':'queued_resources'}}
   if '/task/' in path:return {'status':{'stage':'queued_resources'}}
   if path.endswith('admission-proof'):return {'resource_waiting':True,'reasons':['fresh_old_lease_pending']}
   raise AssertionError('must not admit before fresh actual resource proof')
  tick(state,plan,[20418],api);self.assertEqual(state['scientific_submissions'],1);self.assertNotIn('matrix',state['tasks']);self.assertTrue(any('dependencies' in x for x in state['wait_reasons']))
  self.assertEqual(resolve({'task_ref':['x','results','scene_sets']},{'x':{'results':{'scene_sets':{'test':40}}}}),{'test':40})
class EngineeringDeploymentPureTests(unittest.TestCase):
 def test_three_bounded_engineering_configs_no_formal_eval(self):
  from satellite_rl.t0_comparison.contract import canonical_config
  reference=json.loads(REFERENCE.read_text(encoding='utf8'))
  for method in ('sac','ppo','dqn'):
   cfg=training_config(reference,method);cfg['training'].update(total_steps=512,warmup_steps=256,batch_size=256);cfg['evaluation'].update(validation_count=0,test_count=0,interval_steps=0,extra_steps=[])
   self.assertEqual(canonical_config(cfg,purpose='engineering')['training']['total_steps'],512)
   with self.assertRaises(ValueError):canonical_config(cfg,purpose='science')
 def test_deployment_only_borrowed_two_lanes(self):
  from scripts.stage2_t0_fair24 import deploy
  self.assertEqual(deploy.PORTS,(20418,20424))
  with self.assertRaises(ValueError):deploy.execute_remote(20473,'raise AssertionError()',{})
class CompletedEndpointControllerTests(unittest.TestCase):
 def test_completed_endpoint_catalog_enters_matrix_freeze_body(self):
  from scripts.stage2_t0_fair24.controller import tick
  ref={'env_step':150000,'endpoint_verified':True,'algorithm':'sac','model_files':[{'sha256':'actual_fixture'}]};state={'tasks':{'train':{'executor_port':20418,'run_id':'train','status':{'stage':'completed'}}},'scientific_submissions':1};body_seen=[]
  plan={'tasks':{'train':{'executor_port':20418,'body':{}},'matrix':{'executor_port':20418,'depends_on':['train'],'body':{'request_id':'matrix','endpoint_models':{'sac':{'task_ref':['train','endpoint_model_ref']}}}}}}
  def api(port,path,body=None):
   if path.endswith('/task/train'):return {'job':{'t0_mode':'train'},'status':{'stage':'completed'}}
   if path.endswith('/tasks'):return {'tasks':[{'job':{'run_id':'train'},'endpoint_model_ref':ref}]}
   if path.endswith('/submit'):body_seen.append(body);return {'job':{'run_id':'matrix'},'status':{'stage':'queued_resources'}}
   if path.endswith('/task/matrix'):return {'status':{'stage':'queued_resources'}}
   raise AssertionError(path)
  tick(state,plan,[20418],api);self.assertEqual(body_seen[0]['endpoint_models']['sac'],ref);self.assertEqual(state['tasks']['matrix']['status']['stage'],'queued_resources')
class OriginalLeaseFreshPureTests(unittest.TestCase):
 def test_original_lease_identity_and_future_exclusion(self):
  from unittest.mock import patch
  from satellite_rl.workbench.t0compare_qualification import qualify_original_leases
  prefix='/home/u2024110224/first_stage_satellite_rl/s1_ai17/resource_leases/'
  proof={'gpu_uuid':'fixtureGPU','original_lease_ref':{'path':prefix+'s2_t24_20418.json','sha256':'actualFixture'},'original_future_reservation_ref':{'path':prefix+'S1_future_reservation_after_s2_t24.json','sha256':'actualFixture'}}
  lease={'schema':'S1AI17PersistentResourceLeaseV1','status':'active_lent','port':20418,'borrower':'S2-T24','s1_new_science_allowed':False,'gpu_uuid':'fixtureGPU'};future={'schema':'S1AI17FutureReservationUpdateV1','borrowed_ports':[20418,20424],'new_S1_scientific_dispatch_ports':[20473,20474]}
  with patch('satellite_rl.workbench.t0compare_qualification.pin',side_effect=[lease,future]):self.assertTrue(qualify_original_leases(proof,20418)['original_bytes_checked'])
  with patch('satellite_rl.workbench.t0compare_qualification.pin',side_effect=[{**lease,'s1_new_science_allowed':True},future]):
   with self.assertRaises(ValueError):qualify_original_leases(proof,20418)
  with self.assertRaises(ValueError):qualify_original_leases({**proof,'original_lease_ref':{'path':'copied_stale_snapshot','sha256':'actualFixture'}},20418)
