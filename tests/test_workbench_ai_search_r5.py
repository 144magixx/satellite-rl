"""R5 mock process, IO and finite orchestration checks; zero geometry/CUDA."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from satellite_rl.workbench import coverage_ai_search_r5 as policy
from satellite_rl.workbench.ai_search_r5_runner import method_order,cost_record,stage_value
from satellite_rl.workbench.r5_io import validate_output
from satellite_rl.workbench.r5_stage_guard import run_stage,rss_tree
from satellite_rl.workbench.storage import digest
from satellite_rl.training.run_monitor import atomic_json
from scripts.stage1_ai_search_r5 import plan_builder,controller,ops

class R5MockTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def config(self):return {'schema_version':13,'research_stage':'coverage','algorithm':'ai_search_r5','name':'mock','protocol_ref':{'path':'mock','sha256':policy.DESIGN_SHA},'datasets':{'development':{'role':'development','count':80,'sha256':policy.DATA_SHA}},'workers':1,'resources':{'minimum_free_bytes':8589934592,'parent_children_rss_bytes':2147483648},'hard_watchdogs':{'decoder':15}}
 def test_design_SHA_actual_bytes(self):self.assertEqual(digest(Path(plan_builder.__file__).with_name('round5_design.json')),policy.DESIGN_SHA)
 def test_config_exact(self):self.assertEqual(policy.validate_config(self.config()),self.config())
 def test_config_forbids_future_role(self):
  c=self.config();c['datasets']['train']=c['datasets']['development']
  with self.assertRaises(ValueError):policy.validate_config(c)
 def test_config_forbids_extra_teacher_watchdog(self):
  c=self.config();c['hard_watchdogs']['teacher']=45
  with self.assertRaises(ValueError):policy.validate_config(c)
 def test_case_index_rotation_not_selected_ordinal(self):
  for index in (0,1,16,17,32,44,60,76):self.assertEqual(method_order(index)[0],policy.METHODS[index%7]);self.assertEqual(set(method_order(index)),set(policy.METHODS))
 def inputs(self):
  d=json.loads(Path(plan_builder.__file__).with_name('round5_design.json').read_text('utf-8'));m=['m'+str(i)for i in range(20)for j in range(4)]
  return {'identities':{'mother_ids':m},'identity_ref':{'path':'i','sha256':policy.IDENTITY_SHA},'registry_ref':{'path':'r','sha256':policy.REGISTRY_SHA},'qualification_ref':{'path':'q','sha256':'a'*64},'process_ref':{'path':'s','sha256':'b'*64},'dataset_ref':{'path':'d','role':'development','count':80,'sha256':policy.DATA_SHA}}
 def test_mechanical_4lane_168(self):
  p=plan_builder.build('c'*40,self.inputs());v=plan_builder.audit(p,self.inputs()['identities']);self.assertEqual(v['result_slots'],168);self.assertEqual(len(p['tasks']),4);self.assertTrue(all(len(t['case_indices'])==6 for t in p['tasks'].values()))
 def test_precise_routing_preserves_whole_case_not_whole_mother(self):
  inputs=self.inputs();d=json.loads(Path(plan_builder.__file__).with_name('round5_design.json').read_text('utf-8'));routing={str(i):ops.SSH_PORTS[n%4] for n,i in enumerate(d['P0']['case_indices'])};p=plan_builder.build('c'*40,inputs,routing);self.assertEqual(plan_builder.audit(p,inputs['identities'])['case_routes'],routing)
 def test_duplicate_case_fails_plan(self):
  p=plan_builder.build('c'*40,self.inputs());ts=list(p['tasks'].values());ts[0]['case_indices'][0]=ts[1]['case_indices'][0]
  with self.assertRaises(ValueError):plan_builder.audit(p,self.inputs()['identities'])
 def test_future_mode_fails_plan(self):
  p=plan_builder.build('c'*40,self.inputs());p['phase']='labels'
  with self.assertRaises(ValueError):plan_builder.audit(p,self.inputs()['identities'])
 def test_fixed_rss_function_is_default(self):
  with patch('satellite_rl.workbench.r4_monitor_repair.rss_tree',return_value=123) as fixed:self.assertEqual(rss_tree(1234),123);fixed.assert_called_once_with(1234)
 def test_unknown_WAL_not_replayed(self):
  atomic_json(self.root/'stage.wal.json',{'command':['mock']});spawn=Mock()
  with self.assertRaises(ValueError):run_stage(['mock'],self.root,15,2**31,popen=spawn)
  spawn.assert_not_called()
 def test_orphan_output_not_success(self):
  atomic_json(self.root/'stage_output.json',{'old_W':True});spawn=Mock()
  with self.assertRaises(ValueError):run_stage(['mock'],self.root,15,2**31,popen=spawn)
  spawn.assert_not_called()
 def test_mock_process_exit_without_output(self):
  child=Mock(pid=123,returncode=0);child.poll.return_value=0
  with patch('satellite_rl.workbench.r5_stage_guard.birth',return_value=456):r=run_stage(['mock'],self.root,15,2**31,popen=Mock(return_value=child),memory=Mock(return_value=0))
  self.assertEqual(r['status'],'hard_unpublished');self.assertEqual(r['reason'],'missing_output')
 def test_hardkill_old_output_unpublished(self):
  v,_=stage_value({'status':'hard_unpublished','reason':'hard_time_guard','output_ref':{'path':'old','sha256':'a'},'unknown_algorithm_spend':True},{'method':'D5_EXACT'});self.assertEqual(v['status'],'unpublished');self.assertNotIn('layout',v)
 def test_hard_source_error_stops(self):
  with self.assertRaises(ValueError):stage_value({'hard_interface_error':True},{'method':'D5_RULE'})
 def cost(self,method):
  p=self.root/'old.json';atomic_json(p,{'components':{'input':2.,'prepare':3.,'teacher':7.},'unknown_cost':False,'unknown_algorithm_spend':False});case={'teacher':{'publication_cost':{'path':str(p),'sha256':digest(p)},'stage_output':{'path':'t','sha256':'a'}},'prepared':{'stage_output':{'path':'p','sha256':'b'}}};return cost_record(method,case,{'actual_stage_wall_seconds':5.,'unknown_algorithm_spend':False},1.,.5)
 def test_cold_rule_adds_prepare_not_teacher(self):self.assertEqual(self.cost('D5_RULE')['complete_method_seconds'],11.5)
 def test_cold_diagnostics_add_true_teacher_once(self):
  for m in ('D5_EXACT','D5_OWNER10','D5_JOINT10','D4_OWNER10','D4_JOINT10'):self.assertEqual(self.cost(m)['complete_method_seconds'],18.5)
 def test_controller_first_POST_unknown_stops_successors(self):
  spec={'group':ops.GROUP,'automatic_extension':False,'scientific_commit':'a'*40,'tasks':[{'port':20418,'body':{'request_id':'first'}},{'port':20424,'body':{'request_id':'second'}}]};p=self.root/'spec.json';atomic_json(p,spec)
  with patch.object(controller,'route',return_value={'ready':True}),patch.object(ops,'api',side_effect=RuntimeError('unknown')) as api:r=controller.run(p,self.root/'state')
  self.assertEqual(r['phase'],'attention_required');self.assertEqual(api.call_count,1);self.assertEqual(set(r['intents']),{'first'})
 def test_controller_GET_error_stops_successors(self):
  spec={'group':ops.GROUP,'automatic_extension':False,'scientific_commit':'a'*40,'tasks':[{'port':20418,'body':{'request_id':'first'}},{'port':20424,'body':{'request_id':'second'}}]};p=self.root/'spec.json';atomic_json(p,spec)
  with patch.object(controller,'route',return_value={'ready':True}),patch.object(ops,'api',side_effect=[{'run_id':'sameid'},RuntimeError('geterror')]) as api:r=controller.run(p,self.root/'state')
  self.assertEqual(r['phase'],'attention_required');self.assertEqual(api.call_count,2);self.assertEqual(r['intents']['first']['run_id'],'sameid')
 def test_controller_complete_exits(self):
  spec={'group':ops.GROUP,'automatic_extension':False,'scientific_commit':'a'*40,'tasks':[{'port':20418,'body':{'request_id':'first'}}]};p=self.root/'spec.json';atomic_json(p,spec)
  with patch.object(controller,'route',return_value={'ready':True}),patch.object(ops,'api',side_effect=[{'run_id':'sameid'},{'status':{'stage':'completed'}}]):r=controller.run(p,self.root/'state')
  self.assertEqual(r['phase'],'completed')
if __name__=='__main__':unittest.main()
