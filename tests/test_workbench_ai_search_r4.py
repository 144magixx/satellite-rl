"""Only mock orchestration/contracts/guards: zero geometry/native/CUDA cases."""
import json,tempfile,unittest,hashlib
from pathlib import Path
from unittest.mock import patch,Mock
from satellite_rl.workbench.ai_search_r4_runner import execute_bundle,method_order,cost_record,check_case_identity
from satellite_rl.workbench.ai_search_r4_stage import persist_accepted
from satellite_rl.workbench.r4_io import check_record
from satellite_rl.workbench.r4_stage_guard import run_stage,birth
from satellite_rl.workbench.coverage_ai_search_r4 import validate_config,DESIGN_SHA,DATA_SHA
from satellite_rl.workbench.ai_search_runner import recover_rows,result_key
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.storage import digest
from scripts.stage1_ai_search_r4 import controller
IDENTITY={'case_id':'mock','mother_id':'mockmother','profile':'base','input_sha256':'b'*64,'identity_sha256':'c'*64,'role':'development'};SOURCE='a'*64
class R4MockTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def stage(self,name,value):
  p=self.root/(name+'.json');atomic_json(p,{'result':value});return {'status':'completed','output_ref':{'path':str(p),'sha256':digest(p)},'actual_stage_wall_seconds':1.,'unknown_algorithm_spend':False}
 def test_teacher_unavailable_skips_two_decoder_slots(self):
  calls=[];emitted=[]
  def perform(stage,method):
   calls.append((stage,method))
   if stage=='preparation':return self.stage(stage,{'status':'available','identity':IDENTITY,'source_sha256':SOURCE})
   if stage=='teacher':return self.stage(stage,{'status':'unavailable','reason':'fixture','provenance':{**IDENTITY,'source_sha256':SOURCE}})
   return self.stage(method,{'status':'success',**IDENTITY,'source_sha256':SOURCE})
  execute_bundle(0,perform,lambda *args:emitted.append(args),IDENTITY,SOURCE);self.assertEqual([m for s,m in calls if s=='decoder'],['RULE','RANDOM']);self.assertEqual(len(emitted),5);self.assertEqual([x[1]['reason']for x in emitted if x[0].startswith('TEACHER_')],['unavailable_teacher']*2)
 def test_bad_teacher_tag_is_hard(self):
  def perform(stage,method):return self.stage(stage,{'status':'available','identity':IDENTITY,'source_sha256':SOURCE}if stage=='preparation'else {'status':'bogus'})
  with self.assertRaises(ValueError):execute_bundle(0,perform,lambda *x:None,IDENTITY,SOURCE)
 def test_unavailable_teacher_wrong_source_hard(self):
  with self.assertRaises(ValueError):check_record({'status':'unavailable','provenance':{**IDENTITY,'source_sha256':'d'*64}},'teacher',IDENTITY,SOURCE)
 def test_available_teacher_wrong_identity_hard(self):
  with self.assertRaises(ValueError):check_record({'status':'available',**IDENTITY,'role':'test','source_sha256':SOURCE},'teacher',IDENTITY,SOURCE)
 def test_unavailable_proposal_wrong_source_hard(self):
  with self.assertRaises(ValueError):check_record({'status':'unavailable','provenance':{**IDENTITY,'source_sha256':'z'}},'proposal',IDENTITY,SOURCE)
 def test_unavailable_with_layout_hard(self):
  with self.assertRaises(ValueError):check_record({'status':'unavailable','provenance':{**IDENTITY,'source_sha256':SOURCE},'layout':{}},'proposal',IDENTITY,SOURCE)
 def test_four_method_rotation(self):
  for ordinal in range(24):self.assertEqual(method_order(ordinal)[0],['RULE','RANDOM','TEACHER_EXACT','TEACHER_PERTURBED'][ordinal%4])
 def test_loaded_identity_mismatch_rejected_before_stage(self):
  ids={'case_ids':['mock'],'mother_ids':['mockmother'],'profiles':['base'],'sha256':SOURCE};case={'case_id':'mock','mother_id':'mockmother','demand_profile':'base','input_dataset_sha256':SOURCE};self.assertIs(check_case_identity(case,0,ids),case)
  for key in case:
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'frozen identity'):check_case_identity({**case,key:'wrong'},0,ids)
 def test_prepared_provenance_checked_before_teacher(self):
  with self.assertRaisesRegex(ValueError,'source SHA'):check_record({'status':'available','identity':IDENTITY,'source_sha256':'wrong'},'prepared',IDENTITY,SOURCE)
 def test_unavailable_ready_mask_direction_metrics_rejected(self):
  for key in ('score_mask_nxK_bool','direction_offset_Kx3','metrics'):
   with self.subTest(key=key),self.assertRaises(ValueError):check_record({'status':'unavailable','provenance':{**IDENTITY,'source_sha256':SOURCE},key:[]},'proposal',IDENTITY,SOURCE)
 def test_accepted_snapshot_callback_durable_without_geometry(self):
  from types import SimpleNamespace
  from satellite_rl.coverage.ai_search_r4.core import Ledger,_accepted
  directory=self.root/'teacher';save=persist_accepted(directory,IDENTITY,SOURCE);ledger=Ledger(1,4096,30);ledger.on_accepted=save;record={'centers':[[0.,0.]],'widths_deg':[.3],'assignment':[0],'beam_ids':['mock']};layout=Mock();layout.to_record.return_value=record
  with patch('satellite_rl.coverage.ai_search_r4.core._layout',return_value=layout),patch('satellite_rl.coverage.ai_search_r4.core._key',return_value=(1.,2.)):
   _accepted(SimpleNamespace(scene=object()),{'anchors':['plane']},ledger,'MOVE')
  index=json.loads((directory/'accepted_snapshot_index.json').read_text());self.assertEqual(index['count'],1);ref=index['accepted_refs'][0];self.assertEqual(digest(ref['path']),ref['sha256']);saved=json.loads(Path(ref['path']).read_text());self.assertEqual(saved['snapshot'],ledger.data['accepted'][0]);self.assertTrue(index['partial_evidence_not_final_publication'])
  with self.assertRaisesRegex(ValueError,'implicit replay'):persist_accepted(directory,IDENTITY,SOURCE)
 def test_teacher_bill_exact_once_not_rule(self):
  p={'actual_stage_wall_seconds':2};t={'actual_stage_wall_seconds':30};d={'actual_stage_wall_seconds':5};rule=cost_record('RULE',1,p,t,d,.5);exact=cost_record('TEACHER_EXACT',1,p,t,d,.5);self.assertEqual(rule['complete_method_seconds'],8.5);self.assertEqual(exact['complete_method_seconds'],38.5);self.assertNotIn('regular_publication_eligible',exact)
 def test_unknown_cost_not_zero(self):
  c=cost_record('TEACHER_EXACT',1,{'actual_stage_wall_seconds':2},{'actual_stage_wall_seconds':45,'unknown_algorithm_spend':True},{'actual_stage_wall_seconds':0},.2);self.assertTrue(c['unknown_cost']);self.assertTrue(c['unknown_algorithm_spend'])
 def test_unknown_started_forbids_replay(self):
  (self.root/'cases').mkdir();key=result_key('mock','RULE');atomic_json(self.root/'started'/(key+'.json'),{})
  with self.assertRaisesRegex(ValueError,'unknown spent'):recover_rows(self.root,{key})
 def test_guard_early_exit_birth_missing_completed_reason_none(self):
  directory=self.root/'early';directory.mkdir();atomic_json(directory/'stage_output.json',{'result':'no_algorithm_mock'});child=Mock();child.pid=987654;child.poll.return_value=0;child.returncode=0
  with patch('satellite_rl.workbench.r4_stage_guard.birth',return_value=None):r=run_stage(['mockcommand'],directory,20,2147483648,popen=lambda *a,**kw:child,memory=lambda _:0,sleep=lambda _:None)
  self.assertEqual(r['status'],'completed');self.assertIsNone(r['reason']);self.assertIsNone(r['birth'])
 def test_guard_hardtime_owned_kill_mock(self):
  child=Mock();child.pid=987654;child.poll.return_value=None;child.returncode=-15;clock=iter([0.,0.,21.,21.]);directory=self.root/'timeout'
  with patch('satellite_rl.workbench.r4_stage_guard.birth',return_value=123),patch('satellite_rl.workbench.r4_stage_guard.terminate_owned')as kill:r=run_stage(['mock'],directory,20,2147483648,popen=lambda *a,**kw:child,clock=lambda:next(clock),memory=lambda _:0,sleep=lambda _:None)
  kill.assert_called_once_with(child,123);self.assertEqual(r['status'],'hard_unpublished');self.assertTrue(r['unknown_algorithm_spend'])
 def test_guard_unknown_WAL_no_spawn(self):
  d=self.root/'unknown';d.mkdir();atomic_json(d/'stage.wal.json',{});spawn=Mock()
  with self.assertRaisesRegex(ValueError,'unknown interrupted'):run_stage(['mock'],d,20,10,popen=spawn)
  spawn.assert_not_called()
 def test_birth_linux_exit_read_race(self):
  with patch('pathlib.Path.read_text',side_effect=FileNotFoundError):self.assertIsNone(birth(987654))
 def test_completed_controller_no_route_or_POST(self):
  spec=self.root/'spec.json';atomic_json(spec,{'group':controller.ops.GROUP,'scientific_commit':'a'*40,'automatic_extension':False,'tasks':[]});state=self.root/'control';state.mkdir();atomic_json(state/'state.json',{'group':controller.ops.GROUP,'spec_sha256':digest(spec),'phase':'completed','intents':{}})
  with patch.object(controller,'route')as route,patch.object(controller.ops,'api')as api:controller.run(spec,state)
  route.assert_not_called();api.assert_not_called()
 def test_attention_controller_cannot_automatically_retry(self):
  spec=self.root/'spec.json';atomic_json(spec,{'group':controller.ops.GROUP,'scientific_commit':'a'*40,'automatic_extension':False,'tasks':[{'port':20418,'body':{'request_id':'same_original'}}]});state=self.root/'attention';state.mkdir();atomic_json(state/'state.json',{'group':controller.ops.GROUP,'spec_sha256':digest(spec),'phase':'attention_required','intents':{'same_original':{'phase':'unknown_post_outcome'}}})
  with patch.object(controller,'route')as route,patch.object(controller.ops,'api')as api:
   with self.assertRaisesRegex(ValueError,'root read-only reconciliation'):controller.run(spec,state)
  route.assert_not_called();api.assert_not_called()
 def test_first_controller_POST_error_stops_following(self):
  spec=self.root/'spec.json';atomic_json(spec,{'group':controller.ops.GROUP,'scientific_commit':'a'*40,'automatic_extension':False,'tasks':[{'port':20418,'body':{'request_id':'A'}},{'port':20424,'body':{'request_id':'B'}}]});state=self.root/'failfirst';api=Mock(side_effect=RuntimeError('unknown reply'))
  with patch.object(controller,'route',return_value={'ready':True}),patch.object(controller.ops,'api',api):controller.run(spec,state)
  self.assertEqual(api.call_count,1);d=json.loads((state/'state.json').read_text());self.assertEqual(d['phase'],'attention_required');self.assertNotIn('B',d['intents']);self.assertEqual(d['intents']['A']['phase'],'unknown_post_outcome')
 def test_config_blocks_train_or_more_workers(self):
  c={'schema_version':12,'research_stage':'coverage','algorithm':'ai_search_r4','name':'mock','protocol_ref':{'sha256':DESIGN_SHA},'datasets':{'development':{'role':'development','count':80,'sha256':DATA_SHA}},'workers':1,'resources':{'minimum_free_bytes':8589934592,'parent_children_rss_bytes':2147483648},'hard_watchdogs':{'preparation':20,'teacher':45,'decoder':15}};validate_config(c);c['workers']=8
  with self.assertRaises(ValueError):validate_config(c)
if __name__=='__main__':unittest.main()
