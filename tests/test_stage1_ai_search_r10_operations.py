"""Synthetic native boundary and serial caller integration; no geometry/model execution."""
import ast,importlib,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from scripts.stage1_ai_search_r10 import contracts as c,guard,worker,server

OWNER=dict(pid=12345,birth_ticks=10,state='S',boot_id='mockboot',uid=7)
def ident(i):return dict(case_id='case'+str(i),mother_id='mother'+str(i),profile='uniform21600000',role='train',input_sha256='i',identity_sha256='j',shared_case_content_sha256='s',source_asset_SHA='a')

class Operations(unittest.TestCase):
 def test_complete_owned_ops_parse_and_cold_import(self):
  for name in c.OPS_FILES:
   path=Path(c.__file__).with_name(name);ast.parse(path.read_text(encoding='utf8'));importlib.import_module('scripts.stage1_ai_search_r10'+('' if name=='__init__.py' else '.'+path.stem))
 def test_only_v2_method_names_and_caps(self):
  self.assertEqual(c.ARMS,('Gd','Gr','Gw','S2','SH-C'));self.assertEqual(c.METHOD_HARD_SECONDS,300);self.assertEqual(c.CAPS['global_matching'],108);self.assertEqual(c.CAPS['sequential_kernel'],768)
 def test_gzip_common_budget_consumed_correctly(self):
  from scripts.stage1_ai_search_r10.schedule import Reservations
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);value=dict(run_id='run',original_index=0,identity=ident(0),actual_stage='common');ref=c.write_asset(root/'common.json.gz',value);Reservations(root/'budget','run','epoch',{0:ident(0)}).reserve(0,'Gd',{},ref)
 def test_native_existing_WAL_no_spawn(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);c.immutable(root/'stage.wal.json',{});spawn=Mock()
   with self.assertRaises(ValueError):guard.run_stage(['mock'],root,'common',stop=lambda:None,popen=spawn)
   spawn.assert_not_called()
 def test_native_spawn_error_preserves_receipt(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   with patch.object(guard,'identity',return_value=OWNER):
    with self.assertRaises(RuntimeError):guard.run_stage(['mock'],root,'Gd',stop=lambda:None,popen=Mock(side_effect=OSError('mockspawn')),memory=lambda pid:0)
   receipt=json.loads((root/'guard_receipt.json').read_bytes());self.assertEqual(receipt['status'],'partial_unknown');self.assertFalse(receipt['automatic_retry']);self.assertTrue((root/'stage.wal.json').exists())
 def test_method300_common30_guard_limits(self):
  for stage,limit in [('common',30),('Gd',300),('SH-C',300)]:
   with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);child=Mock(pid=12345);state={'exit':None};child.poll.side_effect=lambda:state['exit'];ticks=iter([0,limit+1,limit+2]);terminate=Mock(side_effect=lambda *a:state.update(exit=-15))
    with patch.object(guard,'identity',return_value=OWNER),patch.object(guard.os,'getpgid',return_value=12345,create=True),patch.object(guard,'observe_owned'):
     with self.assertRaises(RuntimeError):guard.run_stage(['mock'],root,stage,stop=lambda:None,popen=Mock(return_value=child),clock=lambda:next(ticks),memory=lambda pid:0,sleep=lambda seconds:None,terminate=terminate)
    self.assertTrue(terminate.called);self.assertEqual(json.loads((root/'stage.wal.json').read_bytes())['hard_seconds'],limit)
 def endpoint(self,root):
  endpoint=server.Endpoint.__new__(server.Endpoint);endpoint.root=root;endpoint.plan=dict(run_id='r10_mock',request_id='request',archive_root=str(root),executor_port=20473);endpoint.plan_ref=dict(path='plan',sha256='p');endpoint.review_ref=dict(path='review',sha256='r');endpoint.path='/api/ai-search-r10/tasks/r10_mock';return endpoint
 def test_wrong_endpoint_request_never_spawns(self):
  with tempfile.TemporaryDirectory() as tmp:
   endpoint=self.endpoint(Path(tmp));spawn=Mock()
   with self.assertRaises(ValueError):endpoint.post('/api/ai-search-r9/tasks/old/start',{},spawn)
   spawn.assert_not_called()
 def test_worker_no_birth_even_already_exited_unknown(self):
  with tempfile.TemporaryDirectory() as tmp:
   endpoint=self.endpoint(Path(tmp));body=dict(plan_ref=endpoint.plan_ref,root_review_ref=endpoint.review_ref,request_id='request');child=Mock(pid=99);child.poll.return_value=0
   with patch.object(server,'validate',return_value=endpoint.plan),patch.object(server,'boundary'),patch.object(server,'storage_boundary'),patch.object(server,'fresh_route'),patch.object(server,'identity',return_value=None):
    with self.assertRaises(RuntimeError):endpoint.post(endpoint.path+'/start',body,Mock(return_value=child))
   self.assertTrue((endpoint.root/'submission_intent.json').exists());self.assertFalse((endpoint.root/'actual_worker_launch.json').exists())
 def plan(self,root):
  cases=[dict(original_index=i,identity=ident(i),aircraft_count=2,common_input_ref={'path':'qualified','sha256':'q'}) for i in c.MOTHERS]
  return dict(run_id='r10_mock',group=c.GROUP,archive_root=str(root),host_compute_lock=str(root/'compute.lock'),execution_epoch_id='new',cases=cases,mother_host_assignment={str(i):20473 if k<6 else 20474 for k,i in enumerate(c.MOTHERS)},executor_port=20473)
 def run_worker(self,root,fail_stage=None):
  plan=self.plan(root);calls=[];limits=dict(raypairs=0,flow_queries=0,residual_edge_inspections=0,heap_pushes_plus_pops=0,heap_key_comparisons=0,graph_arc_initialization_writes=0,augmentation_capacity_writes=0,full_vector_node_visits=0,sort_key_comparisons=0)
  def native(command,target,stage,**kwargs):
   request=json.loads((target/'request.json').read_bytes());calls.append((request['original_index'],stage));c.immutable(target/'stage.wal.json',{'stage':stage})
   if stage==fail_stage:raise RuntimeError('mockunknown')
   value=dict(run_id=plan['run_id'],original_index=request['original_index'],identity=ident(request['original_index']),actual_stage=stage,new_RULE_D5_W_T_model_calls=0)
   core=dict(identity=value['identity'],original_index=value['original_index'],common_SHA='common')
   if stage=='common':value.update(common=dict(core,common_ledger=dict(limits)))
   else:value.update(common_ref=request['common_ref'],result=dict(core,method=stage,complete=True,ledger=dict(limits)))
   asset=c.write_asset(target/'stage_output.json.gz',value);pointer=c.immutable(target/'stage_output.json',dict(asset_ref=asset));c.immutable(target/'guard_receipt.json',dict(status='completed'));return dict(output_ref=pointer)
  with patch.object(worker,'bind_parent'),patch.object(worker,'validate',return_value=plan),patch.object(worker,'boundary'),patch.object(worker,'storage_boundary'),patch.object(worker,'identity',return_value=OWNER),patch.object(worker,'io_snapshot',return_value=None),patch.object(worker,'run_stage',side_effect=native):
   if fail_stage:
    with self.assertRaises(RuntimeError):worker.run({}, {})
   else:worker.run({}, {})
  return calls
 def test_all36_host_stages_serial_same_typed_common(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);calls=self.run_worker(root);self.assertEqual(len(calls),36);self.assertEqual([s for i,s in calls[:6]],['common',*c.ARMS]);self.assertTrue((root/'result_manifest.json').exists());self.assertTrue((root/'budget_reservations/mothers/0/Gd/stage.wal.json').exists());self.assertTrue((root/'mothers/0/Gd/stage.wal.json').exists())
 def test_unknown_stage_keeps_denominator_no_replay(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);calls=self.run_worker(root,'Gd');self.assertEqual(len(calls),2);value=json.loads((root/'status.json').read_bytes());self.assertEqual(value['stage'],'partial_unknown');self.assertEqual(value['original_denominator'],12);self.assertFalse(value['automatic_retry']);self.assertTrue((root/'worker_finally_cost.json').exists())
 def admitted(self):
  planref={'path':'plan','sha256':'plan'};refs={k:dict(path=k,sha256=k) for k in ('design_ref','root_design_acceptance_ref','numeric_origin_protocol_ref','human_epoch_ref','input_qualification_ref','storage_inventory_ref')};refs['design_ref']['sha256']=c.DESIGN_SHA;refs['numeric_origin_protocol_ref']['sha256']=c.NUMERIC_PROTOCOL_SHA;p=self.plan(Path('/owned/round10/archive'));p.update(schema='R10ExactMechanismPlanV2',source_input_budget_frozen=True,common_hard_seconds=30,method_hard_seconds=300,method_rss_bytes=2147483648,arms=list(c.ARMS),caps=dict(c.CAPS),**refs)
  p['archive_root']='/owned/round10/archive';q=[]
  for case in p['cases']:
   case.update(uniform_request_bps=21600000,origin_kind='original_R9_allpool_best',B_layout_sha256='B');q.append(dict(original_index=case['original_index'],identity=case['identity'],aircraft_count=2,input_ref=case['common_input_ref'],B_layout_sha256='B'))
  review=dict(schema='R10RootExactExecutionAdmissionV1',exact_plan_ref=planref,passed=True,execution_authorized=True,scientific_execution_authorized=True,source_accepted=True,input_accepted=True,runtime_accepted=True,engineering8_accepted=True,**{k:p[k] for k in ('human_epoch_ref','input_qualification_ref','storage_inventory_ref')})
  docs={'plan':p,'review':review,'design_ref':{'group':c.GROUP},'root_design_acceptance_ref':dict(schema='RootR10MathematicalDesignAcceptanceV2',accepted_design=True,design_ref=p['design_ref']),'numeric_origin_protocol_ref':{},'input_qualification_ref':dict(schema='RootR10Exact12ExistingInputsStaticQualificationV1',qualified_static=True,heldout_reads=0,rows=q)}
  return planref,p,review,docs
 def validate_mock(self,planref,docs):
  with patch.object(c,'read',side_effect=lambda r:docs[r['path']]),patch.object(c,'pin'),patch.object(c,'source_check'),patch.object(c,'human_epoch'),patch.object(c,'storage_check'):return c.validate(planref,dict(path='review',sha256='review'))
 def test_exact_new_root_admission_shape_positive(self):
  ref,p,review,docs=self.admitted();self.assertIs(self.validate_mock(ref,docs),p)
 def test_source_acceptance_without_science_flag_not_execution(self):
  ref,p,review,docs=self.admitted();review['scientific_execution_authorized']=False
  with self.assertRaises(ValueError):self.validate_mock(ref,docs)
 def test_root_actual_origin_hash_mismatch_rejected(self):
  ref,p,review,docs=self.admitted();p['cases'][0]['B_layout_sha256']='different'
  with self.assertRaises(ValueError):self.validate_mock(ref,docs)
if __name__=='__main__':unittest.main()
