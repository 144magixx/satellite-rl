"""Pure mock R9 operations tests. No real scene, RULE, D5, kernel or update."""
import errno,gzip,hashlib,json,os,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from scripts.stage1_ai_search_r9 import contracts as c,control,guard,server,plan_builder,train_input,worker

OWNER=dict(pid=12345,birth_ticks=10,state='S',boot_id='mockboot')
class R9Operations(unittest.TestCase):
 def setUp(self):
  # Native lifecycle tests inject process observations; never inspect real
  # Linux groups or spawn/kill a real process on this Windows qualification.
  self.observe_patch=patch.object(guard,'observe_owned',return_value=None);self.observe_patch.start();self.addCleanup(self.observe_patch.stop)
 def test_all_operations_syntax_and_cold_imports_no_science(self):
  import ast,importlib
  for name in c.OPS_FILES:
   p=Path(c.__file__).with_name(name);ast.parse(p.read_text(encoding='utf8'));importlib.import_module('scripts.stage1_ai_search_r9'+('' if name=='__init__.py' else '.'+p.stem))
 def epoch(self,root):return dict(watched_pause_paths=[str(root/'human_hold.json')],historical_hold_exemptions=[])
 def plan(self,root):return dict(group=c.GROUP,stage='M',run_id='r9_M_mock',request_id='root-exact',archive_root=str(root),human_epoch_ref={},executor_port=20474)
 def endpoint(self,root):
  e=server.Endpoint.__new__(server.Endpoint);e.root=root;e.plan=self.plan(root);e.plan_ref={'path':'exact_plan','sha256':'p'};e.review_ref={'path':'root_review','sha256':'r'};e.path='/api/ai-search-r9/tasks/r9_M_mock';return e
 def test_fixed_M_budgets_and_only_train_original_indices(self):
  self.assertEqual(c.M_INDICES,[0,6,12,18,24,36,42,48,54,60,72,78]);self.assertEqual(c.CAPS,dict(RULE=12,D5=12,candidate=384,W=0,T=0,forward=0,updates=0))
 def test_byte_pinning_detects_modification(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'source.json';ref=c.immutable(p,{'actual':1});self.assertEqual(c.read(ref),{'actual':1});p.write_text('{"actual":2}')
   with self.assertRaises(ValueError):c.pin(ref)
 def test_immutable_intent_cannot_be_replaced(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'intent';c.immutable(p,{'first':1})
   with self.assertRaises(FileExistsError):c.immutable(p,{'second':2})
   self.assertEqual(json.loads(p.read_bytes()),{'first':1})
 def test_asset_read_checks_original_content_hash_and_bounds(self):
  with tempfile.TemporaryDirectory() as tmp:
   raw=b'{"value":3}';p=Path(tmp)/'asset.gz';p.write_bytes(gzip.compress(raw));r=dict(path=str(p),sha256=c.digest(p),compression='gzip',uncompressed_bytes=len(raw),canonical_json_sha256=hashlib.sha256(raw).hexdigest());self.assertEqual(c.asset_read(r),{'value':3})
   with self.assertRaises(ValueError):c.asset_read(dict(r,uncompressed_bytes=len(raw)-1))
   with self.assertRaises(ValueError):c.asset_read(dict(r,uncompressed_bytes=268435457))
 def test_proc_disappearance_only_ENOENT_ESRCH(self):
  for code in (errno.ENOENT,errno.ESRCH):
   with patch.object(Path,'read_text',side_effect=OSError(code,'gone')):self.assertIsNone(c.identity(12345))
  with patch.object(Path,'read_text',side_effect=OSError(errno.EACCES,'denied')):
   with self.assertRaises(OSError):c.identity(12345)
 def test_same_PID_different_birth_or_boot_never_owner(self):
  self.assertTrue(c.same_process(OWNER,dict(OWNER,state='R')))
  self.assertFalse(c.same_process(OWNER,dict(OWNER,birth_ticks=11)));self.assertFalse(c.same_process(OWNER,dict(OWNER,boot_id='different')))
 def test_original_design_actual_metadata_and_precise_runtime_alias(self):
  base=Path('D:/paper/outputs/experiments/first_stage_ai_search_20261013/design');accepted=json.loads((base/'round9_root_design_acceptance_v3.json').read_bytes());self.assertTrue(accepted['accepted']);self.assertEqual(accepted['design_ref']['sha256'],c.DESIGN_SHA)
  auth=Path('D:/paper/outputs/experiments/first_stage_ai_search_20261013/round8/resume_after_0700/root_explicit_user_continue_authority.json');self.assertEqual(c.digest(auth),c.AUTHORITY_SHA);self.assertTrue(json.loads(auth.read_bytes())['human_authorization'])
  with tempfile.TemporaryDirectory() as tmp:
   runtime=Path(tmp)/'runtime.json';runtime.write_bytes((base/'round9_design_v3.json').read_bytes());ref=dict(path=str(runtime),sha256=c.digest(runtime),bytes=runtime.stat().st_size);p={'document_aliases':[dict(runtime_ref=ref,origin_ref=accepted['design_ref'])]};self.assertEqual(c.origin_ref(p,ref),accepted['design_ref']);p['document_aliases'].append(p['document_aliases'][0])
   with self.assertRaises(ValueError):c.origin_ref(p,ref)
 def test_empty_pause_inventory_passes_no_cutoff(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   with patch.object(control,'human_epoch',return_value=self.epoch(root)):self.assertTrue(control.boundary(self.plan(root),'mock')['passed'])
 def test_new_pause_and_stop_block(self):
  for name,value in [('human_hold.json',{'user_pause':True}),('stop.request',{})]:
   with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);c.atomic(root/name,value)
    with patch.object(control,'human_epoch',return_value=self.epoch(root)):
     with self.assertRaises(control.Paused):control.boundary(self.plan(root),'mock')
 def test_exact_old_pause_generation_exempt_but_identical_rewrite_blocks(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);old=root/'human_hold.json';c.atomic(old,{'paused':True});snap=control.marker_snapshot(old);archived=root/'old_evidence.json';archived.write_bytes(old.read_bytes());snap.update(reason='historical0700_pause',archived_evidence_ref=dict(path=str(archived),sha256=c.digest(archived)));epoch=self.epoch(root);epoch['historical_hold_exemptions']=[snap]
   with patch.object(control,'human_epoch',return_value=epoch):self.assertTrue(control.boundary(self.plan(root),'old')['passed'])
   # Recreate same bytes: inode/stat generation differs even if SHA does not.
   raw=old.read_bytes();old.unlink();old.write_bytes(raw);os.utime(old,ns=(snap['st_mtime_ns']+1000000,snap['st_mtime_ns']+1000000))
   with patch.object(control,'human_epoch',return_value=epoch):
    with self.assertRaises(control.Paused):control.boundary(self.plan(root),'rewritten')
   self.assertEqual(archived.read_bytes(),raw)
 def test_removed_historical_marker_blocks(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);old=root/'human_hold.json';c.atomic(old,{'paused':True});snap=control.marker_snapshot(old);evidence=root/'evidence';evidence.write_bytes(old.read_bytes());snap.update(reason='historical0700_pause',archived_evidence_ref=dict(path=str(evidence),sha256=c.digest(evidence)));old.unlink();epoch=self.epoch(root);epoch['historical_hold_exemptions']=[snap]
   with patch.object(control,'human_epoch',return_value=epoch):
    with self.assertRaises(control.Paused):control.boundary(self.plan(root),'removed')
 def test_storage_reserve_and_size_fail_no_deletion(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'temp').mkdir();e=root/'protected';e.write_bytes(b'original');s=dict(owned_growth_root=str(root),owned_temporary_root=str(root/'temp'),free_reserve_bytes=8589934592,new_logical_bytes_max=536870912,temporary_bytes_max=268435456)
   with patch.object(control,'storage_check',return_value=s),patch.object(control.shutil,'disk_usage',return_value=SimpleNamespace(free=8589934591)):
    with self.assertRaises(control.Paused):control.storage_boundary(self.plan(root),'mock')
   self.assertEqual(e.read_bytes(),b'original')
 def test_existing_WAL_never_spawns_again(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);c.immutable(root/'stage.wal.json',{'unknown':True});spawn=Mock()
   with self.assertRaises(ValueError):guard.run_stage(['mock'],root,'pool',stop=lambda:None,popen=spawn)
   spawn.assert_not_called()
 def test_spawn_failure_preserves_unknown_reservation_receipt(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   with patch.object(guard,'identity',return_value=OWNER):
    with self.assertRaises(RuntimeError):guard.run_stage(['mock'],root,'common',stop=lambda:None,popen=Mock(side_effect=OSError('mock spawn failed')))
   self.assertTrue((root/'stage.wal.json').exists());r=json.loads((root/'guard_receipt.json').read_bytes());self.assertEqual(r['status'],'partial_unknown');self.assertIsNone(r['algorithm_spend_exact'])
 def mock_child(self,root,exitcode=None):
  child=SimpleNamespace(pid=12345,code=exitcode);child.poll=lambda:child.code;child.wait=lambda timeout:child.code;return child
 def test_pool_hard60_common30_and_RSS_kill_owned_mock(self):
  for stage,limit,memory in [('pool',60,0),('common',30,0),('RULE',15,0),('D5',15,0),('pool',60,2147483649)]:
   with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);child=self.mock_child(root);ticks=iter([0,limit+1,limit+2]);kill=Mock(side_effect=lambda ch,owner:setattr(ch,'code',-9))
    with patch.object(guard,'identity',return_value=OWNER),patch.object(guard.os,'getpgid',return_value=12345,create=True):
     with self.assertRaises(RuntimeError):guard.run_stage(['mock'],root,stage,stop=lambda:None,popen=lambda *a,**kw:child,clock=lambda:next(ticks),memory=lambda pid:memory,terminate=kill)
    kill.assert_called_once_with(child,OWNER);self.assertEqual(json.loads((root/'stage.wal.json').read_bytes())['hard_seconds'],limit)
 def test_pause_after_spawn_terminates_owned_mock(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);child=self.mock_child(root);calls=iter([None,control.Paused('mock human pause')]);kill=Mock(side_effect=lambda ch,owner:setattr(ch,'code',-9))
   def stop():
    e=next(calls)
    if e:raise e
   with patch.object(guard,'identity',return_value=OWNER),patch.object(guard.os,'getpgid',return_value=12345,create=True):
    with self.assertRaises(RuntimeError):guard.run_stage(['mock'],root,'pool',stop=stop,popen=lambda *a,**kw:child,terminate=kill)
   kill.assert_called_once();self.assertEqual(json.loads((root/'guard_receipt.json').read_bytes())['status'],'partial_unknown')
 def test_kill_refuses_reused_PID(self):
  child=SimpleNamespace(pid=12345,poll=lambda:None)
  with patch.object(guard,'identity',return_value=dict(OWNER,birth_ticks=11)),patch.object(guard.os,'killpg',create=True) as kill:
   with self.assertRaises(RuntimeError):guard.terminate_owned(child,OWNER)
   kill.assert_not_called()
 def test_post_wrong_body_and_old_endpoint_never_spawns(self):
  with tempfile.TemporaryDirectory() as tmp:
   e=self.endpoint(Path(tmp));spawn=Mock()
   with self.assertRaises(ValueError):e.post(e.path+'/start',{},spawn)
   body=dict(plan_ref=e.plan_ref,root_review_ref=e.review_ref,request_id=e.plan['request_id'])
   with self.assertRaises(ValueError):e.post('/api/ai-search-r8/tasks/old/start',body,spawn)
   spawn.assert_not_called()
 def test_safe_stop_own_archive_only(self):
  with tempfile.TemporaryDirectory() as tmp:
   e=self.endpoint(Path(tmp));spawn=Mock();body=dict(plan_ref=e.plan_ref,root_review_ref=e.review_ref,request_id=e.plan['request_id'])
   self.assertTrue(e.post(e.path+'/stop',body,spawn)['stop_requested']);spawn.assert_not_called();self.assertTrue((e.root/'stop.request').exists())
 def test_unknown_submission_permanent_no_respawn(self):
  with tempfile.TemporaryDirectory() as tmp:
   e=self.endpoint(Path(tmp));c.immutable(e.root/'submission_intent.json',{'unknown':True});body=dict(plan_ref=e.plan_ref,root_review_ref=e.review_ref,request_id=e.plan['request_id']);spawn=Mock()
   with patch.object(server,'validate',return_value=e.plan),patch.object(server,'boundary'),patch.object(server,'storage_boundary'),patch.object(server,'fresh_route'):
    with self.assertRaises(ValueError):e.post(e.path+'/start',body,spawn)
   spawn.assert_not_called()
 def test_exact_diff_restores_ordinals_and_all_beam_values(self):
  origin=dict(beam_ids=['A','B','C'],anchor_aircraft_ids=['p0','p1','p2'],centers=[[1,2],[3,4],[5,6]],widths_deg=[.3,.4,.5],assignment=[0,1,2,0]);layout=dict(beam_ids=['C','newA','newB'],anchor_aircraft_ids=['p2','p0','p3'],centers=[[5,6],[1,2],[7,8]],widths_deg=[.5,.31,.32],assignment=[1,2,0,2]);delta=train_input.layout_delta(layout,origin);self.assertEqual(train_input.restore_layout(delta,origin),layout);self.assertEqual(delta['changed_aircraft_owners'],[[0,'newA'],[1,'newB'],[3,'newB']]);self.assertEqual(origin['assignment'],[0,1,2,0])
 def test_noop_diff_lossless(self):
  origin=dict(beam_ids=['A'],anchor_aircraft_ids=['p0'],centers=[[1,2]],widths_deg=[.3],assignment=[0,0]);delta=train_input.layout_delta(origin,origin);self.assertEqual(delta['changed_beams'],[]);self.assertEqual(delta['changed_aircraft_owners'],[]);self.assertEqual(train_input.restore_layout(delta,origin),origin)
 def test_role_guard_rejects_dev_before_original_scene_work(self):
  with self.assertRaises(ValueError):train_input.load_original(dict(identity={'role':'development'},uniform_request_bps=21600000))
 def test_plan_builder_metadata_only_exact12_no18_or24(self):
  repo=Path(c.__file__).resolve().parents[2];mapping=json.loads(Path('D:/paper/outputs/experiments/first_stage_ai_search_20261013/round9/implementation_preflight_01/selected96_original_source_mapping.json').read_bytes());cfg=dict(run_id='r9_M_draft',request_id='new',archive_root='/mock/round9/archive',executor_port=20474)
  p=plan_builder.build(mapping,cfg,repo);self.assertEqual([x['original_index'] for x in p['cases']],c.M_INDICES);self.assertFalse(p['scientific_execution_authorized_by_plan']);self.assertFalse(p['automatic_L_dispatch'])
  with self.assertRaises(ValueError):plan_builder.build(mapping,dict(cfg,executor_port=20418),repo)
 def test_train_thin_caller_original_APIs_no_teacher_and_D5_exact5(self):
  module=SimpleNamespace(build_proposal=Mock(return_value={'status':'ready','construction_cost':{'raypairs':200}}),decode_groups=Mock(return_value={'mock_result':True}));prepared=SimpleNamespace(scene=SimpleNamespace(demands=[21600000]))
  with patch.dict('sys.modules',{'satellite_rl.coverage.ai_search_r5.core':module}):
   proposal=train_input.rule(prepared,{'sha256':'prepared'});self.assertEqual(train_input.d5(prepared,proposal),{'mock_result':True})
  module.build_proposal.assert_called_once_with(prepared,'RULE',producer_sha256=c.digest(train_input.__file__),evidence_refs=[{'sha256':'prepared'}]);module.decode_groups.assert_called_once_with(prepared,proposal,time_limit=5.)
 def worker_plan(self,root):
  p=self.plan(root);p['host_compute_lock']=str(root/'host_compute.lock');p['cases']=[dict(original_index=i,identity={'role':'train','case_id':'mock:'+str(i)}) for i in c.M_INDICES];return p
 def mock_guard(self,p,fail_stage=None):
  def run(command,folder,stage,**kw):
   if stage==fail_stage:raise RuntimeError('mock unknown stage')
   req=json.loads((folder/'request.json').read_bytes());asset=dict(path=str(folder/'MOCK_no_algorithm_asset'),sha256='mock',stage=stage,original_index=req['original_index']);out=c.immutable(folder/'stage_output.json',dict(asset_ref=asset));c.immutable(folder/'guard_receipt.json',dict(status='completed',mock_only=True));return dict(output_ref=out)
  def asset(ref):
   case=next(x for x in p['cases'] if x['original_index']==ref['original_index']);value=dict(identity=case['identity'],actual_stage=ref['stage'])
   if ref['stage']=='D5':value['result']={'status':'success'}
   if ref['stage']=='common':value['pool']={'descriptors':[{'position':i} for i in range(32)]}
   if ref['stage']=='pool':value.update(complete=True,descriptor_extractions=32,candidate_refs={str(i):{'path':'pure_mock','sha256':'mock'} for i in range(32)},selected_position=-1)
   return value
  return run,asset
 def test_full_worker_serial12_mock_only_complete_and_no_autoL(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);p=self.worker_plan(root);run,asset=self.mock_guard(p);calls=[]
   def recorded(command,folder,stage,**kw):calls.append((folder.parent.name,stage));return run(command,folder,stage,**kw)
   with patch.object(worker,'bind_parent'),patch.object(worker,'validate',return_value=p),patch.object(worker,'boundary'),patch.object(worker,'storage_boundary'),patch.object(worker,'run_stage',side_effect=recorded),patch.object(worker,'asset_read',side_effect=asset),patch.object(worker,'pin'):
    worker.run({'path':'mockplan','sha256':'mock'},{'path':'mockreview','sha256':'mock'})
   self.assertEqual(calls,[(str(i),stage) for i in c.M_INDICES for stage in ('RULE','D5','common','pool')]);self.assertEqual(len(calls),48);result=json.loads((root/'result_manifest.json').read_bytes());self.assertEqual(result['reservations'],dict(RULE=12,D5=12,candidate=384));self.assertFalse(result['automatic_L_dispatch']);self.assertFalse(result['independent_scientific_gate_evaluated'])
 def test_worker_unknown_pool_stops_with_original12_no_replay(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);p=self.worker_plan(root);run,asset=self.mock_guard(p,fail_stage='pool')
   with patch.object(worker,'bind_parent'),patch.object(worker,'validate',return_value=p),patch.object(worker,'boundary'),patch.object(worker,'storage_boundary'),patch.object(worker,'run_stage',side_effect=run) as native,patch.object(worker,'asset_read',side_effect=asset),patch.object(worker,'pin'):
    with self.assertRaises(RuntimeError):worker.run({'path':'mockplan','sha256':'mock'},{'path':'mockreview','sha256':'mock'})
    self.assertEqual(native.call_count,4)
    with self.assertRaises(FileExistsError):worker.run({'path':'mockplan','sha256':'mock'},{'path':'mockreview','sha256':'mock'})
    self.assertEqual(native.call_count,4)
   status=json.loads((root/'status.json').read_bytes());self.assertEqual(status['original_denominator'],12);self.assertFalse(status['automatic_retry']);self.assertEqual(status['completed_mothers'],0)
 def test_completed_native_still_closes_owned_subtree(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);child=self.mock_child(root,exitcode=0);closed=Mock()
   def spawn(*args,**kw):c.immutable(root/'stage_output.json',{'mock':True});return child
   with patch.object(guard,'identity',return_value=OWNER),patch.object(guard.os,'getpgid',return_value=12345,create=True):r=guard.run_stage(['mock'],root,'common',stop=lambda:None,popen=spawn,terminate=closed,memory=lambda pid:0)
   self.assertEqual(r['status'],'completed');closed.assert_called_once_with(child,OWNER)
 def test_subtree_after_leader_exit_kills_exact_member_not_PGID(self):
  child=SimpleNamespace(pid=12345,poll=lambda:0);grandchild=dict(OWNER,pid=12346,birth_ticks=12)
  with patch.object(guard,'group_members',side_effect=[[grandchild],[]]),patch.object(guard,'identity',return_value=grandchild),patch.object(guard.os,'kill') as kill,patch.object(guard.os,'killpg',create=True) as killpg,patch.object(guard.signal,'SIGKILL',9,create=True):
   guard.terminate_owned(child,OWNER)
  kill.assert_called_once_with(12346,9);killpg.assert_not_called()
 def test_subtree_samePID_rebirth_is_not_killed(self):
  child=SimpleNamespace(pid=12345,poll=lambda:0);grandchild=dict(OWNER,pid=12346,birth_ticks=12)
  with patch.object(guard,'group_members',side_effect=[[grandchild],[]]),patch.object(guard,'identity',return_value=dict(grandchild,birth_ticks=13)),patch.object(guard.os,'kill') as kill:
   guard.terminate_owned(child,OWNER)
  kill.assert_not_called()
 def test_human_epoch_is_newR9_not_remainingR8_contract(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=self.plan(Path(tmp));p['human_resume_authority_ref']={'path':'originalhuman','sha256':c.AUTHORITY_SHA};authority=dict(human_authorization=True,root_accepted=True,user_message_verbatim='继续');epoch=dict(schema='R9NewScientificHumanEpochV1',root_accepted=True,group=c.GROUP,stage='M',new_deadline_unix_seconds=None,human_resume_authority_ref=p['human_resume_authority_ref'],run_id=p['run_id'],request_id=p['request_id'],archive_root=p['archive_root'],execution_epoch_id='newR9epoch',watched_pause_paths=['exact_hold'],historical_hold_exemptions=[])
   with patch.object(c,'read',side_effect=[authority,epoch]):self.assertEqual(c.human_epoch(p),epoch)
   with patch.object(c,'read',side_effect=[authority,dict(epoch,schema='R8ExplicitHumanResumeEpochContractV1')]):
    with self.assertRaises(ValueError):c.human_epoch(p)
 def test_exact_root_review_positive_and_scientific_false_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'round9/archive';p=self.plan(root);p.update(archive_root=root.as_posix(),schema='R9MechanismMExactPlanV1',caps=dict(c.CAPS),case_indices=list(c.M_INDICES),design_ref={'path':'design','sha256':c.DESIGN_SHA},root_design_acceptance_ref={'path':'design_accept','sha256':'mock'},human_resume_authority_ref={'path':'authority','sha256':c.AUTHORITY_SHA},RULE_floor_clarification_ref={'path':'floor','sha256':c.RULE_FLOOR_SHA},input_qualification_ref={'path':'qualification','sha256':'mock'},storage_inventory_ref={'path':'storage','sha256':'mock'})
   p['cases']=[dict(original_index=i,identity={'role':'train'},uniform_request_bps=21600000,mechanism12=True,scene_ref={},prepared_ref={},shared_case_ref={}) for i in c.M_INDICES];pr={'path':'plan','sha256':'mock'};rr={'path':'review','sha256':'mock'};r=dict(schema='R9RootMExecutionAdmissionV1',plan_ref=pr,passed=True,execution_authorized=True,scientific_execution_authorized=True,source_accepted=True,input_accepted=True,runtime_accepted=True,**{k:p[k] for k in ('human_epoch_ref','input_qualification_ref','storage_inventory_ref')});accepted=dict(schema='R9RootFiniteDesignAcceptanceV1',accepted=True,design_ref=p['design_ref'],authority_ref=p['human_resume_authority_ref']);items={'plan':p,'review':r,'design':{},'design_accept':accepted,'floor':{},'qualification':dict(passed=True,heldout_reads=False,cases=p['cases'])}
   with patch.object(c,'read',side_effect=lambda ref:items[ref['path']]),patch.object(c,'pin'),patch.object(c,'source_check') as source,patch.object(c,'storage_check'),patch.object(c,'human_epoch'):
    self.assertEqual(c.validate(pr,rr),p);source.assert_called_once_with(p);r['scientific_execution_authorized']=False
    with self.assertRaises(ValueError):c.validate(pr,rr)
 def test_no_source_manifest_bypass_on_wrong_import_path(self):
  import sys
  with tempfile.TemporaryDirectory() as tmp:
   source=Path(tmp)/'original';source.mkdir();bad=SimpleNamespace(__file__=str(Path(tmp)/'outside.py'))
   p={'source_files':{},'original_source_manifest_ref':{'path':str(source/'manifest.json'),'sha256':'mock'}}
   with patch.object(c,'read',return_value={'files':{}}),patch.dict(sys.modules,{'satellite_rl':bad}):
    with self.assertRaises(ValueError):c.imported_check(p)
 def test_short_child_completed_output_still_fails_hard_deadline(self):
  for stage,limit in [('RULE',15),('D5',15),('common',30),('pool',60)]:
   with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);child=self.mock_child(root,exitcode=0);ticks=iter([0,limit+.01])
    def spawn(*args,**kw):c.immutable(root/'stage_output.json',{'mock':True});return child
    with patch.object(guard,'identity',return_value=OWNER),patch.object(guard.os,'getpgid',return_value=12345,create=True):
     with self.assertRaises(RuntimeError):guard.run_stage(['mock'],root,stage,stop=lambda:None,popen=spawn,terminate=Mock(),memory=lambda pid:0,clock=lambda:next(ticks))
    r=json.loads((root/'guard_receipt.json').read_bytes());self.assertEqual(r['status'],'partial_unknown');self.assertEqual(r['reason'],'hard_time_guard')
 def test_short_child_without_owner_is_unknown_even_success_output(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);child=self.mock_child(root,exitcode=0);closed=Mock()
   def spawn(*args,**kw):c.immutable(root/'stage_output.json',{'mock':True});return child
   with patch.object(guard,'identity',side_effect=[OWNER,None]):
    with self.assertRaises(RuntimeError):guard.run_stage(['mock'],root,'pool',stop=lambda:None,popen=spawn,terminate=closed,memory=lambda pid:0)
   closed.assert_not_called();r=json.loads((root/'guard_receipt.json').read_bytes());self.assertEqual(r['status'],'partial_unknown');self.assertFalse(r['subtree_exit_verified']);self.assertIsNone(r['actual_child'])
 def test_rss_statm_disappearance_only_ENOENT_ESRCH(self):
  for code in (errno.ENOENT,errno.ESRCH,errno.EACCES):
   proc=Mock();statm=Mock();statm.read_text.side_effect=OSError(code,'mock statm');proc.__truediv__=Mock(return_value=statm);base=Mock();base.__truediv__=Mock(return_value=proc)
   with patch.object(guard,'Path',return_value=base),patch.object(guard.os,'sysconf',return_value=4096,create=True):
    if code in (errno.ENOENT,errno.ESRCH):self.assertEqual(guard.rss_tree(12345),0)
    else:
     with self.assertRaises(OSError):guard.rss_tree(12345)
 def test_rss_thread_children_ESRCH_keeps_observed_RSS(self):
  proc=Mock();statm=Mock();statm.read_text.return_value='9 2 0';children=Mock();children.read_text.side_effect=OSError(errno.ESRCH,'mock thread gone');thread=Mock();thread.__truediv__=Mock(return_value=children);task=Mock();task.iterdir.return_value=[thread];proc.__truediv__=Mock(side_effect=lambda name:statm if name=='statm' else task);base=Mock();base.__truediv__=Mock(return_value=proc)
  with patch.object(guard,'Path',return_value=base),patch.object(guard.os,'sysconf',return_value=4096,create=True):self.assertEqual(guard.rss_tree(12345),8192)
 def test_group_session_leader_reused_refuses_enumeration(self):
  boot=Mock();boot.read_text.return_value='mockboot';newleader=dict(OWNER,birth_ticks=999)
  with patch.object(guard,'Path',return_value=boot),patch.object(guard,'identity',return_value=newleader):
   with self.assertRaisesRegex(RuntimeError,'leader PID reused'):guard.group_members(OWNER)
  boot.iterdir.assert_not_called()
 def test_group_session_leader_reused_during_observation_refuses_members(self):
  proc=Mock();proc.read_text.return_value='mockboot';proc.iterdir.return_value=[];newleader=dict(OWNER,birth_ticks=999)
  with patch.object(guard,'Path',return_value=proc),patch.object(guard,'identity',side_effect=[None,newleader]):
   with self.assertRaisesRegex(RuntimeError,'reused during observation'):guard.group_members(OWNER)
 def group_without_leader(self,known):
  boot=Mock();boot.read_text.return_value='mockboot';proc=Mock();proc.name='12346';stat=Mock();fields=['S','1','12345','12345']+['0']*15+['12'];stat.read_text.return_value='12346 (mock) '+' '.join(fields);proc.__truediv__=Mock(return_value=stat);base=Mock();base.iterdir.return_value=[proc];member=dict(OWNER,pid=12346,birth_ticks=12)
  def ident(pid):return None if pid==OWNER['pid'] else member
  with patch.object(guard,'Path',side_effect=lambda name:boot if str(name).endswith('boot_id') else base),patch.object(guard,'identity',side_effect=ident):return guard.group_members(OWNER,[member] if known else [])
 def test_vanished_leader_unregistered_same_numeric_group_is_unknown(self):
  with self.assertRaisesRegex(RuntimeError,'unproven group generation'):self.group_without_leader(False)
 def test_vanished_leader_only_exact_registered_generation_allowed(self):
  self.assertEqual(self.group_without_leader(True),[dict(OWNER,pid=12346,birth_ticks=12)])
if __name__=='__main__':unittest.main()
