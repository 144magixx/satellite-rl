"""Pure operations mocks: pause and ownership failures never start science."""
import tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from scripts.stage1_ai_search_r8 import deadline,contracts,server

class Operations(unittest.TestCase):
 def test_order_complete_samecase_rotation(self):
  self.assertEqual([contracts.order(i) for i in range(3)],[['G','A','R'],['A','R','G'],['R','G','A']])
  self.assertEqual(sum(len(contracts.order(i)) for i in range(20)),60)
 def registration(self,root):return dict(root_accepted=True,archive_root=str(root),group=contracts.GROUP,deadline_unix_seconds=contracts.DEADLINE,shared_pause_paths=[str(root/'shared_pause.json')])
 def test_exact_deadline_and_shared_pause_fail_closed(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);plan=dict(archive_root=str(root),deadline_registration_ref={})
   with patch.object(deadline,'read',return_value=self.registration(root)),patch.object(deadline,'identity',return_value=None),patch.object(deadline.time,'time',return_value=contracts.DEADLINE):
    with self.assertRaises(deadline.Paused):deadline.boundary(plan,'mock_prePOST')
   contracts.atomic(root/'shared_pause.json',dict(paused=True))
   with patch.object(deadline,'read',return_value=self.registration(root)),patch.object(deadline,'identity',return_value=None),patch.object(deadline.time,'time',return_value=contracts.DEADLINE-100):
    with self.assertRaises(deadline.Paused):deadline.boundary(plan,'mock_child')
   contracts.atomic(root/'shared_pause.json',dict(global_dispatch_hold=True))
   with patch.object(deadline,'read',return_value=self.registration(root)),patch.object(deadline,'identity',return_value=None),patch.object(deadline.time,'time',return_value=contracts.DEADLINE-100):
    with self.assertRaises(deadline.Paused):deadline.boundary(plan,'mock_queue')
 def test_registration_cannot_reuse_r7_archive(self):
  with tempfile.TemporaryDirectory() as tmp:
   value=self.registration(Path(tmp));value['archive_root']+='wrong'
   with patch.object(deadline,'read',return_value=value):
    with self.assertRaises(ValueError):deadline.boundary(dict(archive_root=tmp,deadline_registration_ref={}),'worker')
 def test_new_child_refused_only_within20seconds_deadline(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);plan=dict(archive_root=tmp,deadline_registration_ref={})
   with patch.object(deadline,'read',return_value=self.registration(root)),patch.object(deadline,'identity',return_value=None),patch.object(deadline.time,'time',return_value=contracts.DEADLINE-19),patch.object(deadline,'storage_boundary',return_value={}):
    with self.assertRaises(deadline.Paused):deadline.before_start(plan,'local_before_intent')
   fresh=root/'fresh_before_stop';fresh.mkdir();fresh_plan=dict(archive_root=str(fresh),deadline_registration_ref={})
   with patch.object(deadline,'read',return_value=self.registration(fresh)),patch.object(deadline,'identity',return_value=None),patch.object(deadline.time,'time',return_value=contracts.DEADLINE-21),patch.object(deadline,'storage_boundary',return_value={}):self.assertTrue(deadline.before_start(fresh_plan,'D5_before_intent')['passed'])
 def sources(self):
  root=Path(contracts.__file__).resolve().parents[2];files={}
  for name in contracts.OPS_FILES:
   p=Path(contracts.__file__).with_name(name);files['scripts/stage1_ai_search_r8/'+name]=dict(path=str(p),sha256=contracts.digest(p))
  for name in contracts.CORE_FILES:
   p=root/'src/satellite_rl/coverage/ai_search_r8'/name;files['src/satellite_rl/coverage/ai_search_r8/'+name]=dict(path=str(p),sha256=contracts.digest(p))
  return dict(source_files=files,core_source_root=str(root/'src'),core_ref=files['src/satellite_rl/coverage/ai_search_r8/core.py'])
 def test_complete_four_core_and_seven_operations_required(self):
  p=self.sources();contracts.validate_sources(p);del p['source_files']['src/satellite_rl/coverage/ai_search_r8/ledger.py']
  with self.assertRaises(ValueError):contracts.validate_sources(p)
 def test_actual_relative_module_path_cannot_differ(self):
  from types import SimpleNamespace
  p=self.sources();modules={name:SimpleNamespace(__file__=p['source_files']['src/satellite_rl/coverage/ai_search_r8/'+name]['path']) for name in contracts.CORE_FILES};contracts.validate_imported_core(p,modules)
  modules['geometry.py']=SimpleNamespace(__file__=modules['core.py'].__file__)
  with self.assertRaises(ValueError):contracts.validate_imported_core(p,modules)
 def test_storage_cap_and_reserve_pause_without_deletion(self):
  from collections import namedtuple
  Disk=namedtuple('Disk','total used free')
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'evidence').write_bytes(b'protected');plan=dict(archive_root=tmp,storage_inventory_ref={});inventory=dict(owned_growth_root=tmp,additional_owned_roots=[],fixed_other_owned_bytes=1073741825)
   with patch.object(deadline,'storage_inventory',return_value=inventory),patch.object(deadline.shutil,'disk_usage',return_value=Disk(10**11,0,10**11)):
    with self.assertRaises(deadline.Paused):deadline.storage_boundary(plan,'before_local')
   inventory['fixed_other_owned_bytes']=0
   with patch.object(deadline,'storage_inventory',return_value=inventory),patch.object(deadline.shutil,'disk_usage',return_value=Disk(10**11,0,8589934591)):
    with self.assertRaises(deadline.Paused):deadline.storage_boundary(plan,'before_D5')
   self.assertEqual((root/'evidence').read_bytes(),b'protected')
 def endpoint(self,root):
  e=server.Endpoint.__new__(server.Endpoint);e.root=root;e.plan_ref=dict(path='plan',sha256='p');e.review_ref=dict(path='review',sha256='r');e.plan=dict(run_id='r8_pilot_mock',request_id='exact',archive_root=str(root));e.path='/api/ai-search-r8/tasks/r8_pilot_mock';return e
 def test_wrong_post_body_or_old_endpoint_never_spawns(self):
  with tempfile.TemporaryDirectory() as tmp:
   e=self.endpoint(Path(tmp));spawn=Mock()
   with self.assertRaises(ValueError):e.post(e.path+'/start',{},popen=spawn)
   body=dict(plan_ref=e.plan_ref,root_review_ref=e.review_ref,request_id='exact')
   with self.assertRaises(ValueError):e.post('/api/ai-search-r7/tasks/old/resume',body,popen=spawn)
   spawn.assert_not_called()
 def test_safe_stop_only_marks_own_archive(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);e=self.endpoint(root);body=dict(plan_ref=e.plan_ref,root_review_ref=e.review_ref,request_id='exact');spawn=Mock()
   with patch.object(server,'identity',return_value=None):self.assertTrue(e.post(e.path+'/stop',body,popen=spawn)['stop_requested'])
   self.assertTrue((root/'stop.request').exists());spawn.assert_not_called()
 def test_immutable_owner_cannot_be_overwritten(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'intent.json';contracts.immutable(p,dict(request='first'))
   with self.assertRaises(FileExistsError):contracts.immutable(p,dict(request='second'))
   self.assertEqual(contracts.read(dict(path=str(p),sha256=contracts.digest(p))),dict(request='first'))
 def test_stale_resource_rejected(self):
  with patch.object(contracts,'read',return_value=dict(observed_unix_seconds=0,ready=True,root_accepted=True,ssh_port=20474)):
   with self.assertRaises(ValueError):contracts.fresh_route(dict(actual_resource_route_ref={},executor_port=20474))
if __name__=='__main__':unittest.main()
