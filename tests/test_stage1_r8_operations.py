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
