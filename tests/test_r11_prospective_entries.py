"""Actual fresh entry bodies with fake spawn/reap/lifecycle, no native process."""
import importlib.util,json,os,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.stage1_ai_search_r11_prospective import server,controller,noop,outer
from scripts.stage1_ai_search_r10 import contracts as c

PARENT=dict(pid=os.getpid(),birth_ticks=100,boot_id='boot',uid=500902,state='R')
CHILD=dict(pid=888,birth_ticks=200,boot_id='boot',uid=500902,state='R')
WORKER=dict(pid=889,birth_ticks=300,boot_id='boot',uid=500902,state='R')
PR=dict(path='new_plan.json',sha256='plan');RR=dict(path='review.json',sha256='review')
class Thread:
 def __init__(self,target,**kw):self.target=target
 def start(self):return None
class Observer:
 def __init__(self,*a,**kw):pass
 def observe(self,*a):return {}
 def close(self):pass
class Watch:
 def __init__(self,*a):pass

class EntryChecks(unittest.TestCase):
 def test_full_server_post_and_worker_wait_body(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);p=dict(run_id='r11_fake',request_id='request',archive_root=str(root),design_ref=dict(path='design',sha256='d'));e=server.Endpoint.__new__(server.Endpoint);e.plan_ref=PR;e.review_ref=RR;e.plan=p;e.root=root;e.path='/task'
   calls=[]
   def popen(command,**kw):calls.append((command,kw));return types.SimpleNamespace(pid=888,returncode=None)
   with patch.object(server,'validate',return_value=p),patch.object(server,'check_pause'),patch.object(server,'fresh_route'),patch.object(server,'boottime',return_value=100),patch.object(server.c,'identity',side_effect=lambda pid:PARENT if pid==os.getpid() else CHILD),patch.object(server.threading,'Thread',Thread),patch.dict(os.environ,{'R11_LANE_DEADLINE_BOOTTIME':'2500'}):
    out=e.post('/task/start',dict(plan_ref=PR,root_review_ref=RR,request_id='request'),popen=popen)
    self.assertEqual(len(calls),1);self.assertFalse(out['POST_replayed']);self.assertTrue(callable(calls[0][1]['preexec_fn']))
    saved=dict(actual_child=CHILD,raw_wait_status=0,decoded_exit_code=0,wait4_observed_clock_boottime_seconds=101,rusage=dict(maxrss_bytes=1024))
    with patch.object(server,'wait_one',return_value=saved):e.worker_reaper.target()
   record=json.loads((root/'worker_reap.json').read_bytes());self.assertEqual(record['decoded_exit_code'],0);self.assertEqual(record['actual_child'],CHILD)
 def test_full_noop_entry_reaches_only_fake_native_runner(self):
  class ReachedNative(Exception):pass
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp)/'round11/noop';p=dict(output_root=str(root),pause_plan={},actual_resource_route_ref={},payload_ref=dict(path='/owned/unique_noop_payload.py'));calls=[]
   def native(*a,**kw):calls.append((a,kw));raise ReachedNative()
   with patch.object(noop,'check',return_value=p),patch.object(noop,'Watchdog',Watch),patch.object(noop.c,'identity',return_value=PARENT),patch.object(noop.os,'sysconf',return_value=100,create=True),patch.object(noop,'boottime',return_value=1.1),patch.object(noop.sys,'platform','linux'),patch.object(noop.sys,'executable',noop.r.DQN),patch.object(noop.r,'fresh_route'),patch.object(noop.r,'lifecycle',return_value=object()),patch.object(noop,'run_process',side_effect=native):
    with self.assertRaises(ReachedNative):noop.run(PR,RR)
   self.assertEqual(len(calls),1);self.assertEqual(calls[0][1]['rss_bytes'],134217728);self.assertLess(calls[0][1]['hard_seconds'],10);self.assertEqual(calls[0][1]['helper_token'],'/owned/unique_noop_payload.py')
 def test_full_controller_body_durable_worker_and_service_exit(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp)/'round11';archive=root/'archive';control=root/'control';archive.mkdir(parents=True);p=dict(archive_root=str(archive),outer_control_root=str(control),source_repo='/new/source',outer_hard_seconds=2400,workbench_port=9050,design_ref=dict(path='d',sha256='d'));state=dict(live=True)
   class Life:
    def observe_descendants(self,*a):pass
    def registered_rss(self,*a):return 4096
    def exact_terminate(self,*a):state['live']=False
    def final_tree_observation(self,c,p,registry):return dict(registered_live=[CHILD] if state['live'] else [],unknown_live=[],all_execution_ended=not state['live'])
   def identity(pid):return PARENT if pid==os.getpid() else CHILD if pid==888 and state['live'] else None
   def popen(command,**kw):
    c.immutable(archive/'actual_worker_launch.json',dict(plan_ref=PR,actual_worker=WORKER));c.immutable(archive/'worker_reap.json',dict(actual_child=WORKER,plan_ref=PR,decoded_exit_code=0));c.immutable(archive/'status.json',dict(stage='completed_pending_independent_audit'));return types.SimpleNamespace(pid=888,returncode=None)
   def wait(child,owner):
    if state['live']:return None
    child.returncode=-15;return dict(actual_child=owner,raw_wait_status=15,decoded_exit_code=-15,wait4_observed_clock_boottime_seconds=101,rusage={})
   with patch.object(controller,'validate',return_value=p),patch.object(controller,'fresh_route'),patch.object(controller,'check_pause'),patch.object(controller,'lifecycle',return_value=Life()),patch.object(controller,'Observer',Observer),patch.object(controller.c,'identity',side_effect=identity),patch.object(controller.subprocess,'Popen',side_effect=popen) as spawn,patch.object(controller,'wait_one',side_effect=wait),patch.object(controller,'boottime',return_value=100),patch.object(controller.sys,'platform','linux'),patch.object(controller.sys,'executable',controller.DQN),patch.dict(os.environ,{'R11_LANE_DEADLINE_BOOTTIME':'2500'}):controller.run(PR,RR)
   record=json.loads((control/'controller_finished.json').read_bytes());self.assertIsNone(record['error']);self.assertEqual(record['worker_exit_code_exact'],0);self.assertIsNotNone(record['service_reap_ref']);self.assertEqual(spawn.call_count,1)
 def test_reused_exact_terminate_never_signals_reused_PID(self):
  # Reuse precisely the accepted legacy function with fake imports and kernel identity.
  path=Path('D:/paper/outputs/experiments/first_stage_ai_search_20261013/round10/runtime_preparation_01/engineering_persistent_launcher_once.py');stub=types.SimpleNamespace(**{k:None for k in ('DQN','read','sha','validate','bootstrap','boundary')})
  with patch.dict('sys.modules',{'engineering_driver':stub}):
   spec=importlib.util.spec_from_file_location('accepted_exact_terminate_for_mock',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  changed=dict(CHILD,birth_ticks=201);fakec=types.SimpleNamespace(identity=lambda pid:changed,same_process=c.same_process)
  with patch.object(m.os,'kill') as kill:m.exact_terminate(fakec,{'child':CHILD},15);kill.assert_not_called()
if __name__=='__main__':unittest.main()
