"""Full reaper body against fake Popen/wait4/clock/lifecycle only; no native child."""
import errno,json,os,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.stage1_ai_search_r11_prospective import reaper as r

class Clock:
 def __init__(self):self.now=100.
 def __call__(self):return self.now
 def sleep(self,n):self.now+=n
class Watch:
 def __init__(self,deadline,clock):self.deadline=deadline;self.closed=False
 def after_native(self,d):self.deadline=d
 def close(self):self.closed=True
class Life:
 def __init__(self,state):self.state=state;self.signals=[]
 def registered_rss(self,c,registry):return self.state.get('rss',4096)
 def observe_descendants(self,c,owner,registry):return None
 def final_tree_observation(self,c,p,registry):
  live=list(registry.values()) if self.state['live'] else []
  return dict(registered_observations=[],registered_live=live,unknown_live=[dict(pid=99)] if self.state.get('unknown') else [],all_execution_ended=not live and not self.state.get('unknown'))
 def exact_terminate(self,c,registry,sig):
  self.signals.append(sig)
  if sig==getattr(r.signal,'SIGKILL',9) or not self.state.get('ignore_term'):self.state['live']=False;self.state['status']=int(sig)

def linux_status(status):return -(status&127) if status&127 else status>>8
def usage():return types.SimpleNamespace(ru_utime=.01,ru_stime=.01,ru_maxrss=4,ru_minflt=0,ru_majflt=0,ru_nvcsw=0,ru_nivcsw=0)

class ReaperChecks(unittest.TestCase):
 def runfake(self,*,status=0,finish_after=0,interrupted=False,unknown=False,ignore_term=False,owner_boot='boot',publish_failure=None,delayed_exit=False,missing_wait=False,rss_sample=4096,exact_exit=False,tail_delay=False):
  clock=Clock();state=dict(live=True,status=status,unknown=unknown,ignore_term=ignore_term,rss=rss_sample);life=Life(state);parent=dict(pid=os.getpid(),birth_ticks=10,boot_id='boot',uid=500902,state='R');owner=dict(pid=888,birth_ticks=20,boot_id=owner_boot,uid=500902,state='R');child=types.SimpleNamespace(pid=888,returncode=None);calls=[];waitcount=0
  def identity(pid):return parent if pid==os.getpid() else owner
  def popen(command,**kwargs):
   calls.append(dict(command=command,kwargs=kwargs))
   if delayed_exit:r.c.immutable(root/'stage_output.json',dict(asset_ref=dict(path='fake',sha256='fake')))
   return child
  def wait4(pid,flags):
   nonlocal waitcount
   waitcount+=1
   if exact_exit:clock.now=101.
   if missing_wait:raise ChildProcessError(errno.ECHILD,'lost owner wait')
   if interrupted and waitcount==1:raise InterruptedError()
   if state['live'] and clock.now<100+finish_after:return(0,0,usage())
   state['live']=False
   return(pid,state['status'],usage())
  realpub=r.c.immutable
  def publish(path,value):
   if publish_failure and Path(path).name==publish_failure:raise OSError('fake persistence failure')
   value_ref=realpub(path,value)
   if tail_delay and Path(path).name=='guard_receipt.json':clock.now+=3
   return value_ref
  result=None;error=None
  with tempfile.TemporaryDirectory() as tmp,patch.dict('sys.modules',{'resource':types.SimpleNamespace(RUSAGE_SELF=0,getrusage=lambda target:usage())}),patch.object(r.os,'waitstatus_to_exitcode',side_effect=linux_status),patch.object(r.os,'WNOHANG',1,create=True),patch.object(r.signal,'SIGKILL',9,create=True):
   root=Path(tmp)/'native'
   try:result=r.run_process(['DQN','-B','unique_payload.py'],root,'mock',stop=lambda:None,life=life,hard_seconds=1,rss_bytes=134217728,helper_token='unique_payload.py',clock=clock,sleep=clock.sleep,popen=popen,wait4=wait4,publish=publish,identity=identity,watchdog_factory=Watch)
   except Exception as e:error=e
   files={p.name:json.loads(p.read_bytes()) for p in root.glob('*.json')}
  return result,error,files,calls,life,clock
 def test_fast_zero_exit_records_real_wait_and_bound(self):
  value,error,files,calls,life,clock=self.runfake();self.assertIsNone(error);self.assertEqual(len(calls),1);self.assertEqual(value['exit_code'],0);self.assertEqual(files['native_reap.json']['raw_wait_status'],0);self.assertEqual(files['guard_receipt.json']['status'],'completed');self.assertEqual(files['native_reap.json']['rusage']['maxrss_bytes'],4096)
 def test_normal_nonzero_exit_not_completed(self):
  value,error,files,*_=self.runfake(status=7<<8);self.assertIsNotNone(error);self.assertEqual(files['native_reap.json']['decoded_exit_code'],7);self.assertEqual(files['guard_receipt.json']['status'],'partial_unknown')
 def test_term_timeout_preserves_nonzero_actual_status(self):
  value,error,files,calls,life,_=self.runfake(finish_after=99);self.assertIsNotNone(error);self.assertIn(r.signal.SIGTERM,life.signals);self.assertEqual(files['native_reap.json']['decoded_exit_code'],-15)
 def test_kill_after_escalation(self):
  value,error,files,calls,life,_=self.runfake(finish_after=99,ignore_term=True);self.assertIsNotNone(error);self.assertEqual(life.signals,[r.signal.SIGTERM,9]);self.assertEqual(files['native_reap.json']['decoded_exit_code'],-9)
 def test_wait_interruption_retried_without_respawn(self):
  value,error,files,calls,*_=self.runfake(interrupted=True);self.assertIsNone(error);self.assertEqual(len(calls),1);self.assertEqual(files['native_reap.json']['decoded_exit_code'],0)
 def test_persistence_failure_not_success(self):
  value,error,files,*_=self.runfake(publish_failure='native_reap.json');self.assertIsNotNone(error);self.assertNotIn('native_reap.json',files);self.assertNotEqual(files.get('guard_receipt.json',{}).get('status'),'completed')
 def test_guard_output_before_delayed_exit_not_completion(self):
  value,error,files,calls,life,clock=self.runfake(finish_after=.6,delayed_exit=True);self.assertIsNone(error);self.assertGreaterEqual(files['native_reap.json']['conservative_process_exit_upper_seconds'],.6)
 def test_boot_mismatch_fails(self):
  value,error,files,*_=self.runfake(owner_boot='another');self.assertIsNotNone(error);self.assertNotEqual(files.get('guard_receipt.json',{}).get('status'),'completed')
 def test_unknown_descendant_never_signalled(self):
  value,error,files,calls,life,*_=self.runfake(unknown=True);self.assertIsNotNone(error);self.assertEqual(life.signals,[]);self.assertNotEqual(files.get('guard_receipt.json',{}).get('status'),'completed');self.assertIs(files['guard_receipt.json']['observer_final_receipt_IO_after_native_closure'],False);self.assertIs(files['guard_receipt.json']['observer_receipt_IO_not_hidden_native_work'],False)
 def test_delayed_collection_does_not_change_saved_bound(self):
  value,error,files,calls,life,clock=self.runfake();before=files['native_reap.json']['conservative_process_exit_upper_seconds'];clock.now+=70;self.assertEqual(before,files['native_reap.json']['conservative_process_exit_upper_seconds'])
 def test_final_sampled_RSS_checked_even_when_fast_reaped(self):
  value,error,files,*_=self.runfake(rss_sample=134217729);self.assertIsNotNone(error);self.assertEqual(files['guard_receipt.json']['reason'],'final_sampled_parent_children_RSS')
 def test_exact_deadline_is_failure(self):
  value,error,files,*_=self.runfake(exact_exit=True);self.assertIsNotNone(error);self.assertEqual(files['guard_receipt.json']['reason'],'full_native_exit_bound_exceeds_hard_limit')
 def test_observer_receipt_tail_explicitly_bounded(self):
  value,error,files,*_=self.runfake(tail_delay=True);self.assertIsNotNone(error);self.assertIn('bounded tail',str(error))
 def test_missing_wait_status_never_completed(self):
  value,error,files,*_=self.runfake(missing_wait=True);self.assertIsNotNone(error);self.assertNotEqual(files.get('guard_receipt.json',{}).get('status'),'completed')

if __name__=='__main__':unittest.main()
