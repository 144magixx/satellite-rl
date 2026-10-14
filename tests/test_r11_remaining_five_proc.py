"""Real b137 vs production observer, fake/proc and complete reaper exception path."""
import errno,importlib.util,types,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.stage1_ai_search_r11_prospective import contracts as candidate
from tests.test_r11_prospective_reaper import ReaperChecks,Life
P=dict(pid=100,birth_ticks=10,boot_id='b',uid=500902,state='R');C=dict(pid=200,birth_ticks=20,boot_id='b',uid=500902,state='S')
class FakePath:
 state='S';ppid=100;fail=False
 def __init__(self,path):self.value=str(path)
 def __truediv__(self,k):return FakePath(self.value+'/'+str(k))
 def iterdir(self):return [FakePath(self.value+'/1')]
 def read_text(self):
  if self.value.endswith('/children'):
   if self.fail:raise PermissionError(errno.EACCES,'not absent')
   return '200' if self.value.startswith('/proc/100/') else ''
  if self.value=='/proc/200/stat':
   f=['0']*20;f[0]=self.state;f[1]=str(self.ppid);f[19]='20';return '200 (fake) '+' '.join(f)
  raise AssertionError(self.value)
def exercise(known=True,zchild=False,end='Z',reuse=False,permission=False):
 n=[0];cn=[0];FakePath.state='Z' if zchild else 'S';FakePath.ppid=1 if zchild else 100;FakePath.fail=permission;reg={(100,10,'b'):P}
 if known:reg[(200,20,'b')]=C
 def identity(pid):
  if pid==100:
   n[0]+=1
   if n[0]>=3:return None if end=='absent' else dict(P,state=end)
   return P
  cn[0]+=1;return dict(C,state=FakePath.state,birth_ticks=21 if reuse and cn[0]>=2 else 20)
 c=types.SimpleNamespace(identity=identity,same_process=lambda a,b:bool(a and b and all(a[k]==b[k] for k in ('pid','birth_ticks','boot_id','uid'))))
 with patch.object(candidate,'Path',FakePath):candidate.observe_descendants(c,P,reg)
 return reg
class ProcTransitionChecks(unittest.TestCase):
 def test_known_parent_to_Z_no_new_ownership(self):self.assertEqual(set(exercise()),{(100,10,'b'),(200,20,'b')})
 def test_known_parent_disappeared_no_new_ownership(self):self.assertEqual(set(exercise(end='absent')),{(100,10,'b'),(200,20,'b')})
 def test_known_Z_reparented_is_not_live_candidate(self):self.assertEqual(set(exercise(zchild=True)),{(100,10,'b'),(200,20,'b')})
 def test_unknown_live_on_ended_parent_fails(self):
  with self.assertRaisesRegex(RuntimeError,'unregistered live'):exercise(known=False)
 def test_unknown_Z_does_not_get_new_ownership(self):
  with self.assertRaisesRegex(RuntimeError,'unregistered ended'):exercise(known=False,zchild=True)
 def test_reused_PID_never_admitted(self):
  with self.assertRaisesRegex(RuntimeError,'unproven live'):exercise(reuse=True)
 def test_non_ENOENT_error_is_not_absence(self):
  with self.assertRaises(PermissionError):exercise(permission=True)
 def test_stable_parent_exact_child_can_register(self):self.assertIn((200,20,'b'),exercise(known=False,end='R'))
 def test_full_reaper_observe_error_still_persists_direct_wait4_not_completion(self):
  with patch.object(Life,'observe_descendants',side_effect=RuntimeError('fake/proc race')):value,error,files,calls,life,clock=ReaperChecks().runfake()
  self.assertIsNotNone(error);self.assertEqual(files['native_reap.json']['raw_wait_status'],0);self.assertEqual(files['guard_receipt.json']['status'],'partial_unknown');self.assertFalse(files['guard_receipt.json']['subtree_exit_verified']);self.assertIn('closure_unknown',files['guard_receipt.json']['error']);self.assertEqual(len(calls),1)
if __name__=='__main__':unittest.main()
