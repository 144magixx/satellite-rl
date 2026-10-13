"""Fail-closed R8 boundaries registered separately from original R7/archive."""
import json,os,signal,time,ctypes,sys
from pathlib import Path
from .contracts import DEADLINE,read,atomic,identity,GROUP

class Paused(RuntimeError):pass
def boundary(plan,point):
 registration=read(plan['deadline_registration_ref'])
 if registration.get('root_accepted')is not True or registration.get('archive_root')!=plan['archive_root'] or registration.get('group')!=GROUP or registration.get('deadline_unix_seconds')!=DEADLINE:raise ValueError('registered own archive boundary required')
 held=[]
 for path in registration['shared_pause_paths']:
  p=Path(path)
  if p.exists():
   value=json.loads(p.read_bytes())
   if value.get('paused') or value.get('hold') or value.get('pause_required'):held.append(path)
 if time.time()>=DEADLINE or held or (Path(plan['archive_root'])/'stop.request').exists():
  atomic(Path(plan['archive_root'])/'pause_state.json',dict(group=GROUP,paused=True,requires_explicit_human_continue=True,point=point,actual_process=identity(os.getpid()),deadline_unix_seconds=DEADLINE,held_paths=held))
  raise Paused('R8 paused; explicit human continue and new root admission required')
 return dict(passed=True,point=point,deadline_unix_seconds=DEADLINE,actual_process=identity(os.getpid()),deadline_registration_ref=plan['deadline_registration_ref'])
def bind_parent(variable):
 if sys.platform!='linux':raise ValueError('native Linux parent guard required')
 expected=int(os.environ[variable]);libc=ctypes.CDLL(None,use_errno=True)
 if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:raise OSError(ctypes.get_errno(),'parent death guard failed')
 if os.getppid()!=expected:raise ValueError('original child parent changed')
