"""Exact historical pause exemption; all new/changed generations remain blocking."""
import ctypes,hashlib,json,os,shutil,signal,sys
from contextlib import contextmanager
from pathlib import Path
from .contracts import AUTHORITY_SHA,pin,publish,human_epoch,storage_check

class Paused(RuntimeError):pass
FINGERPRINT=('sha256','size','st_dev','st_ino','st_mtime_ns','st_ctime_ns')
FLAGS=('user_pause','paused','hold','pause_required','global_dispatch_hold','dispatch_hold','pause_requested')

def marker_snapshot(path):
 p=Path(path)
 if p.is_symlink():raise ValueError('linked pause forbidden')
 if not p.exists():return None
 first=p.stat();raw=p.read_bytes();last=p.stat()
 if len(raw)>1048576 or any(getattr(first,k)!=getattr(last,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')):raise ValueError('pause changed/oversize')
 value=json.loads(raw) if raw else {}
 if not isinstance(value,dict):raise ValueError('pause metadata must be object')
 return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),size=len(raw),st_dev=last.st_dev,st_ino=last.st_ino,st_mtime_ns=last.st_mtime_ns,st_ctime_ns=last.st_ctime_ns,value=value)

def boundary(epoch,run_id,archive_root,point,snapshot=marker_snapshot):
 if epoch.get('group')!='S1-AI17-R10' or epoch.get('run_id')!=run_id or epoch.get('root_accepted')is not True:raise ValueError('exact new R10 epoch required')
 if epoch['authority_ref']['sha256']!=AUTHORITY_SHA:raise ValueError('original explicit human continue ref')
 pin(epoch['authority_ref']);root=Path(archive_root);own={str(root/'stop.request'),str(root/'pause_state.json')};watched=set(epoch['watched_pause_paths'])|own;exempt={}
 for old in epoch['historical_hold_exemptions']:
  if old['path'] in own or old['reason']!='historical0700_pause':raise ValueError('new epoch stop never exempt')
  if old['archived_evidence_ref']['sha256']!=old['sha256']:raise ValueError('history raw SHA binding')
  pin(old['archived_evidence_ref']);exempt[old['path']]=old;watched.add(old['path'])
 held=[]
 for path in sorted(watched):
  current=snapshot(path);prior=exempt.get(path)
  if prior is not None:
   if current is None or any(current[k]!=prior[k] for k in FINGERPRINT):held.append(path)
   continue
  if current is None:continue
  value=current['value']
  if Path(path).name=='stop.request' or any(value.get(k) for k in FLAGS) or str(value.get('state','')).startswith('paused'):held.append(path)
 if held:
  publish(root/'pause_state.json',dict(group='S1-AI17-R10',run_id=run_id,point=point,paused=True,held_paths=held,requires_explicit_continue=True));raise Paused('new/changed pause; no replay')
 return dict(passed=True,point=point,old_deadline_exempt_only_via_exact_epoch=True)

def plan_boundary(plan,point):return boundary(human_epoch(plan),plan['run_id'],plan['archive_root'],point)

def storage_boundary(plan,point):
 s=storage_check(plan);root=Path(plan['archive_root']);disk=shutil.disk_usage(root);logical=temporary=0;seen=set()
 growths=[Path(x).resolve() for x in s.get('owned_growth_roots',[s['owned_growth_root']])];temps=[Path(x).resolve() for x in s.get('owned_temporary_roots',[s['owned_temporary_root']])]
 if any('/round10/' not in x.as_posix()+'/' for x in growths+temps) or any(not any(t.is_relative_to(g) for g in growths) for t in temps):raise ValueError('only exact owned R10 growth/temporary paths')
 for growth in growths:
  for current,dirs,names in os.walk(growth,followlinks=False):
   if any((Path(current)/x).is_symlink() for x in dirs+names):raise ValueError('linked owned storage')
   for name in names:
    f=(Path(current)/name).resolve()
    if f in seen:continue
    seen.add(f);size=f.stat().st_size
    if any(f.is_relative_to(t) for t in temps):temporary+=size
    else:logical+=size
 record=dict(point=point,disk_free_mount_bytes=disk.free,owned_logical_bytes=logical,owned_temporary_bytes=temporary,physical_occupancy_not_claimed=True,personal_quota='unknown unless separately verified in inventory')
 publish(root/'storage_latest.json',record)
 if disk.free<s['free_reserve_bytes'] or logical>s['new_logical_bytes_max'] or temporary>s['temporary_bytes_max']:
  publish(root/'pause_state.json',dict(paused=True,reason='storage',observation=record));raise Paused('storage cap/reserve; no deletion')
 return record

@contextmanager
def exclusive_lock(path):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a+b') as f:
  if os.name=='nt':
   import msvcrt
   f.seek(0);f.write(b'\0');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
  else:
   import fcntl
   fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:yield
  finally:
   if os.name=='nt':f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
   else:fcntl.flock(f,fcntl.LOCK_UN)
def bind_parent(variable):
 if sys.platform!='linux':raise ValueError('formal Linux lifecycle required')
 expected=int(os.environ[variable]);libc=ctypes.CDLL(None,use_errno=True)
 if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PDEATHSIG failed')
 if os.getppid()!=expected:raise ValueError('bound parent changed')
