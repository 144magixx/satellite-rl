"""Pause generation, storage, kernel owner locks and Linux parent binding."""
import ctypes,hashlib,json,os,shutil,signal,sys
from contextlib import contextmanager
from pathlib import Path
from .contracts import atomic,pin,human_epoch,storage_check,identity

class Paused(RuntimeError):pass
FINGERPRINT=('sha256','size','st_dev','st_ino','st_mtime_ns','st_ctime_ns')
def marker_snapshot(path):
 p=Path(path)
 if not p.exists():return None
 if p.is_symlink():raise ValueError('pause symlink forbidden')
 first=p.stat()
 with p.open('rb') as f:raw=f.read(1048577)
 last=p.stat()
 if len(raw)>1048576 or any(getattr(first,k)!=getattr(last,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')):raise ValueError('pause read changed/oversize')
 return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),size=len(raw),st_dev=last.st_dev,st_ino=last.st_ino,st_mtime_ns=last.st_mtime_ns,st_ctime_ns=last.st_ctime_ns,value=json.loads(raw) if raw else {})
def boundary(plan,point):
 epoch=human_epoch(plan);root=Path(plan['archive_root']);paths=set(epoch['watched_pause_paths'])|{str(root/'stop.request'),str(root/'pause_state.json')};exempt={}
 for old in epoch['historical_hold_exemptions']:
  if old.get('reason')!='historical0700_pause' or old['path'] in (str(root/'stop.request'),str(root/'pause_state.json')):raise ValueError('only exact historical generation can be exempt')
  if pin(old['archived_evidence_ref']).read_bytes()!=pin(dict(path=old['archived_evidence_ref']['path'],sha256=old['sha256'])).read_bytes():raise ValueError('historical bytes')
  exempt[old['path']]=old;paths.add(old['path'])
 held=[]
 for path in sorted(paths):
  snap=marker_snapshot(path);old=exempt.get(path)
  if old and snap and all(snap[k]==old[k] for k in FINGERPRINT):continue
  # A removed/replaced/recreated old pause is never silently accepted.
  if old:held.append(path);continue
  if snap is None:continue
  val=snap['value']
  if Path(path).name=='stop.request' or any(val.get(k) for k in ('user_pause','paused','hold','pause_required','global_dispatch_hold','dispatch_hold','pause_requested')) or str(val.get('state','')).startswith('paused'):held.append(path)
 if held:
  atomic(root/'pause_state.json',dict(group=plan['group'],paused=True,point=point,held_paths=held,requires_explicit_human_continue=True,actual_process=identity(os.getpid())))
  raise Paused('new or changed pause generation; no dispatch/replay')
 return dict(passed=True,point=point,human_epoch_ref=plan['human_epoch_ref'],new_deadline_unix_seconds=None)
def storage_boundary(plan,point):
 s=storage_check(plan);root=Path(plan['archive_root']);disk=shutil.disk_usage(root);logical=0;temporary=0;seen=set()
 growth=Path(s['owned_growth_root']).resolve();tmp=Path(s['owned_temporary_root']).resolve()
 if not tmp.is_relative_to(growth):raise ValueError('temporary root must belong to R9 growth')
 for current,dirs,names in os.walk(growth,followlinks=False):
  if any((Path(current)/x).is_symlink() for x in dirs+names):raise ValueError('undeclared linked owned storage')
  for name in names:
   f=(Path(current)/name).resolve()
   if f in seen:continue
   seen.add(f);size=f.stat().st_size
   if f.is_relative_to(tmp):temporary+=size
   else:logical+=size
 rec=dict(point=point,actual_disk_path=str(root.resolve()),disk_free_bytes=disk.free,owned_logical_bytes=logical,owned_temporary_bytes=temporary,physical_occupancy_not_claimed=True)
 atomic(root/'storage_latest.json',rec)
 if disk.free<s['free_reserve_bytes'] or logical>s['new_logical_bytes_max'] or temporary>s['temporary_bytes_max']:
  atomic(root/'pause_state.json',dict(paused=True,reason='storage',observation=rec));raise Paused('storage cap/reserve; no deletion')
 return rec
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
