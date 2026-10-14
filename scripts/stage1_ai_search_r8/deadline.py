"""Fail-closed R8 boundaries registered separately from original R7/archive."""
import json,os,signal,time,ctypes,sys,shutil,hashlib
from pathlib import Path
from .contracts import DEADLINE,read,atomic,identity,GROUP,storage_inventory,resume_epoch,pin

class Paused(RuntimeError):pass
FINGERPRINT=('sha256','size','st_dev','st_ino','st_mtime_ns','st_ctime_ns')
def marker_snapshot(path):
 p=Path(path)
 if not p.exists():return None
 if p.is_symlink():raise ValueError('pause marker links are not admitted')
 first=p.stat()
 with p.open('rb') as f:raw=f.read(1048576+1)
 last=p.stat()
 fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 if fields(first)!=fields(last) or len(raw)>1048576:raise ValueError('pause marker changed during bounded read')
 return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),size=len(raw),st_dev=last.st_dev,st_ino=last.st_ino,st_mtime_ns=last.st_mtime_ns,st_ctime_ns=last.st_ctime_ns,value=json.loads(raw) if raw else {})
def pause_sink(plan,epoch):return Path(epoch['epoch_root'] if epoch else plan['archive_root'])/'pause_state.json'
def storage_boundary(plan,point):
 inventory=storage_inventory(plan);root=Path(plan['archive_root']);disk=shutil.disk_usage(root)
 # Count declared owned logical bytes once per resolved file, no symlink traversal.
 files=set();logical=inventory.get('fixed_other_owned_bytes',0)
 if not isinstance(logical,int) or logical<0:raise ValueError('explicit fixed owned source/meta byte count')
 for dirname in [inventory['owned_growth_root'],*inventory.get('additional_owned_roots',[])]:
  for current,dirs,names in os.walk(dirname,followlinks=False):
   if any((Path(current)/name).is_symlink() for name in dirs+names):raise ValueError('owned storage cannot follow undeclared directory/file links')
   for name in names:
    p=(Path(current)/name).resolve()
    if p not in files:files.add(p);logical+=p.stat().st_size
 receipt=dict(point=point,actual_disk_path=str(root.resolve()),disk_free_bytes=disk.free,owned_logical_bytes=logical,new_logical_bytes_max=1073741824,free_reserve_bytes=8589934592,physical_occupancy_not_claimed=True,storage_inventory_ref=plan['storage_inventory_ref'])
 atomic(root/'storage_boundary_latest.json',receipt)
 if disk.free<8589934592 or logical>1073741824:
  atomic(pause_sink(plan,resume_epoch(plan)),dict(group=GROUP,paused=True,reason='storage_cap_or_reserve',storage_observation=receipt,no_assets_deleted=True));raise Paused('storage cap or 8GiB reserve reached; no new child/WAL')
 return receipt
def before_start(plan,point,hard_seconds=15,startup_margin_seconds=5):
 admission=boundary(plan,point);epoch=resume_epoch(plan)
 storage_boundary(plan,point)
 cutoff=epoch['new_deadline_unix_seconds'] if epoch else DEADLINE
 if cutoff is not None and cutoff-time.time()<hard_seconds+startup_margin_seconds:
  atomic(pause_sink(plan,epoch),dict(group=GROUP,paused=True,requires_explicit_human_continue=True,point=point,reason='insufficient_time_for_new_child_hard_guard_plus5seconds_startup',deadline_unix_seconds=cutoff))
  raise Paused('safe boundary before new child; insufficient deadline margin')
 return admission
def boundary(plan,point):
 registration=read(plan['deadline_registration_ref'])
 if registration.get('root_accepted')is not True or registration.get('archive_root')!=plan['archive_root'] or registration.get('group')!=GROUP or registration.get('deadline_unix_seconds')!=DEADLINE:raise ValueError('registered own archive boundary required')
 epoch=resume_epoch(plan);held=[];paths=set(registration['shared_pause_paths'])|{str(Path(plan['archive_root'])/'pause_state.json'),str(Path(plan['archive_root'])/'stop.request')};exemptions={}
 if epoch:
  paths.update(epoch['watched_pause_paths']);paths.update([str(Path(epoch['epoch_root'])/'pause_state.json'),str(Path(epoch['epoch_root'])/'stop.request')])
  for entry in epoch['historical_hold_exemptions']:
   if entry.get('reason')!='historical0700_pause' or entry['path'] in (str(Path(epoch['epoch_root'])/'pause_state.json'),str(Path(epoch['epoch_root'])/'stop.request')):raise ValueError('only individually signed historical marker generations')
   pin(entry['archived_evidence_ref'])
   if entry['archived_evidence_ref']['sha256']!=entry['sha256']:raise ValueError('exact old hold snapshot bytes')
   exemptions[entry['path']]=entry;paths.add(entry['path'])
 for path in paths:
  snapshot=marker_snapshot(path);exemption=exemptions.get(path)
  if exemption and snapshot is None:held.append(path);continue
  if snapshot is None:continue
  if exemption and all(snapshot[k]==exemption[k] for k in FINGERPRINT):continue
  value=snapshot['value']
  if Path(path).name=='stop.request' or any(value.get(key) for key in ('user_pause','paused','hold','pause_required','global_dispatch_hold','dispatch_hold','pause_requested')) or str(value.get('state','')).startswith('paused'):held.append(path)
 cutoff=epoch['new_deadline_unix_seconds'] if epoch else DEADLINE
 if (cutoff is not None and time.time()>=cutoff) or held:
  atomic(pause_sink(plan,epoch),dict(group=GROUP,paused=True,requires_explicit_human_continue=True,point=point,actual_process=identity(os.getpid()),deadline_unix_seconds=cutoff,held_paths=held,human_resume_authority_ref=plan.get('human_resume_authority_ref')))
  raise Paused('R8 paused; explicit human continue and new root admission required')
 return dict(passed=True,point=point,deadline_unix_seconds=cutoff,actual_process=identity(os.getpid()),deadline_registration_ref=plan['deadline_registration_ref'],human_resume_authority_ref=plan.get('human_resume_authority_ref'),old_cutoff_exempted_only_this_epoch=epoch is not None)
def bind_parent(variable):
 if sys.platform!='linux':raise ValueError('native Linux parent guard required')
 expected=int(os.environ[variable]);libc=ctypes.CDLL(None,use_errno=True)
 if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:raise OSError(ctypes.get_errno(),'parent death guard failed')
 if os.getppid()!=expected:raise ValueError('original child parent changed')
