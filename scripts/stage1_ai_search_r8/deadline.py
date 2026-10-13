"""Fail-closed R8 boundaries registered separately from original R7/archive."""
import json,os,signal,time,ctypes,sys,shutil
from pathlib import Path
from .contracts import DEADLINE,read,atomic,identity,GROUP,storage_inventory

class Paused(RuntimeError):pass
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
  atomic(root/'pause_state.json',dict(group=GROUP,paused=True,reason='storage_cap_or_reserve',storage_observation=receipt,no_assets_deleted=True));raise Paused('storage cap or 8GiB reserve reached; no new child/WAL')
 return receipt
def before_start(plan,point,hard_seconds=15,startup_margin_seconds=5):
 admission=boundary(plan,point)
 storage_boundary(plan,point)
 if DEADLINE-time.time()<hard_seconds+startup_margin_seconds:
  atomic(Path(plan['archive_root'])/'pause_state.json',dict(group=GROUP,paused=True,requires_explicit_human_continue=True,point=point,reason='insufficient_time_for_new_child_hard_guard_plus5seconds_startup',deadline_unix_seconds=DEADLINE))
  raise Paused('safe boundary before new child; insufficient deadline margin')
 return admission
def boundary(plan,point):
 registration=read(plan['deadline_registration_ref'])
 if registration.get('root_accepted')is not True or registration.get('archive_root')!=plan['archive_root'] or registration.get('group')!=GROUP or registration.get('deadline_unix_seconds')!=DEADLINE:raise ValueError('registered own archive boundary required')
 held=[]
 for path in registration['shared_pause_paths']:
  p=Path(path)
  if p.exists():
   value=json.loads(p.read_bytes())
   if any(value.get(key) for key in ('paused','hold','pause_required','global_dispatch_hold','dispatch_hold','pause_requested')) or str(value.get('state','')).startswith('paused'):held.append(path)
 if time.time()>=DEADLINE or held or (Path(plan['archive_root'])/'stop.request').exists():
  atomic(Path(plan['archive_root'])/'pause_state.json',dict(group=GROUP,paused=True,requires_explicit_human_continue=True,point=point,actual_process=identity(os.getpid()),deadline_unix_seconds=DEADLINE,held_paths=held))
  raise Paused('R8 paused; explicit human continue and new root admission required')
 return dict(passed=True,point=point,deadline_unix_seconds=DEADLINE,actual_process=identity(os.getpid()),deadline_registration_ref=plan['deadline_registration_ref'])
def bind_parent(variable):
 if sys.platform!='linux':raise ValueError('native Linux parent guard required')
 expected=int(os.environ[variable]);libc=ctypes.CDLL(None,use_errno=True)
 if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:raise OSError(ctypes.get_errno(),'parent death guard failed')
 if os.getppid()!=expected:raise ValueError('original child parent changed')
