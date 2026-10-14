"""Native safety guard with durable reservations; unknown attempts never replay."""
import errno,os,signal,subprocess,time
from pathlib import Path
from .contracts import atomic,immutable,identity,same_process,digest,read
from .control import Paused

def rss_tree(pid):
 total=0;pending=[pid];seen=set();page=os.sysconf('SC_PAGE_SIZE')
 while pending:
  current=pending.pop()
  if current in seen:continue
  seen.add(current);proc=Path('/proc')/str(current)
  try:
   total+=int((proc/'statm').read_text().split()[1])*page
   for thread in (proc/'task').iterdir():pending.extend(map(int,(thread/'children').read_text().split()))
  except OSError as e:
   if e.errno in (errno.ENOENT,errno.ESRCH):continue
   raise
 return total
def group_members(expected,known_members=()):
 """One fresh session is owned only by this admitted leader birth+boot."""
 if expected is None:raise RuntimeError('missing owner identity')
 boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 if boot!=expected['boot_id']:raise RuntimeError('boot changed; no PID kills')
 first_leader=identity(expected['pid'])
 if first_leader is not None and not same_process(expected,first_leader):raise RuntimeError('old session leader PID reused; refuse entire group')
 result=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  current=identity(int(proc.name))
  if current is None:continue
  try:raw=(proc/'stat').read_text();f=raw[raw.rfind(')')+2:].split()
  except OSError as e:
   if e.errno in (errno.ENOENT,errno.ESRCH):continue
   raise
  if int(f[2])==expected['pid'] and int(f[3])==expected['pid']:
   if int(f[19])!=current['birth_ticks'] or current['birth_ticks']<expected['birth_ticks']:raise RuntimeError('session member generation changed')
   if current['state']!='Z':result.append(current)
 leader=identity(expected['pid'])
 if leader is not None and not same_process(expected,leader):raise RuntimeError('session leader PID reused during observation; refuse entire group')
 if not (same_process(expected,first_leader) and same_process(expected,leader)):
  # A vanished leader can no longer authenticate a numeric session ID. Only
  # exact member generations observed while that leader was present qualify.
  known={(x['pid'],x['birth_ticks'],x['boot_id']) for x in known_members}
  if any((x['pid'],x['birth_ticks'],x['boot_id']) not in known for x in result):raise RuntimeError('unproven group generation after leader exit; preserve unknown, refuse kill')
 return result
def observe_owned(child,expected):
 registry=getattr(child,'_r9_owned_members',{})
 for member in group_members(expected,registry.values()):registry[(member['pid'],member['birth_ticks'],member['boot_id'])]=member
 child._r9_owned_members=registry
def terminate_owned(child,expected):
 live=child.poll()is None
 if live:
  if not same_process(expected,identity(child.pid)) or os.getpgid(child.pid)!=child.pid:raise RuntimeError('native stage owner birth/boot/PGID changed; refuse kill')
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=2)
  except subprocess.TimeoutExpired:
   if not same_process(expected,identity(child.pid)) or os.getpgid(child.pid)!=child.pid:raise RuntimeError('owner changed before force kill')
   os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=3)
 # The leader can exit before a resistant grandchild. A PGID-only kill based
 # on a vanished leader is forbidden; recheck each surviving exact birth.
 registry=getattr(child,'_r9_owned_members',{});members=group_members(expected,registry.values())
 for member in [*registry.values(),*members]:
  current=identity(member['pid'])
  if same_process(member,current) and current['state']!='Z':
   try:os.kill(member['pid'],signal.SIGKILL)
   except OSError as e:
    if e.errno not in (errno.ENOENT,errno.ESRCH):raise
 deadline=time.monotonic()+3
 while group_members(expected,registry.values()) or any(same_process(member,current) and current['state']!='Z' for member in registry.values() if (current:=identity(member['pid'])) is not None):
  if time.monotonic()>=deadline:raise RuntimeError('owned native subtree exit not verified')
  time.sleep(.02)
def run_stage(command,directory,stage,*,stop,limits=None,popen=subprocess.Popen,clock=time.monotonic,memory=rss_tree,sleep=time.sleep,terminate=terminate_owned):
 limits=limits or dict(RULE=15,D5=15,common=30,pool=60)
 if stage not in limits:raise ValueError('unsupported native stage')
 folder=Path(directory);folder.mkdir(parents=True,exist_ok=True);wal=folder/'stage.wal.json';out=folder/'stage_output.json';receipt=folder/'guard_receipt.json'
 if wal.exists() or receipt.exists() or out.exists():raise ValueError('existing/unknown stage; read-only reuse requires separate root admission, never auto replay')
 began=clock();record=dict(stage=stage,status='intent_before_spawn',command=command,hard_seconds=limits[stage],rss_bytes_max=2147483648,actual_parent=identity(os.getpid()),unknown_never_replayed=True)
 immutable(wal,record);child=None;owner=None;reason=None;error=None;caught=None;peak=0;subtree_closed=False;final_observation_error=None
 try:
  stop()
  env=dict(os.environ,R9_PARENT_PID=str(os.getpid()),PYTHONDONTWRITEBYTECODE='1')
  with (folder/'stage.log').open('ab',buffering=0) as log:
   child=popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
   owner=identity(child.pid)
   if owner is None:raise RuntimeError('child birth/boot identity unavailable; completed exit cannot close unknown descendants')
   if owner and os.getpgid(child.pid)!=child.pid:raise RuntimeError('own new stage PGID required')
   observe_owned(child,owner)
   record.update(status='native_started',actual_child=owner);atomic(wal,record)
   while child.poll()is None:
    stop();observe_owned(child,owner);peak=max(peak,memory(os.getpid()))
    if clock()-began>=limits[stage]:reason='hard_time_guard'
    elif peak>2147483648:reason='parent_children_RSS_guard'
    if reason:terminate(child,owner);subtree_closed=True;break
    sleep(.05)
   if reason is None and owner is not None:terminate(child,owner);subtree_closed=True
 except BaseException as e:
  caught=e;error=type(e).__name__+': '+str(e);reason=reason or 'pause_or_interface_failure'
  if child is not None and owner is not None:terminate(child,owner);subtree_closed=True
 finally:
  # A very short child may exit before the first monitor poll. Completion is
  # subject to the same final operational limits and owner/subtree proof.
  try:
   peak=max(peak,memory(os.getpid()))
   if reason is None:stop()
  except BaseException as e:
   final_observation_error=type(e).__name__+': '+str(e);reason=reason or 'final_observer_failure'
   if caught is None:caught=e
  elapsed=clock()-began
  if reason is None and elapsed>=limits[stage]:reason='hard_time_guard'
  if reason is None and peak>2147483648:reason='parent_children_RSS_guard'
  completed=child is not None and child.poll()==0 and reason is None and owner is not None and subtree_closed and out.is_file()
  result=dict(status='completed' if completed else 'partial_unknown',stage=stage,reason=reason or (None if completed else 'native_error_or_no_output'),error=error,final_observation_error=final_observation_error,actual_stage_wall_seconds=elapsed,peak_parent_children_rss_bytes=peak,peak_RSS_is_sampled=True,actual_child=owner,subtree_exit_verified=subtree_closed,exit_code=child.poll() if child else None,output_ref=dict(path=str(out),sha256=digest(out)) if out.is_file() else None,algorithm_spend_exact=None,automatic_retry=False,nested_timers_not_additive=True)
  immutable(receipt,result)
 if not completed:
  if isinstance(caught,Paused):raise caught
  raise RuntimeError('partial/unknown '+stage+'; no retry: '+str(result['reason']))
 read(result['output_ref'])
 return result
