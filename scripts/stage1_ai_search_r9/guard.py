"""Native safety guard with durable reservations; unknown attempts never replay."""
import os,signal,subprocess,time
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
  except FileNotFoundError:continue
 return total
def group_members(expected):
 """One fresh session is owned only by this admitted leader birth+boot."""
 if expected is None:raise RuntimeError('missing owner identity')
 boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 if boot!=expected['boot_id']:raise RuntimeError('boot changed; no PID kills')
 result=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  current=identity(int(proc.name))
  if current is None:continue
  try:raw=(proc/'stat').read_text();f=raw[raw.rfind(')')+2:].split()
  except OSError as e:
   import errno
   if e.errno in (errno.ENOENT,errno.ESRCH):continue
   raise
  if int(f[2])==expected['pid'] and int(f[3])==expected['pid']:
   if int(f[19])!=current['birth_ticks'] or current['birth_ticks']<expected['birth_ticks']:raise RuntimeError('session member generation changed')
   if current['state']!='Z':result.append(current)
 return result
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
 for member in group_members(expected):
  if same_process(member,identity(member['pid'])):os.kill(member['pid'],signal.SIGKILL)
 deadline=time.monotonic()+3
 while group_members(expected):
  if time.monotonic()>=deadline:raise RuntimeError('owned native subtree exit not verified')
  time.sleep(.02)
def run_stage(command,directory,stage,*,stop,limits=None,popen=subprocess.Popen,clock=time.monotonic,memory=rss_tree,sleep=time.sleep,terminate=terminate_owned):
 limits=limits or dict(RULE=15,D5=15,common=30,pool=60)
 if stage not in limits:raise ValueError('unsupported native stage')
 folder=Path(directory);folder.mkdir(parents=True,exist_ok=True);wal=folder/'stage.wal.json';out=folder/'stage_output.json';receipt=folder/'guard_receipt.json'
 if wal.exists() or receipt.exists() or out.exists():raise ValueError('existing/unknown stage; read-only reuse requires separate root admission, never auto replay')
 began=clock();record=dict(stage=stage,status='intent_before_spawn',command=command,hard_seconds=limits[stage],rss_bytes_max=2147483648,actual_parent=identity(os.getpid()),unknown_never_replayed=True)
 immutable(wal,record);child=None;owner=None;reason=None;error=None;caught=None;peak=0
 try:
  stop()
  env=dict(os.environ,R9_PARENT_PID=str(os.getpid()),PYTHONDONTWRITEBYTECODE='1')
  with (folder/'stage.log').open('ab',buffering=0) as log:
   child=popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
   owner=identity(child.pid)
   if owner is None and child.poll()is None:raise RuntimeError('live child identity unknown')
   if owner and os.getpgid(child.pid)!=child.pid:raise RuntimeError('own new stage PGID required')
   record.update(status='native_started',actual_child=owner);atomic(wal,record)
   while child.poll()is None:
    stop();peak=max(peak,memory(os.getpid()))
    if clock()-began>=limits[stage]:reason='hard_time_guard'
    elif peak>2147483648:reason='parent_children_RSS_guard'
    if reason:terminate(child,owner);break
    sleep(.05)
   if reason is None and owner is not None:terminate(child,owner)
 except BaseException as e:
  caught=e;error=type(e).__name__+': '+str(e);reason=reason or 'pause_or_interface_failure'
  if child is not None and child.poll()is None:terminate(child,owner)
 finally:
  elapsed=clock()-began
  completed=child is not None and child.poll()==0 and reason is None and out.is_file()
  result=dict(status='completed' if completed else 'partial_unknown',stage=stage,reason=reason or (None if completed else 'native_error_or_no_output'),error=error,actual_stage_wall_seconds=elapsed,peak_parent_children_rss_bytes=peak,actual_child=owner,exit_code=child.poll() if child else None,output_ref=dict(path=str(out),sha256=digest(out)) if out.is_file() else None,algorithm_spend_exact=None,automatic_retry=False,nested_timers_not_additive=True)
  immutable(receipt,result)
 if not completed:
  if isinstance(caught,Paused):raise caught
  raise RuntimeError('partial/unknown '+stage+'; no retry: '+str(result['reason']))
 read(result['output_ref'])
 return result
