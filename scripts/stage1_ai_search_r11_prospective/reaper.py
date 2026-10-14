"""Durable direct-parent wait4. No algorithm, input data, queue or retry.

Charged boundary starts before intent/Popen. A native child must finish all
work/publication/exit and all registered descendants before that deadline.
The observer's final immutable receipt IO is outside the ended native tree,
separately measured; a missing receipt is never completion evidence.
"""
import ctypes,errno,os,signal,subprocess,time,threading
from pathlib import Path
from scripts.stage1_ai_search_r10 import contracts as c

def boottime():return time.clock_gettime(time.CLOCK_BOOTTIME)
def usage_record(u):return dict(user_seconds=u.ru_utime,system_seconds=u.ru_stime,maxrss_bytes=u.ru_maxrss*1024,minor_faults=u.ru_minflt,major_faults=u.ru_majflt,voluntary_context_switches=u.ru_nvcsw,involuntary_context_switches=u.ru_nivcsw)
def key(m):return(m['pid'],m['birth_ticks'],m['boot_id'])

class Watchdog:
 """No additional child. Parent death kills the bound native tree if IO stalls."""
 def __init__(self,deadline,clock):self.deadline=deadline;self.clock=clock;self.finished=threading.Event();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
 def loop(self):
  while not self.finished.wait(.01):
   if self.clock()>=self.deadline:os._exit(124)
 def after_native(self,deadline):self.deadline=deadline
 def close(self):self.finished.set();self.thread.join(timeout=.1)

def wait_one(child,owner,*,clock=boottime,wait4=None):
 """Never poll/wait the Popen object: that would discard raw wait4 evidence."""
 wait4=os.wait4 if wait4 is None else wait4
 try:pid,status,u=wait4(child.pid,os.WNOHANG)
 except InterruptedError:return None
 except OSError as e:
  if e.errno==errno.EINTR:return None
  raise
 if not pid:return None
 observed=clock()
 if pid!=owner['pid']:raise RuntimeError('wait4 returned another PID')
 code=os.waitstatus_to_exitcode(status);child.returncode=code
 return dict(schema='R11ExactWait4ExitV1',actual_child=owner,raw_wait_status=status,decoded_exit_code=code,wait4_observed_clock_boottime_seconds=observed,rusage=usage_record(u),kernel_exit_instant_exact=False)

def persist_wait(path,record,*,publish=c.immutable):
 """Failure propagates. Never substitute a later /proc poll or mtime."""
 return publish(path,record)

def run_process(command,directory,stage,*,stop,life,hard_seconds,rss_bytes=2147483648,helper_token=None,clock=boottime,sleep=time.sleep,popen=subprocess.Popen,wait4=None,publish=c.immutable,identity=c.identity,boot_id=None,watchdog_factory=Watchdog,observer_tail_seconds=2):
 started=clock();deadline=started+hard_seconds
 if stage=='Gh' and os.environ.get('R11_LANE_DEADLINE_BOOTTIME'):
  inherited=float(os.environ['R11_LANE_DEADLINE_BOOTTIME'])
  if not 0<inherited<float('inf'):raise ValueError('finite inherited lane deadline')
  deadline=min(deadline,inherited)
 folder=Path(directory);folder.mkdir(parents=True,exist_ok=True);parent=identity(os.getpid());registry={};child=None;owner=None;reaped=None;reap_ref=None;reason=None;error=None;tree=None;closed=False;sample_peak=None;sample_count=0;parent_highwater=None;observer_publication_started=None
 if type(hard_seconds)not in(int,float) or not 0<hard_seconds<=2400 or type(rss_bytes)is not int or not 0<rss_bytes<=2147483648:raise ValueError('fixed tightened native limits')
 if parent is None or (boot_id is not None and parent['boot_id']!=boot_id):raise RuntimeError('actual observer generation/boot')
 watchdog=watchdog_factory(deadline,clock)
 publish(folder/'stage.wal.json',dict(schema='R11ProspectiveNativeIntentV1',stage=stage,command=command,actual_parent=parent,charged_start_clock_boottime_seconds=started,absolute_deadline_clock_boottime_seconds=deadline,hard_seconds=hard_seconds,rss_bytes_max=rss_bytes,unknown_never_replayed=True))
 def sample():
  nonlocal sample_peak,sample_count
  value=life.registered_rss(c,registry);sample_count+=1;sample_peak=value if sample_peak is None else max(sample_peak,value)
 def observe():
  if owner:life.observe_descendants(c,owner,registry)
  return life.final_tree_observation(c,dict(helper_refs={'engineering_child.py':dict(path=helper_token or command[0])},output_root=str(folder)),registry)
 def bound_child():
  libc=ctypes.CDLL(None,use_errno=True)
  if libc.prctl(1,signal.SIGKILL,0,0,0)!=0 or os.getppid()!=parent['pid']:os._exit(126)
 def reap():
  nonlocal reaped,reap_ref
  if child is not None and owner is not None and reaped is None:
   result=wait_one(child,owner,clock=clock,wait4=wait4)
   if result is not None:
    reaped=dict(result,actual_observer=parent,charged_start_clock_boottime_seconds=started,absolute_deadline_clock_boottime_seconds=deadline,conservative_process_exit_upper_seconds=result['wait4_observed_clock_boottime_seconds']-started)
    # Timestamp precedes durable receipt IO. Crash/failure here leaves unknown.
    reap_ref=persist_wait(folder/'native_reap.json',reaped,publish=publish)
 try:
  stop();sample()
  if clock()>=deadline:raise RuntimeError('native deadline before spawn')
  env=dict(os.environ,R10_PARENT_PID=str(parent['pid']),R11_NATIVE_DEADLINE_BOOTTIME=str(deadline),PYTHONDONTWRITEBYTECODE='1')
  if stage=='lane':env['R11_LANE_DEADLINE_BOOTTIME']=str(deadline)
  with (folder/'stage.log').open('xb') as log:
   child=popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,preexec_fn=bound_child,env=env)
   owner=identity(child.pid)
   if owner is None or owner['boot_id']!=parent['boot_id'] or owner['uid']!=parent['uid']:raise RuntimeError('native birth/boot/UID unknown')
   registry[key(owner)]=owner;publish(folder/'actual_native_launch.json',dict(actual_observer=parent,actual_native=owner,executable=command[0],argv=command,absolute_deadline_clock_boottime_seconds=deadline,remaining_seconds_after_spawn=deadline-clock(),parent_death_bound=True))
   cleanup_reserve=min(.5,hard_seconds/10)
   while reaped is None:
    stop();tree=observe();sample();reap()
    if reaped is not None:break
    if tree['unknown_live']:reason='unknown_live_descendant';break
    if sample_peak>rss_bytes:reason='parent_children_RSS';break
    if clock()>=deadline-cleanup_reserve:reason='deadline_cleanup_reserve';break
    sleep(.02)
 except BaseException as e:error=type(e).__name__+': '+str(e);reason=reason or 'observer_or_pause_failure'
 finally:
  try:
   tree=observe()
   if tree['unknown_live']:raise RuntimeError('unregistered live witness: refuse inferred signal')
   if tree['registered_live']:
    reason=reason or 'descendants_live_after_direct_child_exit';life.exact_terminate(c,registry,signal.SIGTERM);term_until=min(deadline,clock()+.1)
    while clock()<term_until:
     reap();tree=observe()
     if not tree['registered_live']:break
     sleep(.01)
    if observe()['registered_live']:life.exact_terminate(c,registry,signal.SIGKILL)
   reap()
   while reaped is None and child is not None and owner is not None and clock()<deadline:
    reap()
    if reaped is None:sleep(.01)
   tree=observe();sample();closed=owner is not None and tree['all_execution_ended'];stop();closure_observed=clock()
   if not closed or reaped is None or reap_ref is None:raise RuntimeError('actual wait4 and all generations closure required')
   import resource
   parent_highwater=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
   if sample_peak is None or sample_peak>rss_bytes:reason=reason or 'final_sampled_parent_children_RSS'
   if parent_highwater+reaped['rusage']['maxrss_bytes']>rss_bytes:reason=reason or 'conservative_parent_plus_child_highwater_RSS'
   if closure_observed>=deadline or reaped['conservative_process_exit_upper_seconds']>=hard_seconds:reason=reason or 'full_native_exit_bound_exceeds_hard_limit'
  except BaseException as e:error=(error or '')+';closure_unknown:'+type(e).__name__+': '+str(e);closure_observed=clock()
  observer_publication_started=clock()
  # Only observer IO remains after the closed native tree. It cannot run forever.
  native_end_proved=bool(closed and reaped is not None and reap_ref is not None)
  observer_tail_deadline=min(deadline+(observer_tail_seconds if native_end_proved else 0),observer_publication_started+observer_tail_seconds)
  watchdog.after_native(observer_tail_deadline)
  completed=reason is None and error is None and closed and reaped is not None and reap_ref is not None and reaped['decoded_exit_code']==0
  if not completed:reason=reason or 'nonzero_or_unknown_native_exit'
  result=dict(schema='R11ProspectiveNativeGuardReceiptV1',status='completed' if completed else 'partial_unknown',stage=stage,reason=reason,error=error,actual_child=owner,actual_observer=parent,external_timeout_wrapper=None,raw_wait_status=None if reaped is None else reaped['raw_wait_status'],exit_code=None if reaped is None else reaped['decoded_exit_code'],native_reap_ref=reap_ref,charged_start_clock_boottime_seconds=started,absolute_deadline_clock_boottime_seconds=deadline,closure_observed_clock_boottime_seconds=closure_observed,actual_stage_wall_seconds=closure_observed-started,complete_native_exit_upper_seconds=closure_observed-started if closed and reaped is not None else None,hard_seconds=hard_seconds,subtree_exit_verified=closed,postwait_process_observations=tree,registered_descendants=list(registry.values()),peak_parent_children_rss_bytes=sample_peak,RSS_sample_count=sample_count,parent_self_highwater_bytes=parent_highwater,wait4_rusage=None if reaped is None else reaped['rusage'],observer_final_receipt_IO_after_native_closure=native_end_proved,observer_publication_hard_tail_seconds=observer_tail_seconds,observer_receipt_IO_not_hidden_native_work=native_end_proved,automatic_retry=False,algorithm_spend_exact=None,nested_timers_not_additive=True)
  out=folder/'stage_output.json'
  result['output_ref']=dict(path=str(out),sha256=c.digest(out)) if out.is_file() else None
  result['guard_ref']=publish(folder/'guard_receipt.json',result)
  # Outside the ended native tree, inside the worker/lane outer accounting.
  c.publish(folder/'observer_publication_cost.json',dict(observed_seconds_to_finally=clock()-observer_publication_started,scope='observer final receipt IO only; native all-generations exit sampled before publication',complete_observer_exit_cost_seconds=None))
  if clock()>=observer_tail_deadline:raise RuntimeError('observer receipt/cost IO exceeds bounded tail; no dependent dispatch')
  watchdog.close()
  if clock()>=observer_tail_deadline:raise RuntimeError('observer close exceeds bounded tail')
 if not completed:raise RuntimeError('unknown/failed native '+stage+': '+str(reason))
 return result

def run_stage(command,directory,stage,*,stop,life,limits):
 return run_process(command,directory,stage,stop=stop,life=life,hard_seconds=limits[stage],helper_token='scripts.stage1_ai_search_r11_prospective.child')
