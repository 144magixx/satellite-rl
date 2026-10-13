"""Owned subprocess guards, independent of the bounded geometry algorithm."""
import os,signal,subprocess,time,json
from pathlib import Path
from .storage import now,read_json
from satellite_rl.training.run_monitor import atomic_json

def birth(pid):
 if os.name=='nt':
  from .runtime import process_start_time
  return process_start_time(pid)
 p=Path('/proc')/str(pid)/'stat'
 try:return int(p.read_text().split(') ',1)[1].split()[19])
 except FileNotFoundError:return None

def rss_tree(pid):
 if os.name=='nt':
  import psutil
  try:p=psutil.Process(pid);nodes=[p,*p.children(recursive=True)]
  except psutil.NoSuchProcess:return 0
  total=0
  for node in nodes:
   try:total+=node.memory_info().rss
   except psutil.NoSuchProcess:continue
  return total
 seen=set();queue=[pid];total=0
 while queue:
  current=queue.pop()
  if current in seen:continue
  seen.add(current);root=Path('/proc')/str(current)
  try:
   total+=next(int(x.split()[1])*1024 for x in (root/'status').read_text().splitlines()if x.startswith('VmRSS:'))
   child=root/'task'/str(current)/'children'
   if child.exists():queue.extend(int(x)for x in child.read_text().split())
  except FileNotFoundError:pass
  except StopIteration:
   try:state=(root/'stat').read_text().split(') ',1)[1].split()[0]
   except FileNotFoundError:state='X'
   if state not in ('Z','X'):raise RuntimeError('live ownedprocess missing RSS')
 return total

def terminate_owned(child,expected_birth):
 if child.poll()is not None:return
 if birth(child.pid)!=expected_birth:raise RuntimeError('owned child birth changed; refuse PID kill')
 if os.name!='nt':
  if os.getpgid(child.pid)!=child.pid:raise RuntimeError('owned stage process group changed')
  os.killpg(child.pid,signal.SIGTERM)
 else:child.terminate()
 try:child.wait(timeout=2)
 except subprocess.TimeoutExpired:
  if birth(child.pid)!=expected_birth:raise RuntimeError('owned child changed before force kill')
  if os.name!='nt':os.killpg(child.pid,signal.SIGKILL)
  else:child.kill()
  child.wait(timeout=3)

def run_stage(command,directory,limit_seconds,memory_bytes,stop=lambda:False,popen=subprocess.Popen,clock=time.monotonic,memory=rss_tree,sleep=time.sleep):
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);wal=directory/'stage.wal.json';output=directory/'stage_output.json';receipt=directory/'guard_receipt.json'
 if receipt.exists():
  old=read_json(wal,{})
  if old.get('command')!=command:raise ValueError('original guarded stage command changed')
  value=read_json(receipt,{})
  if value.get('output_ref'):
   from .coverage_ai_search import pin
   pin(value['output_ref'])
  if value.get('accepted_snapshot_index_ref'):
   from .coverage_ai_search import pin
   pin(value['accepted_snapshot_index_ref'])
  return value
 if wal.exists():raise ValueError('unknown interrupted stage; no automatic replay')
 began=clock();record={'status':'intent_before_spawn','command':command,'started_utc':now(),'hard_seconds':limit_seconds,'owned_parent_children_rss_bytes':memory_bytes,'unknown_spend_preserved':True};atomic_json(wal,record)
 with (directory/'stage.log').open('ab',buffering=0)as log:
  child=popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=os.name!='nt',creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt'else 0)
  identity=birth(child.pid)
  if identity is None and child.poll()is None:
   identity=birth(child.pid)
   if identity is None and child.poll()is None:raise RuntimeError('live child identity unavailable; preserve unknown owned process')
  record.update(status='child_started',pid=child.pid,birth=identity);atomic_json(wal,record);peak=0;reason=None
  monitor_error=None
  try:
   while child.poll()is None:
    peak=max(peak,memory(os.getpid()))
    if stop():reason='requested_stop'
    elif clock()-began>=limit_seconds:reason='hard_time_guard'
    elif peak>memory_bytes:reason='owned_tree_memory_guard'
    if reason:terminate_owned(child,identity);break
    sleep(.05)
  except Exception as error:
   monitor_error=str(error);reason='monitor_error'
   terminate_owned(child,identity)
 elapsed=clock()-began;result={'status':'completed'if not reason and child.returncode==0 and output.is_file()else 'hard_unpublished','reason':None if not reason and child.returncode==0 and output.is_file()else reason or ('child_error'if child.returncode!=0 else'missing_output'),'actual_stage_wall_seconds':elapsed,'peak_parent_children_rss_bytes':peak,'pid':child.pid,'birth':identity,'exit_code':child.returncode,'unknown_algorithm_spend':reason is not None,'output_ref':None,'hard_interface_error':reason=='monitor_error' or (not reason and child.returncode!=0),'monitor_error':monitor_error}
 if output.exists():
  from .storage import digest
  result['output_ref']={'path':str(output),'sha256':digest(output)}
 snapshot_index=directory/'accepted_snapshot_index.json'
 if snapshot_index.exists():
  from .storage import digest
  result['accepted_snapshot_index_ref']={'path':str(snapshot_index),'sha256':digest(snapshot_index)}
 atomic_json(receipt,result);return result
