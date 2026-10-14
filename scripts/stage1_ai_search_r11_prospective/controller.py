"""Bounded Gh parent, exact accepted lifecycle reuse; no POST implementation."""
import argparse,ctypes,json,os,signal,subprocess,sys,time
from pathlib import Path
from scripts.stage1_ai_search_r10 import contracts as c
from scripts.stage1_ai_search_r10.control import exclusive_lock
from .contracts import validate,check_pause,lifecycle,fref,fresh_route,storage_check
from .storage import Observer
from .reaper import wait_one,boottime
DQN='/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python'
def final_limits(elapsed,peak,hard,pause):
 pause()
 if elapsed>=hard:raise RuntimeError('cumulative remaining outer time exhausted')
 if peak>2147483648:raise RuntimeError('whole parent/children2GiB sampled RSS')
def merge_native(p,registry):
 root=Path(p['archive_root']);worker=None
 if (root/'actual_worker_launch.json').exists():
  launch=c.read(fref(root/'actual_worker_launch.json'))
  if launch['plan_ref']!=p['_actual_plan_ref']:raise ValueError('actual Gh worker plan differs')
  worker=launch['actual_worker'];registry[(worker['pid'],worker['birth_ticks'],worker['boot_id'])]=worker
 for path in root.glob('mothers/*/*/guard_receipt.json'):
  member=c.read(fref(path)).get('actual_child')
  if member:registry[(member['pid'],member['birth_ticks'],member['boot_id'])]=member
 return worker
def observe_all(life,p,registry):
 snapshots=[]
 for token,prefix in [('scripts.stage1_ai_search_r11_prospective.child',p['archive_root']),('scripts.stage1_ai_search_r11_prospective.worker',str(Path(p['_actual_plan_ref']['path']).parent)),('scripts.stage1_ai_search_r11_prospective.server',str(Path(p['_actual_plan_ref']['path']).parent))]:snapshots.append(life.final_tree_observation(c,dict(helper_refs={'engineering_child.py':dict(path=token)},output_root=prefix),registry))
 result=snapshots[0];result['unknown_live']=[v for s in snapshots for v in s['unknown_live']];result['all_execution_ended']=not result['registered_live'] and not result['unknown_live'];return result
def run(plan_ref,review_ref):
 began=time.monotonic();began_boot=boottime();p=validate(plan_ref,review_ref);x=p
 if sys.platform!='linux' or sys.executable!=DQN:raise ValueError('actual native DQN interpreter')
 fresh_route(p)
 life=lifecycle(x);p['_actual_plan_ref']=plan_ref;parent=c.identity(os.getpid());control=Path(x['outer_control_root']);control.mkdir(parents=True,exist_ok=True);Path(p['archive_root']).mkdir(parents=True,exist_ok=True);hard=float(os.environ['R11_LANE_DEADLINE_BOOTTIME'])-began_boot;
 if not 0<hard<=p['outer_hard_seconds']:raise ValueError('same finite absolute lane deadline')
 observer=None;child=None;service=None;worker=None;registry={};peak=0;error=None;code=None;tree=None;closed=False;status=None;worker_observed=None
 with exclusive_lock(control/'controller.lock'):
  check_pause(p,'Gh_controller_start');c.immutable(control/'controller_intent.json',dict(actual_controller=parent,plan_ref=plan_ref,root_review_ref=review_ref,design_ref=p['design_ref'],historical_control_costs_are_separate=True,outer_hard_seconds=hard,POST=0,automatic_retry=False))
  def bound():
   libc=ctypes.CDLL(None,use_errno=True)
   if libc.prctl(1,signal.SIGKILL,0,0,0)!=0 or os.getppid()!=parent['pid']:os._exit(126)
  try:
   observer=Observer(p,control,storage_check=storage_check);observer.observe('outer_admission');last_storage=time.monotonic()
   command=[DQN,'-B','-u','-m','scripts.stage1_ai_search_r11_prospective.server','--plan',plan_ref['path'],'--plan-sha256',plan_ref['sha256'],'--review',review_ref['path'],'--review-sha256',review_ref['sha256'],'--port',str(p['workbench_port'])]
   with (control/'service.stdout.log').open('xb') as out,(control/'service.stderr.log').open('xb') as err:
    if boottime()>=float(os.environ['R11_LANE_DEADLINE_BOOTTIME']):raise RuntimeError('lane deadline before service spawn')
    child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,close_fds=True,preexec_fn=bound,env=dict(os.environ,PYTHONPATH=p['source_repo'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'));service=c.identity(child.pid)
    if service is None:raise RuntimeError('service generation unknown; no replay')
    registry[(service['pid'],service['birth_ticks'],service['boot_id'])]=service;c.immutable(control/'actual_service_launch.json',dict(actual_controller=parent,actual_service=service,command=command,service_parent_death_bound=True,controller_POST_count=0))
    while True:
     check_pause(p,'Gh_outer_monitor');life.observe_descendants(c,service,registry);worker=merge_native(p,registry);peak=max(peak,life.registered_rss(c,dict(registry) | {(parent['pid'],parent['birth_ticks'],parent['boot_id']):parent}));final_limits(boottime()-began_boot,peak,hard,lambda:None)
     if time.monotonic()-last_storage>=5:observer.observe('outer_monitor');last_storage=time.monotonic()
     early=wait_one(child,service)
     if early is not None:
      c.immutable(control/'service_reap.json',dict(early,retirement_expected=False));raise RuntimeError('service exited before controlled retirement')
     if worker:
      path=Path(p['archive_root'])/'worker_reap.json'
      if path.exists():
       worker_wait=c.read(fref(path))
       if worker_wait['actual_child']!=worker or worker_wait['plan_ref']!=plan_ref:raise ValueError('exact worker wait4 generation/plan')
       worker_observed=c.identity(worker['pid'])
       if worker_wait['decoded_exit_code']!=0:raise RuntimeError('worker nonzero exit: '+str(worker_wait['decoded_exit_code']))
       break
     time.sleep(.25)
  except BaseException as e:error=type(e).__name__+': '+str(e)
  finally:
   try:
    worker=merge_native(p,registry) or worker
    if service:life.observe_descendants(c,service,registry)
    tree=observe_all(life,p,registry)
    if tree['unknown_live']:raise RuntimeError('unknown live witness; no inferred signal/completion')
    if error is None and any(not c.same_process(service,v) for v in tree['registered_live']):error='worker ended while descendants still live; cleanup is not completion'
    life.exact_terminate(c,registry,signal.SIGTERM);until=time.monotonic()+3
    while time.monotonic()<until and observe_all(life,p,registry)['registered_live']:time.sleep(.05)
    if observe_all(life,p,registry)['registered_live']:life.exact_terminate(c,registry,signal.SIGKILL)
    if child and child.returncode is None:
     until=min(float(os.environ['R11_LANE_DEADLINE_BOOTTIME']),boottime()+2)
     while boottime()<until:
      service_wait=wait_one(child,service)
      if service_wait is not None:
       c.immutable(control/'service_reap.json',dict(service_wait,retirement_expected=True));code=service_wait['decoded_exit_code'];break
      time.sleep(.02)
     if child.returncode is None:raise RuntimeError('bounded service wait4 unknown')
    elif child:code=child.returncode
    tree=observe_all(life,p,registry);closed=service is not None and tree['all_execution_ended'];peak=max(peak,life.registered_rss(c,dict(registry) | {(parent['pid'],parent['birth_ticks'],parent['boot_id']):parent}));final_limits(boottime()-began_boot,peak,hard,lambda:check_pause(p,'Gh_outer_final'))
    if not closed:raise RuntimeError('registered descendants exit not closed')
   except BaseException as e:error=(error or '')+'; cleanup_unknown:'+type(e).__name__+': '+str(e)
   try:
    path=Path(p['archive_root'])/'status.json';status=json.loads(path.read_bytes()) if path.exists() else None
    if status is not None and not isinstance(status,dict):raise ValueError('worker status object')
    if observer:observer.observe('outer_after_status')
    peak=max(peak,life.registered_rss(c,dict(registry) | {(parent['pid'],parent['birth_ticks'],parent['boot_id']):parent}));final_limits(boottime()-began_boot,peak,hard,lambda:check_pause(p,'Gh_outer_after_status'))
   except BaseException as e:error=(error or '')+'; status_or_final_boundary:'+type(e).__name__+': '+str(e)
   if observer:observer.close()
   c.immutable(control/'controller_finished.json',dict(schema='R11ProspectiveLaneControllerExitV1',actual_controller=parent,actual_service=service,actual_worker=worker,actual_worker_observed_before_service_retirement=worker_observed,registered_descendants=list(registry.values()),postwait_process_observations=tree,service_wait_exit_code=code,service_retirement_expected=True,worker_exit_code_exact=None if not (Path(p['archive_root'])/'worker_reap.json').exists() else c.read(fref(Path(p['archive_root'])/'worker_reap.json'))['decoded_exit_code'],worker_status=status,worker_reap_ref=fref(Path(p['archive_root'])/'worker_reap.json') if (Path(p['archive_root'])/'worker_reap.json').exists() else None,service_reap_ref=fref(control/'service_reap.json') if (control/'service_reap.json').exists() else None,registered_subtree_exit_verified=closed,provisional_controller_wall_to_receipt_seconds=time.monotonic()-began,historical_control_costs_are_separate=True,outer_hard_seconds=hard,peak_parent_children_RSS_bytes=peak,peak_RSS_sampled=True,error=error,plan_ref=plan_ref,root_review_ref=review_ref,design_ref=p['design_ref'],automatic_retry=False,controller_POST_count=0,controller_exit_requires_separate_external_observation=True))
  if error or not status or status.get('stage')!='completed_pending_independent_audit':raise SystemExit(2)
def main():
 a=argparse.ArgumentParser();a.add_argument('--plan',required=True);a.add_argument('--plan-sha256',required=True);a.add_argument('--review',required=True);a.add_argument('--review-sha256',required=True);a.add_argument('--execute',action='store_true');v=a.parse_args();pr=dict(path=v.plan,sha256=v.plan_sha256);rr=dict(path=v.review,sha256=v.review_sha256)
 if v.execute:run(pr,rr)
 else:validate(pr,rr);print('readcheck only: no service/POST/algorithm')
if __name__=='__main__':main()
