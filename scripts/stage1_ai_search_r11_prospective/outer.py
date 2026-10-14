"""Persistent direct-parent reaper of the whole charged scientific lane."""
import argparse,os,sys,time
from pathlib import Path
from scripts.stage1_ai_search_r10 import contracts as c
from scripts.stage1_ai_search_r10.control import exclusive_lock
from .contracts import validate,check_pause,fresh_route,lifecycle,storage_check,DQN
from .storage import Observer
from .reaper import run_process,boottime,Watchdog

def run(plan_ref,review_ref):
 setup=boottime();p=validate(plan_ref,review_ref)
 if sys.platform!='linux' or sys.executable!=DQN:raise ValueError('actual native DQN')
 fresh_route(p);check_pause(p,'outer_measurement_observer');life=lifecycle(p);root=Path(p['outer_observer_root']);root.mkdir(parents=True,exist_ok=True);Path(p['archive_root']).mkdir(parents=True,exist_ok=True)
 with exclusive_lock(root/'observer.lock'):
  c.immutable(root/'observer_admission_cost.json',dict(actual_observer=c.identity(os.getpid()),observed_seconds=boottime()-setup,scope='source/root metadata admission; no scientific algorithm or input decode',outside_charged_lane=True))
  observer=Observer(p,root,storage_check=storage_check);last=boottime()
  def stop():
   nonlocal last
   check_pause(p,'outer_native_monitor')
   if boottime()-last>=5:observer.observe('outer_native_monitor');last=boottime()
  def postflight():
   path=Path(p['outer_control_root'])/'controller_finished.json';value=c.read(dict(path=str(path),sha256=c.digest(path)))
   if value['error']is not None or not value['registered_subtree_exit_verified'] or not value['worker_status'] or value['worker_status']['stage']!='completed_pending_independent_audit' or value['worker_exit_code_exact']!=0:raise RuntimeError('complete durable worker/controller outcome required')
  command=[DQN,'-B','-u','-m','scripts.stage1_ai_search_r11_prospective.controller','--plan',plan_ref['path'],'--plan-sha256',plan_ref['sha256'],'--review',review_ref['path'],'--review-sha256',review_ref['sha256'],'--execute']
  os.environ['PYTHONPATH']=p['source_repo']
  try:
   receipt=run_process(command,root/'charged_lane','lane',stop=stop,life=life,hard_seconds=p['outer_hard_seconds'],helper_token='scripts.stage1_ai_search_r11_prospective.controller')
   Watchdog(boottime()+2,boottime)
   postflight()
  finally:observer.close()
  # No native descendant remains. Bound only this outer observer's own exit tail.
  Watchdog(boottime()+2,boottime)
  return receipt

def main():
 a=argparse.ArgumentParser();a.add_argument('--plan',required=True);a.add_argument('--plan-sha256',required=True);a.add_argument('--review',required=True);a.add_argument('--review-sha256',required=True);a.add_argument('--execute',action='store_true');v=a.parse_args();p=dict(path=v.plan,sha256=v.plan_sha256);r=dict(path=v.review,sha256=v.review_sha256)
 if v.execute:run(p,r)
 else:validate(p,r);print('readcheck only: no process/service/POST/science')
if __name__=='__main__':main()
