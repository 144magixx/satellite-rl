"""One root-admitted standard-library child; never imports a science algorithm."""
import argparse,hashlib,json,os,sys
from pathlib import Path
from scripts.stage1_ai_search_r10 import contracts as c
from scripts.stage1_ai_search_r10.control import exclusive_lock
from . import contracts as r
from .reaper import run_process,boottime,Watchdog

def check(plan_ref,review_ref):
 p=c.read(plan_ref);a=c.read(review_ref)
 if p.get('schema')!='R11ProspectiveSingleNoopPlanV1' or p.get('ssh_port')!=20473 or (p.get('maximum_attempts'),p.get('maximum_children'),p.get('hard_total_seconds'),p.get('rss_bytes'),p.get('storage_max_bytes'))!=(1,1,10,134217728,1048576):raise ValueError('exact one73/onechild/10s/128MiB/1MiB engineering scope')
 if a.get('schema')!='RootR11ProspectiveSingleNoopAdmissionV1' or a.get('exact_plan_ref')!=plan_ref or any(a.get(k)is not True for k in ('passed','engineering_noop_execution_authorized','source_accepted','runtime_accepted')) or a.get('scientific_execution_authorized')is not False or a.get('automatic_retry')is not False:raise ValueError('root exact elected noop admission')
 if p.get('registration_ref',{}).get('sha256')!='2dfb7184c47a8467c3f8a35b24f8444cf0d38c38f7253dc5fde0fa72b542f0dc':raise ValueError('explicit second finite noop registration required')
 reg=c.read(p['registration_ref'])
 if reg.get('schema')!='RootR11SecondLifecycleNoopProspectiveRegistrationV1' or reg.get('additional_registration_accepted')is not True or reg.get('prior_registration_exhausted')is not True or reg.get('new_registration_number')!=2:raise ValueError('second registration; failed01 never replenished')
 if (p.get('run_id'),p.get('output_root'),p.get('ssh_port'),p.get('maximum_attempts'),p.get('maximum_children'),p.get('hard_total_seconds'),p.get('rss_bytes'),p.get('storage_max_bytes'))!=(reg['run_id'],reg['archive_root'],reg['ssh_port'],reg['maximum_new_attempts'],reg['maximum_new_payload_children'],reg['hard_total_seconds'],reg['rss_bytes'],reg['storage_max_bytes']) or p['pause_plan'].get('run_id')!=p['run_id'] or a.get('registration_ref')!=p['registration_ref']:raise ValueError('exact second run/root/limits/rootreview binding')
 if p['prospective_lifecycle_acceptance_ref']['sha256']!='a26e12fcc51c5af9b042548187fe1947a74b75e03acd77e8fda2606651089d57':raise ValueError('accepted prospective design')
 if p['output_root']!=p['pause_plan']['archive_root'] or not p['output_root'].startswith('/home/u2024110224/first_stage_satellite_rl/s1_ai17/20473/round11/'):raise ValueError('exact73 noop archive/pause binding')
 c.read(p['prospective_lifecycle_acceptance_ref']);payload=c.pin(p['payload_ref'])
 if payload.resolve()!=Path(__file__).with_name('noop_payload.py').resolve() or p['payload_ref']['sha256']!=p['pause_plan']['source_files']['scripts/stage1_ai_search_r11_prospective/noop_payload.py']['sha256']:raise ValueError('sole frozen standard-library payload')
 r.source_check(p['pause_plan']);r.human_epoch(p['pause_plan']);r.storage_check(p['pause_plan']);return p

def run(plan_ref,review_ref):
 actual=c.identity(os.getpid());hz=os.sysconf('SC_CLK_TCK');born_lower=(actual['birth_ticks']-1)/hz;whole_deadline=born_lower+10;Watchdog(whole_deadline,boottime);p=check(plan_ref,review_ref)
 if sys.platform!='linux' or sys.executable!=r.DQN:raise ValueError('actual native DQN')
 r.fresh_route(dict(p['pause_plan'],executor_port=20473,actual_resource_route_ref=p['actual_resource_route_ref']));root=Path(p['output_root'])
 if '/round11/' not in root.as_posix()+'/' or root.resolve()!=root.absolute():raise ValueError('new exact owned noop root')
 root.mkdir(parents=True,exist_ok=True);life=r.lifecycle(p)
 with exclusive_lock(root/'noop.lock'):
  c.immutable(root/'noop_attempt_intent.json',dict(plan_ref=plan_ref,review_ref=review_ref,actual_observer=actual,max_attempts=1,max_children=1,whole_observer_birth_lower_bound_clock_boottime_seconds=born_lower,whole_deadline_clock_boottime_seconds=whole_deadline,automatic_retry=False))
  remaining=whole_deadline-boottime()-.1
  if not 0<remaining<10:raise RuntimeError('remaining original10s exhausted before sole child')
  command=[r.DQN,'-B',p['payload_ref']['path'],'--unique-evidence',str(root/'native'/'noop_evidence.json')]
  receipt=run_process(command,root/'native','noop',stop=lambda:r.check_pause(p['pause_plan'],'noop'),life=life,hard_seconds=remaining,rss_bytes=134217728,helper_token=p['payload_ref']['path'],boot_id=actual['boot_id'],observer_tail_seconds=.1)
  files=list(root.rglob('*'))
  if any(v.is_symlink() for v in files) or sum(v.stat().st_size for v in files if v.is_file())>1048576:raise RuntimeError('1MiB noop storage ceiling')
  observed=boottime()
  if observed>=whole_deadline:raise RuntimeError('whole observer birth/input/source/child/publication10s exceeded')
  result_ref=c.immutable(root/'noop_result.json',dict(schema='R11ProspectiveSingleNoopResultV1',plan_ref=plan_ref,review_ref=review_ref,actual_observer=actual,actual_child=receipt['actual_child'],native_guard_ref=receipt['guard_ref'],maximum_children=1,actual_payload_children=1,solver_vectors_GEO_models=0,raw_child_wait_status=receipt['raw_wait_status'],decoded_child_exit_code=receipt['exit_code'],native_complete_exit_upper_seconds=receipt['complete_native_exit_upper_seconds'],whole_observer_birth_lower_bound_clock_boottime_seconds=born_lower,observed_whole_to_receipt_seconds=observed-born_lower,complete_observer_process_exit_requires_external_observation=True,historical_four_toy30_still_unknown=True,automatic_retry=False))
  return result_ref

def main():
 a=argparse.ArgumentParser();a.add_argument('--plan',required=True);a.add_argument('--plan-sha256',required=True);a.add_argument('--review',required=True);a.add_argument('--review-sha256',required=True);a.add_argument('--execute',action='store_true');v=a.parse_args();p=dict(path=v.plan,sha256=v.plan_sha256);r=dict(path=v.review,sha256=v.review_sha256)
 if v.execute:print(json.dumps(run(p,r)))
 else:check(p,r);print('readcheck only: no noop child/science')
if __name__=='__main__':main()
