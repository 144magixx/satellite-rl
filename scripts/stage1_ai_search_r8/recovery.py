"""One same-ID corrective epoch after proven pre-local input failure; no replay."""
import json,os,sys,time,subprocess
from pathlib import Path
from .contracts import read,pin,digest,identity,immutable,validate,fresh_route,CORE_FILES
from .deadline import before_start

def validate_recovery(plan):
 recovery=plan['operations_recovery'];c=read(recovery['contract_ref']);r=read(recovery['root_review_ref'])
 if c.get('schema')!='R8SameIDBeforeLocalReaderRecoveryContractV1' or c['run_id']!=plan['run_id'] or c['archive_root']!=plan['archive_root'] or c['epoch_root']!=recovery['epoch_root']:raise ValueError('exact original R8 sameID epoch')
 if r.get('contract_ref')!=recovery['contract_ref'] or r.get('execution_authorized')is not True or r.get('passed')is not True:raise ValueError('root exact recovery admission')
 original=read(c['original_plan_ref'])
 for key in ('run_id','request_id','archive_root','scientific_commit','design_ref','cases','case_indices','arms','checkpoint_ref','original_source_manifest_ref','scientific_new_calls'):
  if plan[key]!=original[key]:raise ValueError('original scientific task contract changed: '+key)
 for name in CORE_FILES:
  key='src/satellite_rl/coverage/ai_search_r8/'+name
  if plan['source_files'][key]['sha256']!=original['source_files'][key]['sha256']:raise ValueError('original four scientific core files changed')
 for key in set(original)-{'source_files','core_source_root','core_ref','actual_resource_route_ref','status'}:
  if key not in plan or plan[key]!=original[key]:raise ValueError('original frozen plan field changed: '+key)
 if c.get('actual_original_calls')!=dict(local=1,D5=1) or c.get('failed_A_actual_local_calls')!=0 or c.get('remaining_actual_calls_max')!=dict(local=59,D5=59) or recovery['remaining_actual_calls_max']!=c['remaining_actual_calls_max']:raise ValueError('no extra algorithm call budget')
 if c['original_slot_index_ref']!=recovery['original_slot_index_ref'] or c['original_failed_status_ref']!=recovery['original_failed_status_ref']:raise ValueError('original exact failed assets')
 oldindex=read(c['original_slot_index_ref'])
 if list(oldindex)!=['0:G']:raise ValueError('sole original completed G reused')
 for ref in c['protected_original_refs']:pin(ref)
 for ref in oldindex.values():pin(ref)
 for process in c['original_processes']:
  current=identity(process['pid'])
  if current and current['birth_ticks']==process['birth_ticks'] and current['boot_id']==process['boot_id'] and current['state'] not in ('Z','X'):raise ValueError('original worker/owned child still executing')
 if not Path(c['epoch_root']).resolve().is_relative_to(Path(plan['archive_root']).resolve()) or Path(c['epoch_root']).resolve()==Path(plan['archive_root']).resolve():raise ValueError('fresh independent corrective epoch inside own archive')
 if (Path(plan['archive_root'])/'slots/0/A/local_state/state.json').exists():raise ValueError('A algorithm previously entered; cannot repeat')
 return c

def resume_once(plan_ref,review_ref):
 plan=validate(plan_ref,review_ref);c=validate_recovery(plan);before_start(plan,'sameID_recovery_before_native_spawn');fresh_route(plan);epoch=Path(c['epoch_root']);epoch.mkdir(parents=True,exist_ok=True)
 source=Path(plan['original_source_manifest_ref']['path']).parent;sys.path[:0]=[str(source/'src'),str(source)]
 from satellite_rl.workbench.geoload_runner import exclusive_lock
 with exclusive_lock(epoch/'unique_resume.lock'):
  if (epoch/'resume_intent.json').exists():raise ValueError('unknown or submitted corrective epoch; no retry')
  immutable(epoch/'resume_intent.json',dict(plan_ref=plan_ref,root_review_ref=review_ref,contract_ref=plan['operations_recovery']['contract_ref'],no_retry=True))
  command=[sys.executable,'-B','-u','-m','scripts.stage1_ai_search_r8.worker','--plan',plan_ref['path'],'--plan-sha256',plan_ref['sha256'],'--review',review_ref['path'],'--review-sha256',review_ref['sha256']]
  env=dict(os.environ,R8_SERVICE_PID=str(os.getpid()),PYTHONDONTWRITEBYTECODE='1')
  with (epoch/'worker.stdout.log').open('xb') as out,(epoch/'worker.stderr.log').open('xb') as err:worker=subprocess.Popen(command,stdout=out,stderr=err,env=env,start_new_session=True)
  launch=immutable(epoch/'actual_worker_launch.json',dict(actual_worker=identity(worker.pid),command=command,plan_ref=plan_ref,contract_ref=plan['operations_recovery']['contract_ref'],new_epoch_only=True,original_G_reused=True))
  return dict(run_id=plan['run_id'],actual_worker=identity(worker.pid),launch_ref=launch,POST_replayed=False)

def main():
 import argparse
 from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
 a=argparse.ArgumentParser();a.add_argument('--plan',required=True);a.add_argument('--plan-sha256',required=True);a.add_argument('--review',required=True);a.add_argument('--review-sha256',required=True);args=a.parse_args();pr=dict(path=args.plan,sha256=args.plan_sha256);rr=dict(path=args.review,sha256=args.review_sha256);p=validate(pr,rr);c=validate_recovery(p);before_start(p,'restricted_recovery_service_start');epoch=Path(c['epoch_root']);epoch.mkdir(parents=True,exist_ok=True)
 immutable(epoch/'actual_restricted_service.json',dict(actual_service=identity(os.getpid()),plan_ref=pr,review_ref=rr,source_files=p['source_files'],port=9021,no_scheduler=True));task='/api/ai-search-r8/tasks/'+p['run_id']
 class Handler(BaseHTTPRequestHandler):
  def dispatch(self):
   try:
    if self.command=='GET' and self.path=='/health':value=dict(run_id=p['run_id'],plan_ref=pr,actual_service=identity(os.getpid()),no_scheduler=True,restricted=True)
    elif self.command=='GET' and self.path==task:value=dict(run_id=p['run_id'],status=json.loads((Path(p['archive_root'])/'status.json').read_bytes()),epoch_root=str(epoch))
    elif self.command=='POST' and self.path==task+'/resume':
     n=int(self.headers.get('Content-Length','0'))
     if not 0<n<=8192:raise ValueError('small exact resume body')
     if json.loads(self.rfile.read(n))!=dict(plan_ref=pr,root_review_ref=rr):raise ValueError('exact root sameID pair only')
     value=resume_once(pr,rr)
    else:raise ValueError('sole sameID GET/one resume only')
    raw=json.dumps(value).encode();self.send_response(200)
   except Exception as error:raw=json.dumps(dict(error=str(error),automatic_retry=False)).encode();self.send_response(409)
   self.send_header('Content-Length',str(len(raw)));self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(raw)
  do_GET=dispatch;do_POST=dispatch;do_DELETE=dispatch;do_PUT=dispatch
 ThreadingHTTPServer(('127.0.0.1',9021),Handler).serve_forever()
if __name__=='__main__':main()
