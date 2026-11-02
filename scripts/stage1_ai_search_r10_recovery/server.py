"""One same-ID recovery resume route; no ordinary start, queue or retry."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from scripts.stage1_ai_search_r10 import contracts as c
from scripts.stage1_ai_search_r10.control import exclusive_lock
from .contracts import validate,check_pause
class Endpoint:
 def __init__(self,plan_ref,review_ref):
  self.plan_ref=plan_ref;self.review_ref=review_ref;self.plan,self.contract,self.rows=validate(plan_ref,review_ref);self.root=Path(self.plan['archive_root']);self.root.mkdir(parents=True,exist_ok=True);self.path='/api/ai-search-r10/tasks/'+self.plan['run_id'];check_pause(self.plan,'recovery_service_startup')
 def get(self,path):
  if path=='/health':return dict(group='S1-AI17-R10',stage='storage_race_same_id_recovery',plan_ref=self.plan_ref,actual_service=c.identity(os.getpid()),no_generic_scheduler=True,ordinary_start_forbidden=True)
  if path!=self.path:raise ValueError('sole original run GET')
  return dict(run_id=self.plan['run_id'],plan_ref=self.plan_ref,status=json.loads((self.root/'status.json').read_bytes()) if (self.root/'status.json').exists() else dict(stage='recovery_not_submitted'))
 def post(self,path,body,popen=subprocess.Popen):
  expected=dict(plan_ref=self.plan_ref,root_review_ref=self.review_ref,request_id=self.plan['request_id'])
  if body!=expected:raise ValueError('sole exact root plan/review/request pair')
  if path==self.path+'/stop':c.immutable(self.root/'stop.request',dict(user_pause=True,run_id=self.plan['run_id'],actual_service=c.identity(os.getpid())));return dict(stop_requested=True)
  if path!=self.path+'/resume':raise ValueError('resume-only same-ID endpoint; no start/retrain/queue')
  p,x,rows=validate(self.plan_ref,self.review_ref);check_pause(p,'API_before_recovery_resume');c.fresh_route(p)
  with exclusive_lock(self.root/'submission.lock'):
   if (self.root/'submission_intent.json').exists():raise ValueError('submitted/unknown resume intent; never replay')
   check_pause(p,'before_recovery_intent');c.fresh_route(p);intent=c.immutable(self.root/'submission_intent.json',dict(request=expected,actual_service=c.identity(os.getpid()),recovery_contract_ref=p['recovery_contract_ref'],unknown_never_replayed=True))
   command=[sys.executable,'-B','-u','-m','scripts.stage1_ai_search_r10_recovery.worker','--plan',self.plan_ref['path'],'--plan-sha256',self.plan_ref['sha256'],'--review',self.review_ref['path'],'--review-sha256',self.review_ref['sha256']]
   with (self.root/'worker.stdout.log').open('xb') as out,(self.root/'worker.stderr.log').open('xb') as err:child=popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,R10_SERVICE_PID=str(os.getpid()),PYTHONDONTWRITEBYTECODE='1'))
   owner=c.identity(child.pid)
   if owner is None:raise RuntimeError('worker birth unknown; preserve intent even if already exited')
   launch=c.immutable(self.root/'actual_worker_launch.json',dict(actual_worker=owner,command=command,intent_ref=intent,plan_ref=self.plan_ref,recovery_contract_ref=p['recovery_contract_ref']));return dict(launch_ref=launch,actual_worker=owner,POST_replayed=False,same_original_run_id=p['run_id'])
def main():
 a=argparse.ArgumentParser();a.add_argument('--plan',required=True);a.add_argument('--plan-sha256',required=True);a.add_argument('--review',required=True);a.add_argument('--review-sha256',required=True);a.add_argument('--port',required=True,type=int);v=a.parse_args()
 if v.port not in (9040,9041):raise ValueError('fresh same-group9040/9041 only')
 endpoint=Endpoint(dict(path=v.plan,sha256=v.plan_sha256),dict(path=v.review,sha256=v.review_sha256))
 if v.port!=endpoint.plan['workbench_port']:raise ValueError('exact root workbench port')
 with exclusive_lock(endpoint.root/'service.lock'):
  c.immutable(endpoint.root/'service_epoch_intent.json',dict(actual_service=c.identity(os.getpid()),plan_ref=endpoint.plan_ref,port=v.port,recovery_contract_ref=endpoint.plan['recovery_contract_ref']))
  class Handler(BaseHTTPRequestHandler):
   def dispatch(self):
    try:
     if self.command=='GET':value=endpoint.get(self.path)
     elif self.command=='POST':
      size=int(self.headers.get('Content-Length','0'))
      if not 0<size<=8192:raise ValueError('bounded exact request')
      value=endpoint.post(self.path,json.loads(self.rfile.read(size)))
     else:raise ValueError('unsupported method')
     raw=json.dumps(value).encode();self.send_response(200)
    except Exception as e:raw=json.dumps(dict(error=str(e),automatic_retry=False)).encode();self.send_response(409)
    self.send_header('Content-Length',str(len(raw)));self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(raw)
   do_GET=dispatch;do_POST=dispatch;do_DELETE=dispatch;do_PUT=dispatch
  ThreadingHTTPServer(('127.0.0.1',v.port),Handler).serve_forever()
if __name__=='__main__':main()
