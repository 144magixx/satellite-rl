"""One exact R9 M task; permanent submission intent, no generic scheduler."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from .contracts import validate,read,immutable,atomic,identity,fresh_route
from .control import boundary,storage_boundary,exclusive_lock

class Endpoint:
 def __init__(self,plan_ref,review_ref):
  self.plan_ref=plan_ref;self.review_ref=review_ref;self.plan=validate(plan_ref,review_ref);self.root=Path(self.plan['archive_root']);self.root.mkdir(parents=True,exist_ok=True);self.path='/api/ai-search-r9/tasks/'+self.plan['run_id'];boundary(self.plan,'service_startup')
 def get(self,path):
  if path=='/health':return dict(group='S1-AI17-R9',stage='M',plan_ref=self.plan_ref,actual_service=identity(os.getpid()),no_generic_scheduler=True)
  if path!=self.path:raise ValueError('sole exact R9 task GET')
  status=json.loads((self.root/'status.json').read_bytes()) if (self.root/'status.json').exists() else dict(stage='not_submitted')
  return dict(run_id=self.plan['run_id'],status=status,plan_ref=self.plan_ref)
 def post(self,path,body,popen=subprocess.Popen):
  expected=dict(plan_ref=self.plan_ref,root_review_ref=self.review_ref,request_id=self.plan['request_id'])
  if body!=expected:raise ValueError('exact immutable request only')
  if path==self.path+'/stop':
   atomic(self.root/'stop.request',dict(user_pause=True,run_id=self.plan['run_id'],actual_service=identity(os.getpid())));return dict(stop_requested=True)
  if path!=self.path+'/start':raise ValueError('sole R9 M start/stop; no old endpoints')
  p=validate(self.plan_ref,self.review_ref);boundary(p,'API_before_POST');storage_boundary(p,'API_before_POST');fresh_route(p)
  with exclusive_lock(self.root/'submission.lock'):
   if (self.root/'submission_intent.json').exists():raise ValueError('submitted/unknown intent; no automatic replay')
   boundary(p,'before_submission_intent');fresh_route(p)
   intent=immutable(self.root/'submission_intent.json',dict(request=expected,actual_service=identity(os.getpid()),unknown_never_replayed=True))
   command=[sys.executable,'-B','-u','-m','scripts.stage1_ai_search_r9.worker','--plan',self.plan_ref['path'],'--plan-sha256',self.plan_ref['sha256'],'--review',self.review_ref['path'],'--review-sha256',self.review_ref['sha256']]
   env=dict(os.environ,R9_SERVICE_PID=str(os.getpid()),SATELLITE_RL_SSH_PORT=str(p['executor_port']),PYTHONDONTWRITEBYTECODE='1');boundary(p,'before_native_worker_spawn')
   with (self.root/'worker.stdout.log').open('ab') as out,(self.root/'worker.stderr.log').open('ab') as err:child=popen(command,stdout=out,stderr=err,env=env,start_new_session=True)
   owner=identity(child.pid)
   if owner is None and child.poll()is None:raise RuntimeError('live worker identity unavailable; preserve intent')
   launch=immutable(self.root/'actual_worker_launch.json',dict(actual_worker=owner,command=command,intent_ref=intent,plan_ref=self.plan_ref))
   return dict(launch_ref=launch,actual_worker=owner,POST_replayed=False)
def main():
 a=argparse.ArgumentParser();a.add_argument('--plan',required=True);a.add_argument('--plan-sha256',required=True);a.add_argument('--review',required=True);a.add_argument('--review-sha256',required=True);a.add_argument('--port',type=int,required=True);v=a.parse_args()
 if v.port not in (9030,9031):raise ValueError('new independent R9 service ports9030/9031; fresh admission still required')
 endpoint=Endpoint(dict(path=v.plan,sha256=v.plan_sha256),dict(path=v.review,sha256=v.review_sha256))
 with exclusive_lock(endpoint.root/'service.lock'):
  immutable(endpoint.root/'service_epoch_intent.json',dict(actual_service=identity(os.getpid()),plan_ref=endpoint.plan_ref,port=v.port))
  class Handler(BaseHTTPRequestHandler):
   def dispatch(self):
    try:
     if self.command=='GET':value=endpoint.get(self.path)
     elif self.command=='POST':
      n=int(self.headers.get('Content-Length','0'))
      if not 0<n<=8192:raise ValueError('small exact request')
      value=endpoint.post(self.path,json.loads(self.rfile.read(n)))
     else:raise ValueError('unsupported method')
     raw=json.dumps(value).encode();self.send_response(200)
    except Exception as e:raw=json.dumps(dict(error=str(e),automatic_retry=False)).encode();self.send_response(409)
    self.send_header('Content-Length',str(len(raw)));self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(raw)
   do_GET=dispatch;do_POST=dispatch;do_DELETE=dispatch;do_PUT=dispatch
  ThreadingHTTPServer(('127.0.0.1',v.port),Handler).serve_forever()
if __name__=='__main__':main()
