"""Single exact R8 task API, durable owner and zero generic scheduler scope."""
import argparse,json,os,sys,subprocess
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from .contracts import validate,read,pin,immutable,atomic,identity,fresh_route
from .deadline import boundary

class Endpoint:
 def __init__(self,plan_ref,review_ref):
  self.plan_ref=plan_ref;self.review_ref=review_ref;self.plan=validate(plan_ref,review_ref);self.root=Path(self.plan['archive_root']);self.root.mkdir(parents=True,exist_ok=True)
  boundary(self.plan,'API_startup');self.path='/api/ai-search-r8/tasks/'+self.plan['run_id']
 def get(self,path):
  if path=='/health':return dict(group='S1-AI17-R8',no_generic_scheduler=True,plan_ref=self.plan_ref,actual_service=identity(os.getpid()),boundary=boundary(self.plan,'API_health'))
  if path!=self.path:raise ValueError('sole exact task GET')
  state=json.loads((self.root/'status.json').read_bytes()) if (self.root/'status.json').exists() else dict(stage='not_submitted')
  return dict(run_id=self.plan['run_id'],status=state,plan_ref=self.plan_ref)
 def post(self,path,body,popen=subprocess.Popen):
  expected=dict(plan_ref=self.plan_ref,root_review_ref=self.review_ref,request_id=self.plan['request_id'])
  if body!=expected:raise ValueError('exact immutable R8 request body')
  if path==self.path+'/stop':
   atomic(self.root/'stop.request',dict(run_id=self.plan['run_id'],requested=True,actual_service=identity(os.getpid())));return dict(stop_requested=True)
  if path!=self.path+'/start':raise ValueError('sole exact R8 start or safe stop')
  p=validate(self.plan_ref,self.review_ref);boundary(p,'API_prePOST');fresh_route(p)
  source=Path(p['original_source_manifest_ref']['path']).parent;sys.path[:0]=[str(source/'src'),str(source)]
  from satellite_rl.workbench.geoload_runner import exclusive_lock
  with exclusive_lock(self.root/'submission.lock'):
   if (self.root/'submission_intent.json').exists():raise ValueError('unknown or submitted original intent; no automatic repeat')
   boundary(p,'queue_before_native_launch')
   # Persistent intent precedes native spawn; no queue or scheduler can create another owner.
   intent=immutable(self.root/'submission_intent.json',dict(request=expected,actual_service=identity(os.getpid()),unknown_outcome_never_replayed=True))
   command=[sys.executable,'-B','-u','-m','scripts.stage1_ai_search_r8.worker','--plan',self.plan_ref['path'],'--plan-sha256',self.plan_ref['sha256'],'--review',self.review_ref['path'],'--review-sha256',self.review_ref['sha256']]
   env=dict(os.environ);env['R8_SERVICE_PID']=str(os.getpid());env['SATELLITE_RL_SSH_PORT']=str(p['executor_port']);env['PYTHONDONTWRITEBYTECODE']='1'
   boundary(p,'native_before_worker_spawn')
   with (self.root/'worker.stdout.log').open('ab') as out,(self.root/'worker.stderr.log').open('ab') as err:worker=popen(command,env=env,stdout=out,stderr=err,start_new_session=True)
   launch=immutable(self.root/'actual_worker_launch.json',dict(actual_worker=identity(worker.pid),command=command,plan_ref=self.plan_ref,root_review_ref=self.review_ref,source_files=p['source_files'],intent_ref=intent,scientific_call_caps=dict(local=60,D5=60),no_generic_scheduler=True))
   return dict(run_id=p['run_id'],launch_ref=launch,actual_worker=identity(worker.pid),POST_replayed=False)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha256',required=True);parser.add_argument('--review',required=True);parser.add_argument('--review-sha256',required=True);parser.add_argument('--port',type=int,default=9020);a=parser.parse_args()
 if a.port!=9020:raise ValueError('independent R8 service only9020')
 endpoint=Endpoint(dict(path=a.plan,sha256=a.plan_sha256),dict(path=a.review,sha256=a.review_sha256))
 atomic(endpoint.root/'actual_service.json',dict(actual_service=identity(os.getpid()),plan_ref=endpoint.plan_ref,source_files=endpoint.plan['source_files'],boundary=boundary(endpoint.plan,'server_before_listen'),port=a.port))
 class Handler(BaseHTTPRequestHandler):
  def dispatch(self):
   try:
    if self.command=='GET':value=endpoint.get(self.path)
    elif self.command=='POST':
     length=int(self.headers.get('Content-Length','0'))
     if not 0<length<=8192:raise ValueError('small exact request')
     value=endpoint.post(self.path,json.loads(self.rfile.read(length)))
    else:raise ValueError('unsupported method')
    raw=json.dumps(value).encode();self.send_response(200)
   except Exception as e:raw=json.dumps(dict(error=str(e),automatic_retry=False)).encode();self.send_response(409)
   self.send_header('Content-Length',str(len(raw)));self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(raw)
  do_GET=dispatch;do_POST=dispatch;do_DELETE=dispatch;do_PUT=dispatch
 ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()
