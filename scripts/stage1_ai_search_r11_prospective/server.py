"""One exact new Gh task, one start intent; no queue/retry/control reexecution."""
import argparse,ctypes,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
from scripts.stage1_ai_search_r10 import contracts as c
from scripts.stage1_ai_search_r10.control import exclusive_lock
from .contracts import validate,check_pause,fresh_route
from .reaper import wait_one,boottime
class Endpoint:
 def __init__(self,plan_ref,review_ref):
  self.plan_ref=plan_ref;self.review_ref=review_ref;self.plan=validate(plan_ref,review_ref);self.root=Path(self.plan['archive_root']);self.root.mkdir(parents=True,exist_ok=True);self.path='/api/ai-search-r11/tasks/'+self.plan['run_id'];check_pause(self.plan,'Gh_service_startup')
 def get(self,path):
  if path=='/health':return dict(group='S1-AI17-R11',stage='new_saved_common_Gh_only',plan_ref=self.plan_ref,actual_service=c.identity(os.getpid()),no_generic_scheduler=True,exact_single_start_only=True)
  if path!=self.path:raise ValueError('sole exact run GET')
  return dict(run_id=self.plan['run_id'],plan_ref=self.plan_ref,status=json.loads((self.root/'status.json').read_bytes()) if (self.root/'status.json').exists() else dict(stage='not_submitted'))
 def post(self,path,body,popen=subprocess.Popen):
  expected=dict(plan_ref=self.plan_ref,root_review_ref=self.review_ref,request_id=self.plan['request_id'])
  if body!=expected:raise ValueError('sole exact root plan/review/request pair')
  if path==self.path+'/stop':c.immutable(self.root/'stop.request',dict(user_pause=True,run_id=self.plan['run_id'],actual_service=c.identity(os.getpid())));return dict(stop_requested=True)
  if path!=self.path+'/start':raise ValueError('single Gh start endpoint; no resume/retrain/queue')
  p=validate(self.plan_ref,self.review_ref);check_pause(p,'API_before_Gh_start');fresh_route(p)
  if threading.get_native_id()!=os.getpid():raise RuntimeError('worker spawn requires persistent service main native thread')
  with exclusive_lock(self.root/'submission.lock'):
   if (self.root/'submission_intent.json').exists():raise ValueError('submitted/unknown start intent; never replay')
   check_pause(p,'before_Gh_intent');fresh_route(p);intent=c.immutable(self.root/'submission_intent.json',dict(request=expected,actual_service=c.identity(os.getpid()),design_ref=p['design_ref'],unknown_never_replayed=True))
   command=[sys.executable,'-B','-u','-m','scripts.stage1_ai_search_r11_prospective.worker','--plan',self.plan_ref['path'],'--plan-sha256',self.plan_ref['sha256'],'--review',self.review_ref['path'],'--review-sha256',self.review_ref['sha256']]
   parent=c.identity(os.getpid());worker_started=boottime()
   def bound_worker():
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGKILL,0,0,0)!=0 or os.getppid()!=parent['pid']:os._exit(126)
   if boottime()>=float(os.environ['R11_LANE_DEADLINE_BOOTTIME']):raise RuntimeError('lane deadline before worker spawn')
   with (self.root/'worker.stdout.log').open('xb') as out,(self.root/'worker.stderr.log').open('xb') as err:child=popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,preexec_fn=bound_worker,env=dict(os.environ,R11_SERVICE_PID=str(os.getpid()),PYTHONDONTWRITEBYTECODE='1'))
   owner=c.identity(child.pid)
   if owner is None:raise RuntimeError('worker birth unknown; preserve intent even if already exited')
   def reap_worker():
    try:
     while True:
      record=wait_one(child,owner)
      if record is not None:
       c.immutable(self.root/'worker_reap.json',dict(record,actual_service=parent,plan_ref=self.plan_ref,worker_start_clock_boottime_seconds=worker_started,complete_worker_exit_upper_seconds=record['wait4_observed_clock_boottime_seconds']-worker_started));return
      time.sleep(.02)
    except BaseException as e:c.immutable(self.root/'worker_reap_failure.json',dict(error=type(e).__name__+': '+str(e),actual_worker=owner,wait_status_unknown=True,automatic_retry=False))
   self.worker_handle=child;self.worker_reaper=threading.Thread(target=reap_worker,daemon=True);self.worker_reaper.start()
   launch=c.immutable(self.root/'actual_worker_launch.json',dict(actual_worker=owner,command=command,intent_ref=intent,plan_ref=self.plan_ref,design_ref=p['design_ref']));return dict(launch_ref=launch,actual_worker=owner,POST_replayed=False,exact_run_id=p['run_id'])
def main():
 a=argparse.ArgumentParser();a.add_argument('--plan',required=True);a.add_argument('--plan-sha256',required=True);a.add_argument('--review',required=True);a.add_argument('--review-sha256',required=True);a.add_argument('--port',required=True,type=int);v=a.parse_args()
 if v.port not in (9050,9051):raise ValueError('fresh same-group9050/9051 only')
 if threading.get_native_id()!=os.getpid():raise RuntimeError('HTTPServer must serve on persistent main native thread')
 endpoint=Endpoint(dict(path=v.plan,sha256=v.plan_sha256),dict(path=v.review,sha256=v.review_sha256))
 if v.port!=endpoint.plan['workbench_port']:raise ValueError('exact root workbench port')
 with exclusive_lock(endpoint.root/'service.lock'):
  c.immutable(endpoint.root/'service_epoch_intent.json',dict(actual_service=c.identity(os.getpid()),plan_ref=endpoint.plan_ref,port=v.port,design_ref=endpoint.plan['design_ref']))
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
  HTTPServer(('127.0.0.1',v.port),Handler).serve_forever()
if __name__=='__main__':main()
