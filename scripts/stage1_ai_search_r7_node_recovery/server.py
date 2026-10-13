"""Restricted9018: GET one original task; one durable same-ID direct spawn, no scheduler."""
import argparse,json,os,subprocess,sys
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from .contracts import RUN_ID,SCIENCE,atomic,pin,read,validate,fresh_resource,process_identity

RESUME_PATH='/api/ai-search-r7/tasks/'+RUN_ID+'/resume'
TASK_PATH='/api/ai-search-r7/tasks/'+RUN_ID
def allowed(method,path):return method=='GET' and path in ('/health',TASK_PATH) or method=='POST' and path==RESUME_PATH

class RecoveryEndpoint:
    def __init__(self,contract_ref,root_review_ref):
        self.contract_ref=contract_ref;self.review_ref=root_review_ref
        self.contract,self.job,self.config=validate(contract_ref,root_review_ref)
        self.directory=Path(self.contract['original_job_ref']['path']).parent
        self.operations=Path(self.contract['operations_directory'])
    def get(self,path):
        if not allowed('GET',path):raise ValueError('sole task GET/health only')
        if path=='/health':return dict(passed=True,run_id=RUN_ID,restricted=True,no_generic_create=True,no_scheduler=True,actual_service=process_identity(os.getpid()),contract_ref=self.contract_ref,science_source=SCIENCE)
        status=json.loads((self.directory/'status.json').read_bytes())
        return dict(run_id=RUN_ID,status=status,operations_launch=json.loads((self.operations/'actual_worker_launch.json').read_bytes()) if (self.operations/'actual_worker_launch.json').exists() else None,original_failure_ref=self.contract['original_failure_ref'])
    def resume(self,path,body,popen=subprocess.Popen):
        if not allowed('POST',path) or body!=dict(contract_ref=self.contract_ref,root_review_ref=self.review_ref):raise ValueError('one exact root same-ID resume body')
        from satellite_rl.workbench.geoload_runner import exclusive_lock
        with exclusive_lock(self.operations/'unique_resume_owner.lock'):
            intent=self.operations/'resume_intent.json'
            if intent.exists():return dict(run_id=RUN_ID,operations_intent=json.loads(intent.read_bytes()),POST_replayed=False)
            contract,job,config=validate(self.contract_ref,self.review_ref);fresh_resource(contract)
            from satellite_rl.workbench.r7_deadline import stop_boundary
            stop_boundary(job,self.directory)
            # Never queue: the idle old9016 scheduler is required gone by contract.
            # Probe existing science locks before spawn; fresh worker acquires both itself.
            with exclusive_lock(self.directory/'worker.lock'),exclusive_lock(Path(job['archive_root']).parent/'ops/compute.lock'):
                atomic(intent,dict(schema='R7NodeSameIDRecoveryDurableIntentV1',run_id=RUN_ID,phase='spawn_pending',unknown_outcome_never_replayed=True,contract_ref=self.contract_ref,root_review_ref=self.review_ref))
                env=dict(os.environ);env['PYTHONPATH']=str(Path(job['source_root'])/'src')+os.pathsep+str(Path(job['source_root']))+os.pathsep+str(Path(__file__).resolve().parents[2]);env['SATELLITE_RL_SSH_PORT']='20474';env['PYTHONDONTWRITEBYTECODE']='1'
                command=[sys.executable,'-B','-u','-m','scripts.stage1_ai_search_r7_node_recovery.bootstrap','--contract',self.contract_ref['path'],'--contract-sha256',self.contract_ref['sha256'],'--root-review',self.review_ref['path'],'--root-review-sha256',self.review_ref['sha256']]
                with (self.operations/'worker.stdout.log').open('ab') as out,(self.operations/'worker.stderr.log').open('ab') as err:
                    child=popen(command,cwd=job['source_root'],env=env,stdout=out,stderr=err,start_new_session=True)
                launch=dict(actual_worker=process_identity(child.pid),command=command,original_scientific_source=job['source_root'],original_source_manifest_ref=contract['original_source_manifest_ref'],operations_source_files=contract['operations_source_files'],contract_ref=self.contract_ref,original_launch_json_unchanged=True,no_queued_window=True)
                atomic(self.operations/'actual_worker_launch.json',launch)
                atomic(intent,dict(run_id=RUN_ID,phase='spawned',actual_worker=launch['actual_worker'],unknown_outcome_never_replayed=True,contract_ref=self.contract_ref,root_review_ref=self.review_ref))
            atomic(self.operations/'parent_locks_released.json',dict(locks_released=True,actual_worker=launch['actual_worker'],contract_ref=self.contract_ref))
            return dict(run_id=RUN_ID,actual_worker=launch['actual_worker'],same_ID=True,POST_replayed=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--contract-sha256',required=True);p.add_argument('--root-review',required=True);p.add_argument('--root-review-sha256',required=True);p.add_argument('--port',type=int,default=9018);a=p.parse_args()
    if a.port!=9018:raise ValueError('restricted recovery only port9018')
    endpoint=RecoveryEndpoint(dict(path=a.contract,sha256=a.contract_sha256),dict(path=a.root_review,sha256=a.root_review_sha256))
    source=Path(endpoint.job['source_root']);sys.path[:0]=[str(source/'src'),str(source)]
    from satellite_rl.workbench.r7_deadline import stop_boundary
    actual_guard=stop_boundary(endpoint.job,endpoint.directory)
    class Handler(BaseHTTPRequestHandler):
        def dispatch(self):
            try:
                if not allowed(self.command,self.path):raise ValueError('restricted method/path')
                if self.command=='GET':result=endpoint.get(self.path)
                else:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<=8192:raise ValueError('small exact typed resume body required')
                    result=endpoint.resume(self.path,json.loads(self.rfile.read(length)))
                raw=json.dumps(result).encode();self.send_response(200)
            except Exception as error:raw=json.dumps(dict(error=str(error),automatic_retry=False)).encode();self.send_response(409)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        do_GET=dispatch;do_POST=dispatch;do_DELETE=dispatch;do_PUT=dispatch;do_PATCH=dispatch
    atomic(endpoint.operations/'actual_restricted_service.json',dict(actual_service=process_identity(os.getpid()),port=9018,contract_ref=endpoint.contract_ref,original9016_required_gone=True,no_generic_scheduler=True,operations_source_files=endpoint.contract['operations_source_files'],actual_startup_guard_admission=actual_guard,
        exact_POST_guard_boundary_installed=True,worker_uses_original_preCUDA_and_each_case_method_boundaries=True))
    ThreadingHTTPServer(('127.0.0.1',9018),Handler).serve_forever()
if __name__=='__main__':main()
