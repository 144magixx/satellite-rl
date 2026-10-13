"""74-only new operational API epoch preserving original Store/task ownership."""
import argparse,sys,json,importlib.util
from pathlib import Path

def install_hooks(app,contract_ref,review_ref,ops):
    from satellite_rl.workbench import coverage_ai_search_r7 as policy,runtime
    from satellite_rl.workbench.coverage_ai_search import pin
    metadata=dict(contract_ref=contract_ref,root_review_ref=review_ref)
    original_api=policy.api;original_popen=runtime.subprocess.Popen
    def api(application,method,parts,body):
        if method=='POST' and parts==['tasks',ops.RUN,'resume']:
            if body!=metadata:raise ValueError('one exact root-authorized recovery request required')
            return ops.resume(application,body)
        if method=='POST':raise ValueError('this operational server only accepts the original74 same-ID recovery')
        return original_api(application,method,parts,body)
    policy.api=api
    def popen(command,*args,**kwargs):
        tokens=list(command)
        if 'satellite_rl.workbench.runner' in tokens and '--job-dir' in tokens:
            directory=Path(tokens[tokens.index('--job-dir')+1])
            if directory.name!=ops.RUN:raise ValueError('operational server refuses another science task')
            job=json.loads((directory/'job.json').read_text());contract=ops.validate(job,metadata)
            sidecar=directory/'operations_recovery.json'
            if not sidecar.is_file() or json.loads(sidecar.read_text())!=metadata:raise ValueError('same-ID operations sidecar missing')
            replacement=[tokens[0],'-B','-u',str(pin(contract['operations_source_refs']['bootstrap'])),'--mode','worker','--job-dir',str(directory),'--contract',contract_ref['path'],'--contract-sha256',contract_ref['sha256'],'--review',review_ref['path'],'--review-sha256',review_ref['sha256']]
            command[:]=replacement
        return original_popen(command,*args,**kwargs)
    runtime.subprocess.Popen=popen

def main():
    p=argparse.ArgumentParser();p.add_argument('--original-source',required=True);p.add_argument('--archive-root',required=True);p.add_argument('--port',type=int,default=9015);p.add_argument('--contract',required=True);p.add_argument('--contract-sha256',required=True);p.add_argument('--review',required=True);p.add_argument('--review-sha256',required=True);args=p.parse_args()
    if args.port!=9015 or Path(args.archive_root).as_posix()!='/home/u2024110224/first_stage_satellite_rl/s1_ai17/20474/round7/archive':raise ValueError('original74 API/archive only')
    source=Path(args.original_source);sys.path[:0]=[str(source/'src'),str(source)]
    from satellite_rl.workbench.coverage_ai_search import pin
    cref=dict(path=args.contract,sha256=args.contract_sha256);rref=dict(path=args.review,sha256=args.review_sha256);contract=json.loads(pin(cref).read_text())
    if Path(__file__).resolve()!=pin(contract['operations_source_refs']['server']).resolve():raise ValueError('actual operations server source/path differs')
    path=pin(contract['operations_source_refs']['module']);name='satellite_rl.workbench.r7_recovery_ops';spec=importlib.util.spec_from_file_location(name,path);ops=importlib.util.module_from_spec(spec);sys.modules[name]=ops;spec.loader.exec_module(ops)
    from satellite_rl.workbench.server import Application,WorkbenchServer
    app=Application(Path(args.archive_root),python_executable=sys.executable,max_workers=1)
    install_hooks(app,cref,rref,ops)
    server=WorkbenchServer(('127.0.0.1',args.port),app);app.scheduler.start()
    try:server.serve_forever(poll_interval=.5)
    finally:app.scheduler.stop_scheduler();server.server_close()

if __name__=='__main__':main()
