"""New root-admitted service: existing shared deadline hooks at every boundary."""
import argparse,os,json,hashlib,inspect
from pathlib import Path
from satellite_rl.workbench import coverage_ai_search_r7 as policy
from satellite_rl.workbench.runtime import Scheduler
from satellite_rl.workbench.r7_deadline import load,install_transition,active_service_guard

def main():
    p=argparse.ArgumentParser();p.add_argument('--guard-module',required=True);p.add_argument('--guard-module-sha256',required=True);p.add_argument('--guard-spec',required=True);p.add_argument('--guard-spec-sha256',required=True);p.add_argument('--timer-admission',required=True);p.add_argument('--timer-admission-sha256',required=True);p.add_argument('--guard-transition');p.add_argument('--guard-transition-sha256');p.add_argument('--guard-transition-admission');p.add_argument('--guard-transition-admission-sha256');args,server_args=p.parse_known_args()
    refs=dict(guard_module_ref=dict(path=args.guard_module,sha256=args.guard_module_sha256),guard_spec_ref=dict(path=args.guard_spec,sha256=args.guard_spec_sha256),root_timer_admission_ref=dict(path=args.timer_admission,sha256=args.timer_admission_sha256))
    if args.guard_transition:
        transition_ref=dict(path=args.guard_transition,sha256=args.guard_transition_sha256);admission_ref=dict(path=args.guard_transition_admission,sha256=args.guard_transition_admission_sha256);install_transition(transition_ref,admission_ref)
        os.environ['R7_ROOT_GUARD_TRANSITION_REFS']=json.dumps(dict(transition_ref=transition_ref,admission_ref=admission_ref))
    original_refs=refs;guard,spec,refs=active_service_guard(refs);policy.SERVICE_GUARD_REFS=refs
    api_receipt=guard.install_policy_hook(policy,spec);scheduler_receipt=guard.install_scheduler_hook(Scheduler,spec)
    from satellite_rl.workbench import r7_runner,server
    worker_receipt=guard.install_worker_preflight(r7_runner,spec)
    from satellite_rl.training.run_monitor import atomic_json
    receipt_root=Path(spec['guard_root'])/'actual_service_hooks';receipt_root.mkdir(parents=True,exist_ok=True)
    actual=dict(schema='R7ActualServiceDeadlineHooksV1',guard_refs=refs,original_guard_refs=original_refs,service_process_identity=guard.process_identity(os.getpid()),
                installers=dict(api=api_receipt,scheduler=scheduler_receipt,worker_preflight=worker_receipt),
                installed_function_source_sha256={k:hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() for k,fn in (('api',policy.api),('queued_launch',Scheduler._launch),('worker_preflight',r7_runner.run_job))},
                wrapper_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fresh_worker_independent_preCUDA_receipt_required=True,source_ref=refs['guard_module_ref'],spec_ref=refs['guard_spec_ref'],root_timer_admission_ref=refs['root_timer_admission_ref'])
    actual['receipt_path']=str(receipt_root/(str(os.getpid())+'.json'));atomic_json(actual['receipt_path'],actual);policy.SERVICE_HOOK_RECEIPT_REF=dict(path=actual['receipt_path'],sha256=hashlib.sha256(Path(actual['receipt_path']).read_bytes()).hexdigest())
    # Each fresh worker also checks its actual pinned guard before CUDA,
    # independently of this service-process installation.
    server.main(server_args)

if __name__=='__main__':main()
