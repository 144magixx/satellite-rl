"""New root-admitted service: existing shared deadline hooks at every boundary."""
import argparse,os,json
from satellite_rl.workbench import coverage_ai_search_r7 as policy
from satellite_rl.workbench.runtime import Scheduler
from satellite_rl.workbench.r7_deadline import load,install_transition

def main():
    p=argparse.ArgumentParser();p.add_argument('--guard-module',required=True);p.add_argument('--guard-module-sha256',required=True);p.add_argument('--guard-spec',required=True);p.add_argument('--guard-spec-sha256',required=True);p.add_argument('--timer-admission',required=True);p.add_argument('--timer-admission-sha256',required=True);p.add_argument('--guard-transition');p.add_argument('--guard-transition-sha256');args,server_args=p.parse_known_args()
    refs=dict(guard_module_ref=dict(path=args.guard_module,sha256=args.guard_module_sha256),guard_spec_ref=dict(path=args.guard_spec,sha256=args.guard_spec_sha256),root_timer_admission_ref=dict(path=args.timer_admission,sha256=args.timer_admission_sha256))
    if args.guard_transition:
        transition_ref=dict(path=args.guard_transition,sha256=args.guard_transition_sha256);install_transition(transition_ref)
        os.environ['R7_ROOT_GUARD_TRANSITION_REF']=json.dumps(transition_ref)
    policy.SERVICE_GUARD_REFS=refs
    guard,spec=load(refs);guard.install_policy_hook(policy,spec);guard.install_scheduler_hook(Scheduler,spec)
    from satellite_rl.workbench import r7_runner,server
    guard.install_worker_preflight(r7_runner,spec)
    # Each fresh worker also checks its actual pinned guard before CUDA,
    # independently of this service-process installation.
    server.main(server_args)

if __name__=='__main__':main()
