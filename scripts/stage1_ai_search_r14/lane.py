"""One persistent owned R14 lane, direct cold children, durable readable status.

No HTTP layer, observer, automatic retry or service renewal. A root-pinned
finite workbench registration is the only input. Child clocks include import,
read/hash, numerical checks, save and direct wait/reap.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import shutil
from .runtime import BudgetLedger, PauseGuard, atomic_bytes, json_bytes, exclusive, require_execution,control_scope
from .pipeline import OnceIntent
from .runtime import development_schedule


def validate_task_order(tasks):
    """Whole dev mother/repeats/three arms on this single lane, frozen Latin."""
    ids=[t['task_id'] for t in tasks]
    if len(set(ids))!=len(ids):raise ValueError('duplicate logical task ID')
    dev=[t for t in tasks if t['task_id'].startswith('dev:')]
    expected_all=[]
    for index in sorted({t['recipe_index'] for t in dev}):
        m=index-576
        if not 0<=m<20:raise ValueError('fixed known development mother')
        expected=[f"dev:{m}:{r['repeat']}:{arm}" for r in development_schedule() if r['mother']==m for arm in r['arms_serial']]
        actual=[t['task_id'] for t in dev if t['recipe_index']==index]
        if actual!=expected:raise ValueError('all9 cold comparisons per mother on one lane in exact Latin order')
        expected_all.extend(expected)
    if [t['task_id'] for t in dev]!=expected_all:raise ValueError('assigned development mothers ascending and contiguous')
    return True

def initial_storage_check(domains):
    seen=set()
    for domain in domains:
        path=Path(domain['path']);device=path.stat().st_dev
        if device!=domain['stat_dev'] or domain['physical_domain_id'] in seen:raise ValueError('actual aggregated physical domain identity')
        seen.add(domain['physical_domain_id'])
        free=shutil.disk_usage(path).free
        if (free<12*1024**3 or domain['reservation_bytes']!=6*1024**3
            or domain['maximum_growth_bytes']!=4966055936
            or domain['running_safety_floor_bytes']!=domain['initial_free_bytes']-domain['reservation_bytes']
            or free<domain['running_safety_floor_bytes']):raise ValueError('fresh initial free/reserve and distinct running floor')
    if not domains:raise ValueError('physical storage domain qualification required')


def pinned(ref, limit=32*1024**2):
    if type(ref.get('bytes')) is not int or not 0 < ref['bytes'] <= limit:
        raise ValueError('exact bounded reference')
    with Path(ref['path']).open('rb') as stream:
        raw = stream.read(ref['bytes']+1)
    if len(raw) != ref['bytes'] or hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('pinned file bytes changed')
    return raw


def identity(pid):
    import psutil
    process = psutil.Process(pid)
    boot = Path('/proc/sys/kernel/random/boot_id')
    return {'pid': pid, 'process_start_identity': process.create_time(),
            'boot_id': boot.read_text().strip() if boot.exists() else None}


def validate_epoch(plan):
    epoch = json.loads(pinned(plan['runtime_epoch_ref'], 65536))
    authority = json.loads(pinned(epoch['human_resume_ref'], 65536))
    if (authority.get('schema') != 'S1AI17ExplicitHumanResumeAuthorityV1'
            or authority.get('user_instruction_verbatim') != '\u7ee7\u7eed'
            or authority.get('observed_after_0700_pause') is not True
            or authority.get('root_first_verified_utc') != '2026-10-15T00:58:11Z'
            or authority.get('future_pause_always_wins') is not True
            or epoch.get('group') != 'S1-AI17-R14'
            or epoch.get('boot_id') != identity(os.getpid())['boot_id']
            or time.time() >= epoch['expires_unix']):
        raise ValueError('exact post-pause human continuation and fresh finite boot epoch')
    required = {plan['own_stop_marker'], plan['global_stop_marker']}
    if not required <= set(epoch['own_paths'] + epoch['global_paths']):
        raise ValueError('new R14 own/global markers mandatory')
    if not all('r14' in str(p).lower() for p in required):
        raise ValueError('new isolated R14 markers, never old epoch aliases')
    return epoch


def terminate_owned(child, registered, *, hard=False,timeout=1.):
    """Only descendants captured from the directly owned child are touched."""
    import psutil
    try:
        parent = psutil.Process(child.pid)
        if parent.create_time() != registered['process_start_identity']:
            raise RuntimeError('PID birth mismatch; refuse unrelated process')
        descendants = parent.children(recursive=True)
        targets = descendants + [parent]
    except psutil.NoSuchProcess:
        targets = []
    for item in registered.get('descendants', []):
        try:
            process=psutil.Process(item['pid'])
            if process.create_time()==item['process_start_identity']:
                targets.append(process)
        except psutil.NoSuchProcess:
            pass
    for process in targets:
        try:
            process.kill() if hard else process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, live = psutil.wait_procs(targets, timeout=timeout)
    if hard and live:
        raise RuntimeError('owned child tree remains live after hard kill')
    return len(live)


def run_child(command, task, root, guard, *, popen=subprocess.Popen,
              clock=time.monotonic, sleeper=time.sleep, identify=identity,
              terminate=terminate_owned, sample=None,started=None):
    """Direct finite parent; injected process funcs permit pure fake checks."""
    started = clock() if started is None else started
    group_deadline=getattr(guard,'started',0.)+getattr(guard,'seconds',float('inf'))
    group_stop=getattr(guard,'started',0.)+.9*getattr(guard,'seconds',float('inf'))
    deadline=min(started+task['outer_seconds'],group_deadline)
    stop_new=min(started+.9*task['outer_seconds'],group_stop)
    def left():return max(0.,deadline-clock())
    def cleanup(child,registered):
        if registered is None:
            if child.poll() is None:child.kill()
            child.wait(timeout=left())
            return None
        kwargs={'hard':True}
        if terminate is terminate_owned:kwargs['timeout']=left()
        live=terminate(child,registered,**kwargs)
        child.wait(timeout=left())
        return live
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    atomic_bytes(root/'intent.json', json_bytes({'command': command, 'task': task, 'parent': identify(os.getpid())}))
    child = None; reason = None; registered = None
    try:
        guard.check()
        if clock()>=stop_new or left()<=min(2.,.1*task['outer_seconds']):
            raise TimeoutError('request preparation consumed stop-new-work deadline before Popen')
        with (root/'stdout.log').open('xb') as stdout, (root/'stderr.log').open('xb') as stderr:
            environment={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
            environment['R14_REQUEST_DEADLINE_MONOTONIC']=str(deadline)
            environment['R14_REQUEST_STOP_NEW_MONOTONIC']=str(stop_new)
            environment['R14_NESTED_AUDIT_SECONDS']=str(task.get('additional_stage_reservations',{}).get('audit',0))
            if task.get('source_root'):
                environment['PYTHONPATH']=os.pathsep.join([str(Path(task['source_root'])/'src'),task['source_root'],*task.get('private_dependency_paths',[])])
            child = popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                          start_new_session=(os.name != 'nt'),env=environment,cwd=task.get('source_root'))
            registered = identify(child.pid)
            atomic_bytes(root/'child.json', json_bytes(registered))
            stop_sent = False
            while child.poll() is None:
                elapsed = clock()-started
                if (root/'stdout.log').stat().st_size+(root/'stderr.log').stat().st_size>32768:
                    reason='owned_debug_log_cap'
                    cleanup(child,registered)
                    break
                for domain in task.get('storage_domains',[]):
                    free=shutil.disk_usage(domain['path']).free
                    if free<domain['running_safety_floor_bytes']:reason='actual_storage_safety_floor'
                if sample is not None:
                    sample(child, registered)
                elif identify is identity:
                    import psutil
                    try:
                        processes=[psutil.Process(child.pid)];processes+=processes[0].children(recursive=True)
                        known={p['pid']:p for p in registered.get('descendants',[])}
                        for process in processes[1:]:known[process.pid]=identity(process.pid)
                        registered['descendants']=list(known.values())
                        rss=sum(p.memory_info().rss for p in processes)
                        registered['peak_tree_rss_bytes']=max(registered.get('peak_tree_rss_bytes',0),rss)
                        if rss>2*1024**3:reason='owned_tree_memory_cap'
                    except psutil.NoSuchProcess:
                        pass
                try:
                    guard.check()
                except RuntimeError as error:
                    reason = str(error)
                if clock()>=stop_new or reason:
                    if not stop_sent:
                        atomic_bytes(root/'stop.request', json_bytes({'reason': reason or 'save_reap_reserve'}))
                        stop_sent = True
                if clock()>=deadline-min(2.,.1*task['outer_seconds']):
                    reason = reason or 'outer_hard_deadline'
                    cleanup(child,registered)
                    break
                sleeper(.02)
            raw_code = child.wait(timeout=left())
            live = cleanup(child,registered)
        facts = {**registered, 'raw_returncode': raw_code, 'reaped': True,
                 'descendants_live': live, 'elapsed_seconds': clock()-started,
                 'stop_reason': reason, 'actual_parent_start_to_reap': True}
        facts.update(task_id=task.get('task_id'),kind=task.get('kind'),request_ref=task.get('request_ref'),
                     source_root=task.get('source_root'))
        facts['outer_seconds']=deadline-started
        facts['effective_hard_deadline_monotonic']=deadline;facts['effective_stop_new_monotonic']=stop_new
        atomic_bytes(root/'exit.json', json_bytes(facts))
        return facts
    except BaseException as error:
        cleanup_error=None
        if child is not None:
            try:cleanup(child,registered)
            except BaseException as cleanup_failure:cleanup_error=repr(cleanup_failure)
        atomic_bytes(root/'failure.json', json_bytes({'error': repr(error), 'elapsed_seconds': clock()-started,
                     'child_entered': child is not None,'identity_known':registered is not None,
                     'cleanup_error':cleanup_error, 'unknown_cost_not_retryable': True}))
        raise


def run_lane(plan_ref):
    lane_started=time.monotonic()
    plan = json.loads(pinned(plan_ref)); require_execution(plan)
    validate_task_order(plan['tasks'])
    review = json.loads(pinned(plan['root_admission_ref'], 131072))
    plan_binding = hashlib.sha256(json_bytes({k:v for k,v in plan.items() if k!='root_admission_ref'})).hexdigest()
    if review.get('accepted') is not True or review.get('plan_binding_sha256') != plan_binding:
        raise ValueError('independent exact final plan acceptance')
    for ref in plan['source_refs']:
        pinned(ref)
    from .bindings import verify_loaded_source,bind_task,verify_private_dependencies
    verify_loaded_source(plan['source_root'],plan['source_refs'])
    for ref in plan['private_dependency_refs']:pinned(ref)
    verify_private_dependencies(plan)
    pinned(plan['interpreter_ref'])
    if Path(sys.executable).resolve()!=Path(plan['interpreter']).resolve():
        raise ValueError('lane must use exact admitted DQN interpreter')
    epoch = validate_epoch(plan)
    initial_storage_check(plan['storage_domains'])
    root = Path(plan['control_root']); root.mkdir(parents=True, exist_ok=True)
    guard = PauseGuard(epoch['own_paths'], epoch['global_paths'], epoch['historical_exact_refs'],
                       seconds=min(plan['outer_seconds'], epoch['expires_unix']-time.time()))
    budget = BudgetLedger(plan['shared_budget_root']); once = OnceIntent(plan['shared_dispatch_root'])
    ops_id=plan['run_id']+'_initialization'
    elapsed=time.monotonic()-lane_started
    budget.reserve(ops_id,'ops',max(elapsed,.001))
    budget.finish_inline_ops(ops_id,elapsed)
    with exclusive(root/'.lane.lock'):
        registration = root/'workbench_registration.json'
        if registration.exists():
            raise ValueError('existing lane registration: status inspection only, never replay')
        atomic_bytes(registration, json_bytes({'schema': 'R14OwnedWorkbenchTaskV1', 'run_id': plan['run_id'],
                     'owner': identity(os.getpid()), 'plan_ref': plan_ref, 'state': 'running',
                     'status_path': str(root/'status.json'), 'isolated_group': 'S1-AI17-R14'}))
        for task in plan['tasks']:
            task_started=time.monotonic()
            task={**task,'source_root':plan['source_root'],'private_dependency_paths':plan['private_dependency_paths'],
                  'storage_domains':plan['storage_domains']}
            if task['stage']=='fit_baseline':
                task['outer_seconds']=min(task['outer_seconds'],budget.available('fit_baseline'))
                if task['outer_seconds']<=2:raise RuntimeError('fit/veto stage exhausted including save/reap reserve')
            def preparation_check():
                guard.check()
                if time.monotonic()-task_started>=.9*task['outer_seconds']:raise TimeoutError('root request stop-new boundary')
            with control_scope(preparation_check):
                guard.check(); validate_epoch(plan)
                task_id = task['task_id'].replace(':', '_')
                once.claim(task_id, task, plan)
                budget.reserve(task_id, task['stage'], task['outer_seconds'],task.get('additional_stage_reservations'))
                request = json.loads(pinned(task['request_ref']))
                bind_task(task,request)
                from .stage_admission import verify_stage
                if request['root_admission_ref']!=plan['child_stage_admission_ref']:raise ValueError('stable child stage differs from final plan approval')
                verify_stage(request)
                if request['own_child_stop_marker']!=str(root/task_id/'stop.request'):
                    raise ValueError('exact child request identity')
                from .dispatcher import command_for_task,check_dependencies,collect_task
                check_dependencies(request)
                command=command_for_task(plan,task)
            facts = run_child(command, task, root/task_id, guard,started=task_started)
            collect_started=time.monotonic()
            collected,_=collect_task(task,request,facts)
            allocations=None
            if task['kind'] in ('engineering','B','teacher'):
                from .bindings import read_json
                audit_seconds=(read_json(collected['output_ref'])['nested_independent_audit_elapsed']
                    if task['kind']=='engineering' else collected['nested_independent_audit_elapsed'])
                allocations={task['stage']:facts['elapsed_seconds'],'audit':audit_seconds}
            budget.finish(task_id,facts['elapsed_seconds'],facts,allocations)
            atomic_bytes(root/'status.json', json_bytes({'run_id': plan['run_id'], 'last_task': task_id, 'exit': facts}))
            collect_elapsed=time.monotonic()-collect_started
            budget.reserve(task_id+'_collect','ops',max(collect_elapsed,.001));budget.finish_inline_ops(task_id+'_collect',collect_elapsed)
            if facts['raw_returncode'] != 0 or facts['stop_reason'] or facts['elapsed_seconds'] > task['outer_seconds']:
                raise RuntimeError('finite lane closes on first failed/unknown request; no retry')
        atomic_bytes(root/'complete.json', json_bytes({'run_id': plan['run_id'], 'state': 'complete', 'owner': identity(os.getpid())}))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--sha256',required=True);p.add_argument('--bytes',type=int,required=True)
    a=p.parse_args();run_lane({'path':a.plan,'sha256':a.sha256,'bytes':a.bytes})
