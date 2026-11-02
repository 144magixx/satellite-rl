"""Adapter to the single root-pinned operational deadline guard, never a timer."""
import importlib.util,os,json
from pathlib import Path
from .coverage_ai_search import pin
from .storage import read_json,digest

KEYS=('guard_module_ref','guard_spec_ref','root_timer_admission_ref')
ACTIVE_TRANSITION_REF=None
ACTIVE_TRANSITION_ADMISSION_REF=None

def install_transition(ref,admission_ref=None):
    """Only a new root-admitted operations bootstrap may install this sidecar."""
    global ACTIVE_TRANSITION_REF,ACTIVE_TRANSITION_ADMISSION_REF
    admission_ref=admission_ref or (ACTIVE_TRANSITION_ADMISSION_REF if ACTIVE_TRANSITION_REF==ref else None)
    if not admission_ref:raise RuntimeError('two external transition/admission refs required; no reciprocal content hash')
    transition=read_json(pin(ref),{})
    if transition.get('schema')!='S1AI17ExplicitHumanContinueGuardTransitionV1' or transition.get('passed')is not True or transition.get('explicit_human_continue')is not True or transition.get('actual_guard_adapter_sha256')!=digest(__file__):raise RuntimeError('explicit human continue and root exact operational adapter transition required')
    root=read_json(pin(admission_ref),{})
    if root.get('passed')is not True or root.get('transition_sha256')!=ref['sha256'] or root.get('explicit_human_continue')is not True:raise RuntimeError('root operational continue admission must bind transition')
    prior=read_json(pin(transition['prior_pause_ref']),{})
    if prior.get('user_pause')is not True or prior.get('dispatch_hold')is not True:raise RuntimeError('actual persistent prior pause proof required')
    for source in transition['new_operations_source_refs'].values():pin(source)
    ACTIVE_TRANSITION_REF=ref
    ACTIVE_TRANSITION_ADMISSION_REF=admission_ref
    return transition

def active_service_guard(refs):
    original_guard,original_spec=load(refs)
    if ACTIVE_TRANSITION_REF is None:return original_guard,original_spec,refs
    transition=install_transition(ACTIVE_TRANSITION_REF);port=str(original_spec['ssh_port'])
    old=transition['old_guard_refs_by_port'][port];new=transition['new_guard_refs_by_port'][port]
    if refs not in (old,new):raise RuntimeError('service guard epoch not in exact root explicit-continue transition')
    guard,spec=load(new)
    return guard,spec,new

def load(refs):
    if not set(KEYS)<=set(refs):raise RuntimeError('root scheduled-pause source/spec/admission hooks required')
    source=pin(refs[KEYS[0]]);spec=read_json(pin(refs[KEYS[1]]),{});admit=read_json(pin(refs[KEYS[2]]),{})
    if admit.get('passed')is not True or admit.get('group')!='S1-AI17-R7' or admit.get('guard_source_SHA')!=refs[KEYS[0]]['sha256'] or admit.get('spec_sha256')!=refs[KEYS[1]]['sha256']:raise RuntimeError('root exact existing deadline guard admission')
    module_spec=importlib.util.spec_from_file_location('r7_pinned_user_deadline_'+digest(source),source)
    guard=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(guard);guard.validate(spec)
    return guard,spec

def require(refs,archive_root,experiment_id='S1-AI17-R7',mode=None,plan_ref=None,job_ref=None):
    if ACTIVE_TRANSITION_REF is None and os.environ.get('R7_ROOT_GUARD_TRANSITION_REFS'):
        pair=json.loads(os.environ['R7_ROOT_GUARD_TRANSITION_REFS']);install_transition(pair['transition_ref'],pair['admission_ref'])
    guard,spec=load(refs)
    if ACTIVE_TRANSITION_REF is not None:
        transition=install_transition(ACTIVE_TRANSITION_REF);port=str(spec['ssh_port'])
        if transition['old_guard_refs_by_port'].get(port)!={k:refs[k] for k in KEYS} or not plan_ref or plan_ref['sha256'] not in transition['original_plan_sha256s']:raise RuntimeError('continue transition must bind unchanged original guard/plan')
        if job_ref is not None:
            pin(job_ref)
            if job_ref['sha256'] not in transition['original_job_sha256s']:raise RuntimeError('continue transition original job SHA required')
        elif transition.get('allow_uncreated_task_in_original_plan')is not True:raise RuntimeError('continue cannot create another unregistered task')
        guard,spec=load(transition['new_guard_refs_by_port'][port])
    guard.require_dispatch_admission(spec,archive_root=archive_root,experiment_id=experiment_id,mode=mode)
    if mode is not None:
        entries=guard.registrations(spec)
        if not any(e['archive_root']==str(archive_root) and e['experiment_id']==experiment_id and mode in e['modes'] for e in entries):raise RuntimeError('phase/mode missing from root deadline registration; no dispatch')
    process=guard.process_identity(os.getpid())
    return dict(schema='R7ActualDeadlineBoundaryAdmissionV1',passed=True,archive_root=str(archive_root),experiment_id=experiment_id,mode=mode,
                actual_guard_refs={k:refs[k] for k in KEYS} if ACTIVE_TRANSITION_REF is None else transition['new_guard_refs_by_port'][str(spec['ssh_port'])],
                original_guard_refs={k:refs[k] for k in KEYS},guard_transition_ref=ACTIVE_TRANSITION_REF,guard_transition_admission_ref=ACTIVE_TRANSITION_ADMISSION_REF,process_identity=process,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip() if os.name!='nt' else None,
                adapter_source_sha256=digest(__file__),scientific_model_or_geometry_calls=0)

def stop_boundary(job,directory):
    """Call before supply cases/updates and after a durable optimizer checkpoint."""
    from pathlib import Path
    if (Path(directory)/'stop.request').exists():raise RuntimeError('USER_PAUSE cooperative stop; explicit root same-ID recovery required')
    return require(job['model_inputs'],job['archive_root'],job['experiment_id'],job['mode'],job['references']['plan'],dict(path=str(Path(directory)/'job.json'),sha256=digest(Path(directory)/'job.json')))
