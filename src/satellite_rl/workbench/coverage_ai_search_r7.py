"""R7 publication policy. Every phase requires its exact root source/data gate."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import os,sys
from .storage import read_json,digest,validate_id
from .coverage_ai_search import pin
from . import frozen_round_tasks
from scripts.stage1_ai_search_r7.contract import GROUP,DESIGN_SHA,PORTS,PRIOR,FIRST_PHASES,UNIFORM_RECIPE_SHA,IMPLEMENTED_PHASES,EVALUATION_PHASES,PHASES
from scripts.stage1_ai_search_r7.plan_builder import audit,FIELDS

MODES=IMPLEMENTED_PHASES
CANONICAL_DATASETS=True
RUNTIME_FIELDS=FIELDS[1:]
SERVICE_GUARD_REFS=None

def task_contract(body):return {k:deepcopy(body.get(k)) for k in FIELDS}
def run_prefix(c):return 'r7_'+c['mode']
def plan_references(p):return {'identity':p['identity_ref'],'prerequisite_gate':p['prerequisite_gate_ref']}
def external_references(c):return {'model_'+k:v for k,v in c['model_inputs'].items()}
def freeze_source(store):
    from .r7_model_freeze import freeze_source
    return freeze_source(store)
def decorate_job(c,p,refs,commit):
    return dict(phase=p['phase'],execution_commit=commit,algorithm='ai_search_r7',
                network='single_set_model',kind='train' if p['phase'] in ('train2000','train6000') else 'prepare' if p['phase'] in ('data_train_dev','labels','train_setup') or p.get('heldout_stage')=='derive' else 'evaluate',
                execution_device='cuda' if p['phase'] in ('train2000','train6000') or (p['phase'] in EVALUATION_PHASES and p.get('heldout_stage')!='derive') else 'cpu',
                methods=c['methods'],model_inputs=c['model_inputs'],model_identity=c['model_identity'])
def verify_metadata(j,p):
    if j['phase']!=p['phase'] or j['execution_commit']!=j['scientific_commit'] or j['algorithm']!='ai_search_r7':raise ValueError('R7 frozen metadata changed')
def expected_results(c):return c['config']['training']['end_update'] if c['mode'] in ('train2000','train6000') else len(c['case_indices'])*4 if c['mode'] in EVALUATION_PHASES and c['config'].get('heldout_stage')!='derive' else len(c['case_indices'])

def model_directory(store,contract,run_id):
    if contract['mode']!='train6000':return store.models_root/run_id
    from .r7_successor_binding import continuation
    refs=contract['model_inputs'];endpoint=read_json(pin(refs['train2000_endpoint']),{})
    return Path(continuation(endpoint,refs,contract['model_identity'],contract['case_indices'],contract['config']['datasets']['train']))

def deployment(store):
    root=store.archive_root.resolve()
    from .r7_resources import lane_from_archive
    port=str(lane_from_archive(root.as_posix()))
    if int(port) not in PORTS or os.environ.get('SATELLITE_RL_SSH_PORT')!=port:raise ValueError('isolated R7 73/74 root required')
    from .coverage import _require_remote_dqn
    _require_remote_dqn();return port

def validate_config(c):
    if (c.get('schema_version'),c.get('research_stage'),c.get('algorithm'),c.get('workers'))!=(16,'coverage','ai_search_r7',1):raise ValueError('R7 single-worker schema16')
    if c.get('resources')!=dict(minimum_free_bytes=8589934592,parent_children_rss_bytes=4294967296):raise ValueError('8GiB/4GiB guards')
    return deepcopy(c)

def validate_task(c,p,port):
    audit(p);validate_config(c['config'])
    if c not in p['tasks'].values() or str(c['executor_port'])!=str(port):raise ValueError('exact original task lane')
    phase=p['phase'];datasets=c['config']['datasets'];refs=c['model_inputs']
    if phase=='data_train_dev':
        if datasets or not {'recipe'}<=set(refs):raise ValueError('first build uses raw recipe only')
        if refs['recipe'].get('sha256')!=UNIFORM_RECIPE_SHA:raise ValueError('exact root596 recipe349370 required')
        from .r7_uniform_data import validate_recipe
        validate_recipe(read_json(pin(refs['recipe']),{}))
    elif p.get('heldout_stage')=='derive':
        if datasets:raise ValueError('sealed build cannot read a scientific dataset')
        from .r7_heldout_data import validate_recipe
        validate_recipe(read_json(pin(refs['heldout_recipe']),{}),phase)
        if refs.get('heldout_unlock_gate')!=p['prerequisite_gate_ref']:raise ValueError('same frozen scientific role unlock gate')
    else:
        role=PHASES[phase]['role'];count=20 if role in ('development','validation') else 40 if role=='test' else 576
        if set(datasets)!={role} or datasets[role].get('role')!=role or datasets[role].get('count')!=count:raise ValueError('one new uniform case per mother')
        data=read_json(pin(datasets[role]),{})
        if data.get('schema')!='R7UniformDatasetV1' or data.get('role')!=role or data.get('count')!=datasets[role]['count'] or len(data.get('cases',[]))!=datasets[role]['count']:raise ValueError('new uniform dataset manifest')
        if 'authority' not in refs or data.get('authority_ref')!=refs['authority']:raise ValueError('root new-case authority required')
        if data.get('fixed_request_bps')!=21600000:raise ValueError('uniform exact integer request')
        if phase in ('validation','test'):
            dg=read_json(pin(refs['heldout_data_gate']),{})
            if dg.get('passed')is not True or dg.get('phase')!=phase+'_data' or dg.get('dataset_ref')!=datasets[role] or dg.get('authority_ref')!=refs['authority'] or dg.get('unlock_gate_ref')!=p['prerequisite_gate_ref']:raise ValueError('root heldout derivation/authority freeze before algorithm read')
    return c['config']

def validate_admission(p,plan_ref,review_ref):
    audit(p);r=read_json(pin(review_ref),{})
    if r.get('passed')is not True or r.get('group')!=GROUP or r.get('scientific_commit')!=p['scientific_commit'] or r.get('plan_sha256')!=plan_ref['sha256'] or r.get('design_sha256')!=DESIGN_SHA or r.get('phase')!=p['phase']:raise ValueError('exact root source/plan/phase admission')
    pin(p['design']);pin(p['identity_ref']);gate=read_json(pin(p['prerequisite_gate_ref']),{})
    if gate.get('passed')is not True or gate.get('phase')!=PRIOR[p['phase']] or gate.get('group')!=GROUP:raise ValueError('previous scientific gate cannot be skipped')
    if p['phase'] not in ('data_train_dev','labels') and gate.get('model_identity')!=p['model_identity']:raise ValueError('one original model identity')
    if p['phase']=='node2000' and gate.get('updates')!=2000:raise ValueError('fixed2000 endpoint')
    for task in p['tasks'].values():
        refs=task['model_inputs']
        from .r7_deadline import require
        port=str(task['executor_port']);archive=p.get('archive_roots_by_port',{}).get(port)
        if not archive:raise RuntimeError('root exact deadline-registered archive required before phase admission')
        require(refs,archive,GROUP,p['phase'],plan_ref)
        if p['phase'] not in ('data_train_dev','labels'):
            from scripts.stage1_ai_search_r7.contract import MODEL_IDENTITY
            if p['model_identity']!=MODEL_IDENTITY:raise ValueError('exact root reserved single seed51001 model')
        if p['phase']=='test':
            statistics=read_json(pin(refs['test_statistical_plan']),{})
            if statistics.get('schema')!='R7FrozenTestBootstrapV1' or statistics.get('matrix_generated_once_before_test_reads')is not True:raise ValueError('test statistical source/draw matrix frozen before source/data reads')
            from scripts.stage1_ai_search_r7 import test_statistics
            if statistics.get('statistical_source_sha256')!=digest(test_statistics.__file__):raise ValueError('actual pre-frozen test statistics source')
        if p['phase']=='train_setup':
            from .r7_train_binding import INPUT_KEYS
            if not set(INPUT_KEYS)<=set(refs) or refs['labels_gate']!=p['prerequisite_gate_ref']:raise ValueError('complete original setup inputs and exact labels gate')
            from .r7_training_dataset import validate_training_dataset
            validate_training_dataset(refs,task['config']['datasets']['train'],p['model_identity'])
        if p['phase']=='train2000':
            from .r7_train_binding import validate_setup
            setup_ref=refs.get('train_setup_endpoint')
            if setup_ref!=gate.get('train_setup_endpoint_ref'):raise ValueError('root prerequisite is not this exact train_setup endpoint')
            validate_setup(read_json(pin(setup_ref),{}),setup_ref,refs,p['model_identity'],task['executor_port'],task['case_indices'],task['config']['datasets']['train'])
        if p['phase'] in EVALUATION_PHASES or p['phase']=='train6000':
            from .r7_successor_binding import gate_requirements
            gate_requirements(p['phase'],gate,p['model_identity'],refs.get('checkpoint'))
        if p['phase'] in EVALUATION_PHASES and p.get('heldout_stage')!='derive':
            from .r7_successor_binding import evaluation_binding
            evaluation_binding(p['phase'],refs,p['model_identity'])
        if p['phase']=='train6000':
            from .r7_successor_binding import continuation
            continuation(read_json(pin(refs['train2000_endpoint']),{}),refs,p['model_identity'],task['case_indices'],task['config']['datasets']['train'])
            if refs.get('node_gate')!=p['prerequisite_gate_ref']:raise ValueError('actual2000 gate is continuation prerequisite')
    root=Path(__file__).resolve().parents[3]
    if not p['qualification_refs']:raise ValueError('source/interface qualification required')
    if set(p['qualification_refs'])!={'complete_source_interface_bridge'}:raise ValueError('one complete-source bridge preserving original interface qualification required')
    from .r7_source_bridge import validate_bridge
    validate_bridge(read_json(pin(p['qualification_refs']['complete_source_interface_bridge']),{}),root,p['scientific_commit'],DESIGN_SHA,p['phase'])

def resource_admission(store,j):
    from .r7_deadline import require
    require(j['model_inputs'],store.archive_root,j['experiment_id'],j['mode'],j['references']['plan'],dict(path=str(store.job_dir(j['run_id'])/'job.json'),sha256=digest(store.job_dir(j['run_id'])/'job.json')))
    port=deployment(store);r=read_json(store.archive_root.parent/'ops/resource_route.json',{})
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(r.get('observed_utc','1970-01-01T00:00:00+00:00'))).total_seconds()
    if r.get('ready')is not True or r.get('root_accepted')is not True or r.get('group')!=GROUP or str(r.get('ssh_port'))!=port or r.get('scientific_commit')!=j['scientific_commit'] or not -5<=age<=120:raise ValueError('fresh root resource/unique task receipt')
    from .r7_resources import observe
    actual=observe(store.archive_root)
    if actual['foreign_workers'] or actual['gpu'].split(',')[0].strip()!=r['gpu_uuid']:raise ValueError('actual foreign worker/GPU changed')
    return actual

def api(app,method,parts,body):
    policy=sys.modules[__name__]
    if method=='GET' and len(parts)==2 and parts[0]=='dispatch-admission':
        if parts[1] not in MODES or SERVICE_GUARD_REFS is None:raise RuntimeError('actual pinned service deadline hooks required')
        from .r7_deadline import load
        guard,spec=load(SERVICE_GUARD_REFS);guard.require_dispatch_admission(spec,archive_root=app.store.archive_root,experiment_id=GROUP,mode=parts[1])
        return dict(group=GROUP,mode=parts[1],passed=True,archive_root=str(app.store.archive_root),actual_guard_refs=SERVICE_GUARD_REFS,science_calls=0)
    if method=='GET' and parts==['schema']:return dict(group=GROUP,schema_version=16,modes=MODES,labels=576,node_cases=20,node_D5=60,test_cases=40,allowed_ports=PORTS,train_port=20474,all_phases_require_exact_root_gate_and_pause_hooks=True)
    if method=='GET' and parts==['tasks']:
        from .coverage_ai_search import observed_status
        return {'tasks':[{'job':j,'status':observed_status(app.store,j,s)} for j,s in app.store.jobs() if j.get('experiment_id')==GROUP]}
    if method=='POST' and parts==['tasks']:
        from .r7_deadline import require
        require(body['model_inputs'],app.store.archive_root,GROUP,body['mode'],body['plan'])
        return frozen_round_tasks.create(app,body,policy)
    if len(parts)>=2 and parts[0]=='tasks':
        run=validate_id(parts[1]);job=read_json(app.store.job_dir(run)/'job.json',{})
        if job.get('experiment_id')!=GROUP:raise ValueError('foreign R7 group')
        if method=='GET' and len(parts)==2:return app.run_detail(run)
        if method=='POST' and parts[2:]==['resume']:
            from .r7_model_resume import resume
            return resume(app,run,body)
        if method=='POST' and parts[2:]==['stop']:return app.scheduler.request_stop(run)
    raise FileNotFoundError('unknown R7 endpoint')
