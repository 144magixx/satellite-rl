"""R7 publication policy. Every phase requires its exact root source/data gate."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import os,sys
from .storage import read_json,digest,validate_id
from .coverage_ai_search import pin
from . import frozen_round_tasks
from scripts.stage1_ai_search_r7.contract import GROUP,DESIGN_SHA,PORTS,PRIOR,FIRST_PHASES
from scripts.stage1_ai_search_r7.plan_builder import audit,FIELDS

MODES=FIRST_PHASES
CANONICAL_DATASETS=True
RUNTIME_FIELDS=FIELDS[1:]

def task_contract(body):return {k:deepcopy(body.get(k)) for k in FIELDS}
def run_prefix(c):return 'r7_'+c['mode']
def plan_references(p):return {'identity':p['identity_ref'],'prerequisite_gate':p['prerequisite_gate_ref']}
def external_references(c):return {'model_'+k:v for k,v in c['model_inputs'].items()}
def freeze_source(store):
    from .r7_model_freeze import freeze_source
    return freeze_source(store)
def decorate_job(c,p,refs,commit):
    return dict(phase=p['phase'],execution_commit=commit,algorithm='ai_search_r7',
                network='single_set_model',kind='train' if p['phase']=='train2000' else 'prepare' if p['phase'] in ('data_train_dev','labels','train_setup') else 'evaluate',
                execution_device='cuda' if p['phase'] in ('train2000','node2000') else 'cpu',
                methods=c['methods'],model_inputs=c['model_inputs'],model_identity=c['model_identity'])
def verify_metadata(j,p):
    if j['phase']!=p['phase'] or j['execution_commit']!=j['scientific_commit'] or j['algorithm']!='ai_search_r7':raise ValueError('R7 frozen metadata changed')
def expected_results(c):return 2000 if c['mode']=='train2000' else len(c['case_indices'])*4 if c['mode']=='node2000' else len(c['case_indices'])

def deployment(store):
    root=store.archive_root.resolve();port=root.parent.parent.name
    expected=Path('/home/u2024110224/first_stage_satellite_rl/s1_ai17')/port/'round7/archive'
    if root!=expected or int(port) not in PORTS or os.environ.get('SATELLITE_RL_SSH_PORT')!=port:raise ValueError('isolated R7 73/74 root required')
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
        from .r7_uniform_data import validate_recipe
        validate_recipe(read_json(pin(refs['recipe']),{}))
    else:
        role='development' if phase=='node2000' else 'train'
        if set(datasets)!={role} or datasets[role].get('role')!=role or datasets[role].get('count')!=(20 if role=='development' else 576):raise ValueError('one new uniform case per mother')
        data=read_json(pin(datasets[role]),{})
        if data.get('schema')!='R7UniformDatasetV1' or data.get('role')!=role or data.get('count')!=datasets[role]['count'] or len(data.get('cases',[]))!=datasets[role]['count']:raise ValueError('new uniform dataset manifest')
        if 'authority' not in refs or data.get('authority_ref')!=refs['authority']:raise ValueError('root new-case authority required')
        if data.get('fixed_request_bps')!=21600000:raise ValueError('uniform exact integer request')
    return c['config']

def validate_admission(p,plan_ref,review_ref):
    audit(p);r=read_json(pin(review_ref),{})
    if r.get('passed')is not True or r.get('group')!=GROUP or r.get('scientific_commit')!=p['scientific_commit'] or r.get('plan_sha256')!=plan_ref['sha256'] or r.get('design_sha256')!=DESIGN_SHA or r.get('phase')!=p['phase']:raise ValueError('exact root source/plan/phase admission')
    pin(p['design']);pin(p['identity_ref']);gate=read_json(pin(p['prerequisite_gate_ref']),{})
    if gate.get('passed')is not True or gate.get('phase')!=PRIOR[p['phase']] or gate.get('group')!=GROUP:raise ValueError('previous scientific gate cannot be skipped')
    if p['phase'] in ('train_setup','train2000','node2000') and gate.get('model_identity')!=p['model_identity']:raise ValueError('one original model identity')
    if p['phase']=='node2000' and gate.get('updates')!=2000:raise ValueError('fixed2000 endpoint')
    for task in p['tasks'].values():
        refs=task['model_inputs']
        if p['phase']=='node2000' and any(refs.get(k)!=gate.get(k+'_ref') for k in ('checkpoint','normalization','initialization_receipt')):raise ValueError('node endpoint exact model/normalization/init chain')
    root=Path(__file__).resolve().parents[3]
    if not p['qualification_refs']:raise ValueError('source/interface qualification required')
    for ref in p['qualification_refs'].values():
        q=read_json(pin(ref),{});members=q.get('source_file_sha256',{})
        if q.get('passed')is not True or q.get('scientific_commit')!=p['scientific_commit'] or q.get('design_sha256')!=DESIGN_SHA or not members or any(digest(root/n)!=s for n,s in members.items()):raise ValueError('actual qualified source changed')

def resource_admission(store,j):
    port=deployment(store);r=read_json(store.archive_root.parent/'ops/resource_route.json',{})
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(r.get('observed_utc','1970-01-01T00:00:00+00:00'))).total_seconds()
    if r.get('ready')is not True or r.get('root_accepted')is not True or r.get('group')!=GROUP or str(r.get('ssh_port'))!=port or r.get('scientific_commit')!=j['scientific_commit'] or not -5<=age<=120:raise ValueError('fresh root resource/unique task receipt')
    from .r7_resources import observe
    actual=observe(store.archive_root)
    if actual['foreign_workers'] or actual['gpu'].split(',')[0].strip()!=r['gpu_uuid']:raise ValueError('actual foreign worker/GPU changed')
    return actual

def api(app,method,parts,body):
    policy=sys.modules[__name__]
    if method=='GET' and parts==['schema']:return dict(group=GROUP,schema_version=16,modes=MODES,labels=576,node_cases=20,node_D5=60,allowed_ports=PORTS,train_port=20474,later_phases_dispatch_allowed=False)
    if method=='GET' and parts==['tasks']:
        from .coverage_ai_search import observed_status
        return {'tasks':[{'job':j,'status':observed_status(app.store,j,s)} for j,s in app.store.jobs() if j.get('experiment_id')==GROUP]}
    if method=='POST' and parts==['tasks']:return frozen_round_tasks.create(app,body,policy)
    if len(parts)>=2 and parts[0]=='tasks':
        run=validate_id(parts[1]);job=read_json(app.store.job_dir(run)/'job.json',{})
        if job.get('experiment_id')!=GROUP:raise ValueError('foreign R7 group')
        if method=='GET' and len(parts)==2:return app.run_detail(run)
        if method=='POST' and parts[2:]==['resume']:raise ValueError('R7 resume requires separate exact root same-ID reconciliation; no blind replay')
        if method=='POST' and parts[2:]==['stop']:return app.scheduler.request_stop(run)
    raise FileNotFoundError('unknown R7 endpoint')
