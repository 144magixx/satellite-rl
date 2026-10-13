"""Exact one-setup input/cache/activity origin for the sole R7 learner."""
import math,time,os
from pathlib import Path
from .storage import read_json,digest
from .coverage_ai_search import pin
from satellite_rl.training.run_monitor import atomic_json

INPUT_KEYS=('qualified','labels_gate','initialization_receipt','source_equivalence','initial_state','authority')
OUTPUT_KEYS=('normalization','feature_cache','supply_endpoint')
TRAIN_KEYS=INPUT_KEYS+OUTPUT_KEYS

def publish_setup(job,config,directory,began,runtime,owner):
    directory=Path(directory);refs=job['model_inputs']
    if not set(INPUT_KEYS)<=set(refs):raise ValueError('complete setup original input refs required')
    for key in INPUT_KEYS:pin(refs[key])
    supply=directory/'supply';generated={k:dict(path=str(supply/(name+'.json')),sha256=digest(supply/(name+'.json'))) for k,name in (('normalization','normalization'),('feature_cache','feature_cache'),('supply_endpoint','supply_endpoint'))}
    terminal=read_json(pin(generated['supply_endpoint']),{});started=supply/'supply_started.json'
    charge=time.monotonic()-began
    if not math.isfinite(charge) or charge<terminal['charged_supply_seconds_including_publication'] or charge>28800:raise ValueError('entire setup activity charge must include supply and fit8h')
    from .r5_stage_guard import birth
    endpoint=dict(schema='R7TrainSetupEndpointV1',group='S1-AI17-R7',phase='train_setup',passed=True,
                  model_identity=job['model_identity'],executor_port=job['executor_port'],run_id=job['run_id'],
                  source_commit=job['scientific_commit'],source_manifest_sha256=job['source_manifest_sha256'],
                  training_refs={**{k:refs[k] for k in INPUT_KEYS},**generated},dataset_ref=config['datasets']['train'],case_indices=job['case_indices'],
                  runtime_identity=runtime,charged_train_setup_seconds=charge,
                  activity_origin=dict(supply_started_ref=dict(path=str(started),sha256=digest(started)),setup_process_pid=os.getpid(),setup_process_birth=birth(os.getpid()),scope='entire_original_train_setup_only_once'),
                  optimizer_updates=0,future_dispatch_allowed=False)
    path=directory/'train_setup_endpoint.json';atomic_json(path,endpoint);setup_ref=dict(path=str(path),sha256=digest(path))
    owner.update(train_setup_endpoint_ref=setup_ref,setup_training_refs=endpoint['training_refs'],setup_activity_origin=endpoint['activity_origin'],
                 setup_runtime_identity=runtime,setup_dataset_ref=endpoint['dataset_ref'],setup_case_indices=endpoint['case_indices'],
                 qualified_ref=refs['qualified'],supply_run_id=job['run_id'],supply_executor_port=job['executor_port'])
    return endpoint,setup_ref

def validate_setup(endpoint,setup_ref,refs,model_identity,executor_port,case_indices,dataset_ref):
    if endpoint.get('schema')!='R7TrainSetupEndpointV1' or endpoint.get('phase')!='train_setup' or endpoint.get('group')!='S1-AI17-R7' or endpoint.get('passed')is not True or endpoint.get('optimizer_updates')!=0:raise ValueError('actual complete successful setup endpoint required')
    if endpoint.get('model_identity')!=model_identity or endpoint.get('executor_port')!=executor_port or executor_port!=20474 or endpoint.get('case_indices')!=case_indices or endpoint.get('dataset_ref')!=dataset_ref or refs.get('train_setup_endpoint')!=setup_ref:raise ValueError('same setup model/GPU/table/dataset/endpoint')
    expected=endpoint.get('training_refs',{})
    if set(expected)!=set(TRAIN_KEYS) or any(refs.get(k)!=expected[k] for k in TRAIN_KEYS):raise ValueError('training refs must all match one exact original train_setup')
    for key in TRAIN_KEYS:pin(expected[key])
    origin=endpoint.get('activity_origin',{})
    if origin.get('scope')!='entire_original_train_setup_only_once' or type(origin.get('setup_process_pid'))is not int or type(origin.get('setup_process_birth'))is not int:raise ValueError('original setup activity process identity missing')
    started=read_json(pin(origin['supply_started_ref']),{})
    if started.get('qualified_ref')!=expected['qualified'] or started.get('gate_ref')!=expected['labels_gate'] or started.get('charged_to_training_activity')is not True:raise ValueError('actual supply activity source differs')
    supply=read_json(pin(expected['supply_endpoint']),{})
    if supply.get('qualified_ref')!=expected['qualified'] or supply.get('normalization_ref')!=expected['normalization'] or supply.get('cache_ref')!=expected['feature_cache']:raise ValueError('same actual supply/cache/normalization endpoint')
    charge=endpoint.get('charged_train_setup_seconds')
    if type(charge) not in (int,float) or not math.isfinite(charge) or not supply['charged_supply_seconds_including_publication']<=charge<=28800:raise ValueError('entire original setup activity clock charge')
    return float(charge)

def validate_owner(owner,endpoint,setup_ref):
    if not owner or owner.get('train_setup_endpoint_ref')!=setup_ref or owner.get('model_identity')!=endpoint['model_identity'] or owner.get('supply_run_id')!=endpoint['run_id'] or owner.get('supply_executor_port')!=20474:raise ValueError('original single setup global owner required before learner')
    if owner.get('setup_training_refs')!=endpoint['training_refs'] or owner.get('qualified_ref')!=endpoint['training_refs']['qualified'] or owner.get('setup_activity_origin')!=endpoint['activity_origin'] or owner.get('setup_runtime_identity')!=endpoint['runtime_identity'] or owner.get('setup_dataset_ref')!=endpoint['dataset_ref'] or owner.get('setup_case_indices')!=endpoint['case_indices']:raise ValueError('owner actual setup refs/activity/runtime/table changed')
    return True
