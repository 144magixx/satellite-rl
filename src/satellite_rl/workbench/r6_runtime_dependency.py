"""Bounded R6 same-ID script dependency; never changes frozen snapshots."""
import hashlib,json,os,subprocess,importlib.util,inspect
from pathlib import Path
from .storage import digest,read_json
from .coverage_ai_search import pin

SCIENCE='7bb5e68456c00846f5a8b249bba19f8800ea13ee'
TARGETS={20418:'labels_ai_search_r6_20261013_190216_ecc1eb4a',20424:'labels_ai_search_r6_20261013_190220_be67ae33',20473:'labels_ai_search_r6_20261013_190225_f760c531',20474:'labels_ai_search_r6_20261013_190229_baf6ffe9'}
BASE=Path('/home/u2024110224/first_stage_satellite_rl/s1_ai17')

def stable_job_sha(job):
    original={k:v for k,v in job.items() if k not in ('resume','resume_attempt')}
    return hashlib.sha256(json.dumps(original,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args]).decode().strip()

def validate_contract(contract,job,config_sha,snapshot_manifest_sha):
    if contract.get('schema')!='R6RuntimeScriptDependencyAdmissionV2' or contract.get('passed')is not True or contract.get('science_commit')!=SCIENCE or contract.get('group')!='S1-AI17-R6':raise ValueError('R6 dependency v2 root admission required')
    if job.get('experiment_id')!='S1-AI17-R6' or job.get('scientific_commit')!=SCIENCE or job.get('git_commit')!=SCIENCE or TARGETS.get(job.get('executor_port'))!=job.get('run_id'):raise ValueError('exact original R6 science/task required')
    if set(contract.get('tasks',{}))!=set(TARGETS.values()):raise ValueError('exact original four IDs required')
    bound=contract['tasks'][job['run_id']]
    if bound!={'stable_job_sha256':stable_job_sha(job),'config_sha256':config_sha,'snapshot_manifest_sha256':snapshot_manifest_sha,'executor_port':job['executor_port']}:raise ValueError('original task/config/source binding differs')

def prepare(store,job,directory,python):
    """Service-only prelaunch validation, imports and manifest; zero algorithms."""
    directory=Path(directory);archive=store.archive_root.resolve();port=job['executor_port']
    if archive!=(BASE/str(port)/'round6/archive').resolve():raise ValueError('exact original lane archive required')
    contract_ref=read_json(archive.parent/'ops/runtime_dependency_admission_ref.json',{})
    contract=read_json(pin(contract_ref),{})
    snapshot=Path(job['source_root']).resolve();manifest_path=snapshot/'source_manifest.json';manifest=read_json(manifest_path,{})
    validate_contract(contract,job,digest(directory/'config.json'),digest(manifest_path))
    if digest(manifest_path)!=job['source_manifest_sha256'] or manifest.get('git_commit')!=SCIENCE:raise ValueError('original frozen manifest changed')
    if contract.get('operational_service_commit')!=git(store.source_root,'rev-parse','HEAD'):raise ValueError('actual operational service epoch differs')
    adapter_sources=contract.get('validation_adapter_source_file_sha256',{})
    required={'src/satellite_rl/workbench/r6_runtime_dependency.py','src/satellite_rl/workbench/r6_runtime_bootstrap.py'}
    if set(adapter_sources)!=required or any(digest(Path(store.source_root)/name)!=sha for name,sha in adapter_sources.items()):raise ValueError('v2 validation adapter source differs')
    repo=Path(manifest.get('source_root','')).resolve()
    if repo!=(BASE/str(port)/'round6/source_epochs'/SCIENCE/'repo').resolve() or git(repo,'rev-parse','HEAD')!=SCIENCE:raise ValueError('original immutable dependency repository differs')
    review=read_json(pin(contract['original_source_review_ref']),{})
    if review.get('passed')is not True or review.get('scientific_commit')!=SCIENCE:raise ValueError('original root source review differs')
    expected=contract['script_file_sha256'];tracked=git(repo,'ls-tree','-r','--name-only',SCIENCE,'--','scripts').splitlines()
    if set(expected)!=set(tracked) or not expected or any(review['source_file_sha256'].get(name)!=sha for name,sha in expected.items()):raise ValueError('entire original tracked script tree binding differs')
    if git(repo,'status','--porcelain','--','scripts'):raise ValueError('dependency script tree dirty')
    for name,sha in expected.items():
        target=(repo/name).resolve()
        if not target.is_relative_to(repo/'scripts') or not target.is_file() or digest(target)!=sha:raise ValueError('script checksum differs: '+name)
        blob=subprocess.check_output(['git','-C',str(repo),'show',SCIENCE+':'+name])
        if hashlib.sha256(blob).hexdigest()!=sha:raise ValueError('script Git blob differs: '+name)
    env=os.environ.copy();env.update(PYTHONPATH=str(snapshot/'src')+os.pathsep+str(repo),PYTHONDONTWRITEBYTECODE='1')
    code="import json,satellite_rl,scripts.stage1_ai_search_r6.plan_builder as p,scripts.stage1_ai_search_r6.label_audit as a,satellite_rl.workbench.coverage_ai_search_r6 as c,satellite_rl.workbench.ai_search_r6_stage as s;print(json.dumps(dict(package=satellite_rl.__file__,plan_builder=p.__file__,label_audit=a.__file__,policy=c.__file__,stage=s.__file__)))"
    probe=subprocess.run([python,'-B','-c',code],cwd=snapshot,env=env,capture_output=True,text=True,timeout=30)
    if probe.returncode:raise ValueError('exact composite imports failed: '+probe.stderr[-1200:])
    identities=json.loads(probe.stdout.splitlines()[-1])
    for key in ('package','policy','stage'):
        if not Path(identities[key]).resolve().is_relative_to(snapshot/'src'):raise ValueError('scientific module escaped original snapshot')
    for key in ('plan_builder','label_audit'):
        if not Path(identities[key]).resolve().is_relative_to(repo/'scripts'):raise ValueError('script import escaped original dependency')
    _,validation=original_validation_policy(store.source_root,archive)
    validation={k:v for k,v in validation.items() if k in ('actual_validation_file','actual_validation_file_sha256','actual_validation_function_sha256','validation_base','original_hardchecks_preserved')}
    return dict(schema='R6CompositeRuntimeDependencyV1',run_id=job['run_id'],science_commit=SCIENCE,
        operational_service_commit=contract['operational_service_commit'],snapshot_root=str(snapshot),snapshot_manifest_sha256=digest(manifest_path),
        dependency_root=str(repo),script_file_sha256=expected,actual_import_identities=identities,
        root_admission_ref=contract_ref,pythonpath=env['PYTHONPATH'],scientific_source_unchanged=True,new_algorithm_calls=0,
        validation_adapter_source_file_sha256=adapter_sources,
        bootstrap_path=str(Path(store.source_root)/'src/satellite_rl/workbench/r6_runtime_bootstrap.py'),**validation)

def original_validation_policy(operational_root,archive_root):
    """Route source validation to the signed original full repository only."""
    archive=Path(archive_root).resolve();port=int(archive.parent.parent.name)
    if archive!=(BASE/str(port)/'round6/archive').resolve() or port not in TARGETS:raise ValueError('original validation lane scope')
    ref=read_json(archive.parent/'ops/runtime_dependency_admission_ref.json',{});contract=read_json(pin(ref),{})
    if contract.get('schema')!='R6RuntimeScriptDependencyAdmissionV2' or contract.get('passed')is not True or contract.get('science_commit')!=SCIENCE or set(contract.get('tasks',{}))!=set(TARGETS.values()):raise ValueError('v2 original validation admission')
    operational_root=Path(operational_root).resolve()
    if operational_root!=(BASE/str(port)/'round6/source_epochs'/contract['operational_service_commit']/'repo').resolve() or git(operational_root,'rev-parse','HEAD')!=contract['operational_service_commit']:raise ValueError('actual operational validation source differs')
    sources=contract.get('validation_adapter_source_file_sha256',{})
    if set(sources)!={'src/satellite_rl/workbench/r6_runtime_dependency.py','src/satellite_rl/workbench/r6_runtime_bootstrap.py'} or any(digest(operational_root/name)!=sha for name,sha in sources.items()):raise ValueError('validation adapter file binding')
    review=read_json(pin(contract['original_source_review_ref']),{})
    repo=BASE/str(port)/'round6/source_epochs'/SCIENCE/'repo'
    if git(repo,'rev-parse','HEAD')!=SCIENCE or git(repo,'status','--porcelain'):raise ValueError('original validation repository changed')
    if review.get('passed')is not True or review.get('scientific_commit')!=SCIENCE:raise ValueError('original validation root review')
    for name,sha in review['source_file_sha256'].items():
        target=(repo/name).resolve()
        if not target.is_relative_to(repo) or digest(target)!=sha:raise ValueError('original full runtime source changed: '+name)
    name='satellite_rl.workbench._r6_original_validation'
    path=repo/'src/satellite_rl/workbench/coverage_ai_search_r6.py'
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,dict(actual_validation_file=str(path),actual_validation_file_sha256=digest(path),
        actual_validation_function_sha256=hashlib.sha256(inspect.getsource(module.validate_admission).encode()).hexdigest(),
        validation_base=str(repo),original_hardchecks_preserved=True,root_admission_ref=ref,science_commit=SCIENCE,
        operational_service_commit=contract['operational_service_commit'])

def service_validate_admission(plan,plan_ref,review_ref,operational_root):
    if plan.get('scientific_commit')!=SCIENCE:raise ValueError('bounded original R6 validation only')
    port=int(os.environ['SATELLITE_RL_SSH_PORT']);archive=BASE/str(port)/'round6/archive'
    module,receipt=original_validation_policy(operational_root,archive)
    module.validate_admission(plan,plan_ref,review_ref)
    return receipt
