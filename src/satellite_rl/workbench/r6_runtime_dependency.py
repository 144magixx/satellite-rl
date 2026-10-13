"""Bounded R6 same-ID script dependency; never changes frozen snapshots."""
import hashlib,json,os,subprocess
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
    if contract.get('schema')!='R6RuntimeScriptDependencyAdmissionV1' or contract.get('passed')is not True or contract.get('science_commit')!=SCIENCE or contract.get('group')!='S1-AI17-R6':raise ValueError('R6 dependency root admission required')
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
    return dict(schema='R6CompositeRuntimeDependencyV1',run_id=job['run_id'],science_commit=SCIENCE,
        operational_service_commit=contract['operational_service_commit'],snapshot_root=str(snapshot),snapshot_manifest_sha256=digest(manifest_path),
        dependency_root=str(repo),script_file_sha256=expected,actual_import_identities=identities,
        root_admission_ref=contract_ref,pythonpath=env['PYTHONPATH'],scientific_source_unchanged=True,new_algorithm_calls=0)
