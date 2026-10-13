"""Reviewable, root-admitted R6 runtime repair; no task POST or resume here."""
import argparse,json,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.r6_runtime_dependency import SCIENCE,TARGETS,stable_job_sha
from . import ops
from .deploy import SOURCE,git,deploy

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))

def draft(evidence,original_review,operational_commit,output):
    """Only archived original metadata is read; no science datasets or APIs."""
    evidence=Path(evidence);review=read(original_review)
    if review.get('passed')is not True or review['scientific_commit']!=SCIENCE:raise ValueError('original source review')
    scripts={p:h for p,h in review['source_file_sha256'].items() if p.startswith('scripts/')}
    tasks={}
    for port,run in TARGETS.items():
        job=read(evidence/str(port)/'job.json')
        if job['run_id']!=run or job['scientific_commit']!=SCIENCE:raise ValueError('original archived ID/source')
        tasks[run]=dict(stable_job_sha256=stable_job_sha(job),config_sha256=sha(evidence/str(port)/'config.json'),snapshot_manifest_sha256=job['source_manifest_sha256'],executor_port=port)
    remote_review=ops.BASE+'/source/round6/operations_repair/runtime_deps_02/'+sha(original_review)+'.json'
    adapter_names=('src/satellite_rl/workbench/r6_runtime_dependency.py','src/satellite_rl/workbench/r6_runtime_bootstrap.py')
    value=dict(schema='R6RuntimeScriptDependencyAdmissionV2',group=ops.GROUP,passed=False,science_commit=SCIENCE,
        operational_service_commit=operational_commit,tasks=tasks,script_file_sha256=scripts,
        previous_operational_service_commit='124f57d2a5646f64253985a02cf23b77e197158f',
        validation_adapter_source_file_sha256={name:sha(SOURCE/name)for name in adapter_names},
        original_source_review_ref=dict(path=remote_review,sha256=sha(original_review)),
        source_review_local_path=str(Path(original_review).resolve()),authority='requires exact root runtime repair approval before passed=true or deployment',
        no_scientific_source_changes=True,no_new_IDs=True,no_algorithm_replay=True)
    atomic_json(output,value);return dict(contract_path=str(Path(output).resolve()),contract_sha256=sha(output),science_submitted=0)

def _remote(port,code):
    r=ops.remote_python(code,port,timeout=80)
    if r['exit_code']:raise RuntimeError(str(port)+': '+r['stderr'][-1800:])
    return json.loads(r['stdout'].splitlines()[-1])

def check_idle(port,contract,terminate=False):
    i=ops.instance(port);target=TARGETS[port];bound=contract['tasks'][target]
    previous=contract['previous_operational_service_commit']
    code=f'''import json,pathlib,hashlib,urllib.request,subprocess,signal,os,time,shutil
root=pathlib.Path({i['root']!r});service_path=root/'ops/service.json';service=json.loads(service_path.read_text());proc=pathlib.Path('/proc')/str(service['pid']);raw=(proc/'stat').read_text();birth=int(raw[raw.rfind(')')+2:].split()[19]);command=(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
assert service['source_commit']=={previous!r} and birth==service['birth_ticks'] and str(root/'archive') in command and 'satellite_rl.workbench.server' in command and '--port {i['api_port']}' in command
assert str((proc/'cwd').resolve())==service['source_root'] and subprocess.check_output(['git','-C',service['source_root'],'rev-parse','HEAD']).decode().strip()=={previous!r}
with urllib.request.urlopen('http://127.0.0.1:{i['api_port']}/api/ai-search-r6/tasks',timeout=5) as r:tasks=json.load(r)['tasks']
assert len(tasks)==1 and tasks[0]['job']['run_id']=={target!r} and tasks[0]['status']['stage']=='interrupted' and tasks[0]['status']['completed_cases']==0
d=root/'archive/outputs/workbench/jobs'/{target!r};job=json.loads((d/'job.json').read_text());stable={{k:v for k,v in job.items() if k not in ('resume','resume_attempt')}}
assert hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()=={bound['stable_job_sha256']!r}
assert hashlib.sha256((d/'config.json').read_bytes()).hexdigest()=={bound['config_sha256']!r} and job['source_manifest_sha256']=={bound['snapshot_manifest_sha256']!r}
assert not list(d.rglob('started.json')) and not (d/'bundles').exists()
workers=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  c=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
  if str(root/'archive') in c and ('satellite_rl.workbench.runner' in c or 'satellite_rl.workbench.ai_search_r6_stage' in c):workers.append(c)
 except (FileNotFoundError,PermissionError):pass
assert not workers
record={{'service':service,'birth_ticks':birth,'command':command,'actual_task':{target!r},'status':'interrupted','workers':workers,'new_science_calls':0}}
if {terminate!r}:
 backup=root/'operations_repair/runtime_deps_02';backup.mkdir(parents=True,exist_ok=True);original=backup/{('service_'+previous+'_original.json')!r}
 if original.exists():assert original.read_bytes()==service_path.read_bytes()
 else:shutil.copy2(service_path,original)
 assert int((proc/'stat').read_text().split(')')[-1].split()[19])==birth
 os.kill(service['pid'],signal.SIGTERM)
 for _ in range(50):
  if not proc.exists():break
  time.sleep(.1)
 assert not proc.exists();record['terminated_only_original_idle_service']=True
print(json.dumps(record))
'''
    return _remote(port,code)

def install_idle(contract_path,repair_review,operational_source_review,output):
    """Root-approved metadata+idle-service replacement only; cannot resume jobs."""
    contract=read(contract_path);approval=read(repair_review);operational=read(operational_source_review);commit=contract['operational_service_commit']
    if (contract.get('passed')is not True or approval.get('passed')is not True or approval.get('runtime_dependency_contract_sha256')!=sha(contract_path)
        or approval.get('science_commit')!=SCIENCE or approval.get('operational_service_commit')!=commit or approval.get('science_POST_allowed')is not False):raise ValueError('exact root repair-only admission required')
    if git('rev-parse','HEAD')!=commit or operational.get('scientific_commit')!=commit:raise ValueError('actual frozen operational source differs')
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(4) as pool:checked=dict(zip(map(str,TARGETS),pool.map(lambda p:check_idle(p,contract),TARGETS)))
    atomic_json(folder/'four_original_jobs_idle.json',checked)
    original=Path(contract['source_review_local_path'])
    if sha(original)!=contract['original_source_review_ref']['sha256']:raise ValueError('original source review bytes changed')
    contract_target=ops.BASE+'/source/round6/operations_repair/runtime_deps_02/'+sha(contract_path)+'.json'
    uploads=[ops.upload(original,contract['original_source_review_ref']['path'],20418),ops.upload(contract_path,contract_target,20418)]
    ref=dict(path=contract_target,sha256=sha(contract_path))
    for port in TARGETS:
        target=ops.instance(port)['ops']+'/runtime_dependency_admission_ref.json'
        _remote(port,f"import pathlib,json,shutil;from satellite_rl.training.run_monitor import atomic_json;p=pathlib.Path({target!r});backup=p.parent.parent/'operations_repair/runtime_deps_02/runtime_dependency_admission_ref_124_original.json';backup.parent.mkdir(parents=True,exist_ok=True);assert not backup.exists() or backup.read_bytes()==p.read_bytes();shutil.copy2(p,backup);atomic_json(p,{ref!r});print(json.dumps({{'installed_metadata_only':True}}))")
    with ThreadPoolExecutor(4) as pool:stopped=dict(zip(map(str,TARGETS),pool.map(lambda p:check_idle(p,contract,True),TARGETS)))
    atomic_json(folder/'four_original_services_stopped.json',stopped)
    deployment=deploy(commit,operational_source_review)
    receipt=dict(science_commit=SCIENCE,operational_service_commit=commit,original_IDs=TARGETS,metadata_uploads=uploads,deployment=deployment,
        science_POST_calls=0,resume_calls=0,root_same_ID_resume_still_required=True)
    atomic_json(folder/'idle_install_receipt.json',receipt);return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('draft','install-idle'))
    for name in ('evidence','original-review','operational-commit','output','contract','repair-review','operational-source-review'):p.add_argument('--'+name)
    a=p.parse_args()
    if a.action=='draft':r=draft(a.evidence,a.original_review,a.operational_commit,a.output)
    else:r=install_idle(a.contract,a.repair_review,a.operational_source_review,a.output)
    print(json.dumps(r))
