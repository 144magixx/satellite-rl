"""Small shared exact-contract publication/recovery for independent rounds.

Policies supply group/modes and validate science. This layer only handles
immutable source/input pins, atomic publication and unique semantic ownership.
"""
from copy import deepcopy
import hashlib,json
from pathlib import Path
import shutil
from satellite_rl.training.run_monitor import atomic_json
from .storage import digest,new_id,now,read_json,validate_id
from .coverage_ai_search import pin
from .geoload_runner import exclusive_lock


def semantic_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def create(app,body,policy):
    store=app.store;port=policy.deployment(store);request=validate_id(body.get('request_id'))
    original_plan=pin(body['plan']);plan=read_json(original_plan,{})
    task_key=validate_id(body.get('task_key'));contract=policy.task_contract(body)
    if plan.get('group')!=policy.GROUP or plan.get('authorized') is not True or plan.get('tasks',{}).get(task_key)!=contract:
        raise ValueError('task differs from authorized frozen round plan')
    policy.validate_admission(plan,body['plan'],body.get('root_review'))
    policy.validate_task(contract,plan,port)
    payload_sha=semantic_hash(body);owner_root=store.archive_root.parent.parent/'ownership'
    owner_root.mkdir(parents=True,exist_ok=True)
    owner_key=semantic_hash([body['plan']['sha256'],task_key]);owner_path=owner_root/(owner_key+'.json')
    with store.lock,exclusive_lock(owner_root/(owner_key+'.lock'),blocking=True):
        owner=read_json(owner_path,{})
        if owner:
            if (owner.get('request_id')!=request or owner.get('request_hash')!=payload_sha or
                    owner.get('archive_root')!=str(store.archive_root)):
                raise ValueError('semantic task already has another owner/request')
            published=read_json(Path(owner['job_path']),{})
            if not published:raise ValueError('unpublished reservation retained; audit before recovery')
            verify(store,published,policy);return published
        for previous,_ in store.jobs():
            if previous.get('request_id')==request:raise ValueError('request already used by another frozen semantic task')
        source,commit=store.freeze_source()
        if commit!=plan.get('scientific_commit'):raise ValueError('scientific epoch differs from source')
        run_id=new_id('prepare_ai_search_r2' if contract['mode']=='prepare_common' else 'evaluate_ai_search_r2')
        directory=store.job_dir(run_id);directory.mkdir()
        model_dir=store.models_root/run_id;model_dir.mkdir()
        atomic_json(owner_path,{'group':policy.GROUP,'run_id':run_id,'request_id':request,'request_hash':payload_sha,
            'archive_root':str(store.archive_root),'job_path':str(directory/'job.json'),'status':'reserved','at':now()})
        refs={};config=deepcopy(contract['config'])
        to_copy={'plan':body['plan'],'design':plan['design'],'publication_root_review':body['root_review'],**config['datasets']}
        to_copy.update({'qualification_'+k:r for k,r in plan.get('qualification_refs',{}).items()})
        for key,ref in to_copy.items():
            original=pin(ref);target=directory/'input_sources'/key/original.name
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
            refs[key]={**ref,'path':str(target),'source_path':str(original)};pin(refs[key])
        # Historical R1 inputs/layouts and common artifacts stay uniquely
        # canonical; only their signatures are new dependencies, not copies.
        external=policy.external_references(contract)
        for key,ref in external.items():pin(ref);refs[key]=deepcopy(ref)
        config['datasets']={role:refs[role] for role in config['datasets']}
        job={'group':policy.GROUP,'experiment_id':policy.GROUP,'run_id':run_id,'request_id':request,
            'request_hash':payload_sha,'task_key':task_key,'original_task_contract':deepcopy(contract),
            'archive_root':str(store.archive_root),'source_root':str(source),'scientific_commit':commit,'git_commit':commit,
            'source_manifest_sha256':digest(Path(source)/'source_manifest.json'),'config_sha256':None,'references':refs,
            'owner_path':str(owner_path),'created_at':now(),'model_dir':str(model_dir),'name':config['name'],
            'kind':'prepare' if contract['mode']=='prepare_common' else 'evaluate','research_stage':'coverage',
            'algorithm':'ai_search_r2','network':'closed_pool','execution_device':'cpu',
            'mode':contract['mode'],'dataset_role':contract['dataset_role'],'case_indices':contract['case_indices'],
            'executor_port':contract['executor_port'],'variants':contract['variants']}
        atomic_json(directory/'config.json',config);job['config_sha256']=digest(directory/'config.json')
        atomic_json(directory/'status.json',{'run_id':run_id,'stage':'queued','pid':None,'completed_cases':0,
            'total_cases':policy.expected_results(contract),'env_step':0,'update_step':0,'heartbeat_at':now()})
        atomic_json(directory/'job.json',job)
        owner=read_json(owner_path);owner.update(status='bound',job_sha256=digest(directory/'job.json'));atomic_json(owner_path,owner)
        return job


def verify(store,job,policy):
    directory=store.job_dir(job['run_id']);owner=read_json(job['owner_path'],{})
    if (job.get('experiment_id')!=policy.GROUP or job.get('archive_root')!=str(store.archive_root) or
            owner.get('run_id')!=job['run_id'] or owner.get('request_hash')!=job['request_hash'] or
            owner.get('job_sha256')!=digest(directory/'job.json')):raise ValueError('unique published job owner changed')
    for ref in job['references'].values():pin(ref)
    plan=read_json(job['references']['plan']['path'],{});contract=job['original_task_contract']
    policy.validate_admission(plan,job['references']['plan'],job['references']['publication_root_review'])
    policy.validate_task(contract,plan,policy.deployment(store))
    if plan.get('tasks',{}).get(job['task_key'])!=contract or plan.get('scientific_commit')!=job['scientific_commit']:
        raise ValueError('original scientific task/source contract changed')
    for key in ('mode','dataset_role','case_indices','executor_port','variants'):
        if job[key]!=contract[key]:raise ValueError('published task field changed: '+key)
    expected=deepcopy(contract['config']);expected['datasets']={role:job['references'][role] for role in expected['datasets']}
    actual=read_json(directory/'config.json',{})
    if actual!=expected or digest(directory/'config.json')!=job['config_sha256']:raise ValueError('frozen task config changed')
    source=Path(job['source_root']).resolve()
    if source!=(store.root/'source_snapshots'/job['scientific_commit']).resolve() or digest(source/'source_manifest.json')!=job['source_manifest_sha256']:
        raise ValueError('frozen scientific source path/manifest changed')
    return actual


def resume(scheduler,run_id,policy):
    from .runtime import pid_alive
    store=scheduler.store;directory=store.job_dir(run_id);job=read_json(directory/'job.json',{});status=read_json(directory/'status.json',{})
    policy.deployment(store);verify(store,job,policy)
    if job.get('resume') and status['stage']=='queued':return status
    if status['stage'] not in ('failed','stopped','interrupted'):raise ValueError('completed tasks cannot resume')
    launch=read_json(directory/'launch.json',{})
    if pid_alive(status.get('pid'),launch.get('process_start_time')):raise ValueError('original worker remains alive')
    attempt=int(job.get('resume_attempt',0))+1;history=directory/'recovery_history'/str(attempt);history.mkdir(parents=True)
    for name in ('job.json','status.json','failure.json'):
        if (directory/name).is_file():shutil.copyfile(directory/name,history/name)
    job.update(resume=True,resume_attempt=attempt);atomic_json(directory/'job.json',job)
    status.update(stage='queued',pid=None,error=None,heartbeat_at=now());atomic_json(directory/'status.json',status)
    owner=read_json(job['owner_path']);owner['job_sha256']=digest(directory/'job.json');atomic_json(job['owner_path'],owner)
    (directory/'stop.request').unlink(missing_ok=True);return status
