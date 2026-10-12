"""Frozen round/task contracts and unique ownership for S1-AI17 only."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

from satellite_rl.training.run_monitor import atomic_json
from .storage import clean_json, digest, new_id, now, read_json, validate_id

GROUP = 'S1-AI17'
PARENT = Path('/home/u2024110224/first_stage_satellite_rl/s1_ai17')
PORTS = ('20418', '20424', '20473', '20474')
MODES = ('development', 'bridge', 'reference', 'teacher', 'validation', 'test', 'dataset', 'train','diagnostic_validation')
METHODS = ('INIT', 'RULE', 'RANDOM', 'AI', 'N_ONLY', 'W_ONLY', 'POST_SHRINK',
           '07-CV', '07-LEX', 'JOINT-CV', 'JOINT-LEX')
TASK_FIELDS = ('config', 'mode', 'dataset_role', 'case_indices', 'methods', 'executor_port')


def pin(reference):
    if (not isinstance(reference, dict) or not isinstance(reference.get('path'), str)
            or not isinstance(reference.get('sha256'), str) or len(reference['sha256']) != 64):
        raise ValueError('immutable path/SHA reference required')
    path = Path(reference['path']).resolve(strict=True)
    if not path.is_file() or digest(path) != reference['sha256']:
        raise ValueError('source SHA differs: ' + str(path))
    return path


def deployment(store):
    root = store.archive_root.resolve()
    port = root.parent.name
    if (root.name != 'archive' or root.parent.parent != PARENT.resolve() or port not in PORTS
            or os.environ.get('SATELLITE_RL_SSH_PORT') != port):
        raise ValueError('AI17 requires its isolated four authorized remote archives')
    from .coverage import _require_remote_dqn
    _require_remote_dqn()
    return port


def validate_config(config):
    value = deepcopy(config)
    required = {'schema_version', 'research_stage', 'algorithm', 'name', 'geometry',
                'search', 'datasets', 'workers', 'resources'}
    if set(value) != required or (value['schema_version'], value['research_stage'],
                                  value['algorithm']) != (9, 'coverage', 'ai_search'):
        raise ValueError('unknown/missing AI17 configuration fields')
    if not isinstance(value['name'], str) or not 1 <= len(value['name']) <= 100:
        raise ValueError('invalid task name')
    from satellite_rl.coverage.ai_search.config import GeometryConfig, SearchConfig
    GeometryConfig(**value['geometry']); SearchConfig(**value['search'])
    if type(value['workers']) is not int or not 1 <= value['workers'] <= 8:
        raise ValueError('bounded 1..8 case workers required')
    resources = value['resources']
    if (set(resources) != {'min_free_disk_bytes', 'max_rss_bytes'} or
            any(type(v) is not int or v <= 0 for v in resources.values())):
        raise ValueError('positive explicit disk/RSS guards required')
    if not isinstance(value['datasets'], dict):
        raise ValueError('explicit input datasets required')
    for role, record in value['datasets'].items():
        if role not in ('train', 'development', 'validation', 'test'):
            raise ValueError('unknown input role')
        if (not isinstance(record, dict) or record.get('role') != role or
                type(record.get('count')) is not int or record['count'] <= 0):
            raise ValueError('input role/count required')
    return value


def validate_task(body, plan, port):
    """Every scientific field must equal this round's frozen task contract."""
    if plan.get('group') != GROUP or plan.get('authorized') is not True:
        raise ValueError('authorized frozen AI17 plan required')
    key = validate_id(body.get('task_key'))
    expected = plan.get('tasks', {}).get(key)
    fields = TASK_FIELDS + (('dataset_spec',) if body.get('mode') == 'dataset' else
                            ('training_spec',) if body.get('mode') == 'train' else ())
    actual = {field: body.get(field) for field in fields}
    if not isinstance(expected, dict) or set(expected) != set(fields) or actual != expected:
        raise ValueError('task differs from frozen round contract')
    if str(actual['executor_port']) != str(port):
        raise ValueError('task belongs to a different physical workbench')
    if plan.get('derived_from'):
        validate_derived_plan(plan)
    config = validate_config(actual['config'])
    mode, role = actual['mode'], actual['dataset_role']
    if mode not in MODES or (mode != 'dataset' and role not in config['datasets']):
        raise ValueError('invalid worker mode/input role')
    if mode == 'dataset':
        spec = actual['dataset_spec']
        if (role not in ('train','development','validation','test') or spec.get('role') != role or
                spec.get('kind') not in ('official_day','merge_existing') or
                type(spec.get('count')) is not int or spec['count'] <= 0):
            raise ValueError('bounded official/packed dataset preparation contract required')
        if spec['kind'] == 'official_day':
            pin(spec.get('parser'))
            for ref in spec.get('metadata_pins',{}).values():pin(ref)
            raw_root = Path(spec.get('raw_shared_root','')).resolve()
            if not raw_root.is_relative_to(PARENT / 'source' / 'raw'):
                raise ValueError('new raw source must use AI17 canonical source directory')
        else:
            for ref in spec.get('sources',[]):pin(ref)
    if mode == 'teacher' and role != 'train':
        raise ValueError('teachers are training-input only')
    if mode=='teacher':
        mothers=plan.get('teacher_mother_order')
        if not isinstance(mothers,list) or not mothers or len(set(mothers))!=len(mothers):
            raise ValueError('teacher requires frozen global mother order for balanced preferences')
        if actual['methods']!=['RULE']:raise ValueError('teacher uses one common RULE task pool per case')
    if mode=='train':
        if role!='train' or set(config['datasets'])!={'train','validation'} or actual['methods']!=['AI']:
            raise ValueError('training requires frozen train/validation and one AI model')
        training=actual['training_spec'];protocol=training.get('protocol',{})
        if (protocol.get('engineering') is True or protocol.get('seed')!=51001 or
                protocol.get('updates')!=10000 or protocol.get('batch',protocol.get('effective_batch'))!=64):
            raise ValueError('formal training requires seed51001/10000/batch64 protocol')
        if training.get('validation_dispatch')!='workbench_async':
            raise ValueError('training diagnostics must dispatch asynchronously through workbenches')
        pin(training.get('teacher_manifest'));gate=read_json(pin(training.get('gate')), {})
        if gate.get('group')!=GROUP or gate.get('passed') is not True:
            raise ValueError('training requires passed pinned teacher gate')
        if sum(t.get('mode')=='train' for t in plan['tasks'].values())!=1:
            raise ValueError('one train task per frozen scientific epoch required')
        validation_indices=training.get('validation_case_indices')
        if (not isinstance(validation_indices,list) or not validation_indices or
                len(set(validation_indices))!=len(validation_indices) or
                any(type(i)is not int or not 0<=i<config['datasets']['validation']['count'] for i in validation_indices)):
            raise ValueError('training needs explicit bounded validation indices')
    if mode in ('test', 'validation', 'bridge','diagnostic_validation') and config['workers'] != 1:
        raise ValueError('common validation/test/bridge timing uses one case worker')
    if mode == 'test' and (role != 'test' or not plan.get('test_admission', {}).get('passed')):
        raise ValueError('test must have frozen candidate admission proof')
    if role == 'test' and mode not in ('test','dataset'):
        raise ValueError('independent test cannot become development implicitly')
    if mode=='diagnostic_validation' and (role!='validation' or actual['methods']!=['AI']):
        raise ValueError('diagnostic nodes are validation-only AI tasks')
    methods = actual['methods']
    if (not isinstance(methods, list) or not methods or len(set(methods)) != len(methods)
            or set(methods) - set(METHODS)):
        raise ValueError('unknown/duplicate method')
    indices = actual['case_indices']
    if mode in ('dataset','train') and indices != []:
        raise ValueError('dataset preparation does not select optimization case indices')
    if mode not in ('dataset','train') and (not isinstance(indices, list) or not indices or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < config['datasets'][role]['count'] for i in indices)):
        raise ValueError('unique bounded case indices required')
    if mode == 'bridge' and set(methods) - {'07-CV', '07-LEX', 'JOINT-CV', 'JOINT-LEX'}:
        raise ValueError('bridge uses original fixed-N/fixed-width methods only')
    if 'AI' in methods and mode!='train' and not isinstance(plan.get('model'), dict):
        raise ValueError('AI requires frozen model reference')
    return config


def validate_derived_plan(plan):
    """Pin model-producing node and the original immutable diagnostic template."""
    derived=plan['derived_from'];original=read_json(pin(derived['source_plan']),{})
    ready=read_json(pin(derived['node_ready']),{})
    cursor=derived['update_step']
    if cursor not in (2500,5000,7500,10000) or ready.get('update_step')!=cursor:
        raise ValueError('diagnostic node is not a frozen training checkpoint')
    if (ready.get('group')!=GROUP or ready.get('producer_run_id')!=derived['producer_run_id'] or
            ready.get('scientific_commit')!=plan['scientific_commit'] or
            original.get('scientific_commit')!=plan['scientific_commit'] or ready.get('model')!=plan.get('model')):
        raise ValueError('model-producing task/node/source identity differs')
    producer=original.get('tasks',{}).get(derived['producer_task_key'],{})
    if producer.get('mode')!='train' or ready.get('producer_task_key')!=derived['producer_task_key']:
        raise ValueError('node lacks its original unique training task')
    template=original.get('node_templates',{}).get(str(cursor))
    if plan.get('tasks')!=template:
        raise ValueError('derived validation task differs from pre-frozen node template')
    for task in template.values():
        if (task['mode']!=('diagnostic_validation' if cursor<10000 else 'validation') or
                task['dataset_role']!='validation' or task['config']['workers']!=1 or
                (cursor<10000 and task['methods']!=['AI'])):
            raise ValueError('diagnostic/final timing or method boundary differs')
    if cursor==10000:
        expected={'INIT','RULE','RANDOM','AI','N_ONLY','W_ONLY','POST_SHRINK'}
        if any(set(task['methods'])!=expected for task in template.values()):
            raise ValueError('10000 node requires one final seven-method common matrix')
    return ready


def observed_status(store, job, status):
    from .runtime import pid_alive
    launch = read_json(store.job_dir(job['run_id']) / 'launch.json', {})
    result = deepcopy(status)
    result['process_alive'] = pid_alive(status.get('pid'), launch.get('process_start_time'))
    if result.get('stage') in ('running', 'starting', 'training') and status.get('pid') and not result['process_alive']:
        result['reported_stage'] = result['stage']; result['stage'] = 'interrupted'
    return result


def create_job(app, body):
    from .geoload_runner import exclusive_lock
    store = app.store; port = deployment(store)
    request = validate_id(body.get('request_id'))
    source_plan = pin(body.get('plan')); plan = read_json(source_plan, {})
    config = validate_task(body, plan, port)
    for ref in config['datasets'].values(): pin(ref)
    payload_sha = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                          separators=(',', ':')).encode()).hexdigest()
    owner_root = store.archive_root.parent.parent / 'ownership'
    owner_root.mkdir(parents=True, exist_ok=True)
    owner_name = hashlib.sha256((body['plan']['sha256']+'\0'+body['task_key']).encode()).hexdigest()
    owner_path = owner_root / (owner_name + '.json')
    with store.lock, exclusive_lock(owner_root / (owner_name + '.lock'), blocking=True):
        owner = read_json(owner_path, {})
        if owner and (owner.get('request_id') != request or owner.get('archive_root') != str(store.archive_root) or
                      owner.get('request_hash') != payload_sha):
            raise ValueError('request already belongs to another owner/configuration')
        for existing, _ in store.jobs():
            if existing.get('request_id') == request:
                if existing.get('request_hash') != payload_sha:
                    raise ValueError('request ID reused for different task')
                verify_job(store, existing)
                return existing
        if owner:
            raise ValueError('unpublished reservation retained; audit it before recovery')
        snapshot, commit = store.freeze_source()
        if commit != plan.get('scientific_commit'):
            raise ValueError('source commit differs from frozen scientific epoch')
        run_id = new_id('train_ai_search' if body['mode']=='train' else
                        'prepare_ai_search' if body['mode'] in ('teacher','dataset') else 'evaluate_ai_search')
        directory = store.job_dir(run_id); directory.mkdir()
        model_dir = store.models_root / run_id; model_dir.mkdir()
        atomic_json(owner_path, {'group': GROUP, 'archive_root': str(store.archive_root),
            'run_id': run_id, 'request_id': request, 'request_hash': payload_sha,
            'job_path': str(directory / 'job.json'), 'status': 'reserved', 'at': now()})
        refs = {}
        freeze_refs = [('plan', body['plan']), *config['datasets'].items()]
        if plan.get('design'):freeze_refs.append(('design',plan['design']))
        if plan.get('derived_from'):
            freeze_refs.extend((('original_node_plan',plan['derived_from']['source_plan']),
                                ('node_ready',plan['derived_from']['node_ready'])))
        dataset_spec=deepcopy(body.get('dataset_spec'))
        training_spec=deepcopy(body.get('training_spec'))
        if training_spec:
            freeze_refs.extend((('teacher_manifest',training_spec['teacher_manifest']),('teacher_gate',training_spec['gate'])))
        if dataset_spec:
            if dataset_spec['kind']=='official_day':
                freeze_refs.append(('parser',dataset_spec['parser']))
                freeze_refs.extend(('metadata_'+key,ref) for key,ref in dataset_spec['metadata_pins'].items())
            else:freeze_refs.extend(('packed_source_'+str(i),ref) for i,ref in enumerate(dataset_spec['sources']))
        for name, ref in freeze_refs:
            original = pin(ref); target = directory / 'input_sources' / name / original.name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(original, target)
            copied = {**ref, 'path': str(target), 'source_path': str(original)}
            pin(copied); refs[name] = copied
        if plan.get('model'):
            original = pin(plan['model']); target = directory / 'model_input' / original.name
            target.parent.mkdir(); shutil.copyfile(original, target)
            refs['model'] = {**plan['model'], 'path': str(target), 'source_path': str(original)}
            pin(refs['model'])
        config['datasets'] = {role: refs[role] for role in config['datasets']}
        job = {'run_id': run_id, 'request_id': request, 'request_hash': payload_sha,
            'experiment_id': GROUP, 'research_stage': 'coverage', 'algorithm': 'ai_search',
            'network': 'ai_search', 'name': config['name'], 'kind': 'train' if body['mode']=='train' else
                'prepare' if body['mode'] in ('teacher','dataset') else 'evaluate',
            'mode': body['mode'], 'dataset_role': body['dataset_role'],
            'case_indices': body['case_indices'], 'methods': body['methods'], 'task_key': body['task_key'],
            'created_at': now(), 'archive_root': str(store.archive_root), 'source_root': str(snapshot),
            'git_commit': commit, 'scientific_commit': commit, 'model_dir': str(model_dir),
            'references': refs, 'executor_port': int(port), 'execution_device': 'cuda' if 'AI' in body['methods'] else 'cpu',
            'owner_path': str(owner_path), 'config_sha256': None,
            'source_manifest_sha256': digest(Path(snapshot) / 'source_manifest.json'),
            'original_task_contract': {field: deepcopy(body[field]) for field in (TASK_FIELDS+(
                ('dataset_spec',) if dataset_spec else ('training_spec',) if training_spec else ()))},
            'source_boundary': {'inputs': ['positions', 'demands', 'aircraft_ids', 'offsets', 'case_ids', 'mother_ids', 'demand_profiles'],
                                'historical_results_teachers_peer_dependencies': False}}
        atomic_json(directory / 'config.json', config)
        if body['mode']=='teacher':job['teacher_mother_order']=deepcopy(plan['teacher_mother_order'])
        if training_spec:
            training_spec['teacher_manifest']=refs['teacher_manifest'];training_spec['gate']=refs['teacher_gate']
            job['training_spec']=training_spec;job['execution_device']='cuda'
        if dataset_spec:
            if dataset_spec['kind']=='official_day':
                dataset_spec['parser']=refs['parser']
                dataset_spec['metadata_pins']={key:refs['metadata_'+key] for key in dataset_spec['metadata_pins']}
            else:dataset_spec['sources']=[refs['packed_source_'+str(i)] for i in range(len(dataset_spec['sources']))]
            job['dataset_spec']=dataset_spec
        job['config_sha256'] = digest(directory / 'config.json')
        atomic_json(directory / 'status.json', {'run_id': run_id, 'stage': 'queued', 'pid': None,
            'completed_cases': 0, 'total_cases': len(body['case_indices']) * len(body['methods']),
            'env_step': 0, 'update_step': 0, 'budget_env_steps': 0, 'heartbeat_at': now()})
        atomic_json(directory / 'job.json', job)  # Publication is last.
        owner = read_json(owner_path); owner.update(status='bound', job_sha256=digest(directory / 'job.json'))
        atomic_json(owner_path, owner)
        return job


def verify_job(store, job):
    """Bind current executable semantics to original plan, owner and source."""
    directory = store.job_dir(job['run_id'])
    if job.get('experiment_id') != GROUP or job.get('archive_root') != str(store.archive_root):
        raise ValueError('task group/archive identity differs')
    owner = read_json(job['owner_path'], {})
    if (owner.get('run_id') != job['run_id'] or owner.get('archive_root') != job['archive_root'] or
            owner.get('request_hash') != job.get('request_hash') or
            owner.get('job_sha256') != digest(directory / 'job.json')):
        raise ValueError('published job differs from original unique owner')
    for ref in job['references'].values(): pin(ref)
    plan = read_json(job['references']['plan']['path'], {})
    contract = job['original_task_contract']
    if plan.get('tasks', {}).get(job['task_key']) != contract:
        raise ValueError('frozen task contract differs')
    if plan.get('scientific_commit') != job['scientific_commit'] or job['git_commit'] != job['scientific_commit']:
        raise ValueError('scientific source epoch changed')
    for field in ('mode', 'dataset_role', 'case_indices', 'methods', 'executor_port'):
        if job.get(field) != contract.get(field): raise ValueError('published task field differs: '+field)
    config = read_json(directory / 'config.json', {})
    expected = deepcopy(contract['config'])
    expected['datasets'] = {role: job['references'][role] for role in expected['datasets']}
    if config != expected or digest(directory / 'config.json') != job['config_sha256']:
        raise ValueError('published config differs from original contract')
    if contract.get('dataset_spec'):
        expected_spec=deepcopy(contract['dataset_spec'])
        if expected_spec['kind']=='official_day':
            expected_spec['parser']=job['references']['parser']
            expected_spec['metadata_pins']={key:job['references']['metadata_'+key] for key in expected_spec['metadata_pins']}
        else:expected_spec['sources']=[job['references']['packed_source_'+str(i)] for i in range(len(expected_spec['sources']))]
        if job.get('dataset_spec')!=expected_spec:raise ValueError('published dataset preparation spec differs')
    if contract.get('training_spec'):
        expected_training=deepcopy(contract['training_spec'])
        expected_training['teacher_manifest']=job['references']['teacher_manifest']
        expected_training['gate']=job['references']['teacher_gate']
        if job.get('training_spec')!=expected_training:raise ValueError('published training protocol/evidence differs')
    if job['mode']=='teacher' and job.get('teacher_mother_order')!=plan.get('teacher_mother_order'):
        raise ValueError('teacher global mother order changed')
    source = Path(job['source_root']).resolve()
    expected_source = store.root / 'source_snapshots' / job['scientific_commit']
    if source != expected_source.resolve() or digest(source / 'source_manifest.json') != job['source_manifest_sha256']:
        raise ValueError('frozen source path/manifest changed')
    return config


def resume_job(scheduler, run_id):
    from .runtime import pid_alive
    directory = scheduler.store.job_dir(run_id)
    job = read_json(directory / 'job.json', {}); status = read_json(directory / 'status.json', {})
    deployment(scheduler.store)
    if job.get('experiment_id') != GROUP:
        raise ValueError('wrong task group')
    verify_job(scheduler.store, job)
    if job.get('resume') and status.get('stage') in ('queued', 'queued_resources'):
        return status
    if status.get('stage') not in ('failed', 'stopped', 'interrupted'):
        raise ValueError('only an original interrupted/stopped/failed task may resume')
    if job['kind']=='train' and not (Path(job['model_dir'])/'checkpoint.pt').is_file():
        raise ValueError('training resume requires complete optimizer/RNG/sampler checkpoint')
    launch = read_json(directory / 'launch.json', {})
    if pid_alive(status.get('pid'), launch.get('process_start_time')):
        raise ValueError('original worker remains alive')
    attempt = int(job.get('resume_attempt', 0)) + 1
    history = directory / 'recovery_history' / str(attempt); history.mkdir(parents=True)
    for name in ('job.json', 'status.json', 'failure.json'):
        if (directory / name).is_file(): shutil.copyfile(directory / name, history / name)
    job.update(resume=True, resume_attempt=attempt)
    atomic_json(directory / 'job.json', job)
    (directory / 'stop.request').unlink(missing_ok=True)
    status.update(stage='queued', pid=None, error=None, heartbeat_at=now())
    atomic_json(directory / 'status.json', status)
    owner = read_json(job['owner_path'], {}); owner['job_sha256'] = digest(directory / 'job.json')
    atomic_json(job['owner_path'], owner)
    return status


def api(app, method, parts, body):
    if method == 'GET' and parts == ['schema']:
        return {'group': GROUP, 'schema_version': 9, 'task_fields': TASK_FIELDS,
                'modes': MODES, 'methods': METHODS, 'remote_only': True,
                'budget_policy': 'exact frozen plan.tasks[task_key] contract'}
    if method == 'GET' and parts == ['tasks']:
        return {'tasks': [{'job': job, 'status': observed_status(app.store, job, status)}
                          for job, status in app.store.jobs() if job.get('experiment_id') == GROUP]}
    if method == 'POST' and parts == ['tasks']: return create_job(app, body)
    if len(parts) >= 2 and parts[0] == 'tasks':
        run_id = validate_id(parts[1]); job = read_json(app.store.job_dir(run_id) / 'job.json', {})
        if job.get('experiment_id') != GROUP: raise ValueError('wrong task group')
        if method == 'GET' and len(parts) == 2:
            value = app.run_detail(run_id)
            value['status'] = observed_status(app.store, job, value['status'])
            return value
        if method == 'POST' and parts[2:] == ['resume']: return app.scheduler.resume(run_id)
        if method == 'POST' and parts[2:] == ['stop']: return app.scheduler.request_stop(run_id)
    raise FileNotFoundError('unknown AI17 API')
