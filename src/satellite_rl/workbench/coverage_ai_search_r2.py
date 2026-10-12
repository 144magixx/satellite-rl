"""R2-only exact contracts: read existing strong starts, then a closed pool."""
from copy import deepcopy
from datetime import datetime,timezone
import os
from pathlib import Path
from .storage import read_json,validate_id
from .coverage_ai_search import pin
from . import frozen_round_tasks

GROUP='S1-AI17-R2'
PARENT=Path('/home/u2024110224/first_stage_satellite_rl/s1_ai18')
PORTS=('20418','20424','20473')
FIELDS=('config','mode','dataset_role','case_indices','executor_port','variants','source_refs','common_ref')
MODES=('prepare_common','evaluate_closed_pool')


def deployment(store):
    root=store.archive_root.resolve();port=root.parent.name
    if (root.name!='archive' or root.parent.parent!=PARENT.resolve() or port not in PORTS or
            os.environ.get('SATELLITE_RL_SSH_PORT')!=port):raise ValueError('R2 requires independent admitted lane/archive')
    from .coverage import _require_remote_dqn
    _require_remote_dqn();return port


def task_contract(body):return {field:deepcopy(body.get(field)) for field in FIELDS}


def validate_config(config):
    expected={'schema_version','research_stage','algorithm','name','geometry','task','datasets','workers','resources'}
    if set(config)!=expected or (config['schema_version'],config['research_stage'],config['algorithm'])!=(10,'coverage','ai_search_r2'):
        raise ValueError('invalid R2 schema/config fields')
    if set(config['datasets'])!={'development'} or config['datasets']['development'].get('role')!='development':
        raise ValueError('R2 pilot cannot access training/validation/test partitions')
    if not isinstance(config['name'],str) or not 1<=len(config['name'])<=100:raise ValueError('task name')
    from satellite_rl.coverage.ai_search.config import GeometryConfig
    from satellite_rl.coverage.ai_search_r2.config import TaskConfig
    GeometryConfig(**config['geometry']);TaskConfig(**config['task'])
    if type(config['workers'])is not int or config['workers']!=1:raise ValueError('R2 paired case timing requires one case worker per physical lane')
    if set(config['resources'])!={'minimum_free_bytes','max_rss_bytes'} or any(type(v)is not int or v<=0 for v in config['resources'].values()):
        raise ValueError('explicit storage/RSS guards required')
    return deepcopy(config)


def validate_task(contract,plan,port):
    config=validate_config(contract['config'])
    design=read_json(pin(plan['design']),{})
    if design.get('group')!=GROUP or design.get('training_admitted')is not False:
        raise ValueError('exact no-training R2 science design required')
    geometry=design['geometry']
    frozen_geometry={'max_beams':geometry['max_beams'],'width_menu_deg':geometry['full_width_menu_deg'],
        'width_min_deg':geometry['width_min_deg'],'width_max_deg':geometry['width_max_deg'],
        'tolerance':geometry['squared_chord_tolerance'],'profile':'europe_v1'}
    if config['geometry']!=frozen_geometry:raise ValueError('geometry feasible set differs from pinned design')
    expected_indices=sorted(4*m+p for m in design['input']['pilot_mother_indices_zero_based'] for p in range(4))
    order=plan.get('mother_order',[])
    selected=[order[i] for i in design['input']['pilot_mother_indices_zero_based']] if len(order)==20 else []
    if (plan.get('phase')!='pilot' or len(order)!=20 or len(set(order))!=20 or
            plan.get('selected_mothers')!=selected or plan.get('selected_case_indices')!=expected_indices):
        raise ValueError('only exact frozen pilot mother/case scope is admitted; extension requires a new reviewed contract')
    tasks=list(plan.get('tasks',{}).values())
    if (len(tasks)!=3 or {str(t['executor_port']) for t in tasks}!=set(PORTS) or
            sorted(i for t in tasks for i in t['case_indices'])!=expected_indices or
            len({t['mode'] for t in tasks})!=1):raise ValueError('exact three-lane24-case pilot matrix required')
    if any(i not in expected_indices for i in contract['case_indices']):raise ValueError('unselected pilot case cannot run')
    frozen_task={'seconds':design['budget']['per_task']['nominal_guard_seconds'],
        'ray_pair_limit':design['budget']['per_task']['ray_pair_checks'],
        'geometry_limit':design['budget']['per_task']['geometry_calls'],
        'candidate_limit':design['budget']['per_task']['complete_candidates'],
        'active_limit':design['decoders']['D1']['active_cap'],'starts':design['decoders']['D1']['starts'],
        'center_iterations':design['decoders']['D1']['center_updates']}
    if config['task']!=frozen_task:raise ValueError('task budget/mechanism differs from pinned design')
    if contract['mode'] not in MODES or contract['dataset_role']!='development' or str(contract['executor_port'])!=port:
        raise ValueError('R2 mode/role/physical owner differs')
    indices=contract['case_indices'];ref=config['datasets']['development']
    if (not isinstance(indices,list) or not indices or len(set(indices))!=len(indices) or
            any(type(i)is not int or not 0<=i<ref['count'] for i in indices)):raise ValueError('unique explicit development indices required')
    if plan.get('training_allowed') is not False or plan.get('test_allowed') is not False:
        raise ValueError('R2 pilot must explicitly exclude training/test')
    if contract['mode']=='prepare_common':
        if contract['variants']!=[] or contract['common_ref'] is not None or set(contract['source_refs'])!={str(i) for i in indices}:
            raise ValueError('common preparation requires only exact R1 sources for every selected case')
        for refs in contract['source_refs'].values():
            if len(refs)!=5 or {r['method'] for r in refs}!={'07-CV','07-LEX','JOINT-CV','JOINT-LEX','POST_SHRINK'}:
                raise ValueError('five existing signed final methods required')
    else:
        if contract['variants']!=['D0','D1'] or contract['source_refs'] or not contract['common_ref']:
            raise ValueError('same common artifact and both closed-pool decoders required')
    for reference in external_references(contract).values():pin(reference)
    return config


def validate_admission(plan,plan_ref,review_ref):
    review=read_json(pin(review_ref),{})
    if (review.get('group')!=GROUP or review.get('passed')is not True or
            review.get('scientific_commit')!=plan['scientific_commit'] or
            review.get('plan_sha256')!=plan_ref['sha256'] or review.get('design_sha256')!=plan['design']['sha256']):
        raise ValueError('exact root plan/source/design admission proof is required')
    if any(t['mode']=='evaluate_closed_pool' for t in plan['tasks'].values()):
        refs=plan.get('qualification_refs',{})
        if set(refs)!={'validator_equivalence','charge_before_budget'}:raise ValueError('both source-bound engineering qualifications required')
        for reference in refs.values():
            value=read_json(pin(reference),{})
            if (value.get('passed')is not True or value.get('scientific_commit')!=plan['scientific_commit'] or
                    value.get('design_sha256')!=plan['design']['sha256'] or not value.get('core_git_blob_sha256')):
                raise ValueError('engineering qualification source/design proof differs')


def external_references(contract):
    result={}
    if contract['common_ref']:result['common_manifest']=contract['common_ref']
    for index,refs in contract['source_refs'].items():
        for i,reference in enumerate(refs):
            result['R1_case_'+index+'_'+str(i)]={'path':reference['path'],'sha256':reference['sha256']}
            result['R1_audit_'+index+'_'+str(i)]={'path':reference['audit_path'],'sha256':reference['audit_sha256']}
    return result


def expected_results(contract):return len(contract['case_indices'])*max(1,len(contract['variants']))


def resource_admission(store,job):
    port=deployment(store);route=read_json(store.archive_root.parent/'ops/resource_route.json',{})
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(route.get('observed_utc','1970-01-01T00:00:00+00:00'))).total_seconds()
    if (route.get('group')!=GROUP or str(route.get('ssh_port'))!=port or route.get('ready') is not True or
            route.get('scientific_commit')!=job['scientific_commit'] or not -5<=age<=120):
        raise ValueError('fresh same-epoch R2 route is required')
    from .ai_search_resources import observe
    observation=observe(store.archive_root)
    if observation['foreign_workers']:raise ValueError('foreign scientific worker still active')
    if observation['gpu'].split(',')[0].strip()!=route['gpu_uuid']:raise ValueError('GPU identity changed')
    from .runtime import pid_alive
    for other,status in store.jobs():
        if other['run_id']==job['run_id'] or other.get('experiment_id')!=GROUP:continue
        launch=read_json(store.job_dir(other['run_id'])/'launch.json',{})
        if status.get('stage') not in ('completed','failed','stopped','cancelled','interrupted','queued') and pid_alive(status.get('pid'),launch.get('process_start_time')):
            raise ValueError('another R2 case task owns this physical lane')
    return observation


def api(app,method,parts,body):
    import sys
    policy=sys.modules[__name__]
    if method=='GET' and parts==['schema']:
        return {'group':GROUP,'schema_version':10,'modes':MODES,'task_fields':FIELDS,'no_training_test':True,'closed_pool_only':True}
    if method=='GET' and parts==['tasks']:
        from .coverage_ai_search import observed_status
        return {'tasks':[{'job':j,'status':observed_status(app.store,j,s)} for j,s in app.store.jobs() if j.get('experiment_id')==GROUP]}
    if method=='POST' and parts==['tasks']:return frozen_round_tasks.create(app,body,policy)
    if len(parts)>=2 and parts[0]=='tasks':
        run_id=validate_id(parts[1]);job=read_json(app.store.job_dir(run_id)/'job.json',{})
        if job.get('experiment_id')!=GROUP:raise ValueError('R2 endpoint cannot operate R1')
        if method=='GET' and len(parts)==2:return app.run_detail(run_id)
        if method=='POST' and parts[2:]==['resume']:return frozen_round_tasks.resume(app.scheduler,run_id,policy)
        if method=='POST' and parts[2:]==['stop']:return app.scheduler.request_stop(run_id)
    raise FileNotFoundError('unknown R2 workbench endpoint')
