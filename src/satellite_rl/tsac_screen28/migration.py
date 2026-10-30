"""Root-accounted paid source requalification; never silently resets an epoch."""
import math
from pathlib import Path
import pickle
from satellite_rl.t0_width.data import read_json_ref,read_ref,file_sha
from satellite_rl.t0_width.logging import content_sha
from .checks import validate_catalog

EXECUTION_FIELDS={'source_commit','source_root','source_manifest','source_migration_ref',
    'source_epoch_index','requalification','prior_source_commit'}

def task_contract_sha(plan):return content_sha({k:v for k,v in plan.items() if k!='source_migration_ref'})

def validate_source_requalification(plan):
    if 'source_migration_ref' not in plan:return None
    receipt=read_json_ref(plan['source_migration_ref'])
    if (receipt.get('schema')!='W28SourceRequalificationV1' or receipt.get('experiment_id')!='S2-W28'
            or receipt.get('root_approved') is not True or receipt.get('same_task_key')!='engineering'
            or receipt.get('mode')!='paid_full_requalification' or receipt.get('new_source_commit')!=plan['source_commit']
            or receipt.get('new_task_contract_SHA')!=task_contract_sha(plan)
            or plan.get('requalification') is not True or plan.get('source_epoch_index')!=2):
        raise ValueError('root exact paid same-ID source requalification authority required')
    manifest=read_json_ref(receipt['snapshot_manifest_ref']);base=Path(receipt['snapshot_manifest_ref']['path']).parent.resolve()
    required={'owner.json','dispatch_000000.json','engineering_config.json','engineering_cursor.pkl',
        'checks/check_00_method/current.pkl','checks/check_00_method/evidence.json','checks/check_01_method/current.pkl',
        'checks/check_01_method/binding_ref.json','checks/check_01_method/setup.intent'}
    if not required<=set(manifest['files']):raise ValueError('complete old failure snapshot inventory missing')
    for rel,meta in manifest['files'].items():
        path=(base/rel).resolve()
        if not path.is_relative_to(base) or file_sha(path)!=meta['sha256'] or path.stat().st_size!=meta['bytes']:raise ValueError('protected original source-epoch snapshot differs')
    def original(name):return dict(path=str(base/name),sha256=manifest['files'][name]['sha256'])
    old=read_json_ref(original('engineering_config.json'))
    if (old['source_commit']!=receipt['old_source_commit'] or plan['prior_source_commit']!=old['source_commit']
            or content_sha(old)!=receipt['old_task_plan_SHA']
            or content_sha({k:v for k,v in old.items() if k not in EXECUTION_FIELDS})!=content_sha({k:v for k,v in plan.items() if k not in EXECUTION_FIELDS})):
        raise ValueError('requalification may change execution epoch only; fixed checks/policy/data/math differ')
    cursor=pickle.loads(read_ref(original('engineering_cursor.pkl')))
    if cursor['identity']['plan_SHA']!=content_sha(old) or cursor['cursor']!=1 or cursor['unknown']:
        raise ValueError('original exact failed cursor is not the approved failure state')
    exit_observation=read_json_ref(receipt['old_owner_exit_observation_ref'])
    if exit_observation.get('exact_owner_live') is not False or content_sha(exit_observation['owner'])!=content_sha(read_json_ref(original('owner.json'))):raise ValueError('exact old worker exit not independently proven')
    actual=dict(query=0,commit=0,forward=0,kernel_preparation=0,unknown_effects=0)
    for index in (0,1):
        prefix='checks/check_%02d_method/'%index;value=pickle.loads(read_ref(original(prefix+'current.pkl')))
        if value['identity']['plan_SHA']!=content_sha(old) or value['transaction'].get('pending') is not None:
            raise ValueError('original effect identity/unknown differs; never assume zero')
        expected=dict(query=0,commit=1 if index==0 else 0,forward=1,proposal=1)
        if value['attempts']!=expected or value['transaction']['stage']!=('committed' if index==0 else 'proposed'):raise ValueError('old known effects/failed proposal point changed')
        for key in ('query','commit','forward'):
            if value['attempts'][key]!=value['completed'][key]:raise ValueError('old effect spend unresolved')
            actual[key]+=value['attempts'][key]
        # The original reference path may point at the still-protected old run.
        binding=read_json_ref(original(prefix+'binding_ref.json'))
        member=manifest['files'][prefix+'binding.pkl']
        if binding['sha256']!=member['sha256']:raise ValueError('old initial kernel binding not proven closed')
        actual['kernel_preparation']+=1
    if actual!=dict(query=0,commit=1,forward=2,kernel_preparation=2,unknown_effects=0) or receipt.get('prior_actual_costs')!=actual:
        raise ValueError('prior actual/unknown costs differ from audited nonzero originals')
    wall=receipt.get('prior_active_wall_seconds');debit=receipt.get('prior_unrecorded_wall_upper_bound_seconds')
    if (isinstance(wall,bool) or not isinstance(wall,(int,float)) or not math.isfinite(wall) or wall!=cursor['elapsed_active_seconds']
            or isinstance(debit,bool) or not isinstance(debit,(int,float)) or not math.isfinite(debit) or debit<0
            or (debit==0 and receipt.get('prior_wall_complete_proven') is not True)):
        raise ValueError('prior wall must be retained plus an explicit unrecorded upper bound')
    bounds=validate_catalog(plan['checks'],read_json_ref(plan['fixture_manifest_ref']))
    if receipt.get('additional_worst_case')!=bounds['worst_case']:raise ValueError('paid requalification finite bound differs')
    totals={key:actual[key]+bounds['worst_case'][key] for key in ('query','commit','forward','kernel_preparation')}
    if totals['query']+totals['commit']>320 or totals['query']>256 or totals['commit']>64 or totals['forward']>64:
        raise ValueError('cumulative original engineering cap exceeded')
    return dict(receipt=receipt,prior_actual=actual,prior_wall_seconds=wall+debit,cumulative_worst_case=totals,
        old_snapshot_manifest_ref=receipt['snapshot_manifest_ref'])
