"""Durable R2 source-only common preparation and closed-pool CPU probe."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,multiprocessing,os
from pathlib import Path
import shutil,sys,threading,time,traceback
from satellite_rl.training.run_monitor import atomic_json
from .storage import digest,read_json,clean_json,Store,now
from .coverage_ai_search import pin
from .ai_search_runner import load_cases,result_key,recover_rows
from .geoload_runner import exclusive_lock
from . import coverage_ai_search_r2 as policy,frozen_round_tasks


def resource_guard(directory,resources):
    if shutil.disk_usage(directory).free<resources['minimum_free_bytes']:raise OSError('R2 disk reserve reached')
    if os.name!='nt':
        rss=next(int(x.split()[1])*1024 for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:'))
        if rss>resources['max_rss_bytes']:raise MemoryError('R2 RSS guard reached')


def abort_pending(directory,futures,error,completed_rows):
    """Persist the stop boundary before touching queued futures."""
    atomic_json(Path(directory)/'abort.json',{'reason':str(error),'at':now(),'completed_rows':completed_rows,
        'unknown_active_work_preserved':True,'queued_case_cancellation_requested':True})
    atomic_json(Path(directory)/'stop.request',{'reason':'hard worker error','at':now()})
    for queued in futures:queued.cancel()


def production_provenance_metadata(job,plan):
    """Wire shared by the actual result producer and independent contract checks."""
    return {'phase':plan['phase'],'mother_order':plan['mother_order'],
        'execution_epoch':job['scientific_commit'],'case_indices':job['case_indices'],
        'algorithm_core_commit':'f02c0dcbf76561c325b5a70a1a5b9bf98882b35d','execution_commit':job['scientific_commit'],
        'admission_ref':job['references'].get('extension_root_gate'),
        'admission_evidence_refs':{k.removeprefix('extension_'):v for k,v in job['references'].items() if k.startswith('extension_')},
        'scientific_core_sha256':plan.get('scientific_core_sha256'),
        'scientific_dependency_sha256':plan.get('scientific_dependency_sha256')}


def one(case,variant,job,config,output):
    if (Path(output).parent/'stop.request').exists():return []
    from satellite_rl.coverage.ai_search.core import Scene
    from satellite_rl.coverage.ai_search.config import GeometryConfig
    from satellite_rl.coverage.ai_search_r2.initials import CommonStart,freeze_common
    from satellite_rl.coverage.ai_search_r2.probe import evaluate_closed_pool
    from satellite_rl.coverage.ai_search_r2.config import TaskConfig
    directory=Path(output).parent;key=result_key(case['case_id'],variant)
    atomic_json(directory/'started'/(key+'.json'),{'case_id':case['case_id'],'method':variant,'started_at':now(),
        'mode':job['mode'],'input_sha256':case['input_dataset_sha256'],'online_solve_expected':job['mode']=='evaluate_closed_pool'})
    latest_check=[0.]
    def stop():
        at=time.monotonic()
        if at-latest_check[0]>=.25:resource_guard(directory,config['resources']);latest_check[0]=at
        return (directory/'stop.request').exists()
    scene=Scene(case['positions'],case['demands'],case['aircraft_ids'],case['profile']);start=time.perf_counter()
    contract=job['original_task_contract']
    if job['mode']=='prepare_common':
        result=freeze_common(scene,contract['source_refs'][str(case['input_case_index'])],GeometryConfig(**config['geometry'])).to_record()
        common_path=directory/'common'/(key+'.json');atomic_json(common_path,result)
        value={'schema':'s1_ai18_prepared_common_case_v1','common':{'path':str(common_path),'sha256':digest(common_path)},
            'input_sha':result['input_sha'],'archive_sha':result['archive_sha'],'parent_sha':result['parent_sha'],
            'acquisition_cost_refs':result['acquisition_cost_refs'],'preparation_seconds':time.perf_counter()-start,
            'new_R1_solver_calls':0}
    else:
        reference=job['references']['common_manifest'];manifest=read_json(pin(reference),{})
        ref=manifest['cases'].get(str(case['input_case_index']))
        if not ref or ref['case_id']!=case['case_id']:raise ValueError('prepared common case/input identity differs')
        common=CommonStart.from_record(read_json(pin(ref['common']),{}))
        if common.input_sha!=scene.input_sha256:raise ValueError('prepared common input SHA differs')
        value=evaluate_closed_pool(scene,common,variant,config=TaskConfig(**config['task']),stop=stop)
        if value.get('training_updates')!=0:raise ValueError('R2 closed-pool cannot train')
    row={'case_id':case['case_id'],'mother_id':case['mother_id'],'demand_profile':case['demand_profile'],
        'input_case_index':case['input_case_index'],'input_sha256':scene.input_sha256,
        'input_dataset_sha256':case['input_dataset_sha256'],'method':variant,'mode':job['mode'],
        'outcome':'stopped' if stop() else 'completed','result':value,'end_to_end_seconds':time.perf_counter()-start}
    plan=read_json(pin(job['references']['plan']),{})
    row.update(phase=plan['phase'],algorithm_core_commit='f02c0dcbf76561c325b5a70a1a5b9bf98882b35d',execution_commit=job['scientific_commit'])
    path=Path(output)/(key+'.json');atomic_json(path,clean_json(row))
    return key,{'case_id':case['case_id'],'method':variant,'path':str(path),'sha256':digest(path),'outcome':row['outcome']}


def paired(case,job,config,output,already_complete):
    """One actual prepared pool, one immutable parent, both decoder orders."""
    if (Path(output).parent/'stop.request').exists():return []
    from satellite_rl.coverage.ai_search.core import Scene
    from satellite_rl.coverage.ai_search_r2.initials import CommonStart
    from satellite_rl.coverage.ai_search_r2.probe import prepare_pool,evaluate_closed_pool
    from satellite_rl.coverage.ai_search_r2.config import TaskConfig
    directory=Path(output).parent;scene=Scene(case['positions'],case['demands'],case['aircraft_ids'],case['profile'])
    manifest=read_json(pin(job['references']['common_manifest']),{})
    selected=manifest['cases'].get(str(case['input_case_index']))
    if not selected or selected['case_id']!=case['case_id']:raise ValueError('prepared common case differs')
    common=CommonStart.from_record(read_json(pin(selected['common']),{}))
    if common.input_sha!=scene.input_sha256:raise ValueError('common scene SHA differs')
    latest=[0.]
    def stop():
        at=time.monotonic()
        if at-latest[0]>=.25:resource_guard(directory,config['resources']);latest[0]=at
        return (directory/'stop.request').exists()
    pool_key=hashlib.sha256(case['case_id'].encode()).hexdigest();pool_path=directory/'pools'/(pool_key+'.json')
    pool_started=directory/'pool_started'/(pool_key+'.json')
    if pool_path.exists():
        prepared=read_json(pool_path,{})
        history=read_json(pool_started,{})
        if (history.get('case_id')!=case['case_id'] or history.get('input_sha256')!=scene.input_sha256
                or history.get('archive_sha')!=common.archive_sha or history.get('parent_sha')!=common.parent_sha
                or history.get('config')!=config['task']):
            raise ValueError('prepared pool original start/source/config differs or is missing')
        ref=history.get('artifact')
        if ref and (ref['path']!=str(pool_path) or digest(pool_path)!=ref['sha256']):
            raise ValueError('completed prepared pool artifact changed')
    else:
        if pool_started.exists():
            atomic_json(directory/'unknown_budget_pool.json',{'case_id':case['case_id'],
                'started_ref':{'path':str(pool_started),'sha256':digest(pool_started)},
                'automatic_reprepare_forbidden':True,'source':'interrupted shared preparation; unknown work retained'})
            raise ValueError('shared pool preparation has unknown spent work; no automatic repeat')
        atomic_json(pool_started,{'case_id':case['case_id'],'input_sha256':scene.input_sha256,
            'archive_sha':common.archive_sha,'parent_sha':common.parent_sha,'config':config['task'],
            'started_at':now(),'pid':os.getpid(),'status':'started_before_any_shared_preparation'})
        prepared=prepare_pool(scene,common,config=TaskConfig(**config['task']),stop=stop)
        atomic_json(pool_path,clean_json(prepared))
    record=read_json(pool_started,{})
    record.update(status='artifact_bound',artifact={'path':str(pool_path),'sha256':digest(pool_path)},
        known_preparation_status=prepared.get('status','completed'),unknown_work=False)
    atomic_json(pool_started,record)
    prepared_file_sha=digest(pool_path);common_bytes=common.frozen_bytes
    order=job['variants'] if case['input_case_index']%2==0 else list(reversed(job['variants']))
    records=[]
    for ordinal,variant in enumerate(order):
        key=result_key(case['case_id'],variant)
        if key in already_complete:continue
        atomic_json(directory/'started'/(key+'.json'),{'case_id':case['case_id'],'method':variant,
            'mode':job['mode'],'started_at':now(),'prepared_pool_ref':{'path':str(pool_path),'sha256':prepared_file_sha}})
        began=time.perf_counter()
        value=evaluate_closed_pool(scene,common,variant,config=TaskConfig(**config['task']),stop=stop,prepared_pool=prepared)
        if digest(pool_path)!=prepared_file_sha or common.frozen_bytes!=common_bytes:
            raise ValueError('common/pool bytes changed across paired decoder')
        if value['pool_sha']!=prepared['pool_sha'] or value['input_sha']!=common.input_sha or value['parent_sha']!=common.parent_sha:
            raise ValueError('paired decoder source/pool hashes differ')
        row={'case_id':case['case_id'],'mother_id':case['mother_id'],'demand_profile':case['demand_profile'],
            'input_case_index':case['input_case_index'],'input_sha256':scene.input_sha256,
            'input_dataset_sha256':case['input_dataset_sha256'],'method':variant,'mode':job['mode'],
            'outcome':'stopped' if stop() else 'completed','result':value,'end_to_end_seconds':time.perf_counter()-began,
            'common_ref':selected['common'],'prepared_pool_ref':{'path':str(pool_path),'sha256':prepared_file_sha},
            'paired_order':order,'paired_order_position':ordinal,'shared_pool_prepared_once':True,
            'pool_and_parent_sha_verified_before_after':True}
        plan=read_json(pin(job['references']['plan']),{})
        row.update(phase=plan['phase'],algorithm_core_commit='f02c0dcbf76561c325b5a70a1a5b9bf98882b35d',execution_commit=job['scientific_commit'])
        destination=Path(output)/(key+'.json');atomic_json(destination,clean_json(row))
        records.append((key,{'case_id':case['case_id'],'method':variant,'path':str(destination),
                             'sha256':digest(destination),'outcome':row['outcome']}))
        if row['outcome']=='stopped':break
    return records


def run_job(directory,resume=False):
    directory=Path(directory).resolve();job=read_json(directory/'job.json',{});config=read_json(directory/'config.json',{})
    if job.get('experiment_id')!=policy.GROUP:raise ValueError('wrong R2 group')
    status=read_json(directory/'status.json',{});mutex=threading.RLock();done=threading.Event()
    def update(**fields):
        with mutex:status.update(fields,pid=os.getpid(),heartbeat_at=now());atomic_json(directory/'status.json',status)
    def heartbeat():
        while not done.wait(10):update()
    try:
        with exclusive_lock(directory/'worker.lock'),exclusive_lock(Path(job['archive_root']).parent/'ops/compute.lock',shared=False,blocking=True):
            store=Store(job['archive_root'],source_root=job['source_root']);frozen_round_tasks.verify(store,job,policy)
            from .runner import strict_cuda_preflight
            atomic_json(directory/'preflight.json',strict_cuda_preflight(job))
            if any(os.environ.get(key)!='1' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')):
                raise ValueError('R2 formal CPU timing requires all native thread controls frozen to1')
            resource_guard(directory,config['resources']);update(stage='running',error=None)
            threading.Thread(target=heartbeat,daemon=True).start()
            cases=load_cases(config['datasets']['development'],job['case_indices']);variants=job['variants'] or ['COMMON']
            output=directory/'cases';output.mkdir(exist_ok=True)
            expected={result_key(case['case_id'],variant) for case in cases for variant in variants}
            index=recover_rows(directory,expected)
            pending=[(case,variant) for case in cases for variant in variants if result_key(case['case_id'],variant) not in index]
            update(total_cases=len(expected),completed_cases=len(index),case_workers=config['workers'],science_updates=0)
            with ProcessPoolExecutor(max_workers=config['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
                if job['mode']=='evaluate_closed_pool':
                    pending_cases=[case for case in cases if any(result_key(case['case_id'],v) not in index for v in variants)]
                    futures={pool.submit(paired,case,job,config,str(output),set(index)):case for case in pending_cases}
                else:futures={pool.submit(one,case,variant,job,config,str(output)):(case,variant) for case,variant in pending}
                for future in as_completed(futures):
                    if future.cancelled():continue
                    try:returned=future.result()
                    except BaseException as error:
                        abort_pending(directory,futures,error,len(index))
                        raise
                    returned=returned if isinstance(returned,list) else [returned]
                    for key,ref in returned:index[key]=ref
                    atomic_json(directory/'case_index.json',index);update(completed_cases=len(index))
                    if (directory/'stop.request').exists():
                        for f in futures:f.cancel()
            if (directory/'stop.request').exists():update(stage='stopped');return
            if len(index)!=len(expected):raise ValueError('R2 frozen matrix incomplete')
            artifacts={'group':policy.GROUP,'completed':True,'count':len(index),'case_index':{'path':str(directory/'case_index.json'),'sha256':digest(directory/'case_index.json')},'science_updates':0,'scientific_commit':job['scientific_commit']}
            if job['mode']=='prepare_common':
                common_cases={}
                for ref in index.values():
                    row=read_json(pin(ref),{});record=row['result'];pin(record['common'])
                    common_cases[str(row['input_case_index'])]={'case_id':row['case_id'],'mother_id':row['mother_id'],
                        'profile':row['demand_profile'],**{k:record[k] for k in ('common','input_sha','archive_sha','parent_sha','acquisition_cost_refs')}}
                path=directory/'common_manifest.json';atomic_json(path,{'group':policy.GROUP,'schema':'s1_ai18_common_collection_v1','completed':True,
                    'dataset_sha256':config['datasets']['development']['sha256'],'cases':common_cases,'scientific_commit':job['scientific_commit'],'source_only_no_R1_resolve':True})
                plan=read_json(pin(job['references']['plan']),{})
                common_manifest=read_json(path,{})
                common_manifest.update(phase=plan['phase'],execution_epoch=job['scientific_commit'],case_indices=job['case_indices'],
                    algorithm_core_commit='f02c0dcbf76561c325b5a70a1a5b9bf98882b35d',execution_commit=job['scientific_commit'],
                    admission_ref=job['references'].get('extension_root_gate'),scientific_core_sha256=plan.get('scientific_core_sha256'))
                atomic_json(path,common_manifest)
                artifacts['common_manifest']={'path':str(path),'sha256':digest(path)}
            else:
                manifest_cases={}
                for reference in index.values():
                    row=read_json(pin(reference),{});case_key=str(row['input_case_index'])
                    record=manifest_cases.setdefault(case_key,{'case_id':row['case_id'],'mother_id':row['mother_id'],
                        'profile':row['demand_profile'],'role':'development','scene_ref':{'packed':config['datasets']['development'],'case_index':row['input_case_index']},
                        'input_sha256':row['input_sha256'],'common_ref':row['common_ref'],'pool_ref':row['prepared_pool_ref'],'raw_refs':{}})
                    record['raw_refs'][row['method']]={'path':reference['path'],'sha256':reference['sha256']}
                    record.update(phase=row['phase'],algorithm_core_commit=row['algorithm_core_commit'],execution_commit=row['execution_commit'])
                path=directory/'result_manifest.json'
                plan=read_json(pin(job['references']['plan']),{})
                atomic_json(path,{'group':policy.GROUP,'schema':'s1_ai18_paired_case_manifest_v1','cases':list(manifest_cases.values()),
                    **production_provenance_metadata(job,plan),
                    'qualification_refs':{k:job['references']['qualification_'+k] for k in plan.get('qualification_refs',{})},
                    'dataset_role':'development','source_commit':job['scientific_commit'],
                    'job_ref':{'path':str(directory/'job.json'),'sha256':digest(directory/'job.json')},
                    'config_ref':{'path':str(directory/'config.json'),'sha256':digest(directory/'config.json')},
                    'preflight_ref':{'path':str(directory/'preflight.json'),'sha256':digest(directory/'preflight.json')},
                    'source_manifest_ref':{'path':str(Path(job['source_root'])/'source_manifest.json'),'sha256':job['source_manifest_sha256']},
                    'complete_case_method_rows':len(index),'solver_task_slots':len(manifest_cases)*16,'training_updates':0})
                artifacts['result_manifest']={'path':str(path),'sha256':digest(path)}
            atomic_json(directory/'artifacts.json',artifacts);update(stage='completed',completed_at=now())
    except BaseException as error:
        atomic_json(directory/'failure.json',{'group':policy.GROUP,'error':str(error),'traceback':traceback.format_exc(),'at':now(),'evidence_retained':True})
        update(stage='failed',error=str(error));raise
    finally:done.set()
