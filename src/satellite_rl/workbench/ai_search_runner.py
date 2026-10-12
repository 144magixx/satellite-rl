"""Remote-only finite-case AI17 worker with pinned source and durable rows."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import threading
import time
import traceback
import numpy as np

from satellite_rl.training.run_monitor import atomic_json
from .storage import clean_json, digest, now, read_json
from .coverage_ai_search import GROUP, pin, validate_config
from .geoload_runner import exclusive_lock

_MODEL_CACHE = {}


def cached_orderer(reference, config, diagnostic=False):
    from satellite_rl.coverage.ai_search.learning import load_orderer
    key=(reference['path'],reference['sha256'],'cuda',diagnostic,json.dumps(config['geometry'],sort_keys=True),
         config['search'].get('variants'))
    if key not in _MODEL_CACHE:
        callback,seconds=load_orderer(reference,device='cuda',allow_intermediate=diagnostic,
            geometry=config['geometry'],variant_count=config['search'].get('variants',8))
        _MODEL_CACHE.clear();_MODEL_CACHE[key]=callback
        return callback,seconds
    return _MODEL_CACHE[key],0.


def result_key(case_id, method):
    return hashlib.sha256((case_id+'\0'+method).encode()).hexdigest()


def random_seed(case_id):
    payload=json.dumps([51001,str(case_id),'random'],ensure_ascii=False,separators=(',',':'))
    sha=hashlib.sha256(payload.encode('utf-8')).hexdigest()
    return int(sha[:16],16),{'training_seed':51001,'basis_utf8_json':payload,
                           'basis_sha256':sha,'rule':'first16hex_SHA256_of_JSON([51001,case_id,random])'}


def load_cases(reference, indices):
    """Only position/request/identity arrays; never import legacy teacher/layout metadata."""
    path = pin(reference)
    if path.stat().st_size > 256 * (1 << 20): raise ValueError('packed input exceeds declared lightweight supply bound')
    with np.load(path, allow_pickle=False) as values:
        required = {'positions','demands','aircraft_ids','offsets','case_ids','mother_ids','demand_profiles'}
        if not required.issubset(values.files): raise ValueError('raw position/request/ID fields missing')
        positions=values['positions'];demands=values['demands'];aircraft=values['aircraft_ids']
        offsets=values['offsets'];case_ids=values['case_ids'];mothers=values['mother_ids'];profiles=values['demand_profiles']
        count=len(case_ids)
        if (count != reference['count'] or len(set(case_ids.tolist())) != count or len(mothers) != count or
                len(profiles) != count or offsets.shape != (count+1,) or offsets[0] != 0 or
                offsets[-1] != len(demands) or (np.diff(offsets)<0).any() or
                positions.shape != (len(demands),2) or len(aircraft) != len(demands)):
            raise ValueError('packed input identity/count/dimensions invalid')
        rows=[]
        for i in indices:
            a,b=int(offsets[i]),int(offsets[i+1])
            rows.append({'case_id':str(case_ids[i]),'mother_id':str(mothers[i]),
                'demand_profile':str(profiles[i]),'positions':positions[a:b].copy(),
                'demands':demands[a:b].copy(),'aircraft_ids':tuple(map(str,aircraft[a:b])),
                'input_dataset_sha256':reference['sha256'],
                'profile':'europe_v1','input_case_index':int(i)})
    return rows


def resource_check(directory, resources):
    directory=Path(directory)
    if shutil.disk_usage(directory).free < resources['min_free_disk_bytes']:
        raise RuntimeError('disk reserve guard reached; data retained')
    if os.name != 'nt':
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:') and int(line.split()[1])*1024 > resources['max_rss_bytes']:
                raise RuntimeError('RSS guard reached; data retained')


def run_case(case, method, config, mode, output_root, model_reference=None):
    """Write each case atomically in its worker; interruption cannot erase prior cases."""
    from satellite_rl.coverage.ai_search.config import GeometryConfig, SearchConfig
    from satellite_rl.coverage.ai_search.core import Scene
    from satellite_rl.coverage.ai_search.search import run_search
    directory=Path(output_root).parent; output=Path(output_root)
    key=result_key(case['case_id'],method); started=directory/'started'/(key+'.json')
    began=time.perf_counter()
    atomic_json(started,{'case_id':case['case_id'],'method':method,'pid':os.getpid(),
        'started_at':now(),'search_limit':config['search'],'status':'started'})
    def stopped():
        resource_check(directory,config['resources'])
        return (directory/'stop.request').exists()
    geometry=GeometryConfig(**config['geometry']);search=SearchConfig(**config['search'])
    identity_scene=Scene(case['positions'],case['demands'],case['aircraft_ids'],case['profile'])
    cold_seconds=0.;seed_evidence=None
    if mode == 'teacher':
        from satellite_rl.coverage.ai_search.teacher import teacher_case
        teacher=teacher_case(case,directory/'teacher'/key,config=search,geometry=geometry,stop=stopped)
        # Canonical compact teacher manifest/evidence stay once in the teacher directory.
        record={field:teacher.get(field) for field in ('status','manifest','budget','states','candidates','censored_rows','stop_reason')}
        if teacher.get('manifest'):pin(teacher['manifest'])
    elif mode == 'bridge':
        from satellite_rl.coverage.geoload.core import evaluate_case
        objective='legacyCV' if method.endswith('-CV') else 'lex'
        # These are the old algorithms in their own fixed-.5/N domain.
        bridge_search={'seconds':search.seconds,'max_attempts':search.max_attempts,
            'max_geometry_calls':search.max_geometry_calls,'objective':objective}
        record=evaluate_case(case,'07' if method.startswith('07-') else 'JOINT',bridge_search)
        record['domain']='legacy_same_N_full_width_0.5'
    else:
        scene=identity_scene
        orderer=None
        if method == 'RANDOM':
            seed,seed_evidence=random_seed(case['case_id']);seed_evidence['derived_seed']=seed
            rng=np.random.default_rng(seed)
            def orderer(scene,current,pool):
                return [pool[i] for i in rng.permutation(len(pool))]
        elif method == 'AI':
            orderer,cold_seconds=cached_orderer(model_reference,config,diagnostic=mode=='diagnostic_validation')
        record=run_search(scene,initial=None,config=search,geometry=geometry,
                          method=method,orderer=orderer,stop=stopped).to_record()
    row={'case_id':case['case_id'],'mother_id':case['mother_id'],'demand_profile':case['demand_profile'],
         'input_case_index':case['input_case_index'],'method':method,'mode':mode,
         'input_sha256':identity_scene.input_sha256,
         'input_dataset_sha256':case['input_dataset_sha256'],
         'aircraft_count':len(case['demands']),'total_requested_bps':float(np.asarray(case['demands']).sum()),
         'outcome':'stopped' if (directory/'stop.request').exists() else 'completed',
         'cold_model_load_seconds':cold_seconds,'end_to_end_seconds':time.perf_counter()-began,
         'finished_at':now(),'result':record}
    if seed_evidence:row['random_seed_evidence']=seed_evidence
    if mode=='teacher' and record.get('status')=='stopped':row['outcome']='stopped'
    resource_check(directory,config['resources'])
    destination=output/(key+'.json');atomic_json(destination,clean_json(row))
    return key,{'case_id':case['case_id'],'method':method,'path':str(destination),
                'sha256':digest(destination),'outcome':row['outcome']}


def recover_rows(directory, expected_keys):
    """Recover an atomic result written just before its parent index update."""
    directory=Path(directory);index=read_json(directory/'case_index.json',{})
    for key,ref in index.items():
        if key not in expected_keys or digest(directory/'cases'/(key+'.json')) != ref['sha256']:
            raise ValueError('completed row identity/SHA changed')
    for path in (directory/'cases').glob('*.json'):
        if path.stem in index:continue
        row=read_json(path,{})
        if path.stem not in expected_keys or result_key(row.get('case_id',''),row.get('method','')) != path.stem:
            raise ValueError('unindexed result identity differs')
        index[path.stem]={'case_id':row['case_id'],'method':row['method'],'path':str(path),
                         'sha256':digest(path),'outcome':row['outcome'],'recovered_atomic_tail':True}
    unknown=[p.name for p in (directory/'started').glob('*.json') if p.stem not in index]
    if unknown:
        atomic_json(directory/'unknown_budget.json',{'category':'interrupted_case_budget_unknown',
            'case_records':unknown,'automatic_repeat_forbidden':True,
            'instruction':'retain source/budget and audit before same-ID continuation; no silent repeat'})
        raise ValueError('interrupted case has unknown spent budget; audit required')
    atomic_json(directory/'case_index.json',index)
    return index


def train_job(directory,job,config,update,resume):
    from satellite_rl.coverage.ai_search.config import GeometryConfig,SearchConfig
    from satellite_rl.coverage.ai_search.core import Scene
    from satellite_rl.coverage.ai_search.search import run_search
    from satellite_rl.coverage.ai_search.learning import train,orderer
    spec=job['training_spec'];geometry=GeometryConfig(**config['geometry']);search=SearchConfig(**config['search'])
    cases=load_cases(config['datasets']['validation'],spec['validation_case_indices'])
    def stopped():
        resource_check(directory,config['resources'])
        return (directory/'stop.request').exists()
    def validation(model,normalizer,cursor,path):
        if spec.get('validation_dispatch')=='workbench_async':
            root=Path(job['model_dir']);source=root/'checkpoint.pt';target=root/'diagnostic_nodes'/str(cursor)/'node_model.pt'
            target.parent.mkdir(parents=True,exist_ok=True)
            sha=digest(source)
            if target.exists():
                if digest(target)!=sha:raise ValueError('immutable diagnostic model checkpoint differs')
            else:
                temporary=target.with_suffix('.part');shutil.copyfile(source,temporary)
                if digest(temporary)!=sha:raise ValueError('diagnostic model copy SHA differs')
                temporary.replace(target)
            ready={'group':GROUP,'status':'requested','producer_run_id':job['run_id'],
                'producer_task_key':job['task_key'],'update_step':cursor,'scientific_commit':job['scientific_commit'],
                'producer_plan':job['references']['plan'],'model':{'path':str(target),'sha256':sha},
                'training_dataset_sha256':config['datasets']['train']['sha256'],
                'validation_dataset_sha256':config['datasets']['validation']['sha256'],
                'mode':'diagnostic_validation' if cursor<10000 else 'validation',
                'timing':'training_overlap_non_strict' if cursor<10000 else 'final_seven_method_common_interleaved',
                'additional_AI_10000_diagnostic_allowed':False,'created_at':now()}
            ready_path=target.parent/'node_ready.json'
            if ready_path.exists():
                previous=read_json(ready_path,{})
                if previous.get('model')!=ready['model'] or previous.get('producer_run_id')!=job['run_id']:
                    raise ValueError('original diagnostic request changed')
                ready=previous
            else:atomic_json(ready_path,ready)
            update(update_step=cursor,diagnostic_node_requested=cursor)
            return {'status':'requested','path':str(ready_path),'sha256':digest(ready_path)}
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        callback=orderer('AI',model,normalizer,device='cuda',geometry=geometry,variant_count=search.variants)
        expected={result_key(case['case_id'],'AI') for case in cases}
        output=path/'cases';output.mkdir(exist_ok=True)
        index=read_json(path/'case_index.json',{})
        for case in cases:
            key=result_key(case['case_id'],'AI')
            if key in index:
                if digest(output/(key+'.json'))!=index[key]['sha256']:raise ValueError('validation row changed')
                continue
            started=path/'started'/(key+'.json')
            destination=output/(key+'.json')
            if started.exists() and not destination.exists():
                raise ValueError('interrupted validation has unknown candidate budget; no silent repeat')
            if destination.exists():
                row=read_json(destination,{})
                if row.get('case_id')!=case['case_id'] or row.get('update_step')!=cursor:
                    raise ValueError('validation atomic tail identity differs')
            else:
                if stopped():raise InterruptedError('stop before validation case; checkpoint retained')
                atomic_json(started,{'case_id':case['case_id'],'update_step':cursor,'at':now()})
                scene=Scene(case['positions'],case['demands'],case['aircraft_ids'],case['profile'])
                began=time.perf_counter()
                result=run_search(scene,initial=None,config=search,geometry=geometry,
                                  method='AI',orderer=callback,stop=stopped).to_record()
                row={'case_id':case['case_id'],'mother_id':case['mother_id'],'demand_profile':case['demand_profile'],
                     'input_sha256':scene.input_sha256,'input_dataset_sha256':config['datasets']['validation']['sha256'],
                     'aircraft_count':len(case['demands']),'total_requested_bps':float(np.asarray(case['demands']).sum()),
                     'update_step':cursor,'method':'AI','result':result,'end_to_end_seconds':time.perf_counter()-began,
                     'outcome':'stopped' if stopped() else 'completed'}
                atomic_json(destination,clean_json(row))
            index[key]={'path':str(destination),'sha256':digest(destination),'case_id':case['case_id']}
            atomic_json(path/'case_index.json',index)
            update(update_step=cursor,validation_completed_cases=len(index),validation_total_cases=len(expected))
            if stopped():raise InterruptedError('stop after committed validation case; checkpoint retained')
        if set(index)!=expected:raise ValueError('validation matrix identity differs')
        atomic_json(path/'artifacts.json',{'completed':True,'count':len(index),'update_step':cursor,
            'dataset_sha256':config['datasets']['validation']['sha256'],
            'index':{'path':str(path/'case_index.json'),'sha256':digest(path/'case_index.json')}})
    result=train(job['references']['teacher_manifest']['path'],Path(job['model_dir']),spec['protocol'],
        device='cuda',resume=resume,validation=validation,stop=stopped,
        progress=lambda cursor,loss:update(stage='training',update_step=cursor,loss=loss,budget_updates=10000))
    artifacts={'group':GROUP,'training':result,'scientific_commit':job['scientific_commit'],
               'teacher_manifest':job['references']['teacher_manifest'],'validation':config['datasets']['validation']}
    if result.get('status')=='completed':
        final=Path(job['model_dir'])/'final.pt'
        artifacts.update(completed=True,model={'path':str(final),'sha256':digest(final)},updates=result['update'])
        update(stage='completed',update_step=result['update'],completed_at=now())
    else:update(stage='stopped',update_step=result.get('update'),stopped_at=now())
    atomic_json(directory/'artifacts.json',artifacts)


def run_job(directory, *, resume=False):
    directory=Path(directory).resolve();job=read_json(directory/'job.json',{})
    if job.get('experiment_id') != GROUP:raise ValueError('wrong AI17 worker group')
    config=validate_config(read_json(directory/'config.json',{}))
    if digest(directory/'config.json') != job['config_sha256']:raise ValueError('published configuration SHA changed')
    status=read_json(directory/'status.json',{});guard=threading.RLock();quit_heartbeat=threading.Event()
    def update(**fields):
        with guard:
            status.update(fields,pid=os.getpid(),heartbeat_at=now());atomic_json(directory/'status.json',status)
    def heartbeat():
        while not quit_heartbeat.wait(10):update()
    try:
        with exclusive_lock(directory/'worker.lock'), exclusive_lock(Path(job['archive_root']).parent/'ops/compute.lock',
                shared=job['mode'] not in ('validation','test','bridge','train','diagnostic_validation'),blocking=True):
            from .runner import strict_cuda_preflight
            atomic_json(directory/'preflight.json',strict_cuda_preflight(job))
            from .coverage_ai_search import verify_job
            from .storage import Store
            verify_job(Store(job['archive_root'], source_root=job['source_root']),job)
            resource_check(directory,config['resources'])
            update(stage='running',error=None,started_at=status.get('started_at',now()),resume=resume)
            threading.Thread(target=heartbeat,daemon=True).start()
            if job['mode']=='train':
                return train_job(directory,job,config,update,resume)
            if job['mode']=='dataset':
                from satellite_rl.coverage.geoload.dataset import prepare_dataset
                def stopped():
                    resource_check(directory,config['resources'])
                    return (directory/'stop.request').exists()
                record=prepare_dataset(job['dataset_spec'],directory/'prepared_dataset',job['source_root'],
                                       progress=update,stopped=stopped)
                atomic_json(directory/'artifacts.json',{'group':GROUP,'dataset':record,
                    'scientific_commit':job['scientific_commit'],'source_boundary':job['source_boundary']})
                if record.get('status')=='prepared':update(stage='completed',completed_at=now())
                elif record.get('status')=='stopped':update(stage='stopped',stopped_at=now())
                else:raise ValueError('official data unavailable; retained preparation evidence')
                return
            cases=load_cases(config['datasets'][job['dataset_role']],job['case_indices'])
            if job['mode']=='teacher':
                mother_map={mother:i for i,mother in enumerate(job['teacher_mother_order'])}
                for case in cases:
                    if case['mother_id'] not in mother_map:raise ValueError('teacher case absent from frozen selected mother order')
                    case['mother_index']=mother_map[case['mother_id']]
            output=directory/'cases';output.mkdir(exist_ok=True)
            expected={result_key(case['case_id'],method) for case in cases for method in job['methods']}
            index=recover_rows(directory,expected)
            pending=[]
            for case in cases:
                methods=job['methods'];shift=case['input_case_index'] % len(methods)
                rotation=methods[shift:]+methods[:shift]
                for method in rotation:
                    if result_key(case['case_id'],method) not in index:pending.append((case,method))
            case_workers=1 if job['mode'] in ('test','validation','bridge','diagnostic_validation') else config['workers']
            update(total_cases=len(expected),completed_cases=len(index),case_workers=case_workers,
                   execution_device=job['execution_device'])
            # Spawn avoids inherited CUDA/RNG state. Case tasks are bounded and share input provenance.
            with ProcessPoolExecutor(max_workers=case_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures={pool.submit(run_case,c,m,config,job['mode'],str(output),job['references'].get('model')):(c,m) for c,m in pending}
                for future in as_completed(futures):
                    if future.cancelled():continue
                    key,ref=future.result();index[key]=ref
                    atomic_json(directory/'case_index.json',index);update(completed_cases=len(index))
                    if (directory/'stop.request').exists():
                        for item in futures:item.cancel()
            if (directory/'stop.request').exists():update(stage='stopped',stopped_at=now());return
            if len(index)!=len(expected):raise ValueError('finite case matrix incomplete')
            atomic_json(directory/'artifacts.json',{'group':GROUP,'completed':True,'count':len(index),
                'case_index':{'path':str(directory/'case_index.json'),'sha256':digest(directory/'case_index.json')},
                'dataset':config['datasets'][job['dataset_role']],'source_boundary':job['source_boundary'],
                'scientific_commit':job['scientific_commit'],'stopped_case_count':sum(x['outcome']=='stopped' for x in index.values())})
            if job['mode']=='teacher':
                refs=[]
                for entry in index.values():
                    row=read_json(entry['path'],{});ref=row.get('result',{}).get('manifest')
                    if not ref:raise ValueError('teacher case lacks canonical compact evidence manifest')
                    pin(ref);refs.append(ref)
                teacher_manifest=directory/'teacher_manifest.json'
                atomic_json(teacher_manifest,{'group':GROUP,'status':'completed','role':'training',
                    'cases':refs,'count':len(refs),'dataset_sha256':config['datasets']['train']['sha256'],
                    'scientific_commit':job['scientific_commit'],'source_boundary':job['source_boundary']})
                artifacts=read_json(directory/'artifacts.json',{})
                artifacts['teacher_manifest']={'path':str(teacher_manifest),'sha256':digest(teacher_manifest)}
                atomic_json(directory/'artifacts.json',artifacts)
            update(stage='completed',completed_at=now())
    except InterruptedError as error:
        update(stage='stopped',stopped_at=now(),stop_reason=str(error))
    except BaseException as error:
        atomic_json(directory/'failure.json',{'run_id':job.get('run_id'),'at':now(),'error':str(error),
            'traceback':traceback.format_exc(),'last_status':status,'scientific_failure':False})
        update(stage='failed',attention_required=True,error=str(error));raise
    finally:quit_heartbeat.set()
