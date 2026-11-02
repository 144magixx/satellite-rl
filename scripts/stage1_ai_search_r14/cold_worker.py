"""One cold request: own case/B/cache reads, online audits, SH-C, durable output.

No future-teacher input is accepted. Offline pairing is a separate stage.
"""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
from .lane import pinned
from .runtime import atomic_bytes, json_bytes, exclusive, require_execution, ObjectStore
from .bindings import read_json,bind_audit_metadata,bind_sealed_B,verify_loaded_source
from .stop_control import child_stop
from .kernel_persistence import call_saved


def execute(request):
    require_execution(request)
    if request.get('phase')=='data_qualification':
        from .data_worker import execute as execute_data
        return execute_data(request)
    if request.get('phase')=='B':
        from .B_worker import execute as execute_B
        return execute_B(request)
    if request.get('phase') in ('fit_baseline','calibration_veto'):
        from .fit_worker import execute as execute_fit
        return execute_fit(request)
    allowed = {'AI','strong_causal_baseline','full_SHC'}
    if request.get('arm') not in allowed or any('teacher' in k.lower() for k in request):
        raise ValueError('only own inputs; future teacher forbidden')
    started=time.monotonic();out=Path(request['output_root']);out.mkdir(parents=True,exist_ok=False)
    store=ObjectStore(request['shared_object_root'])
    def save(name,record):
        ref=store.put_json(record)
        atomic_bytes(out/name,json_bytes(ref))
    stopped=child_stop(request,started)
    import numpy as np
    from satellite_rl.shared_interface.content import read_wire
    from satellite_rl.coverage.ai_search_r14.features import extract_features
    from satellite_rl.coverage.ai_search_r14.tree import route
    from satellite_rl.coverage.ai_search_r14.stop_adapter import execute_stopping_trajectory
    from satellite_rl.coverage.ai_search_r14.policies import ACTIONS,stop_index,quality_preserving
    from satellite_rl.coverage.ai_search_r10 import core as backend
    if Path(backend.__file__).resolve()!=Path(request['source_root'],'src/satellite_rl/coverage/ai_search_r10/core.py').resolve():
        raise ValueError('frozen mathematical import path differs')
    from .audit import AuditLedger,bind_cache,audit_layout
    verify_loaded_source(request['source_root'],request['source_refs'])
    typed=read_wire(pinned(request['case_ref']))
    meta=json.loads(pinned(request['audit_case_metadata_ref']))
    B=read_json(request['B_ref'])
    cache_record=json.loads(pinned(request['cache_ref']))
    case={'identity':meta['identity'],'aircraft_ids':list(typed['aircraft_ids']),
          'positions':typed['lat_lon_deg'].tolist(),'demands':typed['request_bps_int64'].tolist(),
          'physical_parameters':meta['physical_parameters'],'source_ref':request['case_ref']}
    bind_audit_metadata(typed,meta,request['case_ref'])
    bind_sealed_B(read_json(request['sealed_B_receipt_ref']),request['B_ref'],B,
        request['case_ref'],request['cache_ref'],request['expected_cache_SHA'],request['recipe_index'])
    if typed['case_id']!=meta['identity']['case_id'] or typed['content_sha256']!=B['parent_case_content_sha256']:
        raise ValueError('full case/B identity binding')
    cache=bind_cache(case,cache_record,expected_cache_sha=request['expected_cache_SHA'])
    audit_path=Path(request['shared_audit_ledger']);audit_path.parent.mkdir(parents=True,exist_ok=True)
    audit_elapsed=0.
    def online(layout,label):
        nonlocal audit_elapsed
        audit_started=time.monotonic()
        with exclusive(audit_path.with_suffix('.lock')):
            state=json.loads(audit_path.read_bytes()) if audit_path.exists() else None
            ledger=AuditLedger(state,persist=lambda s:atomic_bytes(audit_path,json_bytes(s)))
            receipt=audit_layout(case,layout,cache,ledger,scope='offline' if request.get('phase')=='train_labels' else 'online',envelope='minimal',
                    Ncap=B['Ncap'],Omega_cap=B['Omega_cap'],stop=stopped.fresh)
            save(label+'.audit.json',receipt);ledger.complete()
            if request.get('phase')=='train_labels':audit_elapsed+=time.monotonic()-audit_started
            return receipt
    ownB=online(B['layout'],'own_B')
    if stopped():raise RuntimeError('pause or108s boundary before policy')
    features=extract_features(case['aircraft_ids'],B['layout'],Ncap=B['Ncap'],Omega_cap=B['Omega_cap'])
    selected={'action':'cap32','node_id':None}
    if request['arm']!='full_SHC':
        selected=route(json.loads(pinned(request['policy_ref'])),features)
    scene=SimpleNamespace(aircraft_ids=tuple(case['aircraft_ids']),positions=np.asarray(case['positions'],dtype=np.float64),
          demands=np.asarray(case['demands'],dtype=np.float64),rays=cache.rays,
          input_sha256=B['identity']['input_sha256'],profile='europe_v1')
    common={'n':len(case['aircraft_ids']),'identity':B['identity'],
            'q':max(np.bincount(B['layout']['assignment']).tolist())-1,
            'origin':{'layout':B['layout'],'metrics':ownB['metrics'],'Ncap':B['Ncap'],'Omega_cap':B['Omega_cap']}}
    ledger=backend.Ledger(common['n'],stop=stopped);observations={};native_refs=[]
    def persisted_kernel(*args):
        return call_saved(backend._kernel,args,store,out,native_refs)
    wrapped=SimpleNamespace(_pool=backend._pool,_kernel=persisted_kernel,ResourceUnknown=backend.ResourceUnknown)
    result=None
    try:
        result=execute_stopping_trajectory(scene,common,selected['action'],backend=wrapped,ledger=ledger,
                                           stop=stopped,observations=observations)
        labels=None
        if request.get('phase')=='train_labels':
            if request['arm']!='full_SHC' or selected['action']!='cap32':raise ValueError('train labels require original full trajectory')
            incumbent=B['layout'];metrics_by_position=[ownB['metrics']]
            for position,native in enumerate(result['results'],1):
                if native['status']=='admissible':incumbent=native['terminal']
                checked=online(incumbent,'train_position_'+str(position));metrics_by_position.append(checked['metrics'])
            end={'metrics':metrics_by_position[-1]}
            statuses=[native['status'] for native in result['results']]
            paid=[stop_index(a,statuses) for a in ACTIONS]
            labels={'actions':list(ACTIONS),'paid':paid,
                    'safe':[quality_preserving(metrics_by_position[p],metrics_by_position[-1]) for p in paid],
                    'metrics_by_position':metrics_by_position,'B_and_all_attempt_positions_independently_audited':True}
            save('training_label_row.json',{'case_id':typed['case_id'],'features':features,
                 'safe':labels['safe'],'paid':paid,'actions':list(ACTIONS),
                 'source_task_id':request['task_id'],'raw_label_evidence_requires_parent_exit':True})
        else:
            end=online(result['selected']['terminal'],'own_terminal')
        wire_result={k:v for k,v in result.items() if k!='results'}
        wire_result['native_results']=[{'attempt':i,'ref':ref} for i,ref in enumerate(native_refs)]
        record={'schema':'R14ColdRequestResultV1','task_id':request['task_id'],'arm':request['arm'],
                'case_id':typed['case_id'],'native_object_refs':native_refs,
                'nested_independent_audit_elapsed':audit_elapsed,
                'input_refs':{k:request[k] for k in ('case_ref','B_ref','cache_ref')},'selected_policy':selected,
                'features':features,'result':wire_result,'training_labels':labels,'native_ledger':ledger.record(),'own_terminal_audit':end,
                'child_internal_elapsed':time.monotonic()-started,'complete':True,
                'parent_start_to_reap_required_separately':True}
        save('result.json',record)
    except BaseException as error:
        try:
            save('partial_failure.json',{'task_id':request['task_id'],'error':repr(error),
                         'producer_result':result,'partial_result':getattr(error,'partial_result',None),
                         'partial_kernel':getattr(error,'partial_kernel',None),'partial_ledger':getattr(error,'partial_ledger',None),
                         'native_object_refs':native_refs,'native_ledger':ledger.record(),'elapsed':time.monotonic()-started})
        except BaseException as storage_error:
            atomic_bytes(out/'small_cost_failure.json',json_bytes({'task_id':request['task_id'],'error':repr(error),
                'storage_error':repr(storage_error),'native_object_refs':native_refs,
                'scalar_costs':{k:v for k,v in ledger.record().items() if type(v) in (int,float,str,bool)},
                'elapsed':time.monotonic()-started,'previous_returned_native_objects_preserved':True,
                'current_failed_persistence_not_qualified':True}))
        raise


if __name__=='__main__':
    from .worker_entry import main
    main(execute)
