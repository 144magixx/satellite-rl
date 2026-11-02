"""Original eight fixed fixtures; at most two native kernels each, no W/RULE/D5/fit."""
import math,json,time,hashlib
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
from .runtime import require_execution,ObjectStore,atomic_bytes,json_bytes,exclusive
from .bindings import verify_loaded_source,read_json
from .stop_control import child_stop
from .kernel_persistence import call_saved

def execute(request):
    require_execution(request)
    from satellite_rl.envs.regions import profile_metadata
    from satellite_rl.coverage.ai_search_r10 import core as backend
    from satellite_rl.coverage.ai_search_r14.stop_adapter import execute_engineering_prefix
    from .audit import engineering_fixture_definitions,AuditLedger,build_ray_cache,bind_cache,audit_layout
    verify_loaded_source(request['source_root'],request['source_refs'])
    definitions=engineering_fixture_definitions();j=request['fixture_index']
    if type(j) is not int or j not in range(8):raise ValueError('original8 fixture only')
    f=definitions[j];out=Path(request['output_root']);out.mkdir(parents=True,exist_ok=False)
    if read_json(request['fixture_definition_ref'])!=f:raise ValueError('exact original fixture bytes/definition')
    store=ObjectStore(request['shared_object_root']);source_ref=store.put_json(f);started=time.monotonic()
    stopped=child_stop(request,started)
    identity={'case_id':'r14_fixture_'+str(j),'role':'synthetic_engineering','profile':'uniform21600000'}
    case={'identity':identity,'aircraft_ids':f['aircraft_ids'],'positions':f['positions'],'demands':f['demands'],
          'physical_parameters':profile_metadata('europe_v1'),'source_ref':source_ref}
    auditpath=Path(request['shared_audit_ledger']);auditpath.parent.mkdir(parents=True,exist_ok=True)
    def ledger():return AuditLedger(json.loads(auditpath.read_bytes()) if auditpath.exists() else None,
                                   persist=lambda s:atomic_bytes(auditpath,json_bytes(s)))
    audit_elapsed=0.;audit_started=time.monotonic()
    with exclusive(auditpath.with_suffix('.lock')):
        charged=ledger();record=build_ray_cache(case,charged,stop=stopped.fresh)
        cache_ref=store.put_json(record);atomic_bytes(out/'preteacher_cache_ref.json',json_bytes(cache_ref));charged.complete()
    audit_elapsed+=time.monotonic()-audit_started
    cache=bind_cache(case,record,expected_cache_sha=record['cache_SHA'])
    omega=(len(f['layout']['beam_ids'])+2)*2*math.pi*(1-math.cos(math.radians(.3)/2))
    auditrefs=[]
    def checked(layout):
        nonlocal audit_elapsed
        audit_started=time.monotonic()
        with exclusive(auditpath.with_suffix('.lock')):
            charged=ledger();receipt=audit_layout(case,layout,cache,charged,scope='offline',envelope='minimal',Ncap=f['Ncap'],Omega_cap=omega,stop=stopped.fresh)
            ref=store.put_json(receipt);auditrefs.append(ref);atomic_bytes(out/'audits.json',json_bytes(auditrefs));charged.complete()
            audit_elapsed+=time.monotonic()-audit_started
            return receipt['metrics']
    import numpy as np
    initial=checked(f['layout']);n=len(f['aircraft_ids'])
    scene=SimpleNamespace(aircraft_ids=tuple(f['aircraft_ids']),positions=np.asarray(f['positions']),demands=np.asarray(f['demands']),
                          rays=cache.rays,input_sha256=hashlib.sha256(json_bytes(f)).hexdigest(),profile='europe_v1')
    common={'n':n,'identity':identity,'q':3,'origin':{'layout':f['layout'],'metrics':initial,'Ncap':f['Ncap'],'Omega_cap':omega}}
    native_refs=[]
    def kernel(*args):
        return call_saved(backend._kernel,args,store,out,native_refs)
    wrapped=SimpleNamespace(_pool=backend._pool,_kernel=kernel,ResourceUnknown=backend.ResourceUnknown)
    costs=backend.Ledger(n,stop=stopped)
    try:result=execute_engineering_prefix(scene,common,backend=wrapped,ledger=costs,stop=stopped)
    except BaseException as error:
        evidence={'error':repr(error),'native_refs':native_refs,'native_ledger':costs.record(),
            'partial_kernel':getattr(error,'partial_kernel',None),'partial_ledger':getattr(error,'partial_ledger',None),
            'fixture_index':j,'retry':False,'nested_independent_audit_elapsed':audit_elapsed}
        try:atomic_bytes(out/'failure_ref.json',json_bytes(store.put_json(evidence)))
        except BaseException as storage_error:
            atomic_bytes(out/'small_cost_failure.json',json_bytes({'error':repr(error),'storage_error':repr(storage_error),
                'native_refs':native_refs,'scalar_costs':{k:v for k,v in costs.record().items() if type(v) in (int,float,str,bool)}}))
        raise
    incumbent=f['layout']
    wire_result={k:v for k,v in result.items() if k!='results'}
    wire_result['native_results']=[{'attempt':i,'ref':ref} for i,ref in enumerate(native_refs)]
    atomic_bytes(out/'producer_completed_ref.json',json_bytes(store.put_json({'result':wire_result,'native_ledger':costs.record(),
        'native_refs':native_refs,'independent_terminal_audit_pending':True})))
    for native in result['results']:
        if native['status']=='admissible':incumbent=native['terminal']
        checked(incumbent)
    if len(result['results'])>2 or costs.data['flow_queries']>16:raise ValueError('original engineering kernel/flow cap')
    ref=store.put_json({'schema':'R14OriginalFixtureActualV1','fixture_index':j,'definition_ref':source_ref,
        'cache_ref':cache_ref,'audit_refs':auditrefs,'native_refs':native_refs,'result':wire_result,'native_ledger':costs.record(),
        'nested_independent_audit_elapsed':audit_elapsed,
        'W_RULE_D5_T_fit_calls':0,'kernels':len(result['results']),'flows':costs.data['flow_queries']})
    atomic_bytes(out/'engineering_receipt_ref.json',json_bytes(ref))

if __name__=='__main__':
    from .worker_entry import main
    main(execute)
