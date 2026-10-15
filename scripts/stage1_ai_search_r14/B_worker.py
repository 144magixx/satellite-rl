"""One admitted native W/RULE/D5 request, preteacher cache and three-state audit."""
import json
from pathlib import Path
import time
import hashlib,math
from .lane import pinned
from .runtime import ObjectStore,atomic_bytes,json_bytes,exclusive,require_execution,produce_B
from .identity_adapter import root_authority,phase_identity,task_binding
from .bindings import bind_audit_metadata,verify_loaded_source,validate_native_B_costs
from .stop_control import child_stop


def execute(request):
    require_execution(request)
    from satellite_rl.shared_interface.content import read_wire
    from satellite_rl.shared_interface.stages import algorithm_scene
    from satellite_rl.coverage.ai_search_r5 import core as backend
    from .audit import AuditLedger,build_ray_cache,bind_cache,audit_layout
    verify_loaded_source(request['source_root'],request['source_refs'])
    core_ref=request['native_core_ref']
    if Path(backend.__file__).resolve()!=Path(core_ref['path']).resolve() or hashlib.sha256(pinned(core_ref)).hexdigest()!=request['native_source_sha256']:
        raise ValueError('actual R5 core import and source SHA')
    started=time.monotonic();out=Path(request['output_root']);out.mkdir(parents=True,exist_ok=False)
    store=ObjectStore(request['shared_object_root']);sequence=[]
    stopped=child_stop(request,started)
    def checkpoint(label,value):
        if value is None:
            if stopped():
                raise RuntimeError('B stop-new-work108/own/global pause')
            return
        ref=store.put_json(value);sequence.append({'label':label,'ref':ref})
        atomic_bytes(out/'objects.json',json_bytes(sequence))
    case=read_wire(pinned(request['case_ref']));selected=json.loads(pinned(request['selected_metadata_ref']))
    if selected['recipe_index']!=request['recipe_index']:raise ValueError('exact selected B mother index')
    binding=task_binding(selected,case)
    authority=root_authority(json.loads(pinned(request['authority_envelope_ref'])),
        request['authority_envelope_content_sha256'],request['shared_registry_content_sha256'],request)
    scene=algorithm_scene(case,authority)
    identity,phase=phase_identity(scene,case,binding,dataset_sha256=request['dataset_sha256'],config=request)
    meta=json.loads(pinned(request['audit_case_metadata_ref']))
    bind_audit_metadata(case,meta,request['case_ref'])
    auditcase={'identity':meta['identity'],'aircraft_ids':list(case['aircraft_ids']),
        'positions':case['lat_lon_deg'].tolist(),'demands':case['request_bps_int64'].tolist(),
        'physical_parameters':meta['physical_parameters'],'source_ref':request['case_ref']}
    auditpath=Path(request['shared_audit_ledger']);auditpath.parent.mkdir(parents=True,exist_ok=True)
    audit_elapsed=0.;audit_started=time.monotonic()
    with exclusive(auditpath.with_suffix('.lock')):
        state=json.loads(auditpath.read_bytes()) if auditpath.exists() else None
        ledger=AuditLedger(state,persist=lambda s:atomic_bytes(auditpath,json_bytes(s)))
        cache_record=build_ray_cache(auditcase,ledger,stop=stopped.fresh)
        cache_ref=store.put(json_bytes(cache_record))
        cache_manifest={'case_ref':request['case_ref'],'cache_ref':cache_ref,'inner_cache_SHA':cache_record['cache_SHA'],
                        'created_before_teacher':True}
        checkpoint('preteacher_cache_manifest',cache_manifest);ledger.complete()
    audit_elapsed+=time.monotonic()-audit_started
    cache=bind_cache(auditcase,cache_record,expected_cache_sha=cache_record['cache_SHA'])
    bundle=produce_B(scene,identity,config=request,source_sha256=request['native_source_sha256'],
        producer_sha256=request['native_source_sha256'],phase_manifest=phase,checkpoint=checkpoint,backend=backend)
    if bundle['status']!='success' or bundle['result'].get('proposal_success') is not True:
        checkpoint('B_failed',{'status':bundle['status'],'qualified':False,'producer_projection_records':len(case['aircraft_ids'])})
        raise RuntimeError('fixed mother B not success; dependent science closes without replacement')
    p,result=bundle['prepared'],bundle['result'];audits=[]
    validate_native_B_costs(len(case['aircraft_ids']),p,bundle['proposal'],result)
    for label,layout,envelope in [('raw_W',p.raw_w.to_record(),'raw_W'),('Wenv',p.Wenv.to_record(),'minimal'),
                                 ('D5terminal',result['layout'],'minimal')]:
        if label!='D5terminal':layout={**layout,'anchor_aircraft_ids':list(p.anchors)}
        audit_started=time.monotonic()
        with exclusive(auditpath.with_suffix('.lock')):
            ledger=AuditLedger(json.loads(auditpath.read_bytes()),persist=lambda s:atomic_bytes(auditpath,json_bytes(s)))
            receipt=audit_layout(auditcase,layout,cache,ledger,scope='offline',envelope=envelope,Ncap=p.Ncap,Omega_cap=p.Omega_cap,stop=stopped.fresh)
            if label=='raw_W':
                if p.Ncap!=min(200,len(case['aircraft_ids']),math.ceil(1.25*receipt['metrics']['N'])) or abs(p.Omega_cap-receipt['metrics']['actual_Omega'])>1e-12:
                    checkpoint('raw_W_cap_mismatch',receipt);raise ValueError('raw W independent Ncap/Omega derivation')
            checkpoint(label+'_audit',receipt);ledger.complete();audits.append(label)
        audit_elapsed+=time.monotonic()-audit_started
    published={'schema':'R14QualifiedBOutputV1','parent_case_content_sha256':case['content_sha256'],
       'identity':identity,'layout':result['layout'],'Ncap':p.Ncap,'Omega_cap':p.Omega_cap,
       'status':'success','proposal_success':True,'qualified':True,'recipe_index':selected['recipe_index'],
       'case_id':case['case_id'],'independent_layouts':audits,'cache_manifest':cache_manifest,
       'native_cost':result['cost'],'producer_projection_records':len(case['aircraft_ids']),
       'nested_independent_audit_elapsed':audit_elapsed,
       'actual_parent_exit_required':True,'complete_cost_exit':False}
    checkpoint('qualified_B_before_parent_exit',published)

if __name__=='__main__':
    from .worker_entry import main
    main(execute)
