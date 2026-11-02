"""One exactly admitted source qualification to immutable typed case; no GEO."""
import json
import time
from pathlib import Path
from .lane import pinned
from .runtime import require_execution,ObjectStore,atomic_bytes,json_bytes
from .input_adapter import read_source,seal_source,InputLedger
from .bindings import verify_loaded_source
from .stop_control import child_stop


def execute(request):
    require_execution(request)
    from satellite_rl.shared_interface.content import wire_bytes
    from satellite_rl.shared_interface.stages import make_case
    from satellite_rl.envs.regions import profile_metadata
    from satellite_rl.shared_interface.geometry import GEOMETRY_SHA
    verify_loaded_source(request['source_root'],request['source_refs'])
    selected=json.loads(pinned(request['selected_metadata_ref']))
    if selected['recipe_index']!=request['recipe_index']:raise ValueError('exact selected data mother index')
    metadata=json.loads(pinned(request['source_job_transport_ref'],8*1024**2))
    out=Path(request['output_root']);out.mkdir(parents=True,exist_ok=False)
    store=ObjectStore(request['shared_object_root']);events=[]
    stopped=child_stop(request,time.monotonic())
    def checkpoint(label,value):
        if label.startswith('before') and stopped():raise RuntimeError('new raw input blocked by pause/deadline')
        events.append({'label':label,'value':value})
        atomic_bytes(out/'events.json',json_bytes(events))
    planes=read_source(selected,request['raw_transport_ref'],config=request,
        ledger=InputLedger(request['shared_input_ledger_root']),checkpoint=checkpoint)
    case,binding=seal_source(selected,planes,metadata,request['source_job_transport_ref'],request['raw_transport_ref'])
    case_ref=store.put(wire_bytes(case));binding_ref=store.put(json_bytes(binding))
    metadata={'identity':{'case_id':case['case_id'],'original_mother_id':case['original_mother_id'],
        'source_asset_SHA':case['source_asset_SHA']},'case_content_sha256':case['content_sha256'],
        'case_ref':case_ref,'geometry_config_SHA':GEOMETRY_SHA,'physical_parameters':profile_metadata('europe_v1')}
    audit_metadata_ref=store.put_json(metadata)
    verify_loaded_source(request['source_root'],request['source_refs'])
    atomic_bytes(out/'typed_case_derivation.json',json_bytes({'case_ref':case_ref,'binding_ref':binding_ref,
        'audit_case_metadata_ref':audit_metadata_ref,'qualified_for_algorithm':False,'requires_independent_root_authority':True,'all_aircraft_count':len(planes)}))

if __name__=='__main__':
    from .worker_entry import main
    main(execute)
