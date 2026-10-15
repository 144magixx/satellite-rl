"""Finite derived registry/stage collection using actual completed member receipts."""
from pathlib import Path
from .runtime import require_execution,ObjectStore,atomic_bytes,json_bytes
from .bindings import read_json,verify_loaded_source
from .dispatcher import aggregate,publish_data_authority

def execute(request):
    require_execution(request);store=ObjectStore(request['shared_object_root'])
    if request['kind']=='data_registry':
        result=publish_data_authority(read_json(request['exact140_metadata_ref']),request['member_receipt_refs'],store=store)
        ref=store.put_json(result)
    elif request['kind']=='aggregate':
        ref=aggregate(request['evidence_role'],request['member_receipt_refs'],expected_ids=request['expected_task_ids'],store=store)
    else:raise ValueError('finite evidence kind only')
    verify_loaded_source(request['source_root'],request['source_refs'])
    out=Path(request['output_root']);out.mkdir(parents=True,exist_ok=False)
    atomic_bytes(out/'evidence_receipt_ref.json',json_bytes(ref))

if __name__=='__main__':
    from .worker_entry import main
    main(execute)
