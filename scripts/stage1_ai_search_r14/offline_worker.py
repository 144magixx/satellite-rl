"""Distinct after-the-fact six-layout audit and three-arm teacher pairing."""
from pathlib import Path
import json,time
from .runtime import require_execution,ObjectStore,exclusive,atomic_bytes,json_bytes
from .bindings import read_json,good_exit,same_ref,bind_audit_metadata,verify_loaded_source,validate_cold_pairing
from .lane import pinned
from .dispatcher import verify_receipt
from .stop_control import child_stop

def execute(request):
    require_execution(request)
    from satellite_rl.shared_interface.content import read_wire
    from satellite_rl.coverage.ai_search_r14.policies import quality_preserving
    from .audit import AuditLedger,bind_cache,audit_layout
    verify_loaded_source(request['source_root'],request['source_refs'])
    started=time.monotonic();out=Path(request['output_root']);out.mkdir(parents=True,exist_ok=False)
    stopped=child_stop(request,started)
    store=ObjectStore(request['shared_object_root'])
    receipts=[verify_receipt(r) for r in request['cold_receipt_refs']]
    validate_cold_pairing(receipts,request)
    arms=('AI','strong_causal_baseline','full_SHC')
    if len(receipts)!=3 or set(r['arm'] for r in receipts)!=set(arms):raise ValueError('three actual same-case arms including full teacher')
    for r in receipts:good_exit(r['actual_exit'])
    reference=receipts[0]
    for r in receipts:
        if (r['case_id']!=request['case_id'] or r['recipe_index']!=request['recipe_index']
            or r.get('repeat')!=request.get('repeat') or any(not same_ref(r[k],reference[k]) for k in ('case_ref','B_ref','cache_ref'))):
            raise ValueError('offline original same-case/repeat/source/cache pairing')
    typed=read_wire(pinned(reference['case_ref']));meta=read_json(request['audit_case_metadata_ref'])
    bind_audit_metadata(typed,meta,reference['case_ref'])
    case={'identity':meta['identity'],'aircraft_ids':list(typed['aircraft_ids']),'positions':typed['lat_lon_deg'].tolist(),
        'demands':typed['request_bps_int64'].tolist(),'physical_parameters':meta['physical_parameters'],'source_ref':reference['case_ref']}
    B=read_json(reference['B_ref']);cache=bind_cache(case,read_json(reference['cache_ref']),expected_cache_sha=B['cache_manifest']['inner_cache_SHA'])
    auditpath=Path(request['shared_audit_ledger']);checked={};records=[]
    for r in receipts:
        cold=read_json(r['result_ref'])
        for label,layout in [('B',B['layout']),('terminal',cold['result']['selected']['terminal'])]:
            with exclusive(auditpath.with_suffix('.lock')):
                ledger=AuditLedger(read_json_file(auditpath),persist=lambda s:atomic_bytes(auditpath,json_bytes(s)))
                audit=audit_layout(case,layout,cache,ledger,scope='offline',envelope='minimal',Ncap=B['Ncap'],Omega_cap=B['Omega_cap'],stop=stopped.fresh)
                ref=store.put_json(audit);records.append({'arm':r['arm'],'position':label,'ref':ref});ledger.complete()
                if label=='terminal':checked[r['arm']]=audit['metrics']
    teacher=checked['full_SHC'];safe={arm:quality_preserving(checked[arm],teacher) for arm in arms[:2]}
    calibration=[]
    if request.get('purpose')=='calibration':
        for r in receipts:
            if r['arm']=='full_SHC':continue
            kind='AI' if r['arm']=='AI' else 'baseline'
            if not same_ref(r['policy_ref'],request['initial_policy_refs'][kind]):raise ValueError('calibration initial immutable policy source')
            calibration.append({'mother_id':r['case_id'],'policy':kind,'features':r['features'],
                'route_id':r['selected_policy']['node_id'],'action':r['selected_policy']['action'],'safe':safe[r['arm']]})
    result={'schema':'R14DistinctOfflinePairingV1','case_id':request['case_id'],'recipe_index':request['recipe_index'],
        'repeat':request.get('repeat'),'cold_receipt_refs':request['cold_receipt_refs'],'offline_layouts':records,
        'safe':safe,'terminal_metrics':checked,'calibration_records':calibration,'initial_policy_refs':request.get('initial_policy_refs'),
        'online_calls_reused':False,'actual_offline_layout_calls':6}
    ref=store.put_json(result);atomic_bytes(out/'offline_receipt_ref.json',json_bytes(ref))

def read_json_file(path):return json.loads(Path(path).read_bytes())

if __name__=='__main__':
    from .worker_entry import main
    main(execute)
