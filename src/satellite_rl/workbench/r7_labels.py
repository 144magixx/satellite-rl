"""Finite fresh labels: one prepare, T and EXACT for each new uniform case."""
import sys,time,shutil,os,subprocess
from pathlib import Path
import numpy as np
from satellite_rl.training.run_monitor import atomic_json
from .storage import digest,read_json
from .coverage_ai_search import pin
from .r6_io import read_asset,pin_tree
from .r6_stage_guard import run_stage
from .r7_uniform_data import load_algorithm_case

CORE_SHA='9f187e4cd8122a6fce415b590d0ab0a385724b7d840cc88dd874e34ea936aea5'

def guarded(target,request,limit,stop,module='satellite_rl.workbench.r7_stage',rss=2147483648):
    target=Path(target);target.mkdir(parents=True,exist_ok=True);wal=target/'started.json';receipt=target/'guard.json'
    if wal.exists():
        if not receipt.exists():raise ValueError('unknown scientific call preserved; root reconciliation required')
        if read_json(target/'request.json',{})!=request or digest(target/'request.json')!=read_json(wal,{})['request_sha256']:raise ValueError('original scientific request changed')
        guard=read_json(receipt,{})
        if guard.get('status')!='completed':raise ValueError('failed/unknown original call retained; no automatic retry')
        pin_tree(guard);ref=read_json(pin(guard['output_ref']),{})['asset_ref']
        return {**guard,'reused_completed_call':True},read_asset(ref),ref
    atomic_json(target/'request.json',request);atomic_json(wal,dict(request_sha256=digest(target/'request.json'),science_call_reserved=True))
    def owned_child(command,**kwargs):
        env=os.environ.copy();env['R7_STAGE_PARENT_PID']=str(os.getpid());return subprocess.Popen(command,env=env,**kwargs)
    guard=run_stage([sys.executable,'-B','-u','-m',module,'--request',str(target/'request.json'),'--directory',str(target)],target,limit,rss,stop,popen=owned_child)
    atomic_json(receipt,guard)
    if guard.get('status')!='completed':raise ValueError('failed/unknown science stops successors')
    ref=read_json(pin(guard['output_ref']),{})['asset_ref'];return guard,read_asset(ref),ref

def supply_case(job,config,i,target):
    role=job['dataset_role'];dataset=read_json(pin(config['datasets'][role]),{});row=dataset['cases'][i]
    if row['case_index']!=i or row['role']!=role:raise ValueError('original dataset order/role')
    record,scene=load_algorithm_case(row['case_ref'],job['model_inputs']['authority'])
    from .r7_case_binding import validate_binding
    validate_binding(record,role,i,row)
    if row['case_content_sha256']!=record['content_sha256'] or row['case_id']!=record['case_id']:raise ValueError('root frozen case changed')
    if not np.all(scene.demands==21600000):raise ValueError('exact uniform scene bps')
    target=Path(target);target.mkdir(parents=True,exist_ok=True);raw=target/'scene.npz'
    if not raw.exists():np.savez_compressed(raw,positions=scene.positions,demands=scene.demands,aircraft_ids=scene.aircraft_ids,profile=scene.profile)
    else:
        with np.load(raw,allow_pickle=False) as old:
            if old['positions'].dtype!=scene.positions.dtype or old['demands'].dtype!=scene.demands.dtype or old['positions'].tobytes()!=scene.positions.tobytes() or old['demands'].tobytes()!=scene.demands.tobytes() or tuple(map(str,old['aircraft_ids']))!=scene.aircraft_ids or str(old['profile'])!=scene.profile:raise ValueError('existing raw scene differs from root shared case; no overwrite or science')
    ident=dict(case_id=record['case_id'],mother_id=record['original_mother_id'],profile='uniform21600000',role=role,
               input_sha256=scene.input_sha256,identity_sha256=job['references']['identity']['sha256'],
               shared_case_content_sha256=record['content_sha256'],source_asset_SHA=record['source_asset_SHA'])
    phase_manifest=dict(stage='labels' if role=='train' else 'development',role=role,admitted=True,
                        identity_sha256=ident['identity_sha256'],dataset_sha256=config['datasets'][role]['sha256'],cases={ident['case_id']:ident})
    req=dict(identity=ident,scene_ref=dict(path=str(raw),sha256=digest(raw)),source_sha256=CORE_SHA,
             producer_sha256=job['source_manifest_sha256'],phase_manifest=phase_manifest)
    return record,scene,req,row

def labels(job,config,directory):
    from scripts.stage1_ai_search_r7.label_audit import audit_bundle,audit_exact_proposal
    directory=Path(directory);index=read_json(directory/'case_index.json',{})
    def stop():
        if shutil.disk_usage(directory).free<8589934592:raise OSError('8GiB reserve reached')
        return (directory/'stop.request').exists()
    for i in job['case_indices']:
        target=directory/'bundles'/str(i)
        if str(i) in index:
            row=read_json(pin(index[str(i)]),{})
            if row['source_commit']!=job['scientific_commit'] or row['input_case_index']!=i:raise ValueError('old label identity/source changed')
            for ref in row['guard_refs']:pin_tree(ref)
            endpoint=read_json(pin(index[str(i)]['cost_endpoint_ref']),{})
            if endpoint['raw_ref']['sha256']!=index[str(i)]['sha256'] or not endpoint['cost_closed']:raise ValueError('old label cost chain')
            continue
        if stop():raise RuntimeError('requested stop')
        started=time.monotonic();record,scene,req,source=supply_case(job,config,i,target);input_seconds=time.monotonic()-started
        parent_seconds={};guards={};assets={};refs={};proofs={}
        for name,limit in (('preparation',20),('teacher',45),('decoder',15)):
            request={**req,'stage':name}
            if name!='preparation':request['prepared_ref']=refs['preparation']
            if name=='decoder':request['teacher_ref']=refs['teacher']
            began=time.monotonic();guards[name],assets[name],refs[name]=guarded(target/name,request,limit,stop);parent_seconds[name]=time.monotonic()-began
            proofs[name]={field:dict(path=str(target/name/file),sha256=digest(target/name/file)) for field,file in (('guard_ref','guard.json'),('request_ref','request.json'),('started_ref','started.json'),('wal_ref','stage.wal.json'),('guard_receipt_ref','guard_receipt.json'))}
            if name=='preparation' and assets[name]['result']['status']!='available':break
        audit_started=time.monotonic();plain=dict(positions=scene.positions.tolist(),demands=scene.demands.tolist(),aircraft_ids=list(scene.aircraft_ids),profile=scene.profile)
        from .r7_label_guard_audit import audit_chain
        unavailable=assets['preparation']['result']['status']!='available'
        proof_row=dict(status='preparation_unavailable' if unavailable else 'complete',source_dataset_sha256=config['datasets']['train']['sha256'],stage_proofs=proofs,identity=req['identity'],scene_ref=req['scene_ref'],core_sha256=CORE_SHA,producer_sha256=job['source_manifest_sha256'],prepared_ref=refs['preparation'],original_T_terminal_ref=refs.get('teacher'),D5_qualification_ref=refs.get('decoder'))
        closure=audit_chain(proof_row);cost_closed=closure['closed']
        if unavailable:
            from scripts.stage1_ai_search_r4.audit import audit_ledger
            if not audit_ledger(assets['preparation']['result']['ledger'],len(scene.demands),'prepare')['valid']:raise ValueError('unavailable preparation ledger invalid')
            checked=dict(qualified=False,nontrivial=False,capability=False,mother_id=req['identity']['mother_id'],label_origin='no_T_witness_unavailable',typed_unavailable=True);proposal=None
        else:
            checked=audit_bundle(plain,assets['preparation'],assets['teacher'],assets['decoder'],req['identity'],CORE_SHA,cost_closed)
            proposal=audit_exact_proposal(read_asset(assets['decoder']['proposal_ref']),assets['teacher']['result'],req['identity'],CORE_SHA,job['source_manifest_sha256'])
        row=dict(identity=req['identity'],input_case_index=i,shared_case_ref=source['case_ref'],scene_ref=req['scene_ref'],
                 prepared_ref=refs['preparation'],original_T_terminal_ref=refs.get('teacher'),D5_qualification_ref=refs.get('decoder'),
                 label_audit=checked,proposal_audit=proposal,cost_closed=cost_closed,source_commit=job['scientific_commit'],
                 status=proof_row['status'],source_dataset_sha256=proof_row['source_dataset_sha256'],stage_proofs=proofs,core_sha256=CORE_SHA,
                 actual_calls=closure['actual_calls'],producer_sha256=job['source_manifest_sha256'],guard_refs=[dict(path=str(target/n/'guard.json'),sha256=digest(target/n/'guard.json')) for n in guards])
        components=dict(raw_input_identity_scene_publication=input_seconds,independent_audit=time.monotonic()-audit_started,**{'parent_'+k:v for k,v in parent_seconds.items()},
                        **{'original_reused_child_'+k:g['actual_stage_wall_seconds'] for k,g in guards.items() if g.get('reused_completed_call')})
        if any(not np.isfinite(v) or v<0 for v in components.values()):raise ValueError('closed nonnegative cost required')
        publication=time.monotonic();path=target/'label_receipt.json';atomic_json(path,row);index[str(i)]=dict(path=str(path),sha256=digest(path));atomic_json(directory/'case_index.json',index)
        components['row_index_publication']=time.monotonic()-publication;endpoint=target/'cost_endpoint.json'
        atomic_json(endpoint,dict(identity=req['identity'],raw_ref=index[str(i)],components=components,complete_offline_label_seconds=sum(components.values()),cost_closed=cost_closed))
        index[str(i)]['cost_endpoint_ref']=dict(path=str(endpoint),sha256=digest(endpoint));atomic_json(directory/'case_index.json',index)
    result=dict(group=job['experiment_id'],phase='labels',scientific_commit=job['scientific_commit'],cases=index,source_dataset_sha256=config['datasets']['train']['sha256'],science_updates=0,future_dispatch_allowed=False)
    atomic_json(directory/'result_manifest.json',result);return result
