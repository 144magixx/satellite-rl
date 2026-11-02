"""Exact content/ref, actual-exit and source-physics joins; pure except pinned IO."""
import hashlib,json,sys
from pathlib import Path
from .runtime import json_bytes

def same_ref(a,b):
    return isinstance(a,dict) and isinstance(b,dict) and all(a.get(k)==b.get(k) for k in ('sha256','bytes'))

def read_json(ref):
    from .lane import pinned
    value=json.loads(pinned(ref))
    if isinstance(value,dict) and value.get('schema')=='R14ChunkedJSONV1':
        raw=b''.join(pinned(r) for r in value['chunks'])
        if len(raw)!=value['bytes'] or hashlib.sha256(raw).hexdigest()!=value['sha256']:raise ValueError('chunk bytes binding')
        value=json.loads(raw)
    return value

def good_exit(facts):
    if (facts.get('raw_returncode')!=0 or facts.get('reaped') is not True or facts.get('descendants_live')!=0
        or not facts.get('process_start_identity') or facts.get('stop_reason')
        or not 0<=facts.get('elapsed_seconds',float('inf'))<=facts.get('outer_seconds',120)):
        raise ValueError('actual successful direct parent exit required')
    return True

def bind_task(task,request):
    for key in ('task_id','kind','phase','recipe_index','arm','repeat','fixture_index','purpose'):
        if task.get(key)!=request.get(key):raise ValueError('task/request exact identity '+key)
    return True

def verify_loaded_source(source_root,refs):
    root=Path(source_root).resolve();allowed={}
    for ref in refs:
        path=Path(ref['path']).resolve()
        if not path.is_relative_to(root):raise ValueError('source ref outside frozen source_root')
        raw=path.read_bytes()
        if len(raw)!=ref['bytes'] or hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('source bytes differ')
        allowed[path]=ref
    for name,module in tuple(sys.modules.items()):
        if name.startswith(('scripts.stage1_ai_search_r14','satellite_rl')) and getattr(module,'__file__',None):
            if Path(module.__file__).resolve() not in allowed:raise ValueError('actual import outside exact closure '+name)
    main=sys.modules.get('__main__')
    if getattr(main,'__file__',None) and Path(main.__file__).resolve() not in allowed:
        raise ValueError('actual executing entry source outside exact closure')
    return True

def verify_private_dependencies(config):
    from .lane import pinned
    expected='/home/u2024110224/first_stage_satellite_rl/s1_ai17/20473/round12/v2_01/runtime_dependencies'
    ref=config['private_dependency_manifest_ref']
    if (ref['sha256']!='02c29d7ee923a9d6b15cb265bc8132b82fcbad74dcdb87190293492484b4bd04'
        or ref['bytes']!=5507 or config['private_dependency_paths']!=[expected]):raise ValueError('sole approved private dependency provenance')
    manifest=json.loads(pinned(ref));root=Path(expected)
    if manifest['private_root']!=expected:raise ValueError('private dependency root')
    for entry in manifest['installed_files']:
        p=(root/entry['relative']).resolve()
        if not p.is_relative_to(root.resolve()):raise ValueError('private member escapes root')
        raw=p.read_bytes()
        if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('private dependency member differs')
    if expected not in sys.path:sys.path.append(expected)
    import psutil
    if not Path(psutil.__file__).resolve().is_relative_to(root.resolve()) or psutil.__version__!='7.0.0':raise ValueError('actual psutil import provenance')
    return True

def bind_audit_metadata(typed,meta,case_ref):
    from satellite_rl.envs.regions import profile_metadata
    from satellite_rl.shared_interface.geometry import GEOMETRY_SHA
    expected=profile_metadata('europe_v1')
    if (meta['identity']['case_id']!=typed['case_id'] or meta['case_content_sha256']!=typed['content_sha256']
        or not same_ref(meta['case_ref'],case_ref) or meta['geometry_config_SHA']!=GEOMETRY_SHA
        or meta['physical_parameters']!=expected):raise ValueError('exact case/source/frozen physics binding')
    return True

def bind_sealed_B(receipt,B_ref,B,case_ref,cache_ref,inner_cache_SHA,recipe_index):
    good_exit(receipt['actual_exit'])
    if (receipt.get('kind')!='B' or receipt.get('recipe_index')!=recipe_index or B.get('recipe_index')!=recipe_index or not same_ref(receipt['B_ref'],B_ref)
        or not same_ref(receipt['case_ref'],case_ref) or receipt['case_id']!=B['case_id']
        or B.get('status')!='success' or B.get('proposal_success') is not True or B.get('qualified') is not True
        or not same_ref(B['cache_manifest']['cache_ref'],cache_ref)
        or B['cache_manifest']['inner_cache_SHA']!=inner_cache_SHA
        or B['cache_manifest'].get('created_before_teacher') is not True
        or not same_ref(receipt['cache_ref'],cache_ref) or receipt['inner_cache_SHA']!=inner_cache_SHA):
        raise ValueError('qualified B/source/preteacher cache/actual exit mismatch')
    return True

def bind_fit_labels(refs,rows,receipts,expected_cases):
    if len(refs)!=96 or len(rows)!=96:raise ValueError('fixed96 actual label refs')
    for ref,row,receipt,case_id in zip(refs,rows,receipts,expected_cases):
        good_exit(receipt['actual_exit'])
        if (not same_ref(ref,receipt['label_ref']) or row['case_id']!=case_id or receipt['case_id']!=case_id
            or row['source_task_id']!=receipt['source_task_id'] or receipt.get('all_positions_audited') is not True):
            raise ValueError('fit label actual audited source/ref/exit join')
    return True

def validate_cold_pairing(receipts,request):
    arms=('AI','strong_causal_baseline','full_SHC')
    if len(receipts)!=3 or set(r['arm'] for r in receipts)!=set(arms):raise ValueError('three actual arms with full teacher')
    reference=receipts[0]
    for r in receipts:
        good_exit(r['actual_exit'])
        if (r['case_id']!=request['case_id'] or r['recipe_index']!=request['recipe_index']
            or r.get('repeat')!=request.get('repeat') or any(not same_ref(r[k],reference[k]) for k in ('case_ref','B_ref','cache_ref'))):
            raise ValueError('same-case/repeat/source/cache pairing')
    return True

def validate_native_B_costs(n,prepared,proposal,result):
    """Original hard ledgers only; nominal overrun is recorded, never hidden."""
    for cost,factor,seconds,reserve,proposals,refits in ((prepared.ledger,1024,10.,1,0,0),(result['cost'],2048,5.,3,512,8)):
        expected={'raypairs_max':factor*n,'seconds_nominal':seconds,'terminal_reserve_raypairs':reserve*n,
                  'proposals_max':proposals,'refits_max':refits}
        if any(cost.get(k)!=v for k,v in expected.items()):raise ValueError('exact original native B ledger bound')
        for field,cap in (('raypairs',factor*n),('proposals',proposals),('refits',refits)):
            if type(cost.get(field)) is not int or not 0<=cost[field]<=cap:raise ValueError('actual native B paid cap '+field)
    caps={'INITIAL':0,'EVICT':128,'INSERT':256,'REFIT':8,'SPLIT':64,'MOVE':56,'AUDIT':0}
    for phase in result['cost']['phases']:
        cap=caps.get(phase['operator'])
        if cap is None or phase['proposal_cap']!=cap or not 0<=phase['proposals']<=cap:raise ValueError('native phase nontransferable cap')
    if proposal['construction_cost']['raypairs']!=n*len(prepared.raw_w.beam_ids):raise ValueError('RULE complete score pair work')
    return True
