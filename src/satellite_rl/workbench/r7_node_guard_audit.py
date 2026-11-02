"""Actual node preparation/T guard proofs and independent four-method costs."""
import math
from pathlib import Path
from .storage import read_json,digest
from .coverage_ai_search import pin
from .r6_io import read_asset,pin_tree
from .r7_label_guard_audit import audit_stage,finite_nonnegative

def stage_proof(directory):
    directory=Path(directory)
    return {field:dict(path=str(directory/name),sha256=digest(directory/name)) for field,name in (('guard_ref','guard.json'),('request_ref','request.json'),('started_ref','started.json'),('wal_ref','stage.wal.json'),('guard_receipt_ref','guard_receipt.json'))}

def audit_preparation_teacher(row):
    unavailable=row['status']=='preparation_unavailable'
    expected={'preparation'} if unavailable else {'preparation','teacher'}
    if set(row['stage_proofs'])!=expected:raise ValueError('exact node original preparation/T call set')
    result={}
    for stage in sorted(expected):
        result[stage]=audit_stage(**row['stage_proofs'][stage],stage=stage,identity=row['identity'],scene_ref=row['scene_ref'],
                                  source_sha=row['core_sha256'],producer_sha=row['producer_sha256'],dataset_sha=row['source_dataset_sha256'],
                                  asset_ref=row['prepared_ref'] if stage=='preparation' else row['result_refs']['T'],prepared_ref=row['prepared_ref'] if stage=='teacher' else None,
                                  phase_role=row['identity']['role'],phase_stage=row['identity']['role'])
    return result

def audit_method_proof(method,row,case):
    """Outer cold-method guard and inner D5 original request/WAL closure."""
    proof=row['guard_proof'];request=read_json(pin(proof['request_ref']),{});wal=read_json(pin(proof['wal_ref']),{});started=read_json(pin(proof['started_ref']),{})
    guard=read_json(pin(proof['guard_ref']),{});original=read_json(pin(proof['guard_receipt_ref']),{})
    if guard!=original or started.get('request_sha256')!=proof['request_ref']['sha256'] or started.get('science_call_reserved')is not True or request!=row['request'] or request['identity']!=case['identity'] or request['scene_ref']!=case['scene_ref'] or request['prepared_ref']!=case['prepared_ref'] or request['method']!=method or request['producer_sha256']!=case['producer_sha256']:raise ValueError('original cold method source/request/WAL chain')
    limit=90 if method=='AI' else 15;rss_limit=4294967296 if method=='AI' else 2147483648
    cmd=wal.get('command',[])
    if wal.get('hard_seconds')!=limit or wal.get('owned_parent_children_rss_bytes')!=rss_limit or wal.get('status')!='child_started' or guard.get('pid')!=wal.get('pid') or guard.get('birth')!=wal.get('birth') or len(cmd)!=9 or cmd[1:5]!=['-B','-u','-m','satellite_rl.workbench.r7_model_eval_stage'] or cmd[5:9]!=['--request',proof['request_ref']['path'],'--directory',str(Path(proof['request_ref']['path']).parent)]:raise ValueError('outer original method caps/command/birth')
    closed=bool(guard.get('status')=='completed' and guard.get('reason')is None and guard.get('exit_code')==0 and guard.get('unknown_algorithm_spend')is False and guard.get('hard_interface_error')is False and guard.get('monitor_error')is None and finite_nonnegative(guard.get('actual_stage_wall_seconds')) and guard['actual_stage_wall_seconds']<=limit and finite_nonnegative(guard.get('peak_parent_children_rss_bytes')) and guard['peak_parent_children_rss_bytes']<=rss_limit)
    pin_tree(guard);value=read_asset(row['result_ref']);output=read_json(pin(guard['output_ref']),{})
    if output.get('asset_ref')!=row['result_ref'] or value.get('actual_method')!=method or value.get('identity')!=case['identity'] or value.get('one_D5_call')is not True:raise ValueError('actual cold output is not original guarded result')
    inner_proof=value['D5_proof']
    if set(inner_proof)!=set(('request_ref','wal_ref','guard_ref')) or inner_proof['guard_ref']!=value['D5_guard_ref']:raise ValueError('actual original inner request/WAL/guard pinned proof required')
    for reference in inner_proof.values():pin(reference)
    inner=read_json(pin(inner_proof['guard_ref']),{});inner_dir=Path(inner_proof['guard_ref']['path']).parent
    inner_wal=read_json(pin(inner_proof['wal_ref']),{});inner_request=read_json(pin(inner_proof['request_ref']),{})
    if inner_wal.get('hard_seconds')!=15 or inner_wal.get('owned_parent_children_rss_bytes')!=2147483648 or inner.get('pid')!=inner_wal.get('pid') or inner.get('birth')!=inner_wal.get('birth') or inner_request.get('identity')!=case['identity'] or inner_request.get('scene_ref')!=case['scene_ref'] or inner_request.get('prepared_ref')!=case['prepared_ref'] or inner_request.get('proposal_ref')!=value['proposal_ref'] or inner_request.get('b7_decoder_source_SHA')!=case['core_sha256']:raise ValueError('original inner D5 identity/request/caps/source')
    inner_cmd=inner_wal.get('command',[])
    if len(inner_cmd)!=9 or inner_cmd[1:5]!=['-B','-u','-m','satellite_rl.workbench.r7_model_d5_stage'] or inner_cmd[5:9]!=['--request',str(inner_dir/'request.json'),'--directory',str(inner_dir)]:raise ValueError('inner original D5 guarded command')
    inner_output=read_json(pin(inner['output_ref']),{});decoded=read_asset(inner_output['asset_ref'])
    if decoded.get('result')!=value['result'] or decoded.get('identity')!=case['identity'] or decoded.get('owning_AI_process_pid')!=guard['pid'] or decoded.get('owning_AI_process_birth')!=guard['birth']:raise ValueError('original decoded result/parent identity differs from outer cold method')
    closed=bool(closed and inner.get('status')=='completed' and inner.get('reason')is None and inner.get('exit_code')==0 and inner.get('unknown_algorithm_spend')is False and inner.get('hard_interface_error')is False and inner.get('monitor_error')is None and finite_nonnegative(inner.get('actual_stage_wall_seconds')) and inner['actual_stage_wall_seconds']<=15 and finite_nonnegative(inner.get('peak_parent_children_rss_bytes')) and inner['peak_parent_children_rss_bytes']<=2147483648)
    return dict(closed=closed,outer_wall_seconds=guard['actual_stage_wall_seconds'],inner_wall_seconds=inner['actual_stage_wall_seconds'])

def audit_costs(row,stage_results):
    components=row['cost_components'];unavailable=row['status']=='preparation_unavailable'
    expected={'input','preparation_parent'}|({'T_parent','AI_parent','RULE_parent','RANDOM_parent'} if not unavailable else set())
    if set(components)!=expected or any(not finite_nonnegative(v) for v in components.values()):raise ValueError('complete node raw common/method cold costs')
    if components['preparation_parent']<stage_results['preparation']['wall_seconds']:raise ValueError('preparation actual child wall not paid')
    if unavailable:
        if set(row['costs'])!=set(('AI','RULE','RANDOM','T')) or any(v is not None for v in row['costs'].values()):raise ValueError('unexecuted four methods cannot have invented complete costs')
        return row['costs']
    if components['T_parent']<stage_results['teacher']['wall_seconds']:raise ValueError('teacher actual child wall not paid')
    rebuilt={m:components['input']+components['preparation_parent']+components[m+'_parent'] for m in ('AI','RULE','RANDOM','T')}
    if any(not math.isclose(rebuilt[m],row['costs'][m],rel_tol=1e-12,abs_tol=1e-9) for m in rebuilt):raise ValueError('producer complete cold method sum differs')
    return rebuilt
