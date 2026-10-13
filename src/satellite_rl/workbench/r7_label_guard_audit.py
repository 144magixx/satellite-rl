"""Read-only original call proof; producer closure flags do not grant credit."""
import math
from pathlib import Path
from .coverage_ai_search import pin
from .storage import read_json,digest
from .r6_io import read_asset,pin_tree

CAPS={'preparation':20,'teacher':45,'decoder':15}

def finite_nonnegative(value):return type(value) in (int,float) and math.isfinite(value) and value>=0

def audit_stage(guard_ref,request_ref,started_ref,wal_ref,guard_receipt_ref,*,stage,identity,scene_ref,source_sha,producer_sha,dataset_sha,asset_ref,prepared_ref=None,teacher_ref=None,phase_role='train',phase_stage='labels'):
    if stage not in CAPS:raise ValueError('registered label stage required')
    refs=(guard_ref,request_ref,started_ref,wal_ref,guard_receipt_ref)
    for ref in refs:pin(ref)
    guard=read_json(pin(guard_ref),{});request=read_json(pin(request_ref),{});started=read_json(pin(started_ref),{});wal=read_json(pin(wal_ref),{});child_receipt=read_json(pin(guard_receipt_ref),{})
    if guard!=child_receipt:raise ValueError('wrapper and original guard receipt differ')
    expected=dict(identity=identity,scene_ref=scene_ref,source_sha256=source_sha,producer_sha256=producer_sha,
                  phase_manifest=request.get('phase_manifest'),stage=stage)
    if stage!='preparation':expected['prepared_ref']=prepared_ref
    if stage=='decoder':expected['teacher_ref']=teacher_ref
    if request!=expected or started.get('request_sha256')!=request_ref['sha256'] or digest(pin(request_ref))!=request_ref['sha256'] or started.get('science_call_reserved')is not True:raise ValueError('original stage request/case/source/reservation differs')
    phase=request['phase_manifest']
    if (phase_role,phase_stage) not in (('train','labels'),('development','development'),('validation','validation'),('test','test')) or identity['role']!=phase_role or phase.get('stage')!=phase_stage or phase.get('role')!=phase_role or phase.get('admitted')is not True or phase.get('dataset_sha256')!=dataset_sha or phase.get('identity_sha256')!=identity['identity_sha256'] or phase.get('cases',{}).get(identity['case_id'])!=identity:raise ValueError('original stage phase capability')
    if wal.get('hard_seconds')!=CAPS[stage] or wal.get('owned_parent_children_rss_bytes')!=2147483648 or wal.get('status')!='child_started':raise ValueError('original stage hard caps/WAL state')
    command=wal.get('command',[])
    if len(command)!=9 or command[1:5]!=['-B','-u','-m','satellite_rl.workbench.r7_stage'] or command[5]!='--request' or command[6]!=request_ref['path'] or command[7]!='--directory' or command[8]!=str(Path(request_ref['path']).parent):raise ValueError('original exact guarded command/paths')
    if guard.get('pid')!=wal.get('pid') or guard.get('birth')!=wal.get('birth') or type(wal.get('pid'))is not int or type(wal.get('birth'))is not int:raise ValueError('original child PID/birth identity')
    wall=guard.get('actual_stage_wall_seconds');rss=guard.get('peak_parent_children_rss_bytes')
    closed=bool(guard.get('status')=='completed' and guard.get('reason')is None and guard.get('exit_code')==0
                and guard.get('unknown_algorithm_spend')is False and guard.get('hard_interface_error')is False and guard.get('monitor_error')is None
                and finite_nonnegative(wall) and wall<=CAPS[stage] and finite_nonnegative(rss) and rss<=2147483648)
    if guard.get('output_ref'):
        pin_tree(guard);output=read_json(pin(guard['output_ref']),{})
        if output.get('asset_ref')!=asset_ref or output.get('stage')!=stage or output.get('identity')!=identity:raise ValueError('guard original output does not bind actual stage asset')
        value=read_asset(asset_ref)
        if value.get('stage')!=stage or value.get('identity')!=identity or value.get('source_sha256')!=source_sha or value.get('python')!=command[0]:raise ValueError('actual raw stage identity/source/interpreter')
    else:closed=False
    return dict(closed=closed,stage=stage,wall_seconds=wall,peak_rss_bytes=rss,original_request_ref=request_ref)

def audit_chain(row):
    expected=['preparation'] if row.get('status')=='preparation_unavailable' else ['preparation','teacher','decoder']
    if set(row.get('stage_proofs',{}))!=set(expected):raise ValueError('exact original label call set')
    asset_names={'preparation':'prepared_ref','teacher':'original_T_terminal_ref','decoder':'D5_qualification_ref'}
    checked={}
    for stage in expected:
        proof=row['stage_proofs'][stage]
        checked[stage]=audit_stage(**proof,stage=stage,identity=row['identity'],scene_ref=row['scene_ref'],source_sha=row['core_sha256'],producer_sha=row['producer_sha256'],dataset_sha=row['source_dataset_sha256'],asset_ref=row[asset_names[stage]],
                                   prepared_ref=row.get('prepared_ref') if stage!='preparation' else None,teacher_ref=row.get('original_T_terminal_ref') if stage=='decoder' else None)
    return dict(closed=all(v['closed'] for v in checked.values()),stages=checked,
                actual_calls=dict(prepare=1,T=int('teacher' in checked),D5=int('decoder' in checked)))
