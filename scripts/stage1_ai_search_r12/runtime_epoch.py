"""Exact one-off human resume and finite dual-clock operational lease.

This module does not authorize science, alter a marker, launch, or renew leases.
"""
import hashlib
import json
import math
from pathlib import Path
import time

_DOCUMENTS={}
_STOP_CACHE={}

BASE='/home/u2024110224/first_stage_satellite_rl/s1_ai17/20473/round12/v2_01'
ROOT=BASE+'/resume_20261015_0858'
AUTH_SHA='aa63719a267d0442a7f9849635d4f774184d9195020d4fc074215eb9b562427d'
BINDINGS_SHA='4731a8158665ef7cd5f11f9ca276a3b091cc19647595a0044ae2117ca4dc8e0c'
GLOBAL='/home/u2024110224/first_stage_satellite_rl/s1_ai17/control/'
HISTORICAL={GLOBAL+'user_pause_20261015_0700.json':('6902ca368bdf459105131625f620b3062aa3abc283e25701d0b7e33f78fc9ddd',517),
 '/home/u2024110224/first_stage_satellite_rl/s1_ai17/20473/control/deadline_20261015_0700/user_pause_20261015_0700_state.json':('c41dade127fed70b8a5b6f7a1054d768789063e8760a1235bf524d513fe0053e',1641),
 '/home/u2024110224/first_stage_satellite_rl/s1_ai17/20474/control/deadline_20261015_0700/user_pause_20261015_0700_state.json':('20f2829f9c2c8efc3bbebcc5c474db850dad5f9a1a00fa19c344eda9eebc8064',1641)}
NEW_GLOBAL=GLOBAL+'r12_resume_20261015_0858.stop.request'
WATCH=(*HISTORICAL,BASE+'/stop.request',NEW_GLOBAL,ROOT+'/stop.request')

def pinned_json(ref,sha):
    if ref.get('sha256')!=sha or type(ref.get('bytes')) is not int or not 0<ref['bytes']<=16384:
        raise ValueError('exact bounded epoch document')
    key=(str(ref['path']),sha,ref['bytes'])
    if key in _DOCUMENTS:return _DOCUMENTS[key]
    with Path(ref['path']).open('rb') as f:raw=f.read(ref['bytes']+1)
    if len(raw)!=ref['bytes'] or hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('epoch document changed')
    value=json.loads(raw);_DOCUMENTS[key]=value
    return value

def marker_values(plan):
    """Any presence (including null/empty/malformed) is distinct from absence."""
    result={}
    for name in plan['watched_pause_paths']:
        p=Path(name)
        try:
            if p.is_symlink():raise ValueError('linked pause marker')
            with p.open('rb') as f:raw=f.read(131073)
            if len(raw)>131072:raise ValueError('oversized pause marker')
            result[name]=dict(present=True,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        except FileNotFoundError:result[name]=None
        except (OSError,ValueError) as error:result[name]=dict(present=True,error=type(error).__name__)
    return result

def validate_authority_and_markers(plan,*,markers=None):
    epoch=plan.get('runtime_epoch',{})
    if epoch.get('schema')!='R12ExactOperationalEpochV1' or epoch.get('epoch')!='r12_resume_20261015_0858':raise ValueError('sole accepted operational epoch')
    authority=pinned_json(epoch['resume_ref'],AUTH_SHA)
    if (authority.get('schema')!='S1AI17ExplicitHumanResumeAuthorityV1' or authority.get('group')!='S1-AI17'
        or authority.get('current_round')!='R12' or authority.get('observed_after_0700_pause') is not True
        or authority.get('user_instruction_verbatim')!='继续' or authority.get('future_pause_always_wins') is not True):raise ValueError('direct post-pause human authority')
    binding=pinned_json(epoch['historical_bindings_ref'],BINDINGS_SHA)
    if binding.get('resume_ref',{}).get('sha256')!=AUTH_SHA:raise ValueError('historical authority binding')
    refs=binding.get('historical_pause_refs',[])
    if len(refs)!=3 or {r.get('absolute_path'):(r.get('sha256'),r.get('bytes')) for r in refs}!=HISTORICAL:raise ValueError('three exact historical generations')
    if len(plan.get('watched_pause_paths',[]))!=6 or set(plan['watched_pause_paths'])!=set(WATCH) or plan.get('own_stop_marker')!=ROOT+'/stop.request':raise ValueError('all historical/old own/new stop paths mandatory')
    if plan.get('executor_port')!=20473 or plan.get('run_id')!='r12_v2_scientific_20261015_20473_01':raise ValueError('sole existing logical task')
    if plan.get('archive_root')!=ROOT+'/science' or plan.get('control_root')!=ROOT+'/service':raise ValueError('new epoch owned output roots')
    if not isinstance(plan.get('request_id'),str) or 'resume_20261015_0858' not in plan['request_id']:raise ValueError('unique epoch request')
    values=marker_values(plan) if markers is None else markers
    if set(values)!=set(WATCH):raise ValueError('complete marker read')
    for path,(sha,size) in HISTORICAL.items():
        value=values[path]
        if not isinstance(value,dict) or value.get('sha256')!=sha or value.get('bytes')!=size or value.get('error'):raise ValueError('historical marker disappeared/changed/unreadable')
    if any(values[p] is not None for p in WATCH if p not in HISTORICAL):raise ValueError('new or old own pause always wins')
    return epoch

def validate(plan,*,now=None,monotonic=None,markers=None,boot_id=None):
    epoch=validate_authority_and_markers(plan,markers=markers)
    actual_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip() if boot_id is None else boot_id
    if epoch.get('boot_id')!=actual_boot:raise ValueError('epoch same boot required')
    names=('issued_unix_seconds','issued_monotonic_seconds','deadline_unix_seconds','deadline_monotonic_seconds','stop_new_unix_seconds','stop_new_monotonic_seconds')
    if any(type(epoch.get(k)) not in (int,float) or not math.isfinite(epoch[k]) for k in names):raise ValueError('finite explicit dual-clock lease')
    duration=epoch['deadline_unix_seconds']-epoch['issued_unix_seconds']
    if not 0<duration<=2550 or abs(epoch['deadline_monotonic_seconds']-epoch['issued_monotonic_seconds']-duration)>1e-6:raise ValueError('unrenewable at most2550 lease')
    for clock in ('unix','monotonic'):
        if abs(epoch[f'deadline_{clock}_seconds']-epoch[f'stop_new_{clock}_seconds']-120)>1e-6:raise ValueError('120s save margin')
    now=time.time() if now is None else now;monotonic=time.monotonic() if monotonic is None else monotonic
    if now<epoch['issued_unix_seconds']-5 or monotonic<epoch['issued_monotonic_seconds']-1:raise ValueError('clock generation not established')
    remaining=min(epoch['deadline_unix_seconds']-now,epoch['deadline_monotonic_seconds']-monotonic)
    stop_remaining=min(epoch['stop_new_unix_seconds']-now,epoch['stop_new_monotonic_seconds']-monotonic)
    if remaining<=0:raise ValueError('operational lease expired')
    return dict(remaining_seconds=remaining,stop_remaining_seconds=stop_remaining,deadline_monotonic=monotonic+remaining)

def stopped(plan,*,force=False):
    """At most100ms cooperative marker latency; explicit boundaries force fresh."""
    tick=time.monotonic();cached=_STOP_CACHE.get(id(plan))
    if not force and cached is not None and cached[0] is plan and 0<=tick-cached[1]<.1:return cached[2]
    if 'runtime_epoch' not in plan:
        from .admission import STOP_NEW
        result=time.time()>=STOP_NEW.timestamp() or any(v is not None for v in marker_values(plan).values())
    else:
        try:result=validate(plan)['stop_remaining_seconds']<=0
        except (KeyError,TypeError,ValueError,OSError):result=True
    _STOP_CACHE[id(plan)]=(plan,tick,result)
    return result

def absolute_deadline(plan):
    if 'runtime_epoch' in plan:
        # Reaping still uses the immutable lease even when a pause becomes present.
        e=plan['runtime_epoch'];return min(e['deadline_monotonic_seconds'],time.monotonic()+e['deadline_unix_seconds']-time.time())
    from .admission import DEADLINE
    return time.monotonic()+DEADLINE.timestamp()-time.time()

def prior_dispatch_absent(plan):
    for name in ('service','submission_intent','service_process','science','launch'):
        p=Path(BASE)/name
        if p.exists() or p.is_symlink():raise ValueError('prior dispatch exists/unknown; never replay')
