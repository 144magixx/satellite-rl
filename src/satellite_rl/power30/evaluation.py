"""Conditional 24-terminal confirmation; inaccessible to the training loader."""
from pathlib import Path
from datetime import datetime
from collections import Counter
import json
import torch
from .control import read_ref, RuntimeControl
from .ledger import CallLedger
from .logging import atomic_json, content_sha, EvidenceLogs
from .menus import MENUS
from .agent import Power30Agent
from .environment import Power30Env
from .runner import evaluate_cases, RecoveryAccountingRequired, _sha_file


def validate_confirmation(config):
    manifest=read_ref(config['confirmation_manifest_ref'])
    unlock=read_ref(config['confirmation_unlock_ref'])
    required={f'{menu}_{group}_{seed}' for menu in MENUS for group in ('M0','M1') for seed in (51001,51002,51003)}
    if (manifest.get('schema')!='W30SealedConfirmationV1' or manifest.get('experiment_id')!='S2-W30'
            or manifest.get('root_approved') is not True or manifest.get('role')!='test'
            or len(manifest.get('rows',[]))!=40 or manifest.get('previously_used_for_selection') is not False):
        raise ValueError('Forty new sealed W30 confirmation mothers required')
    rows=manifest['rows']
    if len({r['case_id'] for r in rows})!=40 or any(r['role']!='test' for r in rows):
        raise ValueError('Confirmation identity/role mismatch')
    dates=manifest.get('dates',[])
    if len(set(dates))!=4 or len(manifest.get('historical_date_use_audit_refs',[]))!=4:
        raise ValueError('Four independently audited new dates required')
    audits=[read_ref(r) for r in manifest['historical_date_use_audit_refs']]
    if (any(r.get('never_used_for_selection') is not True for r in audits)
            or {r.get('date') for r in audits}!=set(dates)):
        raise ValueError('New date independence audit failed')
    row_dates=[]
    for row in rows:
        stamp=datetime.fromisoformat(row['UTC'].replace('Z','+00:00'))
        if stamp.tzinfo is None or stamp.utcoffset().total_seconds()!=0:
            raise ValueError('Explicit UTC confirmation mother time required')
        row_dates.append(stamp.date().isoformat())
    if Counter(row_dates)!={date:10 for date in dates}:
        raise ValueError('Exactly ten mothers from each of four sealed dates required')
    refs=config['terminal_policies']
    if set(refs)!=required or set(config['policy_configs'])!=required:
        raise ValueError('All eight cells and all three seeds required')
    if (unlock.get('schema')!='W30ConfirmationUnlockV1' or unlock.get('experiment_id')!='S2-W30'
            or unlock.get('root_approved') is not True or unlock.get('passed') is not True
            or unlock.get('manifest_SHA')!=config['confirmation_manifest_ref']['sha256']
            or unlock.get('terminal_model_SHAs')!={k:v['sha256'] for k,v in refs.items()}
            or unlock.get('first_round_complete') is not True or unlock.get('development_gate_passed') is not True
            or unlock.get('locked_candidate') not in {f'{m}_{g}' for m in MENUS for g in ('M0','M1')}):
        raise ValueError('Root development gate, candidate and 24 terminals must be locked')
    for key,ref in refs.items():
        if set(ref)!={'path','sha256'} or _sha_file(ref['path'])!=ref['sha256']:
            raise ValueError('Frozen terminal bytes differ')
        c=read_ref(config['policy_configs'][key])
        if key!=f"{c['menu']}_{c['group']}_{c['training']['seed']}" or c['training']['total_steps']!=150000:
            raise ValueError('Terminal config cell/seed/budget differs')
    return manifest,unlock


def run_confirmation_task(config,directory,progress_callback=None,stop_requested=None):
    manifest,unlock=validate_confirmation(config)
    directory=Path(directory);control=RuntimeControl(config,directory);control.validate_owner()
    ledger=CallLedger(config['ledger_root'],config['run_id'],'evaluation')
    control.arm_storage_guard()
    def hook(kind,**metadata):ledger.charge(kind,**metadata)
    def loader(row):
        # W26's test unlock is deliberately not used or forged. W30 checks its
        # own sealed manifest and all 24 terminal locks before reading any row.
        from satellite_rl.t0_width.data import load_case,read_ref,load_authority
        from satellite_rl.shared_interface.content import read_wire
        from satellite_rl.shared_interface.contracts import validate_layout
        from satellite_rl.shared_interface.stages import UniformStage2Input
        from satellite_rl.shared_interface.spherical_gaussian import PHYSICS
        case,_=load_case(row,allow_confirmation=True,geometry_check=False)
        if case['UTC']!=row['UTC']:
            raise ValueError('Sealed mother UTC differs from actual case')
        layout=read_wire(read_ref(row['layout_ref']))
        authority=load_authority(row['publication_authority_ref'],row['publication_authority_content_SHA'])
        validate_layout(case,layout,authority)
        if layout['content_sha256']!=row['layout_content_SHA']:
            raise ValueError('Confirmation published layout differs')
        return UniformStage2Input(case,layout,authority,PHYSICS)
    config_sha=content_sha(config);cursor_path=directory/'confirmation_cursor.json'
    state=json.loads(cursor_path.read_text(encoding='utf8')) if cursor_path.exists() else dict(config_sha=config_sha,policy=0,cursor=0,offsets={})
    if state['config_sha']!=config_sha:raise ValueError('Confirmation resume identity differs')
    for name in EvidenceLogs.NAMES:
        log_path=directory/'logs'/(name+'.jsonl.gz')
        if (log_path.stat().st_size if log_path.exists() else 0)!=state['offsets'].get(name,0):
            raise RecoveryAccountingRequired('Unique confirmation evidence suffix preserved; no truncate/replay')
    logs=EvidenceLogs(directory/'logs',state['offsets'])
    keys=sorted(config['terminal_policies'])
    try:
        while state['policy']<len(keys):
            key=keys[state['policy']];c=read_ref(config['policy_configs'][key])
            c['training']['device']=config['device']
            agent=Power30Agent(c,call_hook=hook)
            saved=torch.load(config['terminal_policies'][key]['path'],map_location=agent.device,weights_only=False)
            if saved.get('env_step')!=150000 or saved.get('schema')!=agent.schema or saved.get('update_step')!=148001:
                raise ValueError('Complete terminal policy/schema/update identity required')
            for name in ('actor','critic1','critic2'):
                getattr(agent,name).load_state_dict(saved[name],strict=True)
            def committed(cursor):
                state['cursor']=cursor;state['offsets']=logs.barrier();atomic_json(cursor_path,state);ledger.flush()
                if progress_callback:progress_callback(dict(stage='confirmation',policy=key,completed_episodes=state['policy']*120+cursor,total_episodes=2880))
            _,finished=evaluate_cases(agent,manifest['rows'],c['group'],loader=loader,
                env_factory=lambda value,group:Power30Env(value,c['menu'],group,call_hook=hook),
                seed_base=275000,node=150000,logs=logs,phase='confirmation_'+key,
                stop_requested=lambda:(stop_requested and stop_requested()) or control.stop_requested(),
                start_cursor=state['cursor'],wal_path=directory/'confirmation_wal.pt',
                row_committed=committed,wal_context=dict(policy_SHA=config['terminal_policies'][key]['sha256']),call_ledger=ledger)
            if not finished:
                ledger.close();return dict(stage='stopped_during_evaluation',policy=key)
            state['policy']+=1;state['cursor']=0;state['offsets']=logs.barrier();atomic_json(cursor_path,state)
        ledger.close();return dict(stage='completed',episodes=2880,locked_candidate=unlock['locked_candidate'],calls=ledger.snapshot())
    except BaseException:
        ledger.flush() # Unknown reservations remain charged. No automatic refund.
        raise
