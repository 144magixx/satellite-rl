"""Bounded injected episode executor with idempotent append/adoption logs.

Callers must first qualify the real loader, policy and PHY. Tests inject fakes.
Uncommitted partial log suffixes are preserved and blocked, never truncated.
"""
import os
from pathlib import Path
from satellite_rl.t0_width.logging import canonical_json
from .transaction import restore_memory,close_transaction,UnknownCost,fsync_directory
from .worker import one_decision,candidate_log

def publish_committed(journal,path):
    tx=journal.value['transaction']
    if tx['stage']!='committed':raise ValueError('publication needs exact committed transaction')
    observation,reward,done,info=tx['commit_result']
    row=dict(schema='W28DecisionEvidenceV1',context=tx['context'],candidates=candidate_log(journal),
        reward=reward,done=done,info=info)
    raw=(canonical_json(row)+'\n').encode('utf8')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    offset=journal.value.get('log_offset',0)
    if (path.stat().st_size if path.exists() else 0)<offset:raise ValueError('trusted evidence log truncated')
    if path.exists():
        with path.open('rb') as stream:
            stream.seek(offset);suffix=stream.read()
    else:suffix=b''
    if suffix:
        if suffix!=raw:raise UnknownCost('uncommitted evidence suffix differs; preserve bytes and stop')
        # A fully durable row written before cursor save is adopted, not appended.
    else:
        with path.open('ab') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        fsync_directory(path.parent)
    journal.value['log_offset']=offset+len(raw)
    close_transaction(journal)
    return row

def run_episode(env,observation,method,case_id,repeat,journal,policy_probabilities,
                evidence_path,stop_requested,admission,baseline_action=None):
    """Zero implicit retries/loads/reconstruction. Gate checked every boundary."""
    if admission.get('root_approved') is not True or admission.get('qualification_passed') is not True:
        raise ValueError('root engineering/runtime admission required')
    identity=journal.value['identity']
    for key,value in dict(case_id=case_id,method=method,repeat=repeat,N=env.N).items():
        if identity.get(key)!=value:raise ValueError('same-ID episode binding mismatch: '+key)
    tx=journal.value['transaction']
    if tx is None or tx['stage']!='committed':
        actual=Path(evidence_path).stat().st_size if Path(evidence_path).exists() else 0
        if actual!=journal.value.get('log_offset',0):raise UnknownCost('noncommitted evidence has an unaccounted suffix')
    if tx is not None:
        if tx.get('pending'):raise UnknownCost('preserve interrupted physical/forward intent; no automatic replay')
        restore_memory(env,tx['committed_state'] if tx['stage'] in ('committed','closed') else tx['base'])
        if tx['stage']=='committed':publish_committed(journal,evidence_path)
        if tx['stage']=='closed':
            observation=tx['commit_result'][0]
    while env.beam_idx<env.N:
        if stop_requested():return dict(stage='safe_stopped',processed=env.beam_idx,costs=journal.value['attempts'])
        result=one_decision(env,observation,method,case_id,repeat,journal,policy_probabilities,stop_requested,baseline_action)
        if result is None:return dict(stage='safe_stopped',processed=env.beam_idx,costs=journal.value['attempts'])
        observation,_,_,_=result
        publish_committed(journal,evidence_path)
    return dict(stage='completed',processed=env.beam_idx,costs=journal.value['attempts'],log_offset=journal.value.get('log_offset',0))
