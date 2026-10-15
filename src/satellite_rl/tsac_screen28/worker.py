"""Inert W28 decision/episode wiring. Root admission is mandatory at call time.

No CLI launches a remote task. Loader/Gate/admission/finite run registration are
provided by the independent workbench after W27 closure and engineering gates.
"""
from time import perf_counter
from .proposals import draw,uniform_probabilities
from .transaction import decision,snapshot,state_sha,close_transaction,UnknownCost,rng_snapshot,check_rng

def one_decision(env,observation,method,case_id,repeat,journal,policy_probabilities,stop_requested):
    if method not in ('P1','P4','R1','R4'):raise ValueError('four frozen methods only')
    tx=journal.value['transaction']
    if tx is not None and tx.get('pending'):raise UnknownCost('pending query/commit cannot be replayed')
    context=dict(case_id=case_id,method=method,repeat=repeat,step=env.beam_idx)
    if tx is not None and tx['stage']!='closed':
        if any(tx['context'][k]!=context[k] for k in ('case_id','method','repeat')):raise ValueError('original proposal owner differs')
        context=tx['context'];proposals=tx['proposals']
    else:
        if stop_requested():return None
        base=snapshot(env)
        tx=dict(context=context,base=base,base_SHA=state_sha(base),proposals=None,
            results={},selected=None,stage='proposing',pending=None,commit_result=None,rng=rng_snapshot())
        journal.value['transaction']=tx;journal.intent('proposal')
        if method[0]=='P':
            if journal.value['attempts']['forward']>=journal.value['limits']['forward']:raise ValueError('finite forward budget exhausted')
            journal.value['attempts']['forward']+=1;journal.save()
        mask=env.action_mask();start=perf_counter()
        try:
            fl,p=policy_probabilities(observation,mask) if method[0]=='P' else uniform_probabilities(mask)
            check_rng(tx['rng'])
        except Exception as exc:
            try:check_rng(tx['rng'])
            except ValueError:pass
            tx['failure']=dict(phase='proposal',error=type(exc).__name__,message=str(exc),unknown_not_zero=True)
            journal.save();raise
        proposal_seconds=perf_counter()-start
        proposals=[draw(fl,p,mask,case_id,method[0],repeat,env.beam_idx,k) for k in range(int(method[1]))]
        for row in proposals:row['distribution_and_draw_seconds']=proposal_seconds
        tx.update(proposals=proposals,stage='proposed')
        if method[0]=='P':journal.value['completed']['forward']+=1
        journal.finish('proposal')
    return decision(env,proposals,journal,context,stop_requested)

def candidate_log(journal):
    tx=journal.value['transaction']
    if tx['stage']!='committed':raise ValueError('only complete committed evidence may be published')
    outcomes=[]
    for action,item in tx['results'].items():
        _,reward,done,info=item['result']
        outcomes.append(dict(action=list(action),first_index=item['first_index'],reward=reward,
            done=done,beam_service_step=info['beam_service_step'],satisfaction_increment=info['satisfaction_increment'],
            before_SI_sum=tx['base']['values']['total_satisfaction'],after_SI_sum=item['state']['values']['total_satisfaction'],
            next_state_SHA=state_sha(item['state']),query_seconds=item['query_seconds']))
    return dict(context=tx['context'],base_SHA=tx['base_SHA'],proposals=tx['proposals'],
        candidates=outcomes,selected=tx['selected'],unique_candidates=tx['unique_candidates'],cache_hits=tx['cache_hits'],
        commit_seconds=tx['commit_seconds'],cost_attempts=journal.value['attempts'],cost_completed=journal.value['completed'])
