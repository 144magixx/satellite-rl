"""Admitted A1 science composition; all new work passes the shared store.

Scientific inputs, full Actor bytes and A0 facts are SHA-bound by the workbench.
No Q object, optimizer, new prefix replay or static-kernel build exists here.
"""
from . import core,a1_core as math_core,a1_policy as policy
from .a1_inputs import A0Facts
from .a1_model import ActorOnly


def _domain():
    for start in range(100):
        for count in range(1,11):
            if start+count<=100:
                for power in range(5,51,5):yield start,count,float(power)


def _action_tuple(start,count,power):return (start,count-1,int(power/5)-1)


def _query_id(sid,start,count,power):return sid+'/enum/'+str(start)+'/'+str(count)+'/'+str(int(power))


def _cached_result(context,state,point,spec):
    if point['spec']['valid']!=spec['valid']:raise ValueError('A0/A1 legality disagreement')
    if not spec['valid']:return dict(status='invalid',spec=spec,physical_calls=0)
    return math_core.result_from_a0(context,state,point['result'])


def _check_query(context,state,result,start,count,power):
    if result['status']!='completed':raise ValueError('required original-domain query not completed')
    if result['state_SHA']!=core.content_sha(state) or result['action']!=dict(start=start,count=count,power_w=float(power)):
        raise ValueError('durable query state/action changed')
    return result


def execute(config,store,mode='science'):
    if mode!='science':raise ValueError('independent a1_engineering owns engineering execution')
    if config['binding']['Q_status']!='unavailable_same_node_150000':raise ValueError('missing-Q frozen scope')
    mothers=config['mothers'];lane=config['lane']
    if type(lane) is not int or not 0<=lane<4 or [row['mother'] for row in mothers]!=list(range(20)):
        raise ValueError('exact original ordered twenty mothers and fixed lane required')
    facts=A0Facts(config)
    actor=ActorOnly.load(config['actor_path'],config['binding']['actor_SHA'],device=config.get('actor_device','cuda'),
                         expected_metadata=config.get('actor_metadata'))
    store.record_metadata('Actor_provenance',actor.provenance)
    states_done=0;branches_done=0
    for mother in mothers:
        if mother['mother']%4!=lane:continue
        store.boundary();context,actions=facts.mother(mother)
        for prefix in core.prefix_positions(context.N):
            sid=mother['case_id']+'/'+str(prefix)
            if sid in store.closed_states:
                states_done+=1;continue
            store.boundary();state,snapshot_ref,cache=facts.state(context,mother,prefix)
            expected_ids=[]
            snapshot_id=sid+'/A0_snapshot';store.alias(snapshot_id,snapshot_ref,state=sid);expected_ids.append(snapshot_id)
            mask=math_core.action_mask(context,state);law_id=sid+'/Actor_law'
            law=store.transact(law_id,dict(forward=1,observation=1),
                lambda:actor.forward_probs(math_core.observation(context,state),mask),state=sid)
            expected_ids.append(law_id)
            fl,power=policy.normalize_probabilities(law['frequency_slots'],law['power'],mask)
            reward_max=None;reward_max_action=None
            previous_fl=None;block=None
            for start,count,watts in _domain():
                store.boundary();op_id=_query_id(sid,start,count,watts);expected_ids.append(op_id)
                if previous_fl!=(start,count):previous_fl=(start,count);block=None
                spec=core.candidate_spec(context,state,start,count,watts,'original')
                point=cache.get((start,count,watts))
                if point is not None:
                    store.alias(op_id,point['source_ref'],state=sid);result=_cached_result(context,state,point,spec)
                else:
                    saved=store.get(op_id)
                    if saved is not None:
                        result=saved
                    elif not spec['valid']:
                        result=store.transact(op_id,{},lambda spec=spec:dict(status='invalid',spec=spec,physical_calls=0),state=sid)
                    else:
                        need_block=block is None
                        def query():
                            nonlocal block
                            if block is None:block=math_core.build_fl_coefficients(context,state,start,count)
                            return math_core.evaluate_power(context,state,start,count,watts,block)
                        # A resumed FL may need one new coefficient block. It is
                        # charged inside the next genuinely new query, never a
                        # free rebuild, static kernel, or duplicate PHY replay.
                        result=store.transact(op_id,dict(phy=1,enumeration=1,coefficient=int(need_block)),query,state=sid)
                if spec['valid']:
                    _check_query(context,state,result,start,count,watts)
                    action=_action_tuple(start,count,watts);reward=float(result['reward'])
                    if reward_max is None or reward>reward_max or reward==reward_max and action<reward_max_action:
                        reward_max=reward;reward_max_action=action
                elif result['status']!='invalid':raise ValueError('illegal enumeration point gained numeric score')
            if reward_max_action is None:raise ValueError('original action domain unexpectedly empty')
            selection_id=sid+'/candidate_slots'
            selection=store.transact(selection_id,{},lambda:policy.select_slots(mother['case_id'],prefix,
                actions[prefix],reward_max_action,fl,power,mask),state=sid)
            expected_ids.append(selection_id)
            if selection['Q_available'] or len(selection['unique_actions'])>5 or selection['slots'][2]['status']!='not_available':
                raise ValueError('missing-Q candidate contract')
            for candidate in selection['unique_actions']:
                owner=candidate['slot'];f,l,p=candidate['action'];count=l+1;watts=float((p+1)*5)
                first_id=_query_id(sid,f,count,watts);point=cache.get((f,count,watts))
                if point is not None:first=_cached_result(context,state,point,core.candidate_spec(context,state,f,count,watts,'original'))
                else:first=store.get(first_id)
                _check_query(context,state,first,f,count,watts)
                first_probability=policy.action_probability(fl,power,(f,l,p))
                for repeat in (0,1):
                    branch=sid+'/suffix/slot'+str(owner)+'/repeat'+str(repeat)
                    current=math_core.advance_state(context,state,first)
                    accumulator=policy.accumulate_return(policy.new_return_state(),first['reward'],first_probability['log_probability'])
                    while current['beam_idx']<context.N:
                        store.boundary();absolute=current['beam_idx'];step_id=branch+'/step/'+str(absolute)
                        def suffix_step():
                            current_mask=math_core.action_mask(context,current)
                            probabilities=actor.forward_probs(math_core.observation(context,current),current_mask)
                            rng=policy.uniforms(mother['case_id'],prefix,'suffix',repeat=repeat,absolute_beam_index=absolute)
                            action=policy.draw_action(probabilities['frequency_slots'],probabilities['power'],current_mask,rng['uniforms'])
                            af,al,ap=action;acount=al+1;apower=float((ap+1)*5)
                            current_block=math_core.build_fl_coefficients(context,current,af,acount)
                            result=math_core.evaluate_power(context,current,af,acount,apower,current_block)
                            result.update(action_indices=list(action),probability=policy.action_probability(probabilities['frequency_slots'],probabilities['power'],action),rng=rng)
                            return result
                        result=store.transact(step_id,dict(phy=1,suffix=1,forward=1,coefficient=1,observation=1),suffix_step,state=sid)
                        expected_ids.append(step_id)
                        current=math_core.advance_state(context,current,result)
                        accumulator=policy.accumulate_return(accumulator,result['reward'],result['probability']['log_probability'])
                    terminal_id=branch+'/terminal'
                    def terminal():
                        return dict(status='completed',slot=owner,repeat=repeat,first_query_ref=first_id,
                            first_probability=first_probability,returns=accumulator,terminal=math_core.terminal_summary(context,current),
                            interpretation='two-sample fixed-policy continuation; no exact Q or training claim')
                    store.transact(terminal_id,{},terminal,state=sid);expected_ids.append(terminal_id);branches_done+=1
            store.close_state(sid,expected_ids=expected_ids);states_done+=1
    if states_done!=15:raise ValueError('lane did not resolve its fixed fifteen states')
    store.record_metadata('scientific_complete',dict(status='completed',states=states_done,
        newly_reconstructed_branches=branches_done,Q_available=False,parameter_updates=0))
    return dict(status='completed',states=states_done,branches=branches_done,Q_available=False,parameter_updates=0)
