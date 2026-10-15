"""Independent A1 same-state engineering qualification; no automatic science gate.

Called only by admitted a1_workbench. Budget charged for both new and old paths.
"""
import copy
import time
import numpy as np
from .ledger import Halt


def reference_environment(ctx,state,stage_input=None):
    from satellite_rl.t0_width.environment import T0WidthEnv
    from satellite_rl.t0_width.schema import observation_schema
    env=T0WidthEnv.__new__(T0WidthEnv)
    env.stage_input=stage_input
    if stage_input is None:
        class Stage:
            group_by_beam_id={bid:int(ctx.groups[i]) for i,bid in enumerate(ctx.ids)}
            def export_actual_allocation(self,actual,*,requested_actions_separate):
                # Engineering reference never publishes this allocation or uses
                # it as S3 input. Avoid an unrelated publication authority call.
                return dict(engineering_only=True,actual_by_beam_id=actual)
        env.stage_input=Stage()
    env.group='S10';env.schema=observation_schema('S10');env.layout=ctx.layout;env.physics=ctx.physics
    env.ids=list(ctx.ids);env.order=list(ctx.identity['order']);env.indices=ctx.indices;env.groups=ctx.groups
    env.N=ctx.N;env.demands=ctx.demands;env.beta=float(ctx.demands.mean());env.kernel=ctx.kernel
    env.noise=ctx.physics['k_J_per_K']*ctx.physics['noise_temperature_K']*ctx.physics['slot_Hz']
    for name in ('beam_idx','starts','counts','powers','occupied','rates','total_satisfaction'):setattr(env,name,copy.deepcopy(state[name]))
    # _observation/step before terminal do not consume requested history.
    env.requested=[]
    return env


def reference_forward(actor,obs,mask):
    """Original masked_policy path, independently outside ActorOnly wrapper."""
    import torch
    from satellite_rl.agents.action_masking import masked_policy
    actor.network.eval()
    with torch.inference_mode():
        x=torch.as_tensor(obs,dtype=torch.float32,device=actor.device).reshape(1,-1)
        fm=torch.as_tensor(mask['frequency_slots'],dtype=torch.bool,device=actor.device).unsqueeze(0)
        pm=torch.as_tensor(mask['power'],dtype=torch.bool,device=actor.device).unsqueeze(0)
        fl,power,*_=masked_policy(actor.network(x),fm,pm)
    fl=fl[0].cpu().numpy().astype(np.float64);power=power[0].cpu().numpy().astype(np.float64)
    return dict(frequency_slots=fl/fl.sum(),power=power/power.sum(),forward_calls=1)


def select_states(mothers,lane):
    """Before outcomes: worst registered workload per prefix stratum plus next worst."""
    from .core import prefix_positions
    workload=[]
    for mother,row in enumerate(mothers):
        if mother%4!=lane:continue
        for stratum,prefix in enumerate(prefix_positions(row['N'])):
            slots=sum(action[1]+1 for action in row['actions_indices_by_processing_order'][:prefix])
            workload.append(dict(mother=mother,prefix=prefix,stratum=stratum,N=row['N'],slots=slots,
                score=row['N']**2+(slots+10)*(prefix+1)))
    selected=[]
    for stratum in range(3):selected.append(max((r for r in workload if r['stratum']==stratum),key=lambda r:(r['score'],-r['mother'])))
    selected.append(max((r for r in workload if r not in selected),key=lambda r:(r['score'],-r['mother'],-r['prefix'])))
    return selected,workload


def _same_result(new,old):
    np.testing.assert_allclose(new['capacity_bps'],old['capacity_bps'],rtol=1e-12,atol=1e-6)
    np.testing.assert_allclose(new['reward'],old['reward'],rtol=1e-12,atol=1e-12)
    if new['action']!=old['action']:raise Halt('reference requested/executed action differs')


def execute(config,store,mode='engineering'):
    if mode!='engineering':raise Halt('engineering mode only')
    from . import a1_core as core
    from .a1_inputs import A0Facts
    from .a1_model import ActorOnly
    from .a1_policy import uniforms,draw_action,action_probability,new_return_state,accumulate_return
    started=time.monotonic();before=started
    facts=A0Facts(config);selected,workload=select_states(config['mothers'],config['lane'])
    input_seconds=time.monotonic()-before;before=time.monotonic()
    actor=ActorOnly.load(config['actor_path'],config['binding']['actor_SHA'],device=config.get('actor_device','cuda'),expected_metadata=config.get('actor_metadata'))
    actor_load_seconds=time.monotonic()-before
    durations=[];differences=[];forward_durations=[];observation_durations=[]
    for position,item in enumerate(selected):
        row=config['mothers'][item['mother']]
        before=time.monotonic();ctx,actions=facts.mother(row)
        state,snapshot_ref,cache=facts.state(ctx,row,item['prefix'])
        input_seconds+=time.monotonic()-before
        state_id='engineering/'+str(item['mother'])+'/'+str(item['prefix']);expected=[]
        state_sha=core.content_sha(state)
        before=time.monotonic();env=reference_environment(ctx,state)
        obs=core.observation(ctx,state);old_obs=env._observation();np.testing.assert_array_equal(obs,old_obs)
        mask=core.action_mask(ctx,state);old_mask=env.action_mask()
        for key in mask:np.testing.assert_array_equal(mask[key],old_mask[key])
        observation_durations.append(time.monotonic()-before)
        def law_pair(tag,current,current_env):
            nonlocal forward_durations
            observation=core.observation(ctx,current);old_observation=current_env._observation()
            np.testing.assert_array_equal(observation,old_observation);current_mask=core.action_mask(ctx,current)
            before=time.monotonic()
            new=store.transact(state_id+'/'+tag+'/actor',dict(forward=1),lambda:actor.forward_probs(observation,current_mask),state=state_id)
            old=store.transact(state_id+'/'+tag+'/reference_actor',dict(forward=1),lambda:reference_forward(actor,old_observation,current_mask),state=state_id)
            forward_durations.append(time.monotonic()-before)
            for key in ('frequency_slots','power'):np.testing.assert_allclose(new[key],old[key],rtol=1e-6,atol=1e-7)
            return new,current_mask
        law,mask=law_pair('prefix',state,env)
        legal=np.argwhere(mask['frequency_slots']);original=actions[item['prefix']]
        pairs=[(original[0],original[1]+1)]
        ranked=sorted(((int(f),int(l)+1) for f,l in legal),key=lambda p:(-p[1],p[0]))
        for pair in (ranked[0],ranked[-1]):
            if pair not in pairs:pairs.append(pair)
        powers=sorted(set([5.,float((int(np.flatnonzero(mask['power'])[-1])+1)*5),float((original[2]+1)*5)]))
        original_result=None
        for f,count in pairs:
            block=None
            for power in powers:
                if not mask['power'][int(power/5)-1]:continue
                op_id=state_id+'/query/'+str(f)+'/'+str(count)+'/'+str(int(power));expected.append(op_id)
                before=time.monotonic();needs=block is None
                def new_query():
                    nonlocal block
                    if block is None:block=core.build_fl_coefficients(ctx,state,f,count)
                    return core.evaluate_power(ctx,state,f,count,power,block)
                charges=store.operations[op_id]['charges'] if op_id in store.operations else dict(phy=1,coefficient=1 if needs else 0)
                new=store.transact(op_id,charges,new_query,state=state_id)
                durations.append(time.monotonic()-before)
                if core.content_sha(state)!=state_sha:raise Halt('query mutated shared prefix')
                reference_id=op_id+'/reference';expected.append(reference_id)
                def old_query():
                    old_env=reference_environment(ctx,state)
                    outcome=old_env.step([f,count-1,int(power/5)-1])
                    return dict(capacity_bps=old_env.rates.copy(),reward=float(outcome[1]),
                        action=outcome[3]['actual_action'],observation=outcome[0])
                old=store.transact(reference_id,dict(phy=1,coefficient=1),old_query,state=state_id)
                _same_result(new,old)
                post=core.advance_state(ctx,state,new)
                np.testing.assert_array_equal(core.observation(ctx,post),old['observation'])
                differences.append(dict(mother=item['mother'],prefix=item['prefix'],action=new['action'],
                    max_capacity_abs_diff=float(np.max(np.abs(new['capacity_bps']-old['capacity_bps']))),reward_abs_diff=abs(new['reward']-old['reward'])))
                if (f,count,power)==(original[0],original[1]+1,float((original[2]+1)*5)):original_result=new
        if original_result is None:raise Halt('original engineering action not checked')
        before_counts=dict(store.counts)
        invalid=core.evaluate_power(ctx,state,99,10,.5,None)
        if invalid['physical_calls']!=0 or store.counts!=before_counts:raise Halt('illegal action consumed physical allowance')
        # Continue one sampled next step under frozen common-uniform law.
        post=core.advance_state(ctx,state,original_result);old_env=reference_environment(ctx,post)
        post_law,post_mask=law_pair('post',post,old_env)
        rng=uniforms(config['mothers'][item['mother']]['case_id'],item['prefix'],'suffix',repeat=0,absolute_beam_index=post['beam_idx'])
        chosen=draw_action(post_law['frequency_slots'],post_law['power'],post_mask,rng['uniforms'])
        f,l,p=chosen;op_id=state_id+'/suffix';expected.append(op_id)
        def suffix_query():
            block=core.build_fl_coefficients(ctx,post,f,l+1)
            value=core.evaluate_power(ctx,post,f,l+1,float((p+1)*5),block)
            value.update(logpi=action_probability(post_law['frequency_slots'],post_law['power'],chosen),rng=rng)
            return value
        before=time.monotonic();new=store.transact(op_id,dict(phy=1,coefficient=1),suffix_query,state=state_id);durations.append(time.monotonic()-before)
        def suffix_reference():
            outcome=old_env.step(chosen)
            return dict(capacity_bps=old_env.rates.copy(),reward=float(outcome[1]),action=outcome[3]['actual_action'])
        old=store.transact(op_id+'/reference',dict(phy=1,coefficient=1),suffix_reference,state=state_id);expected.append(op_id+'/reference');_same_result(new,old)
        # Current entropy exclusion and probability-zero representation need no calls.
        ret=accumulate_return(new_return_state(),original_result['reward'],float('-inf'))
        if ret['ordinary_discounted']!=ret['Q_semantic_soft_sample']:raise Halt('initial entropy entered Q-semantic return')
        replayed=store.get(op_id)
        if replayed['capacity_bps'].tobytes()!=new['capacity_bps'].tobytes():raise Halt('lossy numeric block')
        store.close_state(state_id,expected_ids=expected)
    store.record_metadata('engineering_evidence',dict(status='checks_passed_requires_root_acceptance',
        selected=selected,all15_workload_proxies=workload,proxy_is_runtime_bound=False,capacity_reward_differences=differences,
        durable_new_query_seconds=durations,paired_forward_seconds=forward_durations,observation_seconds=observation_durations,
        input_seconds=input_seconds,actor_load_seconds=actor_load_seconds,wall_seconds=time.monotonic()-started,
        counts=dict(store.counts),Actor_node=150000,Critic_used=False,updates=0,science_admitted=False))
