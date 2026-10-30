"""A1 mathematical primitives. Caller meters every coefficient/PHY/observation.

No operation here dispatches, retries or persists. FLBlock is ephemeral: never
serialize its potentially large coefficient matrix per query. Query records keep
one lossless full-N float64 capacity vector and shared-context references.
"""
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from satellite_rl.shared_interface.link import build_link_coefficients as _build, capacity_bps as _capacity, raw_slot_coupling as _coupling
from satellite_rl.shared_interface.spherical_gaussian import PHYSICS_SHA
from satellite_rl.t0_width.environment import sgm_values,legacy_repaired_reward
from satellite_rl.t0_width.logging import content_sha
from . import core

STATE_ARRAYS=('starts','counts','powers','occupied','rates')


def reuse_context(stage_input,kernel):
    """Reuse an externally SHA-verified A0 kernel; zero static-kernel calls."""
    from satellite_rl.shared_interface.stages import UniformStage2Input
    if not isinstance(stage_input,UniformStage2Input):raise TypeError('validated shared input required')
    ids=tuple(stage_input.layout['beam_ids']);order=list(stage_input.processing_index_to_beam_id)
    indices=np.asarray([ids.index(b) for b in order],dtype=np.int64)
    groups=np.asarray([stage_input.group_by_beam_id[b] for b in ids],dtype=np.int64)
    demands=np.asarray(stage_input.layout['request_bps_by_beam_int64'],dtype=np.float64).copy()
    if kernel['layout_SHA']!=stage_input.layout['content_sha256'] or kernel['physics_SHA']!=PHYSICS_SHA:raise ValueError('A0 kernel identity')
    if kernel['gain'].shape!=(len(ids),len(ids)) or kernel['path'].shape!=(len(ids),100):raise ValueError('A0 kernel dimensions')
    for array in (indices,groups,demands,kernel['gain'],kernel['path']):array.setflags(write=False)
    identity=dict(layout_SHA=stage_input.layout['content_sha256'],physics_SHA=PHYSICS_SHA,order=order,groups=groups.tolist())
    return core.Context(stage_input.layout,stage_input.physics,ids,indices,groups,demands,kernel,identity)


def state_from_a0(context,snapshot):
    old=core.restore_snapshot(context,snapshot)
    return dict(schema='W29A1CompactStateV1',identity=deepcopy(context.identity),beam_idx=old['beam_idx'],
        total_satisfaction=float(old['total_satisfaction']),**{key:old[key] for key in STATE_ARRAYS})


def copy_state(context,state):
    if state['schema']!='W29A1CompactStateV1' or state['identity']!=context.identity:raise ValueError('A1 state source')
    k=state['beam_idx']
    if type(k) is not int or not 0<=k<=context.N:raise ValueError('A1 cursor')
    result=dict(schema=state['schema'],identity=deepcopy(state['identity']),beam_idx=k,total_satisfaction=float(state['total_satisfaction']))
    for key in STATE_ARRAYS:
        value=state[key];dtype=bool if key=='occupied' else np.int64 if key in ('starts','counts') else np.float64
        shape=(8,100) if key=='occupied' else (context.N,)
        if value.shape!=shape or value.dtype!=dtype or not np.isfinite(value).all():raise ValueError('A1 array '+key)
        result[key]=value.copy()
    return result


def action_mask(context,state):
    fl=core.legal_frequency_slots(context,state);power=np.zeros(10,dtype=bool);k=state['beam_idx']
    if k<context.N:
        power[:]=np.arange(5,51,5)<=3000-float(state['powers'].sum())-5*(context.N-k-1)
        if not fl.any() or not power.any():raise ValueError('empty original legal action domain')
    return dict(frequency_slots=fl,power=power)


def observation(context,state):
    """Exact original S10 208-state ordering; raw coupling work is wall-metered."""
    k=state['beam_idx']
    if k==context.N:return np.zeros(208,dtype=np.float32)
    c=int(context.indices[k]);g=int(context.groups[c]);lat,lon=context.layout['centres_lat_lon_deg'][c]
    active=np.flatnonzero(state['counts']);incoming=np.zeros(100,dtype=np.float64)
    if len(active):
        coupling=_coupling(context.layout,context.groups,context.physics,[c],active,kernel=context.kernel)[0]
        for j,index in enumerate(active):
            start=int(state['starts'][index]);count=int(state['counts'][index])
            incoming[start:start+count]+=coupling[j,start:start+count]*state['powers'][index]/count
    normalized=np.clip((10*np.log10(incoming+1e-30)+300)/200,0,1)
    remaining=context.indices[k:];beta=float(context.demands.mean())
    parts=[np.asarray([2*(lat-52.5)/55,2*(lon-5)/90,context.demands[c]/550e6,(3000-state['powers'].sum())/3000]),
        normalized,state['occupied'][g].astype(float),np.asarray([beta/6e8,k/context.N,context.demands[remaining].sum()/(context.N*6e8)]),
        np.asarray([context.layout['full_width_deg'][c]])]
    result=np.concatenate(parts).astype(np.float32)
    if result.shape!=(208,) or not np.isfinite(result).all():raise ValueError('S10 observation schema')
    return result


@dataclass(frozen=True)
class FLBlock:
    state_SHA: str
    start: int
    count: int
    coefficients: object


def build_fl_coefficients(context,state,start,count):
    """One charged coefficient build; selected power is not a coefficient input.

    The common function stores powers_initial for convenience, but all signal,
    noise and interference coefficients depend on layout/active counts/FL only.
    evaluate_power always supplies the complete actual power vector explicitly.
    """
    spec=core.candidate_spec(context,state,start,count,5.,'original')
    if not spec['valid']:raise ValueError('illegal FL coefficients: '+repr(spec['invalid_reasons']))
    k=state['beam_idx'];current=int(context.indices[k]);starts=state['starts'].copy();counts=state['counts'].copy();powers=state['powers'].copy()
    starts[current]=start;counts[current]=count;powers[current]=5.
    allocation=dict(beam_ids=list(context.ids),actual_start_slot=starts,actual_slot_count=counts,
        actual_whole_beam_power_w=powers,group_by_beam_id={bid:int(context.groups[i]) for i,bid in enumerate(context.ids)},content_sha256='A1_internal_partial_not_published')
    coefficients=_build(context.layout,allocation,context.physics,kernel=context.kernel)
    return FLBlock(content_sha(state),start,count,coefficients)


def evaluate_power(context,state,start,count,power_w,block):
    """Exactly one charged capacity call; no coefficient rebuild or state replay."""
    spec=core.candidate_spec(context,state,start,count,power_w,'original')
    if not spec['valid']:return dict(status='invalid',spec=spec,physical_calls=0)
    if block.state_SHA!=content_sha(state) or (block.start,block.count)!=(start,count):raise ValueError('coefficient prefix/FL mismatch')
    k=state['beam_idx'];current=int(context.indices[k]);powers=state['powers'].copy();powers[current]=float(power_w)
    rates=np.asarray(_capacity(block.coefficients,power_w=powers),dtype=np.float64)
    if rates.shape!=(context.N,) or not np.isfinite(rates).all() or (rates<0).any():raise ValueError('capacity schema')
    beta=float(context.demands.mean());processed=context.indices[:k+1]
    si=np.minimum(rates/context.demands,1);sgm=sgm_values(rates,context.demands,beta)
    total=float(si[processed].sum());increment=total-state['total_satisfaction']
    current_rate=float(rates[current]);demand=float(context.demands[current])
    reward=legacy_repaired_reward(current_rate,demand,beta,increment,0,count)
    return dict(status='completed',physical_calls=1,state_SHA=block.state_SHA,beam_id=context.ids[current],
        action=dict(start=start,count=count,power_w=float(power_w)),capacity_bps=rates.copy(),reward=float(reward),
        requested_action=dict(start=start,count=count,power_w=float(power_w)),actual_action=dict(start=start,count=count,power_w=float(power_w)),
        reward_parts=dict(SI_threshold=1. if increment>=1 else -1.,current_SGM=float(sgm[current]),
            deviation_penalty=max(abs(current_rate-demand)-5e7,0.)/1e9,occupancy_penalty=0.,satisfaction_increment=increment),
        current=dict(capacity_bps=current_rate,demand_bps=demand,SI=float(si[current]),SGM=float(sgm[current])),
        prefix_SI_sum=total,prefix_SGM_sum=float(sgm[processed].sum()),
        total_power_w=float(powers.sum()),remaining_reserved_power_w=5*(context.N-k-1))


def advance_state(context,state,result):
    if result['status']!='completed' or result['state_SHA']!=content_sha(state):raise ValueError('post-action source mismatch')
    new=copy_state(context,state);k=new['beam_idx'];current=int(context.indices[k]);action=result['action']
    spec=core.candidate_spec(context,new,action['start'],action['count'],action['power_w'],'original')
    if not spec['valid'] or result['beam_id']!=context.ids[current]:raise ValueError('post-action legality/identity')
    new['starts'][current]=action['start'];new['counts'][current]=action['count'];new['powers'][current]=action['power_w']
    new['occupied'][context.groups[current],action['start']:action['start']+action['count']]=True
    new['rates']=np.asarray(result['capacity_bps'],dtype=np.float64).copy();new['total_satisfaction']=float(result['prefix_SI_sum']);new['beam_idx']=k+1
    return new


def result_from_a0(context,state,old_result):
    """Translate a SHA-accepted completed A0 row; zero new physics."""
    if old_result['status']!='completed':raise ValueError('resolve A0 aliases first')
    action=old_result['actual_action'];k=state['beam_idx'];current=int(context.indices[k])
    if action!=old_result['requested_action'] or action['beam_id']!=context.ids[current]:raise ValueError('A0 request/execution source')
    spec=core.candidate_spec(context,state,action['start'],action['count'],action['power_w'],'original')
    if not spec['valid']:raise ValueError('non-original A0 cache point')
    return dict(status='completed',physical_calls=0,state_SHA=content_sha(state),beam_id=action['beam_id'],
        action={key:action[key] for key in ('start','count','power_w')},capacity_bps=np.asarray(old_result['capacity_bps'],dtype=np.float64).copy(),
        requested_action={key:action[key] for key in ('start','count','power_w')},actual_action={key:action[key] for key in ('start','count','power_w')},
        reward=float(old_result['reward']),reward_parts=deepcopy(old_result['reward_parts']),current=deepcopy(old_result['current']),
        prefix_SI_sum=float(old_result['prefix']['SI_sum']),prefix_SGM_sum=float(old_result['prefix']['SGM_sum']),
        total_power_w=float(old_result['constraints']['total_power_w']),remaining_reserved_power_w=5*(context.N-k-1))


def terminal_summary(context,state):
    if state['beam_idx']!=context.N:raise ValueError('incomplete suffix is not terminal')
    rates=state['rates'];si=np.minimum(rates/context.demands,1);sgm=sgm_values(rates,context.demands,float(context.demands.mean()))
    occupied=np.zeros((8,100),dtype=bool);conflicts=0;invalid=0
    for index in range(context.N):
        start=int(state['starts'][index]);count=int(state['counts'][index]);power=float(state['powers'][index]);g=int(context.groups[index])
        if not 1<=count<=10 or not 0<=start<=100-count or power not in range(5,51,5):invalid+=1;continue
        conflicts+=int(occupied[g,start:start+count].any());occupied[g,start:start+count]=True
    constraints=dict(slot_conflicts=conflicts,invalid_original_actions=invalid,total_power_exceeded=float(state['powers'].sum())>3000,
        occupancy_equal=bool(np.array_equal(occupied,state['occupied'])))
    if conflicts or invalid or constraints['total_power_exceeded'] or not constraints['occupancy_equal']:raise ValueError('terminal hard constraint violation')
    return dict(N=context.N,beam_ids=list(context.ids),capacity_bps=rates.copy(),demand_bps=context.demands.copy(),
        layout_SHA=context.identity['layout_SHA'],physics_SHA=PHYSICS_SHA,processing_index_to_beam_id=list(context.identity['order']),
        group_by_beam_id={bid:int(context.groups[i]) for i,bid in enumerate(context.ids)},constraints=constraints,
        SI=si,SGM=sgm,SI_mean=float(si.mean()),SGM_mean=float(sgm.mean()),total_power_w=float(state['powers'].sum()),
        actual_start_slot=state['starts'].copy(),actual_slot_count=state['counts'].copy(),actual_whole_beam_power_w=state['powers'].copy(),
        complete=True,zero_slot_count=int(np.count_nonzero(state['counts']==0)))
