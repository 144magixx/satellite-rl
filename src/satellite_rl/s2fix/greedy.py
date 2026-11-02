"""Capacity-aware eight-group allocation with member-average link gains."""
import numpy as np
from satellite_rl.shared_interface.link import prepare_slot_kernel
from satellite_rl.shared_interface.geometry import directions_and_distance
from satellite_rl.shared_interface.spherical_gaussian import PHYSICS_V2


def proportional_power(demand, budget=3000., cap=50.):
    demand=np.asarray(demand,float)
    power=np.zeros(len(demand))
    free=np.ones(len(demand),bool)
    remaining=min(budget,cap*len(demand))
    while free.any():
        quota=remaining*demand[free]/demand[free].sum()
        indices=np.flatnonzero(free)
        hit=quota>=cap
        if not hit.any():
            power[indices]=quota
            break
        power[indices[hit]]=cap
        free[indices[hit]]=False
        remaining-=cap*hit.sum()
    power*=min(1.,budget/np.nextafter(power.sum(),np.inf))
    return power


def greedy_allocation(stage_input):
    """Return groups, starts, counts, powers in layout beam order."""
    cached=getattr(stage_input,"_s2fix_greedy_allocation",None)
    if cached is not None:return tuple(value.copy() for value in cached)
    layout=stage_input.layout
    config=stage_input.physics
    kernel=getattr(stage_input,'_s2fix_slot_kernel',None)
    if kernel is None:
        kernel=prepare_slot_kernel(layout,config,stage_input.case)
        stage_input._s2fix_slot_kernel=kernel
    demand=np.asarray(layout['request_bps_by_beam_int64'])
    n=len(demand)
    gain=np.maximum(kernel['gain'],kernel['gain'].T)
    power=proportional_power(demand)
    estimated_power=np.minimum(config['power_budget_w']*demand/demand.sum(),50.)
    _,distance=directions_and_distance(layout['centres_lat_lon_deg'],validate=False)
    coefficient=(10**(-config['signal_loss_dB']/10)*np.diag(kernel['gain'])*
        10**(config['rx_gain_dBi']/10)*(config['c_m_per_s']/
        (4*np.pi*distance*config['f0_Hz']))**2)
    noise=config['k_J_per_K']*config['noise_temperature_K']*config['slot_Hz']
    slots=np.full(n,10,int)
    for k in range(1,11):
        snr=coefficient*estimated_power/k/noise
        capacity=k*config['slot_Hz']*np.minimum(np.log2(1+snr*
            10**(-config['implementation_loss_dB']/10)),config['spectral_efficiency_cap'])
        eligible=(slots==10)&(capacity>=demand)
        slots[eligible]=k
    order=np.argsort(-demand,kind='stable')
    groups=np.full(n,-1,int)
    used=np.zeros(8,int)
    penalty=gain.max()*n
    for i in order:
        costs=[gain[i,(groups>=0)&(groups%2==g%2)&(groups!=g)].sum()+
            penalty*max(0,used[g]+slots[i]-100) for g in range(8)]
        groups[i]=g=int(np.argmin(costs))
        used[g]+=slots[i]
    for g in range(8):
        members=np.flatnonzero(groups==g)
        if slots[members].sum()>100:
            quota=(100-len(members))*demand[members]/demand[members].sum()
            count=np.minimum(slots[members],1+np.floor(quota).astype(int))
            while count.sum()<100 and np.any(count<slots[members]):
                score=np.where(count<slots[members],quota-(count-1),-np.inf)
                count[np.argmax(score)]+=1
            slots[members]=count
    starts=np.zeros(n,int)
    cursor=np.zeros(8,int)
    remaining=np.bincount(groups,weights=slots,minlength=8).astype(int)
    assigned=np.zeros(n,bool)
    for i in order:
        g=groups[i]
        other=np.flatnonzero(assigned&(groups%2==g%2)&(groups!=g))
        choices=np.arange(cursor[g],101-remaining[g])
        overlap=np.maximum(0,np.minimum(choices[:,None]+slots[i],starts[other]+slots[other])-
            np.maximum(choices[:,None],starts[other]))
        starts[i]=choices[np.argmin(overlap@gain[i,other])]
        cursor[g]=starts[i]+slots[i]
        remaining[g]-=slots[i]
        assigned[i]=True
    stage_input._s2fix_greedy_allocation=tuple(value.copy() for value in (groups,starts,slots,power))
    return groups,starts,slots,power


def spread_allocation(stage_input):
    """分散摆放规则：槽数估计与比例功率同容量感知贪心；组号为需求排序后的序号 %8，
    起点取组内任意空位中与已放同极化异组波束加权重叠最小者（组内空段放不下整块时缩到最长空段）。"""
    cached=getattr(stage_input,"_s2fix_spread_allocation",None)
    if cached is not None:return tuple(value.copy() for value in cached)
    power=greedy_allocation(stage_input)[3]   # 同时缓存链路核
    layout=stage_input.layout
    kernel=stage_input._s2fix_slot_kernel
    demand=np.asarray(layout['request_bps_by_beam_int64'])
    n=len(demand)
    gain=np.maximum(kernel['gain'],kernel['gain'].T)
    # 贪心里的组内缩减会改槽数；分散规则用缩减前的估计，再按自己的分组缩减
    config=stage_input.physics
    estimated_power=np.minimum(config['power_budget_w']*demand/demand.sum(),50.)
    _,distance=directions_and_distance(layout['centres_lat_lon_deg'],validate=False)
    coefficient=(10**(-config['signal_loss_dB']/10)*np.diag(kernel['gain'])*
        10**(config['rx_gain_dBi']/10)*(config['c_m_per_s']/(4*np.pi*distance*config['f0_Hz']))**2)
    noise=config['k_J_per_K']*config['noise_temperature_K']*config['slot_Hz']
    slots=np.full(n,10,int)
    for k in range(1,11):
        snr=coefficient*estimated_power/k/noise
        capacity=k*config['slot_Hz']*np.minimum(np.log2(1+snr*
            10**(-config['implementation_loss_dB']/10)),config['spectral_efficiency_cap'])
        slots[(slots==10)&(capacity>=demand)]=k
    order=np.argsort(-demand,kind='stable')
    groups=np.empty(n,int)
    groups[order]=np.arange(n)%8
    for g in range(8):
        members=np.flatnonzero(groups==g)
        if slots[members].sum()>100:
            quota=(100-len(members))*demand[members]/demand[members].sum()
            count=np.minimum(slots[members],1+np.floor(quota).astype(int))
            while count.sum()<100 and np.any(count<slots[members]):
                score=np.where(count<slots[members],quota-(count-1),-np.inf)
                count[np.argmax(score)]+=1
            slots[members]=count
    starts=np.zeros(n,int)
    occupied=np.zeros((8,100),bool)
    assigned=np.zeros(n,bool)
    for i in order:
        g=groups[i]
        choices=np.array([s for s in range(101-slots[i]) if not occupied[g,s:s+slots[i]].any()])
        if len(choices)==0:
            free=np.flatnonzero(~occupied[g])
            longest=max(np.split(free,np.flatnonzero(np.diff(free)>1)+1),key=len)
            slots[i]=len(longest)
            choices=np.array([longest[0]])
        other=np.flatnonzero(assigned&(groups%2==g%2)&(groups!=g))
        overlap=np.maximum(0,np.minimum(choices[:,None]+slots[i],starts[other]+slots[other])-
            np.maximum(choices[:,None],starts[other]))
        starts[i]=choices[np.argmin(overlap@gain[i,other])]
        occupied[g,starts[i]:starts[i]+slots[i]]=True
        assigned[i]=True
    stage_input._s2fix_spread_allocation=tuple(value.copy() for value in (groups,starts,slots,power))
    return groups,starts,slots,power
