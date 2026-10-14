"""Variable-N shared-physics environment with unchanged T0 reward semantics."""
import copy
from time import perf_counter
import numpy as np
from satellite_rl.shared_interface.stages import UniformStage2Input
from satellite_rl.shared_interface.link import raw_slot_coupling,build_link_coefficients,capacity_bps,prepare_slot_kernel
from satellite_rl.shared_interface.spherical_gaussian import PHYSICS_SHA
from .schema import GROUPS,observation_schema,state_width

def sgm_values(rates,demands,beta):
    result=np.zeros_like(rates,dtype=np.float64);positive=rates>0
    ratio=rates[positive]/demands[positive]
    real=np.where(ratio>=1,ratio-1,1-1/ratio)
    result[positive]=1-(1-np.exp(-np.abs(real+1j*(rates[positive]-demands[positive])/beta)))**3
    return result

def legacy_repaired_reward(rate,demand,beta,si_increment,occupy=0,origin_slots=1):
    sgm=float(sgm_values(np.asarray([rate]),np.asarray([demand]),beta)[0])
    return (1. if si_increment>=1 else -1.)+sgm-max(abs(rate-demand)-5e7,0.)/1e9-occupy/origin_slots

class T0WidthEnv:
    def __init__(self,stage_input,group='S00'):
        if not isinstance(stage_input,UniformStage2Input):raise TypeError('validated UniformStage2Input required')
        self.stage_input=stage_input;self.group=group;self.schema=observation_schema(group)
        self.layout=stage_input.layout;self.physics=stage_input.physics
        self.ids=list(self.layout['beam_ids']);self.order=list(stage_input.processing_index_to_beam_id)
        self.indices=np.asarray([self.ids.index(b) for b in self.order],np.int64)
        self.groups=np.asarray([stage_input.group_by_beam_id[b] for b in self.ids],np.int64)
        self.N=len(self.ids);self.demands=self.layout['request_bps_by_beam_int64'].astype(np.float64)
        self.beta=float(self.demands.mean());self.noise=self.physics['k_J_per_K']*self.physics['noise_temperature_K']*self.physics['slot_Hz']
        self.kernel=prepare_slot_kernel(self.layout,self.physics)
        self.reset()

    def reset(self):
        self.beam_idx=0;self.starts=np.zeros(self.N,np.int64);self.counts=np.zeros(self.N,np.int64);self.powers=np.zeros(self.N,np.float64)
        self.occupied=np.zeros((8,100),bool);self.rates=np.zeros(self.N,np.float64);self.requested=[];self.total_satisfaction=0.
        return self._observation()

    def action_mask(self):
        fl=np.zeros((100,10),bool);power=np.zeros(10,bool)
        if self.beam_idx==self.N:return dict(frequency_slots=fl,power=power)
        c=self.indices[self.beam_idx];g=self.groups[c];remaining=self.indices[self.beam_idx:]
        reserve_slots=int(np.count_nonzero(self.groups[remaining]==g))-1
        free=~self.occupied[g];max_count=int(free.sum())-reserve_slots
        for count in range(1,min(10,max_count)+1):
            for start in range(101-count):fl[start,count-1]=bool(free[start:start+count].all())
        power[:]=np.arange(5,51,5)<=3000-float(self.powers.sum())-5*(self.N-self.beam_idx-1)
        if not fl.any() or not power.any():raise RuntimeError('nonempty reserved T0 legal domain required')
        return dict(frequency_slots=fl,power=power)

    def _coupling(self,receivers,transmitters):
        return raw_slot_coupling(self.layout,self.groups,self.physics,receivers,transmitters,kernel=self.kernel)

    def _observation(self):
        if self.beam_idx==self.N:return np.zeros(state_width(self.group),np.float32)
        c=self.indices[self.beam_idx];g=self.groups[c];lat,lon=self.layout['centres_lat_lon_deg'][c]
        active=np.flatnonzero(self.counts)
        incoming=np.zeros(100,np.float64)
        if len(active):
            K=self._coupling([c],active)[0]
            for j,index in enumerate(active):incoming[self.starts[index]:self.starts[index]+self.counts[index]]+=K[j,self.starts[index]:self.starts[index]+self.counts[index]]*self.powers[index]/self.counts[index]
        normalized=np.clip((10*np.log10(incoming+1e-30)+300)/200,0,1)
        remaining=self.indices[self.beam_idx:]
        parts=[np.asarray([2*(lat-52.5)/55,2*(lon-5)/90,self.demands[c]/550e6,(3000-self.powers.sum())/3000]),normalized,self.occupied[g].astype(float),np.asarray([self.beta/6e8,self.beam_idx/self.N,self.demands[remaining].sum()/(self.N*6e8)])]
        width,relations=GROUPS[self.group]
        if width:parts.append(np.asarray([self.layout['full_width_deg'][c]]))
        if relations:
            out=np.zeros(100,np.float64)
            if len(active):
                K=self._coupling(active,[c])[:,0,:]
                for j,index in enumerate(active):out[self.starts[index]:self.starts[index]+self.counts[index]]+=K[j,self.starts[index]:self.starts[index]+self.counts[index]]/self.noise
            future=self.indices[self.beam_idx+1:];future_in=np.zeros(100,np.float64)
            if len(future):future_in=(self._coupling([c],future)[0]*(~self.occupied[self.groups[future]])).sum(axis=0)/self.noise
            outgoing_model=np.log1p(out);future_model=np.log1p(future_in)
            # Audit-only values never enter policy/critic outside the selected
            # schema. Nonrelation groups do not allocate these records.
            self.relation_audit=dict(out_raw_w_float64=out*self.noise,future_in_raw_w_float64=future_in*self.noise,
                out_observation_float64=outgoing_model.copy(),future_in_observation_float64=future_model.copy(),
                out_observation_float32=outgoing_model.astype(np.float32),future_in_observation_float32=future_model.astype(np.float32))
            parts.extend([outgoing_model,future_model])
        result=np.concatenate(parts)
        if not np.isfinite(result).all():raise ValueError('finite observation required')
        return result.astype(np.float32)

    def export_allocation(self):
        actual={bid:dict(group=int(self.groups[i]),start=int(self.starts[i]),count=int(self.counts[i]),power_w=float(self.powers[i])) for i,bid in enumerate(self.ids)}
        return self.stage_input.export_actual_allocation(actual,requested_actions_separate=copy.deepcopy(self.requested))

    def step(self,action):
        step_started=perf_counter()
        if self.beam_idx==self.N:raise RuntimeError('episode complete')
        if len(action)!=3 or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) for x in action):raise ValueError('three integer action indices required')
        f,l,p=map(int,action);mask=self.action_mask()
        if not (0<=f<100 and 0<=l<10 and 0<=p<10 and mask['frequency_slots'][f,l] and mask['power'][p]):raise ValueError('illegal action; clipping forbidden')
        c=int(self.indices[self.beam_idx]);bid=self.ids[c];self.starts[c]=f;self.counts[c]=l+1;self.powers[c]=(p+1)*5
        self.occupied[self.groups[c],f:f+l+1]=True
        self.requested.append(dict(beam_id=bid,frequency_index=f,count_index=l,power_index=p,start=f,count=l+1,power_w=(p+1)*5))
        # The case/layout were independently audited at import. Internal rows
        # are created only through the legal mask; terminal export revalidates
        # the complete versioned graph without repeating coverage each step.
        allocation=dict(beam_ids=self.ids,actual_start_slot=self.starts,actual_slot_count=self.counts,
            actual_whole_beam_power_w=self.powers,group_by_beam_id=self.stage_input.group_by_beam_id,
            content_sha256='internal_partial_not_published')
        coeff=build_link_coefficients(self.layout,allocation,self.physics,kernel=self.kernel);self.rates=capacity_bps(coeff)
        satisfaction=float(np.minimum(self.rates/self.demands,1).sum());increment=satisfaction-self.total_satisfaction
        reward=legacy_repaired_reward(float(self.rates[c]),float(self.demands[c]),self.beta,increment,0,l+1)
        self.total_satisfaction=satisfaction;self.beam_idx+=1;done=self.beam_idx==self.N
        info=dict(beam_id=bid,requested_action=list(action),actual_action=dict(start=f,count=l+1,power_w=(p+1)*5),rate_bps=float(self.rates[c]),required_rate_bps=float(self.demands[c]),satisfaction_increment=increment,metrics=self.metrics())
        info['beam_service_step']=dict(beam_id=bid,rate_bps=float(self.rates[c]),required_rate_bps=float(self.demands[c]),SI=float(min(self.rates[c]/self.demands[c],1)),SGM=float(sgm_values(self.rates[[c]],self.demands[[c]],self.beta)[0]))
        if done:info['beam_results']=self.beam_results();info['allocation']=self.export_allocation()
        observation=self._observation()
        info['timing']=dict(environment_step_seconds=perf_counter()-step_started)
        return observation,float(reward),done,info

    def metrics(self):
        sgms=sgm_values(self.rates,self.demands,self.beta);si=np.minimum(self.rates/self.demands,1)
        return dict(SI_mean=float(si.mean()),SI_sum=float(si.sum()),SGM=float(sgms.mean()),throughput_bps=float(self.rates.sum()),total_power_w=float(self.powers.sum()),N=self.N,processed=self.beam_idx,complete=self.beam_idx==self.N,zero_slot_rate=float((self.counts==0).mean()))

    def beam_results(self):
        sgms=sgm_values(self.rates,self.demands,self.beta)
        return [dict(beam_id=bid,group=int(self.groups[i]),full_width_deg=float(self.layout['full_width_deg'][i]),demand_bps=int(self.demands[i]),capacity_bps=float(self.rates[i]),SI=float(min(self.rates[i]/self.demands[i],1)),SGM=float(sgms[i]),start=int(self.starts[i]),count=int(self.counts[i]),power_w=float(self.powers[i])) for i,bid in enumerate(self.ids)]

    def state_dict(self):
        return dict(schema=copy.deepcopy(self.schema),layout_SHA=self.layout['content_sha256'],physics_SHA=PHYSICS_SHA,beam_idx=self.beam_idx,starts=self.starts.copy(),counts=self.counts.copy(),powers=self.powers.copy(),occupied=self.occupied.copy(),rates=self.rates.copy(),requested=copy.deepcopy(self.requested),total_satisfaction=self.total_satisfaction)

    def load_state_dict(self,state):
        if state['schema']!=self.schema or state['layout_SHA']!=self.layout['content_sha256'] or state['physics_SHA']!=PHYSICS_SHA:raise ValueError('environment checkpoint identity mismatch')
        if not 0<=state['beam_idx']<=self.N or len(state['requested'])!=state['beam_idx']:raise ValueError('checkpoint processing count')
        self.reset()
        for row in state['requested']:self.step([row['frequency_index'],row['count_index'],row['power_index']])
        for name in ('starts','counts','powers','occupied','rates'):
            if not np.array_equal(getattr(self,name),state[name]):raise ValueError('checkpoint state differs from actual action replay: '+name)
        if self.requested!=state['requested'] or self.total_satisfaction!=state['total_satisfaction']:raise ValueError('checkpoint service/history identity')
