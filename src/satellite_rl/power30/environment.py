"""Variable-N shared-physics environment with unchanged T0 reward semantics."""
import copy
from time import perf_counter
import numpy as np
from satellite_rl.shared_interface.stages import UniformStage2Input
from satellite_rl.shared_interface.link import raw_slot_coupling,build_link_coefficients,capacity_bps,prepare_slot_kernel
from satellite_rl.shared_interface.spherical_gaussian import PHYSICS_SHA
from .schema import GROUPS,state_width, schema_identity
from .menus import power_table, action_schema, legal_power_mask, validate_actual_power

def sgm_values(rates,demands,beta):
    result=np.zeros_like(rates,dtype=np.float64);positive=rates>0
    ratio=rates[positive]/demands[positive]
    real=np.where(ratio>=1,ratio-1,1-1/ratio)
    result[positive]=1-(1-np.exp(-np.abs(real+1j*(rates[positive]-demands[positive])/beta)))**3
    return result

def legacy_repaired_reward(rate,demand,beta,si_increment,occupy=0,origin_slots=1):
    sgm=float(sgm_values(np.asarray([rate]),np.asarray([demand]),beta)[0])
    return (1. if si_increment>=1 else -1.)+sgm-max(abs(rate-demand)-5e7,0.)/1e9-occupy/origin_slots

class Power30Env:
    def __init__(self,stage_input,menu='P00',group='M0',call_hook=None):
        if not isinstance(stage_input,UniformStage2Input):raise TypeError('validated UniformStage2Input required')
        self.stage_input=stage_input;self.group=group;self.menu=menu;self.power_table=power_table(menu)
        self.schema=schema_identity(menu,group,plain=getattr(stage_input,'plain',False));self.action_schema=action_schema(menu,plain=getattr(stage_input,'plain',False));self.call_hook=call_hook;self.call_counts={}
        self.last_coefficients=None
        self.layout=stage_input.layout;self.physics=stage_input.physics
        self.ids=list(self.layout['beam_ids']);self.order=list(stage_input.processing_index_to_beam_id)
        self.indices=np.asarray([self.ids.index(b) for b in self.order],np.int64)
        self.groups=np.asarray([stage_input.group_by_beam_id[b] for b in self.ids],np.int64)
        self.N=len(self.ids);self.demands=self.layout['request_bps_by_beam_int64'].astype(np.float64)
        self.beta=float(self.demands.mean());self.noise=self.physics['k_J_per_K']*self.physics['noise_temperature_K']*self.physics['slot_Hz']
        self._newcall("kernel")
        if getattr(stage_input,'plain',False):
            self.kernel=getattr(stage_input,'_s2fix_slot_kernel',None)
            if self.kernel is None:
                self.kernel=prepare_slot_kernel(self.layout,self.physics,case=stage_input.case)
                stage_input._s2fix_slot_kernel=self.kernel
        else:
            self.kernel=prepare_slot_kernel(self.layout,self.physics)
        self.reset()

    def _newcall(self, kind, **metadata):
        if self.call_hook is not None:self.call_hook(kind, **metadata)
        self.call_counts[kind]=self.call_counts.get(kind,0)+1

    def reset(self):
        self.last_coefficients=None
        self.beam_idx=0;self.starts=np.zeros(self.N,np.int64);self.counts=np.zeros(self.N,np.int64);self.powers=np.zeros(self.N,np.float64)
        self.occupied=np.zeros((8,100),bool);self.rates=np.zeros(self.N,np.float64);self.requested=[];self.total_satisfaction=0.
        return self._observation()

    def action_mask(self):
        fl=np.zeros((100,10),bool);power=np.zeros(len(self.power_table),bool)
        if self.beam_idx==self.N:return dict(frequency_slots=fl,power=power)
        c=self.indices[self.beam_idx];g=self.groups[c];remaining=self.indices[self.beam_idx:]
        reserve_slots=int(np.count_nonzero(self.groups[remaining]==g))-1
        free=~self.occupied[g];max_count=int(free.sum())-reserve_slots
        for count in range(1,min(10,max_count)+1):
            for start in range(101-count):fl[start,count-1]=bool(free[start:start+count].all())
        power[:]=legal_power_mask(self.menu,float(self.powers.sum()),self.N-self.beam_idx-1)
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
        parts.append(np.asarray([self.layout['full_width_deg'][c]]))
        if GROUPS[self.group]:
            self._newcall('feature')
            parts.append(self._margin_summary(c))
        result=np.concatenate(parts)
        if not np.isfinite(result).all():raise ValueError('finite observation required')
        return result.astype(np.float32)

    def _margin_summary(self, current):
        result=np.zeros(100,np.float64)
        coeff=self.last_coefficients
        if coeff is None:return result
        denominator=coeff.noise+coeff.interference@self.powers
        for j in self.indices[:self.beam_idx]:
            rows=np.flatnonzero(coeff.slot_owner==j)
            slots=coeff.occupied_slots[rows]
            eligible=(self.groups[j]%2==self.groups[current]%2 and self.groups[j]!=self.groups[current])
            if eligible:
                kappa=(self.kernel['gain'][j,current]*10**(self.physics['rx_gain_dBi']/10)
                       *self.kernel['path'][j,slots])
                weight=np.exp(-abs(self.rates[j]/self.demands[j]-1.0))
                result[slots]+=weight*kappa/denominator[rows]
        return np.log1p(result)

    def export_allocation(self):
        actual={bid:dict(group=int(self.groups[i]),start=int(self.starts[i]),count=int(self.counts[i]),power_w=float(self.powers[i])) for i,bid in enumerate(self.ids)}
        validate_actual_power(self.menu,self.powers,active=self.counts>0)
        result=self.stage_input.export_actual_allocation(actual,requested_actions_separate=copy.deepcopy(self.requested))
        return result

    def step(self,action):
        step_started=perf_counter()
        if self.beam_idx==self.N:raise RuntimeError('episode complete')
        if len(action)!=3 or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) for x in action):raise ValueError('three integer action indices required')
        f,l,p=map(int,action);mask=self.action_mask()
        if not (0<=f<100 and 0<=l<10 and 0<=p<len(self.power_table) and mask['frequency_slots'][f,l] and mask['power'][p]):raise ValueError('illegal action; clipping forbidden')
        return self._place(f,l,p,action,step_started)

    def _place(self,f,l,p,action,step_started):
        """Apply a validated action; shared with the optional group-choice environment."""
        c=int(self.indices[self.beam_idx]);bid=self.ids[c];self.starts[c]=f;self.counts[c]=l+1;self.powers[c]=self.power_table[p]
        self.occupied[self.groups[c],f:f+l+1]=True
        self.requested.append(dict(beam_id=bid,frequency_index=f,count_index=l,power_index=p,start=f,count=l+1,power_w=self.power_table[p]))
        # The case/layout were independently audited at import. Internal rows
        # are created only through the legal mask; terminal export revalidates
        # the complete versioned graph without repeating coverage each step.
        allocation=dict(beam_ids=self.ids,actual_start_slot=self.starts,actual_slot_count=self.counts,
            actual_whole_beam_power_w=self.powers,group_by_beam_id=self.stage_input.group_by_beam_id,
            content_sha256='internal_partial_not_published')
        self._newcall('coeff')
        coeff=build_link_coefficients(self.layout,allocation,self.physics,kernel=self.kernel)
        self.last_coefficients=coeff
        self._newcall('phy')
        self.rates=capacity_bps(coeff)
        # Match 85f calculate_total_satisfaction's serviced processing prefix:
        # layout storage order and extra unprocessed zeros can change float64
        # reduction roundoff at the exact >=1 reward threshold.
        processed=self.indices[:self.beam_idx+1]
        satisfaction=float(np.minimum(self.rates[processed]/self.demands[processed],1).sum());increment=satisfaction-self.total_satisfaction
        reward=legacy_repaired_reward(float(self.rates[c]),float(self.demands[c]),self.beta,increment,0,l+1)
        self.total_satisfaction=satisfaction;self.beam_idx+=1;done=self.beam_idx==self.N
        info=dict(beam_id=bid,requested_action=list(action),actual_action=dict(start=f,count=l+1,power_w=self.power_table[p]),rate_bps=float(self.rates[c]),required_rate_bps=float(self.demands[c]),satisfaction_increment=increment,metrics=self.metrics())
        info['beam_service_step']=dict(beam_id=bid,rate_bps=float(self.rates[c]),required_rate_bps=float(self.demands[c]),SI=float(min(self.rates[c]/self.demands[c],1)),SGM=float(sgm_values(self.rates[[c]],self.demands[[c]],self.beta)[0]))
        if done:info['beam_results']=self.beam_results();info['allocation']=self.export_allocation()
        observation=self._observation()
        info['action_schema']=self.action_schema
        info['timing']=dict(environment_step_seconds=perf_counter()-step_started)
        return observation,float(reward),done,info

    def metrics(self):
        sgms=sgm_values(self.rates,self.demands,self.beta);si=np.minimum(self.rates/self.demands,1)
        return dict(SI_mean=float(si.mean()),SI_sum=float(si.sum()),SGM=float(sgms.mean()),throughput_bps=float(self.rates.sum()),total_power_w=float(self.powers.sum()),N=self.N,processed=self.beam_idx,complete=self.beam_idx==self.N,zero_slot_rate=float((self.counts==0).mean()))

    def beam_results(self):
        sgms=sgm_values(self.rates,self.demands,self.beta)
        return [dict(beam_id=bid,group=int(self.groups[i]),full_width_deg=float(self.layout['full_width_deg'][i]),demand_bps=int(self.demands[i]),capacity_bps=float(self.rates[i]),SI=float(min(self.rates[i]/self.demands[i],1)),SGM=float(sgms[i]),start=int(self.starts[i]),count=int(self.counts[i]),power_w=float(self.powers[i])) for i,bid in enumerate(self.ids)]

    def state_dict(self):
        if getattr(self.stage_input,'plain',False):
            return copy.deepcopy({name:getattr(self,name) for name in
                ('beam_idx','starts','counts','powers','occupied','rates','requested',
                 'total_satisfaction','last_coefficients')})
        return dict(schema=copy.deepcopy(self.schema),layout_SHA=self.layout['content_sha256'],physics_SHA=PHYSICS_SHA,beam_idx=self.beam_idx,starts=self.starts.copy(),counts=self.counts.copy(),powers=self.powers.copy(),occupied=self.occupied.copy(),rates=self.rates.copy(),requested=copy.deepcopy(self.requested),total_satisfaction=self.total_satisfaction,last_coefficients=self.last_coefficients)

    def load_state_dict(self,state):
        if getattr(self.stage_input,'plain',False):
            for name,value in copy.deepcopy(state).items():
                setattr(self,name,value)
            return
        if state['schema']!=self.schema or state['layout_SHA']!=self.layout['content_sha256'] or state['physics_SHA']!=PHYSICS_SHA:
            raise ValueError('environment checkpoint identity mismatch')
        count=state['beam_idx']
        if not isinstance(count,int) or not 0<=count<=self.N or len(state['requested'])!=count:
            raise ValueError('checkpoint processing count')
        starts=np.zeros(self.N,np.int64);counts=np.zeros(self.N,np.int64);powers=np.zeros(self.N,np.float64)
        occupied=np.zeros((8,100),bool)
        for step,row in enumerate(state['requested']):
            c=self.indices[step];f=row['frequency_index'];l=row['count_index'];p=row['power_index']
            if row['beam_id']!=self.ids[c] or not (0<=f<100 and 0<=l<10 and 0<=p<len(self.power_table)) or f+l+1>100:
                raise ValueError('checkpoint action identity')
            if row['start']!=f or row['count']!=l+1 or row['power_w']!=self.power_table[p]:
                raise ValueError('checkpoint requested/actual mismatch')
            g=self.groups[c]
            future_same=int(np.count_nonzero(self.groups[self.indices[step+1:]]==g))
            if occupied[g,f:f+l+1].any() or 100-int(occupied[g].sum())-(l+1)<future_same:
                raise ValueError('checkpoint slot reservation')
            if not legal_power_mask(self.menu,float(powers.sum()),self.N-step-1)[p]:
                raise ValueError('checkpoint power reservation')
            starts[c]=f;counts[c]=l+1;powers[c]=self.power_table[p];occupied[g,f:f+l+1]=True
        for name,value in dict(starts=starts,counts=counts,powers=powers,occupied=occupied).items():
            if not np.array_equal(value,state[name]):raise ValueError('checkpoint ledger mismatch: '+name)
        rates=np.asarray(state['rates'])
        if rates.shape!=(self.N,) or not np.isfinite(rates).all() or (rates<0).any() or (rates[counts==0]!=0).any():
            raise ValueError('checkpoint rate shape/value')
        satisfaction=float(np.minimum(rates[self.indices[:count]]/self.demands[self.indices[:count]],1).sum())
        if satisfaction!=state['total_satisfaction']:raise ValueError('checkpoint satisfaction mismatch')
        for name,value in dict(starts=starts,counts=counts,powers=powers,occupied=occupied,rates=rates.copy()).items():setattr(self,name,value)
        self.beam_idx=count;self.requested=copy.deepcopy(state['requested']);self.total_satisfaction=satisfaction
        self.last_coefficients=state.get('last_coefficients')
        if count:
            coeff=self.last_coefficients
            if coeff is None:raise ValueError('checkpoint lacks summary coefficient cache')
            expected_owner=np.repeat(np.arange(self.N,dtype=np.int64),counts)
            expected_slots=np.concatenate([np.arange(starts[i],starts[i]+counts[i],dtype=np.int64) for i in range(self.N)])
            if (tuple(coeff.beam_ids)!=tuple(self.ids) or not np.array_equal(coeff.powers_initial,powers)
                    or not np.array_equal(coeff.active,counts>0) or not np.array_equal(coeff.demand_bps,self.demands)
                    or not np.array_equal(coeff.slot_owner,expected_owner) or not np.array_equal(coeff.occupied_slots,expected_slots)
                    or coeff.metadata['parent_layout_SHA']!=self.layout['content_sha256']
                    or coeff.metadata['physics_config_SHA']!=PHYSICS_SHA):
                raise ValueError('checkpoint coefficient cache identity/allocation mismatch')
            if (coeff.interference.shape!=(len(expected_owner),self.N) or coeff.noise.shape!=(len(expected_owner),)
                    or not np.isfinite(coeff.interference).all() or (coeff.interference<0).any()
                    or not np.isfinite(coeff.noise).all() or (coeff.noise<=0).any()):
                raise ValueError('checkpoint coefficient cache values')
        elif self.last_coefficients is not None:raise ValueError('empty prefix has coefficient cache')
