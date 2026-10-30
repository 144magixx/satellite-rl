"""Full 60-state composition with fake Actor, facts, store and physical aliases.

The structural iterator is patched to two powers solely inside this unit test;
production exposes no reduced-domain config switch. All actual scientific calls=0.
"""
import unittest
from unittest.mock import patch
import numpy as np
from satellite_rl.bottleneck29 import a1_worker as worker,a1_core as a,core
from test_bottleneck29_a1_core import fixture


def compact_result(ctx,state,power):
    k=state['beam_idx'];index=int(ctx.indices[k]);powers=state['powers'].copy();powers[index]=power
    counts=state['counts'].copy();counts[index]=1;rates=powers*counts*1e6
    si=np.minimum(rates/ctx.demands,1);sgm=a.sgm_values(rates,ctx.demands,float(ctx.demands.mean()))
    total=float(si[ctx.indices[:k+1]].sum());increment=total-state['total_satisfaction']
    reward=a.legacy_repaired_reward(float(rates[index]),float(ctx.demands[index]),float(ctx.demands.mean()),increment,0,1)
    return dict(status='completed',physical_calls=0,state_SHA=core.content_sha(state),beam_id=ctx.ids[index],
        action=dict(start=0,count=1,power_w=float(power)),capacity_bps=rates,reward=reward,
        reward_parts=dict(SI_threshold=1. if increment>=1 else -1.,current_SGM=float(sgm[index]),deviation_penalty=0.,occupancy_penalty=0.,satisfaction_increment=increment),
        current=dict(capacity_bps=float(rates[index]),demand_bps=float(ctx.demands[index]),SI=float(si[index]),SGM=float(sgm[index])),
        prefix_SI_sum=total,prefix_SGM_sum=float(sgm[ctx.indices[:k+1]].sum()),total_power_w=float(powers.sum()),remaining_reserved_power_w=5*(ctx.N-k-1))


class FakeFacts:
    def __init__(self,config):pass
    def mother(self,mother):
        ctx,_=fixture(3);return ctx,[[0,0,0]]*3
    def state(self,ctx,mother,prefix):
        _,state=fixture(3)
        for _ in range(prefix):state=a.advance_state(ctx,state,compact_result(ctx,state,5.))
        ref=dict(source_SHA='A0',snapshot_SHA='snapshot',row_id=mother['case_id']+'/'+str(prefix)+'/snapshot',record_SHA='record')
        result=compact_result(ctx,state,5.)
        old=dict(status='completed',actual_action=dict(beam_id=result['beam_id'],**result['action']),
            requested_action=dict(beam_id=result['beam_id'],**result['action']),capacity_bps=result['capacity_bps'],reward=result['reward'],
            reward_parts=result['reward_parts'],current=result['current'],prefix=dict(SI_sum=result['prefix_SI_sum'],SGM_sum=result['prefix_SGM_sum']),
            constraints=dict(total_power_w=result['total_power_w']))
        point=dict(spec=core.candidate_spec(ctx,state,0,1,5.,'original'),result=old,source_ref=dict(ref,row_id=ref['row_id']+'/point'))
        return state,ref,{(0,1,5.):point}


class FakeActor:
    provenance=dict(test='fake Actor; no actual model load/forward')
    @classmethod
    def load(cls,*args,**kwargs):return cls()
    def forward_probs(self,obs,mask):
        return dict(frequency_slots=mask['frequency_slots']/mask['frequency_slots'].sum(),
            power=mask['power']/mask['power'].sum(),forward_calls=1)


class FakeStore:
    def __init__(self):self.values={};self.operations={};self.closed_states=set();self.counts={};self.metadata={}
    def boundary(self):pass
    def get(self,key):return self.values.get(key)
    def transact(self,key,charges,operation,*,state):
        if key in self.values:return self.values[key]
        self.values[key]=operation();self.operations[key]=dict(done=True,charges=charges)
        for kind,count in charges.items():self.counts[kind]=self.counts.get(kind,0)+count
        return self.values[key]
    def alias(self,key,source,*,state):
        self.values[key]=dict(status='alias',source=source);self.operations[key]=dict(alias=source,charges={})
    def record_metadata(self,key,value):self.metadata[key]=value
    def close_state(self,state,expected_ids=None):
        if not all(key in self.values for key in expected_ids):raise AssertionError('incomplete expected rows')
        self.closed_states.add(state)


def small_mask(ctx,state):
    fl=np.zeros((100,10),dtype=bool);power=np.zeros(10,dtype=bool)
    if state['beam_idx']<ctx.N:fl[0,0]=True;power[:2]=True
    return dict(frequency_slots=fl,power=power)


class A1WorkerPureTests(unittest.TestCase):
    def test_sixty_states_missing_Q_two_suffixes_resume_no_new_work(self):
        mothers=[dict(mother=i,case_id='stub'+str(i)) for i in range(20)]
        stores=[]
        def build(layout,allocation,physics,kernel):return allocation
        def capacity(coeff,power_w):return np.asarray(coeff['actual_slot_count']*power_w*1e6,dtype=np.float64)
        with patch.object(worker,'A0Facts',FakeFacts),patch.object(worker,'ActorOnly',FakeActor),\
             patch.object(worker,'_domain',side_effect=lambda:iter([(0,1,5.),(0,1,10.)])),\
             patch.object(a,'action_mask',side_effect=small_mask),\
             patch.object(a,'observation',side_effect=lambda ctx,state:np.zeros(208,dtype=np.float32)),\
             patch.object(a,'_build',side_effect=build),patch.object(a,'_capacity',side_effect=capacity):
            for lane in range(4):
                config=dict(lane=lane,mothers=mothers,actor_path='fake',binding=dict(Q_status='unavailable_same_node_150000',actor_SHA='fake'))
                store=FakeStore();result=worker.execute(config,store)
                self.assertEqual(result['status'],'completed');self.assertEqual(len(store.closed_states),15)
                self.assertEqual(sum(key.endswith('/terminal') for key in store.values),60)
                for key,value in store.values.items():
                    if key.endswith('/candidate_slots'):
                        self.assertEqual(value['slots'][2]['status'],'not_available');self.assertEqual(len(value['proposals']),8)
                    if key.endswith('/terminal'):
                        self.assertTrue(value['terminal']['complete']);self.assertEqual(value['terminal']['zero_slot_count'],0)
                counts=dict(store.counts);worker.execute(config,store);self.assertEqual(store.counts,counts)
                # Simulate the narrow boundary after all durable results but
                # before the state-closed marker. Reconstructing state/returns
                # must reuse enum/Actor/suffix data without coefficient or PHY.
                store.closed_states.remove(sorted(store.closed_states)[0])
                worker.execute(config,store);self.assertEqual(store.counts,counts)
                self.assertEqual(len(store.closed_states),15)
                stores.append(store)
        self.assertEqual(sum(len(s.closed_states) for s in stores),60)
        self.assertEqual(sum(s.counts['forward'] for s in stores),300)
        self.assertEqual(sum(s.counts['phy'] for s in stores),300)
        self.assertEqual(sum(s.counts['enumeration'] for s in stores),60)
        self.assertEqual(sum(s.counts['suffix'] for s in stores),240)

    def test_production_structural_domain_is_fixed_9550(self):
        points=list(worker._domain());self.assertEqual(len(points),9550)
        self.assertEqual(points[0],(0,1,5.));self.assertEqual(points[-1],(99,1,50.))


if __name__=='__main__':unittest.main()
