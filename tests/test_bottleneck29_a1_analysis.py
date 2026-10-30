"""Synthetic numerical fixtures only; zero PHY, Actor, kernel or updates."""
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from satellite_rl.bottleneck29 import core,a1_core,a1_policy
from satellite_rl.bottleneck29.a1_stream import StreamStore
from satellite_rl.bottleneck29.a1_analysis import gate,verify_state,_qid
from satellite_rl.bottleneck29.ledger import Halt
from satellite_rl.t0_width.environment import sgm_values,legacy_repaired_reward


def fixture():
    ids=('b0','b1','b2','b3');groups=np.arange(4,dtype=np.int64)
    identity=dict(layout_SHA='synthetic',physics_SHA='synthetic',order=list(ids),groups=groups.tolist())
    ctx=core.Context(dict(content_sha256='synthetic'),{},ids,np.arange(4,dtype=np.int64),groups,np.full(4,21600000.,dtype=np.float64),{},identity)
    snap=core.initial_snapshot(ctx);return ctx,a1_core.state_from_a0(ctx,snap)


def result(ctx,state,point):
    f,count,power=point;rates=state['rates'].copy();index=int(ctx.indices[state['beam_idx']]);rates[index]=count*power*1e6
    si=np.minimum(rates/ctx.demands,1);sgm=sgm_values(rates,ctx.demands,float(ctx.demands.mean()));processed=ctx.indices[:state['beam_idx']+1]
    total=float(si[processed].sum());increment=total-state['total_satisfaction'];current=float(rates[index]);demand=float(ctx.demands[index])
    return dict(status='completed',physical_calls=1,state_SHA=core.content_sha(state),beam_id=ctx.ids[index],
        action=dict(start=f,count=count,power_w=float(power)),capacity_bps=rates,reward=float(legacy_repaired_reward(current,demand,float(ctx.demands.mean()),increment,0,count)),
        reward_parts=dict(SI_threshold=1. if increment>=1 else -1.,current_SGM=float(sgm[index]),deviation_penalty=max(abs(current-demand)-5e7,0.)/1e9,occupancy_penalty=0.,satisfaction_increment=increment),
        current=dict(capacity_bps=current,demand_bps=demand,SI=float(si[index]),SGM=float(sgm[index])),prefix_SI_sum=total,prefix_SGM_sum=float(sgm[processed].sum()),
        total_power_w=float(state['powers'].sum())+power,remaining_reserved_power_w=5*(ctx.N-state['beam_idx']-1))


def build_state(store,ctx,state,mother):
    sid=mother['case_id']+'/0';expected=[];snapshot_ref=dict(source_SHA='A0',snapshot_SHA='snap',row_id=sid+'/snapshot',record_SHA='snap')
    store.alias(sid+'/A0_snapshot',snapshot_ref,state=sid);expected.append(sid+'/A0_snapshot')
    mask=a1_core.action_mask(ctx,state);fl=mask['frequency_slots'].astype(float)/mask['frequency_slots'].sum();p=mask['power'].astype(float)/mask['power'].sum()
    law=dict(frequency_slots=fl,power=p,forward_calls=1);op_id=sid+'/Actor_law'
    store.transact(op_id,dict(forward=1,observation=1),lambda:law,state=sid);expected.append(op_id)
    rows={};best=None;best_reward=None
    for f in range(100):
        for count in range(1,11):
            if f+count>100:continue
            for watts in range(5,51,5):
                point=(f,count,float(watts));op_id=_qid(sid,*point);value=result(ctx,state,point);rows[point]=value
                store.transact(op_id,dict(phy=1,enumeration=1,coefficient=1),lambda value=value:value,state=sid);expected.append(op_id)
                action=(f,count-1,watts//5-1)
                if best is None or value['reward']>best_reward:best=action;best_reward=value['reward']
    selection=a1_policy.select_slots(mother['case_id'],0,mother['actions_indices_by_processing_order'][0],best,fl,p,mask)
    op_id=sid+'/candidate_slots';store.transact(op_id,{},lambda:selection,state=sid);expected.append(op_id)
    for candidate in selection['unique_actions']:
        owner=candidate['slot'];f,l,pi=candidate['action'];point=(f,l+1,float((pi+1)*5));first=rows[point];first_prob=a1_policy.action_probability(fl,p,(f,l,pi))
        for repeat in (0,1):
            branch=sid+'/suffix/slot'+str(owner)+'/repeat'+str(repeat);current=a1_core.advance_state(ctx,state,first);ret=a1_policy.accumulate_return(a1_policy.new_return_state(),first['reward'],first_prob['log_probability'])
            while current['beam_idx']<ctx.N:
                absolute=current['beam_idx'];m=a1_core.action_mask(ctx,current);uf=m['frequency_slots'].astype(float)/m['frequency_slots'].sum();up=m['power'].astype(float)/m['power'].sum()
                rng=a1_policy.uniforms(mother['case_id'],0,'suffix',repeat=repeat,absolute_beam_index=absolute);action=a1_policy.draw_action(uf,up,m,rng['uniforms'])
                f,l,pi=action;value=result(ctx,current,(f,l+1,float((pi+1)*5)));value.update(action_indices=list(action),probability=a1_policy.action_probability(uf,up,action),rng=rng)
                op_id=branch+'/step/'+str(absolute);store.transact(op_id,dict(phy=1,suffix=1,forward=1,coefficient=1,observation=1),lambda value=value:value,state=sid);expected.append(op_id)
                current=a1_core.advance_state(ctx,current,value);ret=a1_policy.accumulate_return(ret,value['reward'],value['probability']['log_probability'])
            terminal=dict(status='completed',slot=owner,repeat=repeat,first_query_ref=_qid(sid,*point),first_probability=first_prob,returns=ret,terminal=a1_core.terminal_summary(ctx,current),interpretation='fixture')
            op_id=branch+'/terminal';store.transact(op_id,{},lambda terminal=terminal:terminal,state=sid);expected.append(op_id)
    store.close_state(sid,expected_ids=expected);return snapshot_ref


class A1AnalysisTests(unittest.TestCase):
    def test_real_sized_complete_state_and_tamper(self):
        ctx,state=fixture();mother=dict(mother=0,case_id='fixture',actions_indices_by_processing_order=[[0,0,0]]*4)
        with tempfile.TemporaryDirectory() as root,patch('os.fsync'):
            limits=dict(phy=12000,enumeration=10000,suffix=1000,coefficient=12000,forward=1000,observation=1000,seconds=120)
            store=StreamStore(root,dict(source='fixture'),limits,minimum_free_bytes=0);store.begin()
            snapshot_ref=build_state(store,ctx,state,mother);store.finish()
            class Facts:
                def state(self,*args):return state,snapshot_ref,{}
            reader=StreamStore(root,dict(source='fixture'),limits,read_only=True)
            checked,terminals=verify_state(reader,Facts(),ctx,mother,0)
            self.assertEqual(checked['tally']['structural'],9550);self.assertTrue(len(terminals)%2==0)
            reader.operations[_qid('fixture/0',0,1,5.)]['charges']['phy']=0
            with self.assertRaises(Halt):verify_state(reader,Facts(),ctx,mother,0)

    def test_gate_same_metric_twelve_and_power(self):
        def states(ds,dg,power=100):
            return [dict(reference=dict(SI=.8,SGM=.5,power=100),causal=dict(SI=.8+ds,SGM=.5+dg,power=power)) for _ in range(3)]
        mothers={i:states(.004 if i<12 else 0,.011 if i<12 else 0) for i in range(20)}
        self.assertFalse(gate(mothers,'causal')['passed'])  # Grand means below registered effect gates.
        mothers={i:states(.006 if i<12 else 0,0) for i in range(20)}
        self.assertTrue(gate(mothers,'causal')['passed'])
        for value in mothers.values():
            for state in value:state['causal']['power']=111
        self.assertFalse(gate(mothers,'causal')['passed'])

    def test_gate_cannot_union_signs_or_omit_mother(self):
        mothers={i:[dict(reference=dict(SI=.8,SGM=.5,power=100),oracle=dict(SI=.81 if i<10 else .8,SGM=.52 if i>=10 else .5,power=100)) for _ in range(3)] for i in range(20)}
        checked=gate(mothers,'oracle');self.assertFalse(checked['passed']);self.assertEqual(checked['SI_improved_mothers'],10);self.assertEqual(checked['SGM_improved_mothers'],10)
        del mothers[19]
        with self.assertRaises(Halt):gate(mothers,'oracle')


if __name__=='__main__':unittest.main()
