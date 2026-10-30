"""Entire engineering controller with fake physical and fake Actor operations."""
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from satellite_rl.bottleneck29 import core,a1_core
from satellite_rl.bottleneck29 import a1_engineering as eng
from satellite_rl.bottleneck29.a1_stream import StreamStore


def context_and_state(prefix):
    n=16;ids=tuple('b'+str(i) for i in range(n));groups=np.arange(n,dtype=np.int64)%8
    layout=dict(content_sha256='stub',beam_ids=list(ids),centres_lat_lon_deg=np.zeros((n,2)),full_width_deg=np.full(n,.5))
    physics=dict(k_J_per_K=1.,noise_temperature_K=1.,slot_Hz=1.)
    identity=dict(layout_SHA='stub',physics_SHA='stub',order=list(ids),groups=groups.tolist())
    ctx=core.Context(layout,physics,ids,np.arange(n,dtype=np.int64),groups,np.full(n,21600000.,dtype=np.float64),{},identity)
    state=core.initial_snapshot(ctx)
    for i in range(prefix):
        state['starts'][i]=i//8;state['counts'][i]=1;state['powers'][i]=5.;state['occupied'][i%8,i//8]=True
        state['requested'].append(dict(beam_id=ids[i],start=i//8,count=1,power_w=5.))
    state['rates']=state['powers']*state['counts']*1e6;state['beam_idx']=prefix
    state['total_satisfaction']=float(np.minimum(state['rates'][:prefix]/ctx.demands[:prefix],1).sum())
    return ctx,a1_core.state_from_a0(ctx,state)


def law(obs,mask):
    return dict(frequency_slots=mask['frequency_slots'].astype(float)/mask['frequency_slots'].sum(),
        power=mask['power'].astype(float)/mask['power'].sum(),forward_calls=1)


class EngineeringMockTests(unittest.TestCase):
    def test_full_engineering_bounded_and_lossless(self):
        import satellite_rl.t0_width.environment as old
        import satellite_rl.bottleneck29.a1_inputs as inputs
        import satellite_rl.bottleneck29.a1_model as model
        class Facts:
            def __init__(self,config):pass
            def mother(self,mother):
                if not isinstance(mother,dict):raise TypeError('real A0Facts requires mother row dict')
                return context_and_state(0)[0],[[i//8,0,0] for i in range(16)]
            def state(self,ctx,mother,prefix):
                if not isinstance(mother,dict):raise TypeError('real A0Facts requires mother row dict')
                return context_and_state(prefix)[1],{},{}
        class Actor:
            forward_probs=staticmethod(law)
        def build(layout,allocation,physics,kernel):return allocation
        def capacity(coeff,power_w=None):
            if power_w is None:power_w=coeff['actual_whole_beam_power_w']
            return np.asarray(coeff['actual_slot_count']*power_w*1e6,dtype=np.float64)
        def coupling(layout,groups,physics,receivers,transmitters,kernel=None):return np.zeros((len(receivers),len(transmitters),100))
        mothers=[dict(mother=i,N=16,case_id='mock'+str(i),actions_indices_by_processing_order=[[j//8,0,0] for j in range(16)]) for i in range(20)]
        with tempfile.TemporaryDirectory() as root:
            limits=dict(phy=114,forward=16,coefficient=114,kernel=3,seconds=60)
            store=StreamStore(root,dict(source_SHA='stub'),limits,minimum_free_bytes=0);store.begin()
            config=dict(mothers=mothers,lane=0,actor_path='never-opened',binding=dict(actor_SHA='stub'))
            with patch.object(inputs,'A0Facts',Facts),patch.object(model.ActorOnly,'load',return_value=Actor()),patch.object(eng,'reference_forward',side_effect=lambda actor,obs,mask:law(obs,mask)),patch.object(a1_core,'_build',side_effect=build),patch.object(a1_core,'_capacity',side_effect=capacity),patch.object(a1_core,'_coupling',side_effect=coupling),patch.object(old,'build_link_coefficients',side_effect=build),patch.object(old,'capacity_bps',side_effect=capacity),patch.object(old,'raw_slot_coupling',side_effect=coupling):
                eng.execute(config,store)
            self.assertEqual(store.counts['forward'],16)
            self.assertLessEqual(store.counts['phy'],80)
            self.assertEqual(len(store.closed_states),4)
            self.assertFalse(store.metadata['engineering_evidence']['science_admitted'])
            store.finish()


if __name__=='__main__':unittest.main()
