"""A1 pure stub checks: no true physical function, model load or forward."""
import math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from satellite_rl.bottleneck29 import core,a1_core as a,a1_policy as p
from satellite_rl.bottleneck29.a1_model import ActorOnly


def fixture(n=4):
    ids=tuple('b'+str(i) for i in range(n));groups=np.arange(n,dtype=np.int64)%8
    layout=dict(content_sha256='stub',centres_lat_lon_deg=np.zeros((n,2)),full_width_deg=np.full(n,.4))
    identity=dict(layout_SHA='stub',physics_SHA=a.PHYSICS_SHA,order=list(ids),groups=groups.tolist())
    ctx=core.Context(layout,{},ids,np.arange(n,dtype=np.int64),groups,np.full(n,21600000.,dtype=np.float64),{},identity)
    return ctx,a.state_from_a0(ctx,core.initial_snapshot(ctx))


class A1CorePureTests(unittest.TestCase):
    def test_coeff_reuse_explicit_actual_power_and_no_snapshot_history(self):
        ctx,state=fixture();builds=[];powers=[]
        def build(layout,allocation,physics,kernel):builds.append(allocation);return allocation
        def capacity(coeff,power_w):powers.append(power_w.copy());return np.asarray(coeff['actual_slot_count']*power_w*1e6,dtype=np.float64)
        before=core.content_sha(state)
        with patch.object(a,'_build',side_effect=build),patch.object(a,'_capacity',side_effect=capacity):
            block=a.build_fl_coefficients(ctx,state,0,2)
            r5=a.evaluate_power(ctx,state,0,2,5.,block);r50=a.evaluate_power(ctx,state,0,2,50.,block)
            next_state=a.advance_state(ctx,state,r50)
        self.assertEqual(len(builds),1);self.assertEqual(len(powers),2)
        self.assertEqual(powers[0][0],5);self.assertEqual(powers[1][0],50)
        self.assertEqual(r5['current']['capacity_bps'],10e6);self.assertEqual(r50['current']['capacity_bps'],100e6)
        self.assertEqual(core.content_sha(state),before);self.assertNotIn('requested',next_state)
        self.assertNotIn('demand_bps',r50);self.assertNotIn('next_snapshot',r50)
        self.assertEqual(next_state['beam_idx'],1)

    def test_original_domain_rejects_half_watt_without_capacity(self):
        ctx,state=fixture()
        with patch.object(a,'_capacity',side_effect=AssertionError('must not call')):
            result=a.evaluate_power(ctx,state,0,1,.5,None)
        self.assertEqual(result['status'],'invalid');self.assertEqual(result['physical_calls'],0)

    def test_observation_exact_old_order_without_real_coupling(self):
        import satellite_rl.t0_width.environment as old_module
        ctx,state=fixture();state['beam_idx']=1;state['starts'][0]=2;state['counts'][0]=2;state['powers'][0]=10.;state['occupied'][0,2:4]=True
        def coupling(layout,groups,physics,receivers,transmitters,kernel=None):return np.full((len(receivers),len(transmitters),100),1e-20)
        old=old_module.T0WidthEnv.__new__(old_module.T0WidthEnv)
        for key in ('layout','physics','indices','groups','demands','kernel'):setattr(old,key,getattr(ctx,key))
        old.N=ctx.N;old.beta=float(ctx.demands.mean());old.group='S10'
        for key in ('beam_idx','starts','counts','powers','occupied','rates','total_satisfaction'):setattr(old,key,state[key])
        with patch.object(a,'_coupling',side_effect=coupling),patch.object(old_module,'raw_slot_coupling',side_effect=coupling):
            np.testing.assert_array_equal(a.observation(ctx,state),old._observation())
        np.testing.assert_array_equal(a.action_mask(ctx,state)['frequency_slots'],old.action_mask()['frequency_slots'])

    def test_soft_sample_excludes_first_negative_infinity_entropy(self):
        state=p.new_return_state()
        for reward,logp in [(1.,-math.inf),(2.,math.log(.5)),(3.,math.log(.25))]:state=p.accumulate_return(state,reward,logp)
        ordinary=1+.99*2+.99**2*3
        expected=ordinary-.01*(.99*math.log(.5)+.99**2*math.log(.25))
        self.assertAlmostEqual(state['ordinary_discounted'],ordinary);self.assertAlmostEqual(state['Q_semantic_soft_sample'],expected)

    def test_exact_8_proposals_missing_Q_and_deterministic_aliases(self):
        fl=np.zeros((100,10));power=np.zeros(10);fl[3,2]=1;power[0]=1
        mask=dict(frequency_slots=fl.astype(bool),power=power.astype(bool))
        result=p.select_slots('stub',2,[3,2,0],[3,2,0],fl,power,mask)
        self.assertEqual(len(result['slots']),6);self.assertEqual(len(result['proposals']),8)
        self.assertEqual(result['slots'][2]['status'],'not_available');self.assertEqual(len(result['unique_actions']),1)
        self.assertEqual(result['slots'][1]['alias_slot'],0);self.assertEqual(result['slots'][4]['status'],'all_proposals_duplicate')

    def test_mode_tie_lexicographic_and_zero_probability_preserved(self):
        fl=np.zeros((100,10));power=np.zeros(10);fl[0,1]=.5;fl[1,0]=.5;power[:2]=.5
        mask=dict(frequency_slots=fl.astype(bool),power=power.astype(bool))
        slots=p.select_slots('stub',0,[1,0,1],[1,0,1],fl,power,mask)['slots']
        self.assertEqual(slots[1]['action'],[0,1,0])
        zero=p.action_probability(fl,power,(0,0,0));self.assertEqual(zero['log_probability'],-math.inf)
        self.assertTrue(zero['log_probability_is_negative_infinity'])

    def test_common_uniforms_and_rng_isolation(self):
        before=np.random.get_state()
        x=p.uniforms('stub',4,'suffix',repeat=1,absolute_beam_index=8)
        y=p.uniforms('stub',4,'suffix',repeat=1,absolute_beam_index=8)
        self.assertEqual(x,y);np.testing.assert_array_equal(before[1],np.random.get_state()[1])
        self.assertNotEqual(x['SHA256'],p.uniforms('stub',4,'suffix',repeat=0,absolute_beam_index=8)['SHA256'])
        with self.assertRaises(TypeError):p.uniforms('stub',4,'suffix',repeat=1,absolute_beam_index=8,candidate=0)

    def test_full_SHA_required_before_any_torch_or_weight_load(self):
        with self.assertRaises(ValueError):ActorOnly.load('does-not-exist','5e056ee')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fake';path.write_bytes(b'not-model')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):ActorOnly.load(path,'0'*64)


if __name__=='__main__':unittest.main()
