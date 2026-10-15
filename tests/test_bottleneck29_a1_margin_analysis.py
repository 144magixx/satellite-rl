"""Prespecified margin arithmetic fixtures: no PHY/model/kernel/observation."""
import math
import unittest
import numpy as np
from satellite_rl.bottleneck29.a1_margin_analysis import features,consequences,average_tie_ranks,spearman,analyze_state,summarize


def inputs(k=2):
    n=3;prefix=dict(beam_idx=k,starts=np.zeros(n,dtype=np.int64),counts=np.zeros(n,dtype=np.int64),powers=np.zeros(n,dtype=np.float64),rates=np.zeros(n,dtype=np.float64))
    prefix['counts'][:k]=1;prefix['powers'][:k]=5.;prefix['rates'][:k]=[10.,5.][:k]
    return prefix,np.full(n,10.),np.arange(n,dtype=np.int64),np.asarray([0,2,4]),dict(gain=np.ones((n,n)),path=np.ones((n,100))),dict(k_J_per_K=1.,noise_temperature_K=1.,slot_Hz=1.,rx_gain_dBi=0.,signal_loss_dB=999.)


class MarginTests(unittest.TestCase):
    def test_saved_kernel_receiver_transmitter_units_noise_and_weight(self):
        args=inputs();f=features(*args)
        self.assertEqual(f['status'],'available')
        self.assertAlmostEqual(f['X'][0],math.log1p((1+math.exp(-.5))/6))
        self.assertAlmostEqual(f['U'][0],math.log1p(2/6))
        np.testing.assert_array_equal(f['X'][1:],np.zeros(99));self.assertTrue((f['X']<=f['U']).all())
        # Desired signal loss must have no effect on outgoing interference.
        args[-1]['signal_loss_dB']=-999.;same=features(*args);np.testing.assert_array_equal(f['X'],same['X'])

    def test_group_eligibility_and_missing_fields_not_zero(self):
        args=list(inputs());args[3]=np.asarray([0,2,0]);f=features(*args)
        self.assertAlmostEqual(f['U'][0],math.log1p(1/6))
        del args[4]['path'];missing=features(*args)
        self.assertEqual(missing['status'],'unavailable');self.assertNotIn('X',missing)

    def test_no_prior_demand_fraction_is_undefined(self):
        args=inputs(0);f=features(*args);self.assertEqual(f['status'],'available');self.assertTrue((f['X']==0).all())
        consequence=consequences(args[0],args[1],args[2],np.asarray([10.,0.,0.]))
        self.assertEqual(consequence['SI_damage'],0)
        for row in consequence['affected_bins']:
            self.assertEqual(row['receiver_count'],0);self.assertEqual(row['demand_mass_bps'],0);self.assertIsNone(row['fraction_of_processed_demand'])

    def test_spearman_ties_and_constants_are_not_zero(self):
        np.testing.assert_array_equal(average_tie_ranks([1,1,3]),[1.5,1.5,3])
        self.assertAlmostEqual(spearman([1,1,3],[3,2,1])['value'],-math.sqrt(.75))
        self.assertIsNone(spearman([1,1],[2,3])['value']);self.assertIsNone(spearman([1,2],[3,3])['value'])

    def test_complete_fixed_lp_groups_damage_and_missing_table(self):
        args=inputs();prefix,demand,indices,groups,kernel,physics=args
        def queries():
            for f in range(100):
                for count in range(1,11):
                    if f+count>100:continue
                    for power in range(5,51,5):
                        capacity=prefix['rates'].copy();capacity[2]=10.
                        if f==0:capacity[:2]-=power/count*.01
                        yield (f,count,float(power)),capacity
        selected=dict(original=(0,1,5.),reward_max=(1,1,5.),oracle=(0,1,50.))
        checked=analyze_state('fixture',0,prefix,demand,indices,groups,kernel,physics,queries(),selected)
        self.assertEqual(checked['status'],'available');self.assertEqual(len(checked['groups']),100);self.assertEqual(checked['valid_actions'],9550)
        self.assertAlmostEqual(checked['rho_X'],1.);self.assertAlmostEqual(checked['paired_rho_X_minus_U'],0.)
        self.assertLess(checked['reward_max_minus_original']['SI_damage'],0)
        for row in checked['selected_actions']['original']['affected_bins']:
            self.assertIn('fraction_of_processed_demand',row)
        incomplete=analyze_state('fixture',0,prefix,demand,indices,groups,kernel,physics,iter([next(queries())]),selected)
        self.assertEqual(incomplete['status'],'unavailable');self.assertFalse(incomplete['legal_table_complete'])

    def test_equal_mother_paired_aggregation_and_unavailable(self):
        states=[]
        for mother in range(20):
            for position in range(3):
                states.append(dict(status='available',mother=mother,prefix=position,rho_X=.2,rho_U=.1,paired_rho_X_minus_U=.1,
                    nontrivial_groups=1,defined_X_groups=1,defined_U_groups=1,affected_bins_equal_action_mean=[[0.,0.,0.]]*3,
                    selected_actions=dict(original=dict(SI_damage=1e-5,SGM_signed_change=0.))))
        a1=dict(status='complete_verified',reward_max_causal_single_intervention_gate=dict(passed=True))
        summary=summarize(states,dict(status='available',structural_support=True),a1)
        self.assertTrue(summary['conditional_B_preparation_evidence']);self.assertFalse(summary['training_admitted'])
        states[0]=dict(status='unavailable',mother=0,prefix=0,reason='missing kernel')
        summary=summarize(states,dict(status='available',structural_support=True),a1)
        self.assertEqual(summary['status'],'evidence_unavailable');self.assertFalse(summary['conditional_B_preparation_evidence'])


if __name__=='__main__':unittest.main()
