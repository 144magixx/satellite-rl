import unittest
from unittest.mock import patch
import numpy as np
from satellite_rl.shared_interface.fixtures import object_recipe
from satellite_rl.shared_interface.contracts import Authority
from satellite_rl.shared_interface.stages import UniformStage2Input
from satellite_rl.shared_interface.spherical_gaussian import PHYSICS
from satellite_rl.shared_interface.link import raw_slot_coupling,build_link_coefficients
from satellite_rl.t0_width.environment import T0WidthEnv,legacy_repaired_reward

def fixture(n=9):
    obj=object_recipe(dict(id='widthenv',N=n,expected={},widths_rule='[.3,.5,1.0][i%3]'))
    authority=Authority(obj['authority_registry'],obj['authority_registry_sha'])
    return UniformStage2Input(obj['case'],obj['layout'],authority,PHYSICS)

class EnvironmentTests(unittest.TestCase):
    def test_shapes_and_natural_fields(self):
        for group,width in [('S00',207),('S10',208),('S01',407),('S11',408)]:
            env=T0WidthEnv(fixture(),group)
            self.assertEqual(env.reset().shape,(width,))
            self.assertNotIn('out',env.__dict__)
            self.assertNotIn('future_in',env.__dict__)

    def test_episode_mask_export_restore(self):
        env=T0WidthEnv(fixture(),'S11')
        with self.assertRaises(ValueError):env.step([99,9,0])
        for _ in range(4):env.step([0,0,0])
        recovered=T0WidthEnv(fixture(),'S11');recovered.load_state_dict(env.state_dict())
        np.testing.assert_array_equal(env._observation(),recovered._observation())
        for index in range(4,9):
            mask=env.action_mask();f,l=np.argwhere(mask['frequency_slots'])[0];p=np.flatnonzero(mask['power'])[0]
            a=env.step([int(f),int(l),int(p)]);b=recovered.step([int(f),int(l),int(p)])
            np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[1],b[1])
        self.assertTrue(a[2]);self.assertEqual(len(env.beam_results()),9)
        self.assertEqual(env.export_allocation()['actual_slot_count'].sum(),9)

    def test_n_one_and_200(self):
        for n in [1,200]:
            env=T0WidthEnv(fixture(n))
            for _ in range(n):
                f,l=np.argwhere(env.action_mask()['frequency_slots'])[0]
                _,_,done,_=env.step([int(f),int(l),0])
            self.assertTrue(done);self.assertEqual(env.metrics()['processed'],n)

    def test_direction_and_shared_kernel(self):
        env=T0WidthEnv(fixture())
        K=raw_slot_coupling(env.layout,env.groups,PHYSICS)
        self.assertTrue(np.all(K[0,8]==0)) # same group excluded
        self.assertGreater(K[0,2,0],0)
        self.assertNotEqual(K[0,2,0],K[2,0,0]) # transmitting width belongs to column
        for _ in range(3):env.step([0,0,0])
        coeff=build_link_coefficients(env.layout,env.export_allocation(),PHYSICS)
        for r,(receiver,slot) in enumerate(zip(coeff.slot_owner,coeff.occupied_slots)):
            for transmitter in np.flatnonzero(env.counts):
                expected=K[receiver,transmitter,slot]/env.counts[transmitter]
                self.assertAlmostEqual(coeff.interference[r,transmitter],expected,places=25)

    def test_prior_shared_formula_equivalence(self):
        from satellite_rl.shared_interface.geometry import directions_and_distance
        from satellite_rl.shared_interface.spherical_gaussian import linear_gain
        env=T0WidthEnv(fixture())
        for _ in range(5):env.step([0,1,1])
        coeff=build_link_coefficients(env.layout,env.export_allocation(),PHYSICS)
        rays,distances=directions_and_distance(env.layout['centres_lat_lon_deg'])
        angles=2*np.arcsin(np.minimum(np.linalg.norm(rays[:,None,:]-rays[None,:,:],axis=-1)/2,1))
        gains,_=linear_gain(angles,np.deg2rad(env.layout['full_width_deg'])[None,:])
        prior=np.zeros_like(coeff.interference)
        for r,(receiver,slot) in enumerate(zip(coeff.slot_owner,coeff.occupied_slots)):
            for j in np.flatnonzero(env.counts):
                if env.groups[receiver]%2==env.groups[j]%2 and env.groups[receiver]!=env.groups[j] and env.starts[j]<=slot<env.starts[j]+env.counts[j]:
                    frequency=PHYSICS['f0_Hz']+(slot+.5)*PHYSICS['slot_Hz']
                    prior[r,j]=gains[receiver,j]*10**(PHYSICS['rx_gain_dBi']/10)*(PHYSICS['c_m_per_s']/(4*np.pi*distances[receiver]*frequency))**2/env.counts[j]
        np.testing.assert_allclose(coeff.interference,prior,rtol=2e-15,atol=0)

    def test_relation_future_domain_and_outgoing(self):
        env=T0WidthEnv(fixture(),'S01')
        obs=env.reset();K=raw_slot_coupling(env.layout,env.groups,PHYSICS)
        np.testing.assert_array_equal(obs[207:307],np.zeros(100))
        expected=np.log1p(K[0,env.indices[1:],:].sum(axis=0)/env.noise)
        np.testing.assert_allclose(obs[307:],expected,rtol=1e-7)
        env.step([0,0,0]);env.step([0,0,0])
        obs=env._observation();current=env.indices[2]
        self.assertGreater(obs[207],0)
        self.assertEqual(obs[208],0)
        expected=np.log1p(K[0,current,0]/env.noise)
        self.assertAlmostEqual(float(obs[207]),float(expected),places=6)

    def test_hand_reward(self):
        self.assertEqual(legacy_repaired_reward(1e8,1e8,1e8,1),2.)
        self.assertEqual(legacy_repaired_reward(0,1e8,1e8,0),-1.05)
        expected=-1+1-(1-np.exp(-np.sqrt(2)))**3-.05
        self.assertAlmostEqual(legacy_repaired_reward(2e8,1e8,1e8,.5),expected)

    def test_processing_prefix_reward_reduction_independent_of_storage(self):
        from satellite_rl.shared_interface.content import seal,content_sha
        source=fixture(17)
        permutation=np.asarray([16,1,15,3,13,5,11,7,9,8,6,10,4,12,2,14,0])
        layout=dict(source.layout)
        for name in ('beam_ids','source_slot_ids','anchor_aircraft_ids'):layout[name]=[source.layout[name][i] for i in permutation]
        for name in ('centres_lat_lon_deg','full_width_deg','aircraft_count_by_beam','request_bps_by_beam_int64'):layout[name]=source.layout[name][permutation].copy()
        layout=seal(layout)
        registry=dict(schema='R7PublishedParentRegistryV1',case_sha_by_id={source.case['case_id']:source.case['content_sha256']},layout_shas=[layout['content_sha256']],allocation_shas=[])
        reordered=UniformStage2Input(source.case,layout,Authority(registry,content_sha(registry)),PHYSICS)
        envs=[T0WidthEnv(source),T0WidthEnv(reordered)]
        # A fixed nonuniform capacity fixture makes prefix reduction sensitive
        # to ordering and includes already-served zero rates. No optimizer.
        ratios=np.asarray([0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,.125,.25,.375,.625,.875,np.nextafter(1.,0.),1.])
        previous_sums=[0.,0.]
        for index in range(17):
            observed=[]
            prefix=ratios[:index+1]
            for storage,env in enumerate(envs):
                rates=np.zeros(env.N,np.float64)
                rates[env.indices[:index+1]]=prefix*env.demands[env.indices[:index+1]]
                # Division is repeated exactly as the original numpy routine.
                expected=float(np.minimum(rates[env.indices[:index+1]]/env.demands[env.indices[:index+1]],1).sum())
                f,l=np.argwhere(env.action_mask()['frequency_slots'])[0]
                with patch('satellite_rl.t0_width.environment.capacity_bps',return_value=rates):
                    _,reward,_,info=env.step([int(f),int(l),0])
                self.assertEqual(env.total_satisfaction,expected)
                self.assertEqual(info['satisfaction_increment'],expected-previous_sums[storage])
                previous_sums[storage]=expected
                observed.append((env.total_satisfaction,info['satisfaction_increment'],reward))
            self.assertEqual(observed[0],observed[1])
        # Exact threshold remains exact: below1 is -1 even by one ULP.
        self.assertEqual(legacy_repaired_reward(0,1e8,1e8,np.nextafter(1.,0.)),-1.05)
        self.assertEqual(legacy_repaired_reward(0,1e8,1e8,1.),.95)

if __name__=='__main__':unittest.main()
