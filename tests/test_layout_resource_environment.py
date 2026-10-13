"""Synthetic engineering checks only; no frozen validation/test inputs."""
import unittest
import numpy as np
from satellite_rl.shared_interface.content import content_sha
from satellite_rl.shared_interface.contracts import Authority, validate_allocation
from satellite_rl.shared_interface.geometry import cone_omega
from satellite_rl.shared_interface.spherical_gaussian import PHYSICS, PHYSICS_SHA
from satellite_rl.shared_interface.stages import make_case, make_layout, UniformStage2Input, UniformStage3FixedInput
from satellite_rl.layout_resource.environment import LayoutResourceEnv, encode_action, decode_action


def synthetic_input(n, reverse=False):
    ids = ['a%03d'%i for i in range(n)]
    bids = ['b%03d'%i for i in range(n)]
    case = make_case(dict(case_id='engineering', actual_role='synthetic_engineering',
        source_role='synthetic_engineering', original_mother_id='engineering',
        UTC='2026-10-13T00:00:00Z', source_asset_SHA='a'*64,
        source=dict(synthetic=True)), ids, [[52., 8.]]*n)
    order = list(range(n))[::-1] if reverse else list(range(n))
    widths = np.array([(.3, .5, 1.)[i%3] for i in order])
    layout = make_layout(case, beam_ids=[bids[i] for i in order], source_slot_ids=[str(i) for i in order],
        anchor_aircraft_ids=[ids[i] for i in order], centres_lat_lon_deg=[[52., 8.]]*n,
        full_width_deg=widths, owner_aircraft_to_beam_id=list(zip(ids, bids)), Ncap=n,
        geometry_Omega_cap_sr=cone_omega(np.ones(n)), geometry_Omega_sr=cone_omega(widths),
        geometry_audit=dict(synthetic=True), source_and_cost=dict(synthetic=True))
    registry = dict(schema='R7PublishedParentRegistryV1', case_sha_by_id={'engineering':case['content_sha256']},
        layout_shas=[layout['content_sha256']], allocation_shas=[])
    authority = Authority(registry, content_sha(registry))
    return UniformStage2Input(case, layout, authority, PHYSICS)


class LayoutResourceEnvironmentChecks(unittest.TestCase):
    def test_reset_zero_service_and_explicit_rule_budget(self):
        env = LayoutResourceEnv(synthetic_input(2), dict(power_w_per_slot=3.75))
        env.step(encode_action(0, 1))
        env.step(0)
        self.assertEqual(env.metrics()['sgm_by_beam'][1], 0)
        env.reset()
        env.step(0); env.step(0)
        self.assertEqual(env.metrics()['SI'], 0)
        self.assertEqual(env.metrics()['SGM'], 0)
        self.assertEqual(env.metrics()['power_w'], 0)
        for config in (dict(power_w_per_slot=True), dict(power_w_per_slot=0),
                       dict(power_w_per_slot=float('nan')), dict(power_w_per_slot=3000),
                       dict(reward='legacy'), dict(unexpected=1)):
            with self.assertRaises(ValueError): LayoutResourceEnv(synthetic_input(1), config)

    def test_variable_n_shapes_and_real_terminal_capacity(self):
        for n in (1, 2, 200):
            env = LayoutResourceEnv(synthetic_input(n))
            obs = env.reset()
            self.assertEqual(obs['tokens'].shape, (n, 16))
            self.assertEqual(obs['relations'].shape, (n, n, 4))
            self.assertEqual(obs['global_features'].shape, (12,))
            for i in range(n):
                obs, reward, done, info = env.step(encode_action(i//8, 1))
            self.assertTrue(done)
            self.assertEqual(obs['action_mask'].sum(), 1)
            self.assertTrue(obs['action_mask'][0])
            self.assertEqual(len(info['metrics']['capacity_bps']), n)
            self.assertTrue((info['metrics']['capacity_bps']>0).all())
            self.assertAlmostEqual(info['metrics']['power_w'], n*3.75)
            allocation = env.export_allocation()
            validate_allocation(env.stage.case, env.stage.layout, allocation, env.stage.authority, PHYSICS_SHA, published=False)
            with self.assertRaises(RuntimeError): env.step(0)

    def test_joint_mask_same_group_saturation_and_noop_all_n(self):
        env = LayoutResourceEnv(synthetic_input(200))
        for i in range(80):
            env.step(encode_action(i//8*10, 10))
        self.assertEqual(env.powers.sum(), 3000.)
        self.assertEqual(env.action_mask().sum(), 1)
        for i in range(120): env.step(0)
        self.assertEqual(env.metrics()['allocation_fraction'], .4)
        self.assertEqual(np.count_nonzero(env.metrics()['capacity_bps']), 80)
        self.assertLessEqual(env.metrics()['SI'], .4)

    def test_invalid_action_is_atomic_and_frequency_bounds_joint(self):
        env = LayoutResourceEnv(synthetic_input(9))
        self.assertFalse(env.action_mask()[1000]) # start99,count10 invalid
        for action in (True, 1001, -1, 1000):
            before = env.export_allocation()['content_sha256']
            with self.assertRaises(ValueError): env.step(action)
            self.assertEqual(before, env.export_allocation()['content_sha256'])
        env.step(encode_action(0, 10))
        for i in range(7): env.step(0)
        self.assertFalse(env.action_mask()[encode_action(9, 2)])
        self.assertTrue(env.action_mask()[encode_action(10, 10)])
        self.assertFalse(env.action_mask()[encode_action(0, 1)])

    def test_permutation_identity_and_independent_s3_authority(self):
        normal = LayoutResourceEnv(synthetic_input(9))
        reverse = LayoutResourceEnv(synthetic_input(9, True))
        np.testing.assert_array_equal(normal.reset()['tokens'], reverse.reset()['tokens'])
        for i in range(9):
            action = encode_action(i//8, 1)
            normal.step(action); reverse.step(action)
        np.testing.assert_array_equal(normal.capacities, reverse.capacities)
        allocation = normal.export_allocation()
        registry = dict(schema='R7PublishedParentRegistryV1', case_sha_by_id={'engineering':normal.stage.case['content_sha256']},
            layout_shas=[normal.stage.layout['content_sha256']], allocation_shas=[allocation['content_sha256']])
        authority = Authority(registry, content_sha(registry))
        third = UniformStage3FixedInput.from_layout_and_allocation(normal.stage.case, normal.stage.layout,
            allocation, authority, PHYSICS).fixed_initial_evaluation()
        np.testing.assert_array_equal(third['capacity_bps'], normal.capacities)
        self.assertEqual(allocation['group_by_beam_id']['b008'], 0)
        self.assertEqual(normal.stage.layout['full_width_deg'].tobytes(), synthetic_input(9).layout['full_width_deg'].tobytes())

    def test_cross_group_interference_updates_previously_served_beam(self):
        env = LayoutResourceEnv(synthetic_input(3))
        env.step(encode_action(0, 2))
        before = env.capacities[0]
        env.step(0)
        obs, _, _, _ = env.step(encode_action(0, 2))
        self.assertLess(env.capacities[0], before)
        self.assertGreater(obs['relations'][0, 2, 1], 0)
        self.assertEqual(obs['relations'][0, 2, 2], 1)
        self.assertAlmostEqual(float(obs['relations'][0, 2, 3]), .2)
        self.assertNotEqual(obs['relations'][0, 2, 1], obs['relations'][2, 0, 1])


if __name__ == '__main__': unittest.main()
