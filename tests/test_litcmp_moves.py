import unittest
import numpy as np
from satellite_rl.shared_interface.stages import make_case, make_layout, UniformStage2Input
from satellite_rl.shared_interface.spherical_gaussian import PHYSICS_V2
from satellite_rl.s2fix.greedy import greedy_allocation
from satellite_rl.s2fix.search_env import SearchEnv
from satellite_rl.litcmp.moves import Alloc, legal, propose, pack, uniform_start


def scene():
    ids = [f'a{i}' for i in range(20)]
    beams = [f'b{i}' for i in range(10)]
    centres = np.column_stack((np.linspace(44, 48, 10), np.linspace(8, 14, 10)))
    positions = np.repeat(centres, 2, axis=0) + np.tile([[0., 0.], [.01, .01]], (10, 1))
    case = make_case(dict(case_id='test', actual_role='dev', source_role='dev',
                         original_mother_id='test', UTC='2024-01-12T00:00:00Z',
                         source_asset_SHA='synthetic', source='synthetic'), ids, positions,
                     request_bps=43200000, plain=True)
    layout = make_layout(case, beam_ids=beams, source_slot_ids=list(range(10)),
                         anchor_aircraft_ids=ids[::2], centres_lat_lon_deg=centres,
                         full_width_deg=np.full(10, .3),
                         owner_aircraft_to_beam_id=[[a, beams[i // 2]] for i, a in enumerate(ids)],
                         Ncap=20, geometry_Omega_cap_sr=1., geometry_Omega_sr=.001,
                         geometry_audit={}, source_and_cost={}, plain=True)
    return UniformStage2Input(case, layout, None, PHYSICS_V2, plain=True)


class MovesTest(unittest.TestCase):
    def test_neighbors_and_legal_parity(self):
        alloc = Alloc(*greedy_allocation(scene()))
        rng = np.random.default_rng(51001)
        env = SearchEnv()
        total_power = alloc.powers.sum()
        for _ in range(500):
            alloc = propose(alloc, rng)
            self.assertTrue(legal(alloc))
            self.assertTrue(env._legal(alloc.starts, alloc.counts, alloc.groups))
            self.assertTrue(np.all((alloc.powers >= 0) & (alloc.powers <= 50)))
            self.assertAlmostEqual(alloc.powers.sum(), total_power)
        for _ in range(100):
            state = Alloc(rng.integers(8, size=10), rng.integers(-1, 101, size=10),
                          rng.integers(0, 12, size=10), np.ones(10))
            self.assertEqual(legal(state), env._legal(state.starts, state.counts, state.groups))
        # SearchEnv checks interval occupancy only for groups 0..7; match that contract.
        outside = Alloc(np.array([-1, 8]), np.zeros(2, int), np.ones(2, int), np.ones(2))
        self.assertEqual(legal(outside), env._legal(outside.starts, outside.counts, outside.groups))
        self.assertTrue(legal(outside))

    def test_pack_and_uniform(self):
        groups = np.zeros(20, int)
        starts, counts = pack(groups, np.full(20, 6), np.arange(20) + 1)
        self.assertEqual(counts.sum(), 100)
        self.assertTrue(np.all(counts >= 1))
        self.assertTrue(legal(Alloc(groups, starts, counts, np.ones(20))))
        alloc = uniform_start(scene())
        self.assertTrue(legal(alloc))
        self.assertLessEqual(alloc.powers.sum(), 3000)


if __name__ == '__main__':
    unittest.main()
