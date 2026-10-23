import json
from pathlib import Path
import unittest

import numpy as np
import torch

from satellite_rl.sac_improvement.config import default_config, validate_config
from satellite_rl.sac_improvement.networks import (CandidateNetwork, build_network, changed_beams, gather_rows,
                                                   parameter_count)
from satellite_rl.s2fix.search_env import ACTION_WIDTH, SearchEnv

DATA = Path("D:/paper/outputs/experiments/s2_pure_tabu_20261021/data")
L1_ACTOR = Path("D:/paper/outputs/experiments/s2_orig_compare_20261022/i0_actors/L1-s51001_actor.pt")
FREE8_CONF = Path("D:/paper/outputs/experiments/s2_orig_compare_20261022/cross/free8_in_v2_conf_d432_n150.jsonl")


def config(**overrides):
    return validate_config(dict(default_config(), schema_version="s2fix_i0_v1", **overrides))


def random_inputs(n_beams=30, batch=3, actions=17):
    state = torch.zeros(batch, 2812)
    beams = state[:, :2800].view(batch, 200, 14)
    beams[:, :n_beams] = torch.rand(batch, n_beams, 14)
    beams[:, :n_beams, 0] = 1
    state[:, 2800:] = torch.rand(batch, 12)
    features = torch.zeros(batch, actions, ACTION_WIDTH)
    features[:, :-1, 0] = torch.randint(0, n_beams, (batch, actions - 1)) / 200
    features[:, :-1, 2] = 0.3
    features[:, -1, -1] = 1
    return state, features


class NetworkTest(unittest.TestCase):
    def test_default_is_original_mlp_and_loads_s2fix_weights(self):
        network = build_network(config(), ACTION_WIDTH)
        self.assertIsInstance(network, CandidateNetwork)
        network.load_state_dict(torch.load(L1_ACTOR, map_location="cpu", weights_only=True))
        state, features = random_inputs()
        self.assertEqual(tuple(network(state, features).shape), (3, 17))

    def test_set_networks_shapes_and_padding_invariance(self):
        for kind in ("deepsets", "transformer"):
            for pointer in (False, True):
                network = build_network(config(network=kind, embed_dim=32, num_layers=2, num_heads=4,
                                               pointer=pointer), ACTION_WIDTH).eval()
                state, features = random_inputs()
                out = network(state, features)
                self.assertEqual(tuple(out.shape), (3, 17))
                noisy = state.clone()
                noisy[:, 30 * 14:2800] = torch.rand(3, 2800 - 30 * 14) * (torch.arange(2800 - 30 * 14) % 14 != 0)
                torch.testing.assert_close(network(noisy, features), out)

    def test_pointer_gathers_changed_beam_rows(self):
        rows = torch.arange(2 * 200 * 1, dtype=torch.float32).view(2, 200, 1)
        features = torch.zeros(2, 2, ACTION_WIDTH)
        features[0, 0, 0], features[0, 0, 2] = 7 / 200, 0.1
        features[0, 0, 5], features[0, 0, 7] = 9 / 200, 0.2
        features[1, 1, 0], features[1, 1, 2] = 3 / 200, 0.1
        index, present = changed_beams(features)
        out = gather_rows(rows, index, present)
        self.assertEqual(out[0, 0, 0].item(), 8.0)
        self.assertEqual(out[1, 1, 0].item(), 203.0)
        self.assertEqual(out[0, 1, 0].item(), 0.0)

    def test_parameter_counts_grow_with_size(self):
        small = parameter_count(build_network(config(network="transformer", embed_dim=64, num_heads=4), ACTION_WIDTH))
        large = parameter_count(build_network(config(network="transformer", embed_dim=256, num_layers=4), ACTION_WIDTH))
        self.assertLess(small, large)

    def test_invalid_network_rejected(self):
        with self.assertRaises(ValueError):
            config(network="lstm")


class SpreadStartTest(unittest.TestCase):
    def test_spread_start_is_legal_and_matches_earlier_rule(self):
        from satellite_rl.s2fix.data import read_records, load_input
        from satellite_rl.s2fix.evaluate import metrics
        from satellite_rl.s2fix.greedy import spread_allocation
        record = read_records(DATA, "conf_d432_n150")[0]
        stage_input = load_input(record)
        groups, starts, counts, power = spread_allocation(stage_input)
        self.assertTrue(((groups >= 0) & (groups < 8)).all() and (counts >= 1).all() and (counts <= 10).all())
        self.assertTrue((starts >= 0).all() and (starts + counts <= 100).all())
        for g in range(8):
            occupied = np.zeros(100, int)
            for i in np.flatnonzero(groups == g):
                occupied[starts[i]:starts[i] + counts[i]] += 1
            self.assertLessEqual(occupied.max(), 1)
        earlier = {json.loads(l)["case_id"]: json.loads(l) for l in FREE8_CONF.open(encoding="utf8")}
        self.assertAlmostEqual(metrics(stage_input, groups, starts, counts, power)["W"],
                               earlier[record["case_id"]]["贪心＋8组＋任意空位"]["W"], places=12)
        env = SearchEnv(dict(start="spread"))
        env.reset(stage_input)
        self.assertAlmostEqual(env.base_eval.W, earlier[record["case_id"]]["贪心＋8组＋任意空位"]["W"], places=12)


if __name__ == "__main__":
    unittest.main()
