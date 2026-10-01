"""Physical and input-contract checks for the full-observation graph encoder."""

import math
import unittest

import torch

from satellite_rl.agents.graph_encoder import (GraphEncoder, PHYSICAL_CONSTANTS,
                                               validate_physical_contract)
from satellite_rl.agents.workbench_networks import Encoder, MultiHeadNetwork
from satellite_rl.envs.regions import get_profile
from satellite_rl.envs import satellite as legacy
from satellite_rl.workbench.config import default_config
from satellite_rl.workbench.environment import WorkbenchEnv
from satellite_rl.state_layouts import natural_shapes
from tests.test_workbench_natural_context import make_config


def config(mode, small=True):
    result = default_config()
    result["network"].update(graph_mode=mode, kind="transformer",
                             transformer_layout="tsac_natural_context",
                             embedding_dim=16 if small else 128,
                             num_heads=4 if small else 8,
                             num_layers=1 if small else 2,
                             feedforward_dim=32 if small else 512,
                             activation="relu", dropout=0)
    result["environment"]["profile"] = "europe_v1"
    result["state"].update(context_level="full", demand_visibility="known",
                           features=list(natural_shapes("full")))
    return result


def observation():
    state = torch.zeros(2946)
    profile = get_profile("europe_v1")
    rows = state[214:1314].reshape(100, 11)
    rows[:, 0] = 2 * (50 - sum(profile.latitude_bounds_deg) / 2) / (
        profile.latitude_bounds_deg[1] - profile.latitude_bounds_deg[0])
    rows[:, 1] = 2 * (13 - sum(profile.longitude_bounds_deg) / 2) / (
        profile.longitude_bounds_deg[1] - profile.longitude_bounds_deg[0])
    rows[:, 2] = 0.5
    rows[:, 3] = torch.arange(100) % 8 / 7
    rows[:, 4] = torch.arange(100) / 99
    state[205] = 2 / 100  # current processing index
    state[:2] = rows[2, :2]
    rows[:2, 5] = 1
    rows[0, 6:9] = torch.tensor([10 / 99, 4 / 10, 20 / 50])
    rows[1, 6:9] = torch.tensor([12 / 99, 4 / 10, 20 / 50])
    return state


class GraphEncoderTests(unittest.TestCase):
    def test_real_environment_physics_and_scene_contract(self):
        base = make_config("full")
        env = WorkbenchEnv(base)
        state = env.reset(seed=103)
        spec = make_config("full")
        spec["network"]["graph_mode"] = "interference_bias"
        model = GraphEncoder(spec)
        profile = get_profile("europe_v1")
        constants = dict(PHYSICAL_CONSTANTS, beam_width_deg=legacy.bw,
                         carrier_start_hz=legacy.f_min, light_speed_m_s=legacy.c,
                         latitude_bounds_deg=profile.latitude_bounds_deg,
                         longitude_bounds_deg=profile.longitude_bounds_deg,
                         satellite_longitude_deg=profile.satellite_longitude_deg,
                         earth_radius_km=profile.earth_radius_km,
                         orbit_radius_km=profile.orbit_radius_km)
        self.assertEqual(validate_physical_contract(env.scene(), spec, constants)["beam_width_deg"], 0.5)
        altered = env.scene()
        altered["beam_G_Tx_max"][2, 0] = 49
        with self.assertRaisesRegex(ValueError, "beam_G_Tx_max"):
            validate_physical_contract(altered, spec, constants)
        with self.assertRaisesRegex(ValueError, "slot_bandwidth_hz"):
            validate_physical_contract(env.scene(), spec,
                                       dict(constants, slot_bandwidth_hz=20e6))

        # Real actions leave a single same-polarization source in slot 10 for
        # current receiver 8. Compare to the existing environment calculation,
        # which uses its own direction pattern and per-slot free-space loss.
        for beam_id in range(8):
            freq = {0: 11, 2: 10, 4: 30, 6: 40}.get(beam_id, 50 + beam_id)
            state, _, _, _ = env.step((freq, 0, 0))
        edge = model.edge_features(torch.as_tensor(state, dtype=torch.float32)[None])[0, 8, 2]
        raw_w = env.raw_env._calc_group_slot_interference(0)[10]
        self.assertAlmostEqual(edge[0].item() * 200 - 300, 10 * math.log10(raw_w), delta=2e-4)
        self.assertEqual(edge[1].item(), 0)  # current action is still unknown
        conflicts, _, _ = env.raw_env.interfere_beams([0, 10, 1, 10 * math.log10(5)])
        raw_edge = env.raw_env.interfere_power(conflicts, [0, 10, 1, 10 * math.log10(5)])[0]
        self.assertEqual(len(raw_edge), 1)
        self.assertEqual(raw_edge[0][3], 2)
        self.assertAlmostEqual(edge[0].item() * 200 - 300, raw_edge[0][2], delta=2e-4)
        state, _, _, _ = env.step((10, 0, 0))
        actual = model.edge_features(torch.as_tensor(state, dtype=torch.float32)[None])[0, 8, 2]
        self.assertEqual(actual[1].item(), 1)

    def test_token_contract_and_gradients(self):
        for mode in ("beam_tokens", "interference_bias"):
            with self.subTest(mode=mode):
                model = GraphEncoder(config(mode), graph_mode=mode)
                states = observation().repeat(2, 1)
                self.assertEqual(model._tokens(states).shape, (2, 122, 16))
                output = model(states)
                self.assertEqual(output.shape, (2, 16))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(model.beam_embed.weight.grad)
                self.assertTrue(torch.isfinite(model.beam_embed.weight.grad).all())
                if mode == "interference_bias":
                    self.assertIsNotNone(model.edge_weight.grad)
                    self.assertTrue(torch.isfinite(model.edge_weight.grad).all())
        with self.assertRaises(ValueError):
            GraphEncoder(config("beam_tokens"), state_dim=207)

    def test_same_raw_fields_and_common_initialization(self):
        torch.manual_seed(91)
        g1 = GraphEncoder(config("beam_tokens"))
        torch.manual_seed(91)
        g2 = GraphEncoder(config("interference_bias"))
        for name, weight in g1.state_dict().items():
            self.assertTrue(torch.equal(weight, g2.state_dict()[name]), name)
        state = observation().unsqueeze(0)
        self.assertTrue(torch.equal(g1._tokens(state), g2._tokens(state)))
        self.assertTrue(torch.allclose(g1(state), g2(state), atol=1e-6))
        # Every beam retains all eleven original fields in its own token.
        rows = state[:, 214:1314].reshape(1, 100, 11)
        with torch.no_grad():
            for field in range(11):
                changed = state.clone()
                changed[:, 214 + 37 * 11 + field] += 0.25
                self.assertFalse(torch.equal(g1._tokens(state)[:, 14 + 37],
                                             g1._tokens(changed)[:, 14 + 37]))
        self.assertEqual(rows.shape, (1, 100, 11))

    def test_g0_g1_g2_common_weights_and_head_rng(self):
        plain = config("beam_tokens")
        plain["network"]["graph_mode"] = "none"
        torch.manual_seed(507)
        g0 = MultiHeadNetwork(plain, 2946, (100, 10, 10))
        graphs = []
        for mode in ("beam_tokens", "interference_bias"):
            torch.manual_seed(507)
            graph = MultiHeadNetwork(config(mode), 2946, (100, 10, 10))
            graphs.append(graph)
            self.assertEqual(torch.get_rng_state().tolist(),
                             # Head initialization after the graph must consume
                             # the same stream as G0.
                             self._g0_final_rng(plain))
            for old_name, new_name in (("geo_embed", "geo_embed"),
                                       ("rate_embed", "rate_embed"),
                                       ("power_embed", "power_embed"),
                                       ("slot_embed", "slot_embed"),
                                       ("global_embed", "global_embed"),
                                       ("resource_embed", "resource_embed")):
                old = getattr(g0.encoder, old_name).state_dict()
                new = getattr(graph.encoder.graph, new_name).state_dict()
                for key in old:
                    self.assertTrue(torch.equal(old[key], new[key]), (mode, old_name, key))
            self.assertTrue(torch.equal(g0.encoder.pos_embed, graph.encoder.graph.base_position))
            self.assertTrue(torch.equal(g0.encoder.extra_pos_embed[:, :1],
                                        graph.encoder.graph.global_position))
            self.assertTrue(torch.equal(g0.encoder.extra_pos_embed[:, 11:19],
                                        graph.encoder.graph.resource_position))
            for old_layer, new_layer in zip(g0.encoder.net.layers, graph.encoder.graph.layers):
                self.assertTrue(torch.equal(old_layer.self_attn.in_proj_weight, new_layer.qkv.weight))
                self.assertTrue(torch.equal(old_layer.self_attn.in_proj_bias, new_layer.qkv.bias))
                for old_mod, new_mod in ((old_layer.self_attn.out_proj, new_layer.out),
                                         (old_layer.linear1, new_layer.feedforward[0]),
                                         (old_layer.linear2, new_layer.feedforward[3]),
                                         (old_layer.norm1, new_layer.norm1),
                                         (old_layer.norm2, new_layer.norm2)):
                    for key, value in old_mod.state_dict().items():
                        self.assertTrue(torch.equal(value, new_mod.state_dict()[key]))
            for old_head, new_head in zip(g0.heads, graph.heads):
                for key, value in old_head.state_dict().items():
                    self.assertTrue(torch.equal(value, new_head.state_dict()[key]))
        for name, value in graphs[0].encoder.graph.state_dict().items():
            self.assertTrue(torch.equal(value, graphs[1].encoder.graph.state_dict()[name]), name)

    @staticmethod
    def _g0_final_rng(plain):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(507)
            MultiHeadNetwork(plain, 2946, (100, 10, 10))
            return torch.get_rng_state().tolist()

    def test_sdpa_layer_matches_g0_without_bias_on_same_tokens(self):
        torch.manual_seed(731)
        old = Encoder({**config("beam_tokens"),
                       "network": {**config("beam_tokens")["network"], "graph_mode": "none"}}, 2946)
        torch.manual_seed(731)
        graph = GraphEncoder(config("beam_tokens"))
        tokens = torch.randn(2, 32, graph.output_dim)
        old.eval()
        graph.eval()
        with torch.no_grad():
            for old_layer, new_layer in zip(old.net.layers, graph.layers):
                reference = old_layer(tokens.transpose(0, 1)).transpose(0, 1)
                actual = new_layer(tokens)
                self.assertTrue(torch.allclose(reference, actual, atol=2e-6, rtol=2e-6))
                tokens = reference

    def test_directed_physics_overlap_and_future_mask(self):
        model = GraphEncoder(config("interference_bias"))
        state = observation()
        rows = state[214:1314].reshape(100, 11)
        # Beam 0 (group 0) transmits; beam 8 (also group 0) has overlapping
        # slots. Beam 1 is opposite polarization despite slot overlap.
        rows[8, 5] = 1
        rows[8, 6:9] = torch.tensor([12 / 99, 3 / 10, 20 / 50])
        features = model.edge_features(state.unsqueeze(0))[0]
        self.assertAlmostEqual(features[8, 0, 1].item(), 2 / 4, places=5)
        self.assertEqual(features[1, 0].tolist(), [0, 0])
        self.assertEqual(features[8, 8].tolist(), [0, 0])
        self.assertGreater(features[2, 0, 0].item(), 0)
        self.assertEqual(features[2, 0, 1].item(), 0)  # future target has no slots
        future_changed = state.clone()
        future_changed[214 + 24 * 11 + 6:214 + 24 * 11 + 11] = 0.9
        self.assertTrue(torch.equal(features, model.edge_features(future_changed[None])[0]))
        # Changing transmitter power changes j->i and not an unallocated j.
        louder = state.clone()
        louder[214 + 0 * 11 + 8] = 40 / 50
        changed = model.edge_features(louder[None])[0]
        self.assertGreater(changed[8, 0, 0].item(), features[8, 0, 0].item())
        self.assertEqual(changed[0, 24].tolist(), [0, 0])

    def test_potential_power_matches_fixed_direction_and_db_units(self):
        model = GraphEncoder(config("interference_bias"))
        state = observation()
        row = state[214:1314].reshape(100, 11)
        row[0, 0] += 0.0001  # nonzero beam angle
        features = model.edge_features(state[None])[0]
        profile = get_profile("europe_v1")
        def xyz(normalized):
            lat = (normalized[0].item() + 1) * 0.5 * (
                profile.latitude_bounds_deg[1] - profile.latitude_bounds_deg[0]) + profile.latitude_bounds_deg[0]
            lon = (normalized[1].item() + 1) * 0.5 * (
                profile.longitude_bounds_deg[1] - profile.longitude_bounds_deg[0]) + profile.longitude_bounds_deg[0]
            lat, lon = math.radians(lat), math.radians(lon)
            return [profile.earth_radius_km * math.cos(lat) * math.cos(lon),
                    profile.earth_radius_km * math.cos(lat) * math.sin(lon),
                    profile.earth_radius_km * math.sin(lat)]
        sat = [profile.orbit_radius_km * math.cos(math.radians(profile.satellite_longitude_deg)),
               profile.orbit_radius_km * math.sin(math.radians(profile.satellite_longitude_deg)), 0]
        source = [s - g for s, g in zip(sat, xyz(row[0]))]
        target = [s - g for s, g in zip(sat, xyz(row[8]))]
        distance = math.sqrt(sum(x * x for x in target)) * 1000
        cosine = sum(a * b for a, b in zip(source, target)) / (
            math.sqrt(sum(x * x for x in source)) * math.sqrt(sum(x * x for x in target)))
        angle = math.degrees(math.acos(min(1.0, max(-1.0, cosine))))
        sigma = 0.5 / (2 * math.sqrt(2 * math.log10(2)))
        attenuation = (3 * (angle / 0.25) ** 2 if angle <= 0.25 else
                       3 + 10 / math.log(10) * (angle * angle - 0.25 ** 2) / (2 * sigma * sigma))
        received_w = sum((20 / 4) * 10 ** ((50 - attenuation + 40) / 10)
                         * (2.99792458e8 /
                            (4 * math.pi * distance * (1.77e10 + 2.5e7 * (slot + 0.5)))) ** 2
                         for slot in range(10, 14)) / 4
        dbw = 10 * math.log10(received_w)
        self.assertAlmostEqual(features[8, 0, 0].item(), (max(-300, min(-100, dbw)) + 300) / 200,
                               delta=1e-5)

    def test_storage_permutation_preserves_identity(self):
        for mode in ("beam_tokens", "interference_bias"):
            model = GraphEncoder(config(mode)).eval()
            if mode == "interference_bias":
                with torch.no_grad():
                    model.edge_weight[:] = torch.tensor([0.8, 0.4])
            state = observation().unsqueeze(0)
            swapped = state.clone()
            permutation = torch.randperm(100, generator=torch.Generator().manual_seed(17))
            swapped[:, 214:1314] = state[:, 214:1314].reshape(1, 100, 11)[:, permutation].reshape(1, -1)
            self.assertTrue(torch.allclose(model(state), model(swapped), atol=2e-5, rtol=2e-5))


if __name__ == "__main__":
    unittest.main()
