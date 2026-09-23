"""Small independent oracles for beam power and interference in both environments."""

import importlib
import math
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


MODULE_NAMES = ("satellite", "satellite_variant_103")
SLOT_BANDWIDTH = 2.5e7
OTHER_LOSS_DB = 5.0


def db(value):
    return 10 * math.log10(value)


def link_loss_db(module, distance_km, frequency_hz):
    return 20 * math.log10(4 * math.pi * distance_km * 1000 * frequency_hz / module.c)


class EnvironmentPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_root = tempfile.TemporaryDirectory(prefix="satellite_physics_test_")
        cls.output_patch = patch.dict(os.environ, {
            "SATELLITE_RL_OUTPUT_ROOT": cls.output_root.name,
            "MPLBACKEND": "Agg",
        })
        cls.output_patch.start()
        cls.modules = [importlib.import_module("satellite_rl.envs." + name)
                       for name in MODULE_NAMES]

    @classmethod
    def tearDownClass(cls):
        cls.output_patch.stop()
        cls.output_root.cleanup()

    def setUp(self):
        saved_random_state = np.random.get_state()
        self.addCleanup(np.random.set_state, saved_random_state)

    def make_env(self, module):
        np.random.seed(20260924)
        scene = module.BeamInfo()
        env = module.Env(module.SatelliteFreqState(), scene,
                         module.CurrentBeamState(scene.beam_info))
        env.reset(scene)
        return env, scene

    def test_power_is_shared_only_across_successful_slots(self):
        for module in self.modules:
            with self.subTest(module=module.__name__):
                env, _ = self.make_env(module)
                power_w = 20.0
                received = []
                for start, slots in ((22, 2), (19, 8)):
                    action = [0, start, slots, db(power_w)]
                    received.append(float(env.receiving_power(action)))
                    center = 1.77e10 + (start + slots / 2) * SLOT_BANDWIDTH
                    expected = (db(power_w / slots)
                                + float(env.beam_info["beam_G_Tx_max"][0][0])
                                + float(env.beam_info["UE_G_Rx_max"][0][0])
                                - link_loss_db(module, float(env.beam_info["distances"][0]), center))
                    self.assertAlmostEqual(received[-1], expected, places=10)
                self.assertAlmostEqual(received[0] - received[1], db(4), places=10)

    def test_zero_slots_do_not_radiate_but_keep_existing_budget_charge(self):
        for module in self.modules:
            with self.subTest(module=module.__name__):
                env, _ = self.make_env(module)
                self.assertEqual(env.receiving_power([0, 20, 0, db(20)]), -math.inf)
                env.satellite_freq_state.freq_pool_real[0][20] = 1
                action = [0, 20, 2, db(20)]
                next_state, _, _, info = env.step(action)
                self.assertEqual(info["success_slots"], 0)
                self.assertEqual(float(env.beam_bps[0]), 0.0)
                self.assertEqual(len(env.history_cinr), 0)
                self.assertAlmostEqual(float(env.total_power), 20.0)
                self.assertEqual(next_state.shape, (204,))
                np.testing.assert_array_equal(env._calc_group_slot_interference(2), np.zeros(100))

    def test_clipped_request_uses_only_allocated_slots_for_signal(self):
        for module in self.modules:
            with self.subTest(module=module.__name__):
                env, _ = self.make_env(module)
                env.satellite_freq_state.freq_pool_real[0][22] = 1
                action = [0, 20, 5, db(20)]
                _, _, _, info = env.step(action)
                self.assertEqual(info["success_slots"], 2)
                self.assertAlmostEqual(float(env.total_power), 20)
                center = 1.77e10 + 21 * SLOT_BANDWIDTH
                signal_db = (db(20 / 2)
                    + float(env.beam_info["beam_G_Tx_max"][0][0])
                    + float(env.beam_info["UE_G_Rx_max"][0][0])
                    - link_loss_db(module, float(env.beam_info["distances"][0]), center))
                self.assertAlmostEqual(float(env.history_cinr[0][3]), signal_db, places=10)
                self.assertEqual(len(env.history_cinr), 2)

    def test_antenna_boundary_is_continuous_and_nonincreasing(self):
        for module in self.modules:
            with self.subTest(module=module.__name__):
                antenna = module.AntennaPattern()
                edge = module.bw / 2
                gains = [antenna.transmitting_antenna_pattern(50, phi)
                         for phi in (0, edge - 1e-8, edge, edge + 1e-8, edge + 0.01, 1)]
                self.assertAlmostEqual(gains[0], 50)
                self.assertAlmostEqual(gains[2], 47)
                self.assertLess(abs(gains[2] - gains[3]), 1e-5)
                self.assertTrue(all(a >= b for a, b in zip(gains, gains[1:])))
                self.assertEqual(antenna.transmitting_antenna_pattern(50, -1), gains[-1])

    def test_reset_recomputes_demand_scale(self):
        for module in self.modules:
            with self.subTest(module=module.__name__):
                env, scene = self.make_env(module)
                scene.beam_info["rate"][:] = 3e8
                env.reset(scene)
                self.assertAlmostEqual(np.asarray(env.beta).item(), 3e8)
                scene.beam_info["rate"][:] = 1e8
                env.reset(scene)
                self.assertAlmostEqual(np.asarray(env.beta).item(), 1e8)
                self.assertEqual(env.get_combined_state(
                    env.current_beam_state, env.satellite_freq_state).shape, (204,))

    def test_final_rates_match_independent_link_oracle_in_both_orders(self):
        # A and B have distinct slot counts, power, Tx/Rx gains, distance and noise.
        # Their boresights coincide so the geometry contribution is exactly 0 dB.
        identities = {
            "A": {"group": 0, "start": 20, "slots": 2, "power": 20., "tx": 50.,
                  "rx": 40., "distance": 36000., "noise": 200.},
            "B": {"group": 2, "start": 19, "slots": 4, "power": 30., "tx": 45.,
                  "rx": 37., "distance": 36500., "noise": 240.},
        }
        for module in self.modules:
            with self.subTest(module=module.__name__):
                oracle = {}
                for target_name, target in identities.items():
                    other = identities["B" if target_name == "A" else "A"]
                    center = 1.77e10 + (target["start"] + target["slots"] / 2) * SLOT_BANDWIDTH
                    signal_db = (db(target["power"] / target["slots"]) + target["tx"] + target["rx"]
                                 - link_loss_db(module, target["distance"], center))
                    rate = 0.0
                    for slot in range(target["start"], target["start"] + target["slots"]):
                        noise = module.k * target["noise"] * SLOT_BANDWIDTH
                        interference = 0.0
                        if other["start"] <= slot < other["start"] + other["slots"]:
                            frequency = 1.77e10 + (slot + 0.5) * SLOT_BANDWIDTH
                            interference = 10 ** ((db(other["power"] / other["slots"])
                                + other["tx"] + target["rx"]
                                - link_loss_db(module, target["distance"], frequency)) / 10)
                        cinr_db = signal_db - db(noise + interference) - OTHER_LOSS_DB
                        rate += module.MCS().get_spectral_efficiency(cinr_db) * SLOT_BANDWIDTH
                    oracle[target_name] = rate

                for order in (("A", "B"), ("B", "A")):
                    env, scene = self.make_env(module)
                    for index, identity in zip((0, 2), order):
                        spec = identities[identity]
                        scene.beam_info["beam_G_Tx_max"][index][0] = spec["tx"]
                        scene.beam_info["UE_G_Rx_max"][index][0] = spec["rx"]
                        scene.beam_info["distances"][index] = spec["distance"]
                        scene.beam_info["noise_temperature"][index] = spec["noise"]
                    scene.beam_info["beam_location"][2] = scene.beam_info["beam_location"][0]
                    env.reset(scene)
                    with patch.object(env, "calculate_beam_angle", return_value=0.0):
                        for index in range(3):
                            if index == 1:
                                action = [1, 20, 1, db(5)]
                            else:
                                spec = identities[order[0 if index == 0 else 1]]
                                action = [spec["group"], spec["start"], spec["slots"], db(spec["power"])]
                            _, _, _, info = env.step(action)
                            self.assertEqual(info["success_slots"], action[2])
                        for index, identity in zip((0, 2), order):
                            self.assertAlmostEqual(float(env.beam_bps[index]), oracle[identity],
                                                   delta=1e-5)

    def test_observed_interference_uses_same_cochannel_rule(self):
        for module in self.modules:
            with self.subTest(module=module.__name__):
                env, scene = self.make_env(module)
                scene.beam_info["beam_location"][2] = scene.beam_info["beam_location"][0]
                env.reset(scene)
                env.beam_idx = 2
                env.history_beam = [(0, 0, 20, 4, db(20))]
                with patch.object(env, "calculate_beam_angle", return_value=0.0), \
                     patch.object(env, "beamwidth_to_ground_distance", side_effect=AssertionError(
                         "No distance cutoff belongs in the interference observation")):
                    observed = env._calc_group_slot_interference(2)
                distance = float(scene.beam_info["distances"][2])
                tx = float(scene.beam_info["beam_G_Tx_max"][0][0])
                rx = float(scene.beam_info["UE_G_Rx_max"][2][0])
                for slot in range(20, 24):
                    frequency = 1.77e10 + (slot + 0.5) * SLOT_BANDWIDTH
                    expected = 10 ** ((db(20 / 4) + tx + rx
                        - link_loss_db(module, distance, frequency)) / 10)
                    self.assertAlmostEqual(observed[slot], expected, delta=expected * 1e-12)
                self.assertEqual(int(np.count_nonzero(observed)), 4)


if __name__ == "__main__":
    unittest.main()
