"""Deterministic checks for the second-stage basic-service reward."""
import math
import unittest
from types import SimpleNamespace

import numpy as np

from satellite_rl.envs import satellite
from satellite_rl.evaluation.terminal_metrics import terminal_beam_arrays, terminal_metrics


class CoverageRewardTests(unittest.TestCase):
    def make_env(self, reward_variant):
        np.random.seed(20260924)
        scene = satellite.BeamInfo()
        env = satellite.Env(satellite.SatelliteFreqState(), scene,
                            satellite.CurrentBeamState(scene.beam_info),
                            reward_variant=reward_variant)
        env.reset(scene)
        return env

    def test_basic_service_caps_at_ten_percent_and_tracks_interference_loss(self):
        env = self.make_env("fused")
        env.beam_info['rate'][:2] = 100e6
        env.beta = 100e6
        env.beam_bps = [5e6, 10e6]
        basic, satisfaction, _ = env.service_totals()
        self.assertAlmostEqual(basic, 1.5)
        self.assertAlmostEqual(satisfaction, .15)
        env.beam_bps = [10e6, 50e6]
        self.assertAlmostEqual(env.service_totals()[0], 2.0)
        env.beam_bps[0] = 5e6  # 后续干扰令旧波束跌破最低容量。
        self.assertAlmostEqual(env.service_totals()[0], 1.5)

    def test_zero_slot_uses_actual_power_in_both_repaired_rewards(self):
        for variant in ("legacy_repaired", "fused"):
            with self.subTest(variant=variant):
                env = self.make_env(variant)
                env.satellite_freq_state.freq_pool_real[0][20] = 1
                _, reward, _, info = env.step([0, 20, 2, 10 * math.log10(20)])
                self.assertEqual(info['success_slots'], 0)
                self.assertEqual(info['actual_power'], 0.0)
                self.assertEqual(env.total_power, 0.0)
                self.assertEqual(env.beam_bps, [0])
                if variant == "fused":
                    self.assertAlmostEqual(reward, 0.0)

    def test_fused_step_matches_independent_delta_formula(self):
        env = self.make_env("fused")
        before = env.service_totals()
        _, reward, _, info = env.step([0, 20, 1, 10 * math.log10(20)])
        after = env.service_totals()
        expected = (2 * (after[0] - before[0]) + (after[1] - before[1])
                    + .2 * (after[2] - before[2]) - info['actual_power'] / 3000)
        self.assertAlmostEqual(reward, expected, places=10)
        self.assertEqual(env.get_combined_state(env.current_beam_state,
                         env.satellite_freq_state).shape, (204,))

    def test_terminal_coverage_includes_unprocessed_beams(self):
        env = SimpleNamespace(beam_info={'rate': np.array([100., 200., 300.])},
                              beam_bps=[10., 19.], beta=200., history_beam=[],
                              total_power=0., overload=False,
                              satellite_freq_state=SimpleNamespace(freq_pool_real=np.zeros((8, 100))))
        arrays = terminal_beam_arrays(env)
        np.testing.assert_array_equal(arrays['basic_met'], [True, False, False])
        summary = terminal_metrics(env)
        self.assertEqual(summary['basic_service_count'], 1)
        self.assertEqual(summary['below_basic_indices'], [1, 2])


if __name__ == '__main__':
    unittest.main()
