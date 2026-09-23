"""验证终止步资源登记、裁剪和两套环境各自的终止规则。"""

from contextlib import redirect_stdout
import copy
import importlib
import io
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class TerminalAllocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_root = tempfile.TemporaryDirectory(prefix="satellite_terminal_test_")
        cls.output_environment = patch.dict(os.environ, {
            "SATELLITE_RL_OUTPUT_ROOT": cls.output_root.name,
            "MPLBACKEND": "Agg",
        })
        cls.output_environment.start()
        cls.modules = tuple(importlib.import_module(f"satellite_rl.envs.{name}") for name in (
            "satellite", "satellite_variant_103",
        ))

    @classmethod
    def tearDownClass(cls):
        cls.output_environment.stop()
        cls.output_root.cleanup()

    def setUp(self):
        self.random_state = np.random.get_state()
        self.addCleanup(np.random.set_state, self.random_state)

    def make_env(self, module):
        np.random.seed(20260924)
        beam_info = module.BeamInfo()
        env = module.Env(
            module.SatelliteFreqState(), beam_info,
            module.CurrentBeamState(beam_info.beam_info),
        )
        env.reset(beam_info)
        return env

    def advance_to_last_beam(self, env, first_action=None):
        for index in range(99):
            action = first_action if index == 0 and first_action is not None else [
                index % 8, index // 8, 1, 10 * np.log10(5),
            ]
            _, _, done, info = env.step(action.copy())
            self.assertFalse(done)
            self.assertEqual(info["success_slots"], 1)

    def test_completed_episode_records_all_100_allocations(self):
        for module in self.modules:
            with self.subTest(environment=module.__name__):
                env = self.make_env(module)
                self.advance_to_last_beam(env)
                pool = env.satellite_freq_state
                self.assertEqual(pool.freq_pool_real.sum(), 99)
                final_beam_state = env.current_beam_state.get_state().copy()
                beam_id = env.current_beam_state.beam_id
                distance_before = pool.freq_pool_distance.copy()
                with patch.object(pool, "distance") as distance, patch.object(
                    env.current_beam_state, "update_state"
                ) as advance:
                    state, _, done, info = env.step([3, 12, 1, 10 * np.log10(5)])
                self.assertTrue(done)
                self.assertEqual(env.beam_idx, 100)
                self.assertEqual(len(env.history_beam), 100)
                self.assertEqual(pool.freq_pool_real.sum(), 100)
                self.assertEqual(info["frequency_utilization"], 100 / 800)
                self.assertEqual(pool.freq_pool_real[3, 12], 1)
                latitude, longitude = env.beam_info["beam_location"][99]
                self.assertEqual(pool.freq_pool_lat[3, 12], latitude)
                self.assertEqual(pool.freq_pool_lon[3, 12], longitude)
                distance.assert_not_called()
                advance.assert_not_called()
                self.assertEqual(env.current_beam_state.beam_id, beam_id)
                np.testing.assert_array_equal(pool.freq_pool_distance, distance_before)
                np.testing.assert_array_equal(state[:3], final_beam_state)
                # 保留原规则：处理完第 100 个波束后，终止状态的干扰/占用向量归零。
                np.testing.assert_array_equal(state[4:], np.zeros(200))

    def test_power_overload_records_allocation_and_preserves_variant_rules(self):
        for module in self.modules:
            with self.subTest(environment=module.__name__):
                env = self.make_env(module)
                for index in range(60):
                    _, _, done, _ = env.step([index % 8, index // 8, 1, 10 * np.log10(50)])
                    self.assertFalse(done)
                current_state = env.current_beam_state.get_state().copy()
                is_main = module.__name__.endswith(".satellite")
                with redirect_stdout(io.StringIO()), patch.object(
                    env.satellite_freq_state, "distance",
                    wraps=env.satellite_freq_state.distance,
                ) as distance:
                    state, _, done, info = env.step([4, 30, 3, 10 * np.log10(50)])
                self.assertEqual(done, is_main)
                self.assertTrue(env.overload)
                self.assertAlmostEqual(env.total_power, module.power_threshold + 50)
                self.assertEqual(env.satellite_freq_state.freq_pool_real.sum(), 63)
                self.assertEqual(info["frequency_utilization"], 63 / 800)
                self.assertEqual(info["success_slots"], 3)
                self.assertEqual(len(env.history_beam), 61)
                if is_main:
                    distance.assert_not_called()
                    self.assertEqual(env.current_beam_state.beam_id, 61)
                    np.testing.assert_array_equal(state[:3], current_state)
                else:
                    distance.assert_called_once()
                    self.assertEqual(env.current_beam_state.beam_id, 62)

    def test_terminal_allocation_uses_only_successful_slots(self):
        for module in self.modules:
            env = self.make_env(module)
            self.advance_to_last_beam(env)
            # 第 99 步后，组 3 的 0..11 已占用，其余为空。
            for name, action, expected_slots, expected_indices in (
                ("boundary", [3, 98, 5, 10 * np.log10(5)], 2, [98, 99]),
                ("occupied", [3, 0, 5, 10 * np.log10(5)], 0, []),
                # 从空槽向已占用槽前进，只登记第一个冲突之前的两个槽。
                ("partial", [4, 50, 5, 10 * np.log10(5)], 2, [50, 51]),
            ):
                with self.subTest(environment=module.__name__, case=name):
                    candidate = copy.deepcopy(env)
                    if name == "partial":
                        candidate = self.make_env(module)
                        self.advance_to_last_beam(candidate, [4, 52, 1, 10 * np.log10(5)])
                    pool = candidate.satellite_freq_state
                    before = pool.freq_pool_real.copy()
                    latitude_before = pool.freq_pool_lat.copy()
                    longitude_before = pool.freq_pool_lon.copy()
                    state, _, done, info = candidate.step(action.copy())
                    self.assertTrue(done)
                    self.assertEqual(info["success_slots"], expected_slots)
                    self.assertEqual(candidate.history_beam[-1][3], expected_slots)
                    expected = before.copy()
                    expected[action[0], expected_indices] += 1
                    np.testing.assert_array_equal(pool.freq_pool_real, expected)
                    latitude, longitude = candidate.beam_info["beam_location"][99]
                    latitude_before[action[0], expected_indices] += latitude
                    longitude_before[action[0], expected_indices] += longitude
                    np.testing.assert_array_equal(pool.freq_pool_lat, latitude_before)
                    np.testing.assert_array_equal(pool.freq_pool_lon, longitude_before)
                    self.assertEqual(info["frequency_utilization"], expected.mean())
                    self.assertEqual(state.shape, (204,))

    def test_nonterminal_step_advances_and_records_current_beam(self):
        for module in self.modules:
            with self.subTest(environment=module.__name__):
                env = self.make_env(module)
                pool = env.satellite_freq_state
                first_location = env.beam_info["beam_location"][0]
                second_location = env.beam_info["beam_location"][1]
                with patch.object(pool, "distance", wraps=pool.distance) as distance:
                    state, _, done, info = env.step([0, 20, 3, 10 * np.log10(5)])
                self.assertFalse(done)
                self.assertEqual(env.current_beam_state.beam_id, 2)
                self.assertEqual(info["frequency_utilization"], 3 / 800)
                distance.assert_called_once_with(*second_location)
                np.testing.assert_array_equal(pool.freq_pool_real[0, 20:23], [1, 1, 1])
                np.testing.assert_array_equal(pool.freq_pool_lat[0, 20:23], [first_location[0]] * 3)
                np.testing.assert_array_equal(pool.freq_pool_lon[0, 20:23], [first_location[1]] * 3)
                np.testing.assert_allclose(state[:3], [
                    (second_location[0] - 29) / 25,
                    (second_location[1] - 104) / 31,
                    env.beam_info["rate"][1].item() / 550e6,
                ], rtol=1e-7)

    def test_legacy_update_state_interface_keeps_allocation_and_distance(self):
        for module in self.modules:
            with self.subTest(environment=module.__name__):
                pool = module.SatelliteFreqState()
                with patch.object(pool, "distance", wraps=pool.distance) as distance:
                    pool.update_state(2, 40, 2, 30, 110, 31, 111)
                self.assertEqual(pool.freq_pool_real.sum(), 2)
                np.testing.assert_array_equal(pool.freq_pool_lat[2, 40:42], [30, 30])
                np.testing.assert_array_equal(pool.freq_pool_lon[2, 40:42], [110, 110])
                distance.assert_called_once_with(31, 111)
                self.assertTrue(np.all(pool.freq_pool_distance[2, 40:42] > 0))


if __name__ == "__main__":
    unittest.main()
