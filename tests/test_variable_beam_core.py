"""Variable beam count checks without running a training job."""

import unittest

import numpy as np

from satellite_rl.envs.satellite import BeamInfo
from satellite_rl.state_layouts import (S2P18_VARIABLE_FEATURES,
                                        S2P18_VARIABLE_STATE_DIM,
                                        VARIABLE_BEAM_FEATURES)
from satellite_rl.workbench.config import (config_summary, default_config, validate_config,
                                           variable_beam_features)
from satellite_rl.workbench.environment import WorkbenchEnv, validate_scene


def variable_config():
    config = default_config()
    config["environment"].update(profile="europe_v1", beam_count_max=200,
                                  action_mask="joint_feasible")
    config["network"].update(kind="transformer", transformer_layout="tsac_14_variable_context",
                             embedding_dim=128, num_heads=8, num_layers=2,
                             feedforward_dim=512, dropout=0)
    config["state"]["features"] = list(VARIABLE_BEAM_FEATURES)
    return config


def s2p18_variable_config():
    config = variable_config()
    config["state"]["features"] = list(S2P18_VARIABLE_FEATURES)
    return config


class VariableBeamCoreTests(unittest.TestCase):
    def test_dynamic_scene_and_state(self):
        for count in (1, 10, 25, 49, 50, 100, 200):
            with self.subTest(count=count):
                np.random.seed(51001)
                source = BeamInfo(profile="europe_v1", count=count)
                scene = {key: np.array(value, copy=True) for key, value in source.beam_info.items()}
                scene["original_indices"] = np.array(source.shuffle_indices, copy=True)
                validate_scene(scene, "europe_v1")
                env = WorkbenchEnv(variable_config())
                state = env.reset(scene=scene)
                self.assertEqual(state.shape, (210,))
                self.assertEqual(env.raw_env.beam_num, count)
                self.assertAlmostEqual(float(state[-3]), count / 200)
                self.assertAlmostEqual(float(state[-2]), count / 200)
                # S2-P18 修正2：参考功率改为公平份额 budget/N；N=100 时与旧的写死
                # 30 W 逐位相同，其余束数按新口径。这里同时核对新旧两种口径。
                legacy_reference = 3000 / (3000 + 30 * count)
                corrected_reference = 3000 / (3000 + (3000 / count) * count)
                self.assertAlmostEqual(float(state[-1]), corrected_reference)
                if count == 100:
                    self.assertEqual(float(state[-1]), legacy_reference)
                else:
                    self.assertNotEqual(float(state[-1]), legacy_reference)
                self.assertEqual(env.action_mask()["frequency_slots"].shape, (100, 10))

    def test_s2p18_state_profile_adds_group_and_reservation_scalars(self):
        for count in (1, 7, 100, 200):
            with self.subTest(count=count):
                np.random.seed(51001)
                source = BeamInfo(profile="europe_v1", count=count)
                scene = {key: np.array(value, copy=True) for key, value in source.beam_info.items()}
                scene["original_indices"] = np.array(source.shuffle_indices, copy=True)
                config = s2p18_variable_config()
                env = WorkbenchEnv(config)
                state = env.reset(scene=scene)
                self.assertEqual(state.shape, (S2P18_VARIABLE_STATE_DIM,))
                self.assertEqual(state.shape, (213,))
                self.assertEqual(config_summary(config)["state_dim"], 213)
                # 追加的三项：组号/7、本组剩余预留义务、剩余合法槽上限。
                self.assertAlmostEqual(float(state[-3]), (0 % 8) / 7)
                self.assertAlmostEqual(float(state[-2]), len(range(0, count, 8)) / count)
                expected_slots = max(0, 100 - (len(range(0, count, 8)) - 1))
                self.assertAlmostEqual(float(state[-1]), expected_slots / 100)
                self.assertEqual(list(variable_beam_features("s2p18")),
                                 list(S2P18_VARIABLE_FEATURES))
                self.assertEqual(list(variable_beam_features()),
                                 list(VARIABLE_BEAM_FEATURES))

    def test_s2p18_reservation_scalars_match_the_action_mask(self):
        """观测里的预留义务与剩余槽上限必须和掩码实际使用的口径一致。"""
        np.random.seed(7)
        source = BeamInfo(profile="europe_v1", count=90)
        scene = {key: np.array(value, copy=True) for key, value in source.beam_info.items()}
        scene["original_indices"] = np.array(source.shuffle_indices, copy=True)
        env = WorkbenchEnv(s2p18_variable_config())
        env.reset(scene=scene)
        for turn in range(6):
            mask = env.action_mask()
            state = env._state()
            idx, count = env.raw_env.beam_idx, env.raw_env.beam_num
            remaining_group = len(range(idx, count, 8))
            occupied = np.asarray(env.raw_env.satellite_freq_state.freq_pool_real[idx % 8]) > 0
            max_slots = max(0, int(np.count_nonzero(~occupied)) - (remaining_group - 1))
            self.assertAlmostEqual(float(state[-2]), remaining_group / count)
            self.assertAlmostEqual(float(state[-1]), max_slots / 100)
            # 掩码里任何合法长度都不得超过观测给出的剩余合法槽上限。
            legal_lengths = np.nonzero(mask["frequency_slots"])[1] + 1
            self.assertTrue(len(legal_lengths))
            self.assertLessEqual(int(legal_lengths.max()), max(1, max_slots))
            freq, slot = int(np.nonzero(mask["frequency_slots"])[0][0]), int(
                np.nonzero(mask["frequency_slots"])[1][0])
            power = int(np.nonzero(mask["power"])[0][-1])
            env.step((freq, slot, power))

    def test_bad_counts_and_identity_are_rejected(self):
        np.random.seed(51001)
        source = BeamInfo(profile="europe_v1", count=10)
        scene = {key: np.array(value, copy=True) for key, value in source.beam_info.items()}
        scene["original_indices"] = np.array(source.shuffle_indices, copy=True)
        with self.assertRaises(ValueError):
            BeamInfo(profile="europe_v1", count=201)
        bad = dict(scene, original_indices=np.zeros(10, dtype=int))
        with self.assertRaises(ValueError):
            validate_scene(bad, "europe_v1")
        bad = dict(scene, rate=scene["rate"][:-1])
        with self.assertRaises(ValueError):
            validate_scene(bad, "europe_v1")
        with self.assertRaises(ValueError):
            validate_scene(scene, "europe_v1", max_beams=9)

    def test_no_business_has_no_si_or_sgm(self):
        scene = {"beam_location": np.zeros((0, 2)),
                 "beam_G_Tx_max": np.zeros((0, 1)),
                 "UE_G_Rx_max": np.zeros((0, 1)),
                 "distances": np.zeros(0), "rate": np.zeros((0, 1)),
                 "noise_temperature": np.zeros((0, 1)),
                 "original_indices": np.zeros(0, dtype=int)}
        env = WorkbenchEnv(variable_config())
        state = env.reset(scene=scene)
        self.assertEqual(state.shape, (210,))
        self.assertEqual(env.beam_results(), [])
        self.assertIsNone(env.metrics()["final_si_mean"])
        self.assertIsNone(env.metrics()["final_sgm_mean"])
        self.assertEqual(env.metrics()["termination_reason"], "no_business")
        with self.assertRaises(RuntimeError):
            env.step((0, 0, 0))

    def test_legacy_layout_is_still_207_and_new_layout_is_210(self):
        old = variable_config()
        old["network"]["transformer_layout"] = "tsac_14_context"
        old["state"]["features"] = list(VARIABLE_BEAM_FEATURES[:7])
        self.assertEqual(config_summary(old)["state_dim"], 207)
        self.assertEqual(config_summary(variable_config())["state_dim"], 210)
        self.assertEqual(config_summary(variable_config())["action_dims"], [100, 10, 10])

    def test_100_beam_first_207_fields_and_physics_match(self):
        np.random.seed(51001)
        source = BeamInfo(profile="europe_v1", count=100)
        scene = {key: np.array(value, copy=True) for key, value in source.beam_info.items()}
        scene["original_indices"] = np.array(source.shuffle_indices, copy=True)
        old_config = variable_config()
        old_config["network"]["transformer_layout"] = "tsac_14_context"
        old_config["state"]["features"] = list(VARIABLE_BEAM_FEATURES[:7])
        old, new = WorkbenchEnv(old_config), WorkbenchEnv(variable_config())
        old_state, new_state = old.reset(scene=scene), new.reset(scene=scene)
        np.testing.assert_array_equal(old_state, new_state[:207])
        for _ in range(4):
            old_state, _, old_done, _ = old.step((0, 0, 0))
            new_state, _, new_done, _ = new.step((0, 0, 0))
            self.assertEqual(old_done, new_done)
            np.testing.assert_array_equal(old_state, new_state[:207])
            self.assertEqual(old.beam_results(), new.beam_results())

    def test_scientific_training_rejects_held_out_counts_and_wrong_budget(self):
        config = variable_config()
        config["training"].update(beam_count_mode="mixed_four_buckets", scene_order="variable_beams",
                                  seed=51001, total_steps=400000, alpha_mode="fixed",
                                  checkpoint_interval=10000)
        config["evaluation"].update(validation_count=144, validation_seed=53001,
                                    interval_steps=20000, extra_steps=[0], action_mode="stochastic",
                                    test_count=0)
        validate_config(config)
        config["training"]["held_out_beam_counts"] = [60, 90, 140]
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
