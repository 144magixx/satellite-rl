"""Variable beam count checks without running a training job."""

import unittest

import numpy as np

from satellite_rl.envs.satellite import BeamInfo
from satellite_rl.state_layouts import VARIABLE_BEAM_FEATURES
from satellite_rl.workbench.config import config_summary, default_config, validate_config
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
                self.assertAlmostEqual(float(state[-1]), 3000 / (3000 + 30 * count))
                self.assertEqual(env.action_mask()["frequency_slots"].shape, (100, 10))

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
