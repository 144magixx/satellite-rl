"""Joint slot legality and minimum resource reserves in real 100-beam scenes."""

import copy
import hashlib
import json
import unittest

import numpy as np

from satellite_rl.workbench.config import default_config
from satellite_rl.workbench.environment import WorkbenchEnv
from satellite_rl.workbench.checkpoints import config_hash
from satellite_rl.workbench.runner import _evaluation_episode


class ActionMaskEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = WorkbenchEnv(default_config())
        source.reset(seed=20260925)
        cls.scene = source.scene()

    def make_env(self, budget=3000):
        config = default_config()
        config["environment"].update(action_mask="joint_feasible", power_budget_w=budget)
        env = WorkbenchEnv(config)
        env.reset(scene=self.scene)
        return env

    def test_pair_mask_checks_entire_interval_and_boundary(self):
        env = self.make_env()
        env.raw_env.satellite_freq_state.freq_pool_real[0, 4] = 1
        mask = env.action_mask()
        self.assertTrue(mask["frequency_slots"][3, 0])   # [3] is free
        self.assertFalse(mask["frequency_slots"][3, 1])  # [3,4] conflicts
        self.assertFalse(mask["frequency_slots"][4].any())
        self.assertTrue(mask["frequency_slots"][5, 1])
        self.assertTrue(mask["frequency_slots"][99, 0])
        self.assertFalse(mask["frequency_slots"][99, 1:].any())
        with self.assertRaisesRegex(ValueError, "掩码"):
            env.step([3, 1, 0])
        self.assertEqual(env.raw_env.beam_idx, 0)
        env.step([3, 0, 0])

    def test_reserve_counts_use_current_group_and_true_budget(self):
        env = self.make_env(budget=500)
        # Group 0 has 13 beams left, therefore 12 free slots must be reserved.
        env.raw_env.satellite_freq_state.freq_pool_real[0, 15:] = 1
        mask = env.action_mask()
        self.assertTrue(mask["frequency_slots"][0, 2])
        self.assertFalse(mask["frequency_slots"][:, 3:].any())
        self.assertEqual(np.flatnonzero(mask["power"]).tolist(), [0])
        with self.assertRaisesRegex(ValueError, "掩码"):
            env.step([0, 0, 1])

    def test_no_feasible_set_is_explicit_error(self):
        env = self.make_env()
        env.raw_env.satellite_freq_state.freq_pool_real[0, :] = 1
        with self.assertRaisesRegex(RuntimeError, "无合法动作"):
            env.action_mask()

    def test_custom_small_power_grid_does_not_allow_budget_tolerance_excess(self):
        config = default_config()
        config["environment"].update(action_mask="joint_feasible", power_budget_w=1e-6,
                                     power_levels_w=[1e-8, 1.05e-8])
        env = WorkbenchEnv(config)
        env.reset(scene=self.scene)
        self.assertEqual(env.action_mask()["power"].tolist(), [True, False])
        env.step([0, 0, 0])
        self.assertEqual(env.action_mask()["power"].tolist(), [True, False])

    def test_full_scene_greedy_resource_consumption_preserves_all_beams(self):
        for budget in (500, 3000):
            with self.subTest(budget=budget):
                env = self.make_env(budget)
                for index in range(100):
                    mask = env.action_mask()
                    valid = np.argwhere(mask["frequency_slots"])
                    # Adversarially take the largest feasible slot/power allocation.
                    pair = max(valid.tolist(), key=lambda item: (item[1], -item[0]))
                    action = [*pair, int(np.flatnonzero(mask["power"])[-1])]
                    _, _, done, info = env.step(action)
                    self.assertEqual(info["actual_action"]["slots"], action[1] + 1)
                    self.assertEqual(info["actual_action"]["indices"], action)
                    self.assertEqual(info["out"], 0)
                    self.assertEqual(info["occupy"], 0)
                    self.assertGreater(info["actual_action"]["power_w"], 0)
                    self.assertEqual(done, index == 99)
                result = env.metrics()
                self.assertTrue(result["complete"])
                self.assertFalse(result["overload"])
                self.assertEqual(result["zero_slot_count"], 0)
                self.assertEqual(result["clipped_count"], 0)
                self.assertLessEqual(result["power_w"], budget)
                self.assertIsNone(env.action_mask())

    def test_mask_does_not_depend_on_observation_feature_order(self):
        config = default_config()
        config["environment"]["action_mask"] = "joint_feasible"
        config["state"]["features"] = ["beam"]
        env = WorkbenchEnv(config)
        self.assertEqual(env.reset(scene=self.scene).shape, (3,))
        self.assertEqual(env.action_mask()["frequency_slots"].sum(), 955)

    def test_mask_off_preserves_clipping(self):
        config = default_config()
        env = WorkbenchEnv(config)
        env.reset(scene=self.scene)
        self.assertIsNone(env.action_mask())
        env.raw_env.satellite_freq_state.freq_pool_real[0, 3] = 1
        _, _, _, info = env.step([3, 1, 0])
        self.assertEqual(info["actual_action"]["slots"], 0)

    def test_evaluation_passes_mask_to_policy(self):
        env = self.make_env()

        class MaskPolicy:
            last_action_info = {}
            calls = 0

            def act(self, state, deterministic=True, action_mask=None):
                self.calls += 1
                if action_mask is None:
                    raise AssertionError("mask missing in evaluation")
                f, l = np.argwhere(action_mask["frequency_slots"])[0]
                p = np.flatnonzero(action_mask["power"])[0]
                return int(f), int(l), int(p)

        policy = MaskPolicy()
        result = _evaluation_episode(env, policy, self.scene, "mask_test", 1, "deterministic")
        self.assertEqual(policy.calls, 100)
        self.assertEqual(result["metrics"]["zero_slot_count"], 0)
        self.assertTrue(all(row["action_mask"] == "joint_feasible" for row in result["trace"]))

    def test_legacy_hash_preserved_only_for_disabled_features(self):
        config = default_config()
        historical = copy.deepcopy(config)
        historical["environment"].pop("action_mask", None)
        historical["environment"].pop("profile", None)
        historical["environment"].pop("beam_count_max", None)
        historical["network"].pop("transformer_layout", None)
        historical["network"].pop("sac_policy_structure", None)
        historical["network"].pop("sac_critic_structure", None)
        # Newly optional inactive experiment defaults retain the old hash.
        historical["network"].pop("temporal_mode", None)
        historical["network"].pop("temporal_window", None)
        historical["network"].pop("graph_mode", None)
        historical["state"].pop("context_level", None)
        historical["evaluation"].pop("extra_steps", None)
        historical["training"].pop("scene_order", None)
        historical["training"].pop("scene_order_seed", None)
        historical["training"].pop("alpha_mode", None)
        historical["training"].pop("fixed_alpha", None)
        for key in ("beam_count_mode", "beam_count_min", "beam_count_max", "held_out_beam_counts"):
            historical["training"].pop(key, None)
        historical.pop("telemetry", None)
        expected = hashlib.sha256(json.dumps(historical, ensure_ascii=False, sort_keys=True,
                                            separators=(",", ":")).encode("utf-8")).hexdigest()
        self.assertEqual(config_hash(config), expected)
        changed = copy.deepcopy(config)
        changed["environment"]["action_mask"] = "joint_feasible"
        self.assertNotEqual(config_hash(changed), expected)
        changed = copy.deepcopy(config)
        changed["network"]["transformer_layout"] = "tsac_13_tokens"
        self.assertNotEqual(config_hash(changed), expected)
        changed = copy.deepcopy(config)
        changed["training"]["scene_order"] = "shuffle_each_cycle"
        self.assertNotEqual(config_hash(changed), expected)
        changed = copy.deepcopy(config)
        changed["state"]["context_level"] = "full"
        self.assertNotEqual(config_hash(changed), expected)
        masked = copy.deepcopy(config)
        masked["environment"]["action_mask"] = "joint_feasible"
        structure_hashes = {config_hash(masked)}
        for key, value in (("sac_policy_structure", "conditional"),
                           ("sac_critic_structure", "joint")):
            changed = copy.deepcopy(masked)
            changed["network"][key] = value
            structure_hashes.add(config_hash(changed))
        self.assertEqual(len(structure_hashes), 3)


if __name__ == "__main__":
    unittest.main()
