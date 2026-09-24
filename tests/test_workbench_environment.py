import unittest
from types import SimpleNamespace

import numpy as np

from satellite_rl.envs import satellite as legacy
from satellite_rl.workbench.config import default_config
from satellite_rl.workbench.environment import WorkbenchEnv, validate_scene


class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = WorkbenchEnv(default_config())
        cls.scene.reset(seed=103)
        cls.scene = cls.scene.scene()

    def test_scene_copy_and_validation(self):
        copy = validate_scene(self.scene)
        copy["rate"][0] = 1
        self.assertNotEqual(copy["rate"][0, 0], self.scene["rate"][0, 0])
        with self.assertRaises(ValueError):
            validate_scene({k: v for k, v in self.scene.items() if k != "original_indices"})
        inconsistent = {k: v.copy() for k, v in self.scene.items()}
        inconsistent["distances"][0] += 100
        with self.assertRaises(ValueError):
            validate_scene(inconsistent)
        complex_scene = {k: v.copy() for k, v in self.scene.items()}
        complex_scene["rate"] = complex_scene["rate"].astype(complex) + 1j
        with self.assertRaises(ValueError):
            validate_scene(complex_scene)
        config = default_config()
        env = WorkbenchEnv(config)
        env.reset(scene=self.scene)
        self.assertTrue(np.array_equal(env.scene()["rate"], self.scene["rate"]))

    def test_legacy_equivalence_and_global_budget_restore(self):
        config = default_config()
        original_budget = legacy.power_threshold
        wrapped = WorkbenchEnv(config)
        state = wrapped.reset(scene=self.scene)
        beam = SimpleNamespace(beam_info={key: val.copy() for key, val in self.scene.items() if key != "original_indices"})
        raw = legacy.Env(legacy.SatelliteFreqState(), beam, legacy.CurrentBeamState(beam.beam_info),
                         reward_variant="legacy_repaired")
        raw_state = raw.reset(beam)
        self.assertTrue(np.allclose(state, raw_state))
        for indices in ([0, 0, 0], [99, 9, 5], [0, 0, 0]):
            group = raw.beam_idx % 8
            request = [group, indices[0], indices[1] + 1, 10 * np.log10((indices[2] + 1) * 5)]
            expected_state, expected_reward, expected_done, expected_info = raw.step(request.copy())
            state, reward, done, info = wrapped.step(indices)
            self.assertTrue(np.allclose(state, expected_state))
            self.assertAlmostEqual(reward, expected_reward)
            self.assertEqual(done, expected_done)
            self.assertEqual(info["actual_action"]["slots"], expected_info["success_slots"])
            self.assertAlmostEqual(sum(t["weighted"] for t in info["reward_terms"].values()), reward)
        self.assertEqual(legacy.power_threshold, original_budget)

    def test_feature_visibility_and_no_future_result_leakage(self):
        config = default_config()
        config["state"] = {"features": ["beam", "served_shortfall", "progress", "beta"],
                           "demand_visibility": "known"}
        env = WorkbenchEnv(config)
        first = env.reset(scene=self.scene)
        changed = {k: v.copy() for k, v in self.scene.items()}
        changed["rate"][20:] *= 2
        second = env.reset(scene=changed)
        self.assertTrue(np.array_equal(first[:5], second[:5]))
        self.assertNotEqual(first[-1], second[-1])
        config["state"] = {"features": ["beam", "served_shortfall", "progress"],
                           "demand_visibility": "online"}
        env = WorkbenchEnv(config)
        self.assertTrue(np.array_equal(env.reset(scene=self.scene), env.reset(scene=changed)))

    def test_feature_permutation_is_real(self):
        config = default_config()
        config["state"]["features"] = ["power", "beam", "occupancy", "interference"]
        permuted = WorkbenchEnv(config)
        state = permuted.reset(scene=self.scene)
        normal = WorkbenchEnv(default_config())
        ordinary = normal.reset(scene=self.scene)
        for _ in range(2):
            expected = np.concatenate((ordinary[3:4], ordinary[:3], ordinary[104:], ordinary[4:104]))
            self.assertEqual(state.shape, (204,))
            self.assertTrue(np.array_equal(state, expected))
            state = permuted.step([0, 0, 0])[0]
            ordinary = normal.step([0, 0, 0])[0]

    def test_budget_scoped_across_alternating_environments(self):
        baseline = legacy.power_threshold
        low = default_config()
        low["environment"]["power_budget_w"] = 11
        high = default_config()
        high["environment"]["power_budget_w"] = 7000
        a, b = WorkbenchEnv(low), WorkbenchEnv(high)
        a.reset(scene=self.scene)
        b.reset(scene=self.scene)
        self.assertEqual(legacy.power_threshold, baseline)
        self.assertAlmostEqual(a.raw_env.alpha, 0.11)
        self.assertAlmostEqual(b.raw_env.alpha, 70)
        a.step([0, 0, 0])
        _, _, _, info_b = b.step([0, 0, 0])
        self.assertAlmostEqual(info_b["actual_action"]["power_w"], 5)
        self.assertAlmostEqual(b.metrics()["power_budget_w"], 7000)
        self.assertEqual(legacy.power_threshold, baseline)

    def test_custom_reward_sum_terminal_all_beams(self):
        config = default_config()
        config["reward"] = {"preset": "custom", "terms": [
            {"id": "si_delta", "weight": 2, "scale": 1, "timing": "step"},
            {"id": "terminal_si", "weight": 3, "scale": 1, "timing": "terminal"},
            {"id": "power_cost", "weight": 1, "scale": 1, "timing": "step"},
        ]}
        env = WorkbenchEnv(config)
        env.reset(scene=self.scene)
        state, reward, done, info = env.step([0, 0, 0])
        self.assertFalse(done)
        self.assertEqual(info["reward_terms"]["terminal_si"]["raw"], 0)
        self.assertAlmostEqual(reward, sum(v["weighted"] for v in info["reward_terms"].values()))
        all_beams = env.beam_results()
        self.assertEqual(len(all_beams), 100)
        self.assertEqual(all_beams[99]["rate_bps"], 0)
        self.assertAlmostEqual(env.metrics()["final_si_mean"], all_beams[0]["si"] / 100)

    def test_feasible_projection_no_service(self):
        config = default_config()
        config["environment"]["action_mode"] = "feasible_projection"
        config["environment"]["power_budget_w"] = 5
        config["reward"] = {"preset": "custom", "terms": [
            {"id": "zero_slot_penalty", "weight": 1, "scale": 1, "timing": "step"}]}
        env = WorkbenchEnv(config)
        env.reset(scene=self.scene)
        env.step([0, 0, 0])
        _, reward, _, info = env.step([0, 0, 0])
        self.assertEqual(info["actual_action"]["slots"], 0)
        self.assertEqual(info["actual_action"]["power_w"], 0)
        self.assertEqual(reward, -1)
        self.assertAlmostEqual(env.metrics()["power_w"], 5)
        self.assertEqual(legacy.power_threshold, 3000)

    def test_terminal_reward_uses_all_targets_after_early_stop(self):
        config = default_config()
        config["environment"]["power_budget_w"] = 4
        config["reward"] = {"preset": "custom", "terms": [
            {"id": "terminal_si", "weight": 1, "scale": 1, "timing": "terminal"},
            {"id": "incomplete_penalty", "weight": 1, "scale": 1, "timing": "terminal"},
        ]}
        env = WorkbenchEnv(config)
        env.reset(scene=self.scene)
        _, reward, done, info = env.step([0, 0, 0])
        self.assertTrue(done)
        metrics = env.metrics()
        self.assertEqual(metrics["processed_count"], 1)
        self.assertEqual(metrics["zero_capacity_count"], 99)
        self.assertAlmostEqual(info["reward_terms"]["terminal_si"]["raw"], metrics["final_si_mean"])
        self.assertAlmostEqual(info["reward_terms"]["incomplete_penalty"]["raw"], -0.99)
        self.assertAlmostEqual(reward, metrics["final_si_mean"] - 0.99)

    def test_legacy_original_zero_slot_power_accounting(self):
        config = default_config()
        config["reward"]["preset"] = "legacy_original"
        original = WorkbenchEnv(config)
        original.reset(scene=self.scene)
        beam = SimpleNamespace(beam_info={key: val.copy() for key, val in self.scene.items() if key != "original_indices"})
        raw = legacy.Env(legacy.SatelliteFreqState(), beam, legacy.CurrentBeamState(beam.beam_info),
                         reward_variant="legacy_original")
        raw.reset(beam)
        config["reward"]["preset"] = "legacy_repaired"
        repaired = WorkbenchEnv(config)
        repaired.reset(scene=self.scene)
        for _ in range(9):
            expected_state, expected_reward, _, _ = raw.step([raw.beam_idx % 8, 0, 1, 10 * np.log10(5)])
            state, reward, _, info_a = original.step([0, 0, 0])
            _, _, _, info_b = repaired.step([0, 0, 0])
            self.assertTrue(np.allclose(state, expected_state))
            self.assertAlmostEqual(reward, expected_reward)
        self.assertEqual(info_a["actual_action"]["slots"], 0)
        self.assertEqual(info_b["actual_action"]["slots"], 0)
        self.assertAlmostEqual(original.metrics()["power_w"] - repaired.metrics()["power_w"], 5)
        self.assertAlmostEqual(original.metrics()["radiated_power_w"], repaired.metrics()["radiated_power_w"])

    def test_fused_preset_matches_legacy_physics(self):
        config = default_config()
        config["reward"]["preset"] = "fused"
        wrapped = WorkbenchEnv(config)
        wrapped.reset(scene=self.scene)
        beam = SimpleNamespace(beam_info={key: val.copy() for key, val in self.scene.items() if key != "original_indices"})
        raw = legacy.Env(legacy.SatelliteFreqState(), beam, legacy.CurrentBeamState(beam.beam_info),
                         reward_variant="fused")
        raw.reset(beam)
        for indices in ([0, 0, 0], [1, 1, 1], [0, 0, 0], [99, 9, 9]):
            previous_basic = raw.total_basic_service
            previous_sgm = raw.total_sgm
            action = [raw.beam_idx % 8, indices[0], indices[1] + 1,
                      10 * np.log10((indices[2] + 1) * 5)]
            expected_state, expected_reward, done, _ = raw.step(action)
            state, reward, actual_done, info = wrapped.step(indices)
            self.assertTrue(np.allclose(state, expected_state))
            self.assertAlmostEqual(reward, expected_reward)
            terms = info["reward_terms"]
            self.assertAlmostEqual(terms["basic_delta"]["raw"], raw.total_basic_service - previous_basic)
            self.assertAlmostEqual(terms["sgm_delta"]["raw"], raw.total_sgm - previous_sgm)
            self.assertAlmostEqual(terms["basic_delta"]["weighted"], 2 * terms["basic_delta"]["raw"])
            self.assertAlmostEqual(terms["sgm_delta"]["weighted"], 0.2 * terms["sgm_delta"]["raw"])
            self.assertAlmostEqual(terms["basic_delta"]["scaled"], terms["basic_delta"]["raw"])
            self.assertAlmostEqual(terms["sgm_delta"]["scaled"], terms["sgm_delta"]["raw"])
            self.assertAlmostEqual(sum(t["weighted"] for t in terms.values()), reward)
            self.assertEqual(done, actual_done)

    def test_final_result_is_recomputed_after_later_interference(self):
        config = default_config()
        config["reward"] = {"preset": "custom", "terms": [
            {"id": "terminal_si", "weight": 1, "scale": 1, "timing": "terminal"},
            {"id": "terminal_sgm", "weight": 1, "scale": 1, "timing": "terminal"},
        ]}
        env = WorkbenchEnv(config)
        env.reset(scene=self.scene)
        for step in range(100):
            _, reward, done, info = env.step([step % 100, 0, 0])
            if done:
                break
        beams = env.beam_results()
        metrics = env.metrics()
        self.assertEqual(len(beams), 100)
        self.assertAlmostEqual(metrics["final_si_mean"], sum(b["si"] for b in beams) / 100)
        self.assertAlmostEqual(metrics["final_sgm_mean"], sum(b["sgm"] for b in beams) / 100)
        self.assertAlmostEqual(reward, sum(x["weighted"] for x in info["reward_terms"].values()))
        self.assertAlmostEqual(info["reward_terms"]["terminal_si"]["raw"], metrics["final_si_mean"])

    def test_reward_signs_scales_and_terminal_gate(self):
        ids = ("si_delta", "basic_delta", "sgm_delta", "current_si", "power_cost",
               "slot_cost", "out_penalty", "occupy_penalty", "zero_slot_penalty", "terminal_si")
        config = default_config()
        config["reward"] = {"preset": "custom", "terms": [
            dict(id=term, weight=2, scale=4, timing="terminal" if term.startswith("terminal_") else "step")
            for term in ids]}
        env = WorkbenchEnv(config)
        env.reset(scene=self.scene)
        _, reward, _, info = env.step([99, 9, 0])
        terms = info["reward_terms"]
        self.assertEqual(set(terms), set(ids))
        for item in terms.values():
            self.assertAlmostEqual(item["scaled"], item["raw"] / 4)
            self.assertAlmostEqual(item["weighted"], item["raw"] / 2)
        self.assertAlmostEqual(reward, sum(x["weighted"] for x in terms.values()))
        self.assertEqual(terms["terminal_si"]["raw"], 0)
        self.assertLess(terms["power_cost"]["raw"], 0)
        self.assertLess(terms["slot_cost"]["raw"], 0)
        self.assertLess(terms["out_penalty"]["raw"], 0)
        self.assertEqual(terms["zero_slot_penalty"]["raw"], 0)

    def test_projection_logs_request_and_bounds_resource(self):
        config = default_config()
        config["environment"]["action_mode"] = "feasible_projection"
        config["environment"]["power_budget_w"] = 12
        env = WorkbenchEnv(config)
        env.reset(scene=self.scene)
        _, _, done, info = env.step([99, 9, 9])
        self.assertFalse(done)
        self.assertEqual(info["requested_action"]["power_w"], 50)
        self.assertEqual(info["requested_action"]["slots"], 10)
        self.assertLessEqual(info["actual_action"]["power_w"], 10)
        self.assertEqual(info["actual_action"]["slots"], 1)
        self.assertTrue(info["projection_applied"])
        self.assertLessEqual(env.metrics()["power_w"], 12 + 1e-9)
        self.assertEqual(env.metrics()["clipped_count"], 1)
        self.assertEqual(env.metrics()["allocated_slots"], 1)
        self.assertAlmostEqual(env.metrics()["spectral_efficiency"],
                               env.metrics()["throughput_bps"] / 2.5e7)

    def test_no_feasible_projection_finishes_without_power_or_slot_leak(self):
        config = default_config()
        config["environment"]["action_mode"] = "feasible_projection"
        config["environment"]["power_budget_w"] = 1
        env = WorkbenchEnv(config)
        env.reset(scene=self.scene)
        for step in range(100):
            state, reward, done, info = env.step([0, 0, 0])
            self.assertTrue(np.isfinite(state).all())
            self.assertEqual(info["actual_action"]["slots"], 0)
            self.assertEqual(done, step == 99)
        metrics = env.metrics()
        self.assertTrue(metrics["complete"])
        self.assertFalse(metrics["overload"])
        self.assertEqual(metrics["power_w"], 0)
        self.assertEqual(metrics["allocated_slots"], 0)
        self.assertEqual(metrics["zero_slot_rate"], 1)
        self.assertEqual(metrics["clipped_count"], 100)
        with self.assertRaises(RuntimeError):
            env.step([0, 0, 0])

    def test_jain_reward_uses_all_beams_including_unserved(self):
        config = default_config()
        config["reward"] = {"preset": "custom", "terms": [
            {"id": "jain_delta", "weight": 1, "scale": 1, "timing": "step"}]}
        env = WorkbenchEnv(config)
        env.reset(scene=self.scene)
        _, reward, _, info = env.step([0, 0, 0])
        self.assertAlmostEqual(reward, 0.01)
        self.assertAlmostEqual(info["reward_terms"]["jain_delta"]["raw"], env.metrics()["jain"])


if __name__ == "__main__":
    unittest.main()
