"""Four information levels share a 32-token model and causal observations."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from satellite_rl.agents.workbench_algorithms import build_agent
from satellite_rl.agents.workbench_networks import MultiHeadNetwork, model_description
from satellite_rl.workbench.checkpoints import export_policy, load_policy
from satellite_rl.workbench.config import config_summary, default_config, validate_config
from satellite_rl.workbench.environment import WorkbenchEnv


LEVELS = ("compact", "summary", "layout", "full")


def context_config(level):
    config = default_config()
    config["network"].update(kind="transformer", transformer_layout="tsac_32_context",
                             embedding_dim=128, num_heads=8, num_layers=2,
                             feedforward_dim=512, activation="relu", dropout=0)
    config["state"].update(features=["beam", "power", "interference", "occupancy",
                                     "global_context", "beam_context", "resource_context"],
                           demand_visibility="known", context_level=level)
    config["environment"].update(profile="europe_v1", action_mask="joint_feasible")
    return validate_config(config)


def legal_action(environment, power_index=0):
    mask = environment.action_mask()
    frequency, slots = np.argwhere(mask["frequency_slots"])[0]
    assert mask["power"][power_index]
    return [int(frequency), int(slots), power_index]


class Tsac32ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = WorkbenchEnv(context_config("full"))
        env.reset(seed=103)
        cls.scene = env.scene()

    def test_dimensions_trace_and_identical_initial_weights(self):
        networks = []
        descriptions = []
        for level in LEVELS:
            config = context_config(level)
            summary = config_summary(config)
            self.assertEqual((summary["state_dim"], summary["network_token_count"]), (2946, 32))
            self.assertEqual([(part["id"], part["offset"]) for part in summary["state_layout"][-3:]],
                             [("global_context", 204), ("beam_context", 214), ("resource_context", 1314)])
            torch.manual_seed(51001)
            networks.append(MultiHeadNetwork(config, 2946, summary["action_dims"]).eval())
            descriptions.append(model_description(config, 2946, summary["action_dims"]))
        reference = networks[0].state_dict()
        for network in networks[1:]:
            self.assertEqual(set(network.state_dict()), set(reference))
            for key, value in reference.items():
                self.assertTrue(torch.equal(value, network.state_dict()[key]), key)
        self.assertEqual(len({item["parameter_count"] for item in descriptions}), 1)
        self.assertEqual(len({item["variant"] for item in descriptions}), 1)
        self.assertEqual(descriptions[0]["variant"], "workbench_sac_transformer_tsac_32_context_v1")
        attention = next(layer for layer in descriptions[0]["branches"]["策略"]
                         if layer["type"] == "MultiheadAttention")
        self.assertEqual(attention["input_shape"], [32, 128])
        with self.assertRaisesRegex(ValueError, "2946"):
            MultiHeadNetwork(context_config("full"), 207, [100, 10, 10])

    def test_level_zeroing_and_same_physics_rewards(self):
        environments = {level: WorkbenchEnv(context_config(level)) for level in LEVELS}
        states = {level: env.reset(scene=self.scene) for level, env in environments.items()}
        for step in range(3):
            reference = states["full"]
            for level, state in states.items():
                np.testing.assert_array_equal(state[:204], reference[:204])
                np.testing.assert_array_equal(state[204:207], reference[204:207])
                if level == "compact":
                    self.assertTrue(np.all(state[207:] == 0))
                elif level == "summary":
                    self.assertTrue(np.all(state[214:1314] == 0))
                    self.assertTrue(np.all(state[1314:].reshape(8, 204)[:, 4:] == 0))
                elif level == "layout":
                    np.testing.assert_array_equal(state[214:1314].reshape(100, 11)[:, :5],
                                                  reference[214:1314].reshape(100, 11)[:, :5])
                    self.assertTrue(np.all(state[214:1314].reshape(100, 11)[:, 5:] == 0))
                    self.assertTrue(np.all(state[1314:].reshape(8, 204)[:, 4:] == 0))
            action = legal_action(environments["full"])
            outcomes = {level: env.step(action) for level, env in environments.items()}
            rewards = [outcome[1] for outcome in outcomes.values()]
            self.assertTrue(all(reward == rewards[0] for reward in rewards))
            self.assertEqual(len({outcome[2] for outcome in outcomes.values()}), 1)
            for level in LEVELS:
                self.assertEqual(environments[level].metrics(), environments["full"].metrics())
            states = {level: outcome[0] for level, outcome in outcomes.items()}

    def test_historical_actions_current_service_and_polarization(self):
        environment = WorkbenchEnv(context_config("full"))
        state = environment.reset(scene=self.scene)
        for power in (0, 1):
            action = legal_action(environment, power)
            state, _, done, _ = environment.step(action)
            self.assertFalse(done)
        self.assertEqual(environment.raw_env.beam_idx, 2)
        rows = state[214:1314].reshape(100, 11)
        results = environment.beam_results()
        for index in range(2):
            actual = results[index]
            self.assertEqual(rows[index, 5], 1)
            self.assertAlmostEqual(float(rows[index, 6]), actual["freq"] / 99, places=6)
            self.assertAlmostEqual(float(rows[index, 7]), actual["slots"] / 10, places=6)
            self.assertAlmostEqual(float(rows[index, 8]), actual["power_w"] / 50, places=6)
            self.assertAlmostEqual(float(rows[index, 9]), actual["rate_bps"] / 6e8, places=6)
            self.assertAlmostEqual(float(rows[index, 10]), actual["si"], places=6)
        self.assertTrue(np.all(rows[2:, 5:] == 0))
        groups = state[1314:].reshape(8, 204)
        for polarization in (0, 1):
            raw = environment.raw_env._calc_group_slot_interference(polarization)
            expected = np.clip((10 * np.log10(raw + 1e-30) + 300) / 200, 0, 1)
            for group in range(polarization, 8, 2):
                np.testing.assert_allclose(groups[group, 104:204], expected, rtol=0, atol=3e-8)
                np.testing.assert_array_equal(groups[group, 4:104],
                                              environment.raw_env.satellite_freq_state.freq_pool_real[group])
        np.testing.assert_array_equal(groups[2, 104:204], state[4:104])

    def test_future_demand_and_historical_action_visibility(self):
        changed = {key: value.copy() for key, value in self.scene.items()}
        changed["rate"][1:] *= 0.5
        for level in LEVELS:
            env = WorkbenchEnv(context_config(level))
            original = env.reset(scene=self.scene)
            altered = env.reset(scene=changed)
            np.testing.assert_array_equal(original[:204], altered[:204])
            self.assertNotEqual(original[204], altered[204])
            self.assertNotEqual(original[206], altered[206])
            if level in ("compact", "summary"):
                np.testing.assert_array_equal(original[214:1314], altered[214:1314])
            else:
                self.assertFalse(np.array_equal(original[214:1314], altered[214:1314]))

        for level in LEVELS:
            env = WorkbenchEnv(context_config(level))
            env.reset(scene=self.scene)
            low, _, _, _ = env.step(legal_action(env, 0))
            env.reset(scene=self.scene)
            high, _, _, _ = env.step(legal_action(env, 1))
            if level != "full":
                np.testing.assert_array_equal(low[214:1314].reshape(100, 11)[:, 5:],
                                              high[214:1314].reshape(100, 11)[:, 5:])
            else:
                self.assertNotEqual(low[214 + 8], high[214 + 8])

    def test_terminal_state_has_no_future_beam_interference(self):
        environment = WorkbenchEnv(context_config("full"))
        state = environment.reset(scene=self.scene)
        for index in range(100):
            state, _, done, _ = environment.step(legal_action(environment))
            self.assertEqual(done, index == 99)
        self.assertEqual(state.shape, (2946,))
        self.assertEqual(state[205], 1)
        self.assertEqual(state[206], 0)
        self.assertTrue(np.all(state[[208, 209, 213]] == 0))
        rows = state[214:1314].reshape(100, 11)
        self.assertTrue(np.all(rows[:, 5] == 1))
        np.testing.assert_allclose(rows[:, 10], [row["si"] for row in environment.beam_results()],
                                   rtol=0, atol=3e-8)
        self.assertTrue(np.all(state[1314:].reshape(8, 204)[:, 104:] == 0))

    def test_policy_package_binds_information_level_and_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            for level in LEVELS:
                with self.subTest(level=level):
                    config = context_config(level)
                    summary = config_summary(config)
                    agent = build_agent(config, 2946, summary["action_dims"], "cpu")
                    path = Path(tmp) / level
                    export_policy(path, agent, config, {"kind": "unit_test"})
                    loaded = load_policy(path, config=config)
                    self.assertEqual(loaded.config["state"]["context_level"], level)
                    with self.assertRaisesRegex(ValueError, "config differs"):
                        load_policy(path, config=context_config("full" if level != "full" else "compact"))
                    metadata_path = path / "model.json"
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    metadata["variant"] = "workbench_sac_transformer_wrong_v1"
                    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "variant"):
                        load_policy(path)


if __name__ == "__main__":
    unittest.main()
