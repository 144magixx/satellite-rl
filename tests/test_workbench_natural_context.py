"""Natural observations contain exactly the visible values and train normally."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from satellite_rl.agents.workbench_algorithms import build_agent
from satellite_rl.agents.workbench_networks import MultiHeadNetwork, model_description
from satellite_rl.state_layouts import NATURAL_CONTEXT_LEVELS, natural_dim, natural_shapes, natural_token_count
from satellite_rl.workbench.checkpoints import export_policy, load_policy
from satellite_rl.workbench.config import catalog, config_summary, default_config, validate_config
from satellite_rl.workbench.environment import WorkbenchEnv


def make_config(level, legacy=False):
    config = default_config()
    config["network"].update(kind="transformer", transformer_layout="tsac_32_context" if legacy else "tsac_natural_context",
                             embedding_dim=128, num_heads=8, num_layers=2,
                             feedforward_dim=512, dropout=0)
    config["state"].update(context_level=level,
                           features=(["beam", "power", "interference", "occupancy",
                                      "global_context", "beam_context", "resource_context"]
                                     if legacy else list(natural_shapes(level))))
    config["environment"].update(profile="europe_v1")
    config["training"].update(scene_order="shuffle_each_cycle", batch_size=2, warmup_steps=0)
    return validate_config(config)


class NaturalContextTests(unittest.TestCase):
    def test_catalog_layout_and_validation(self):
        levels = catalog()["natural_context_levels"]
        self.assertEqual([row["id"] for row in levels], list(NATURAL_CONTEXT_LEVELS))
        for row in levels:
            level = row["id"]
            config = make_config(level)
            summary = config_summary(config)
            self.assertEqual(summary["state_dim"], row["state_dim"])
            self.assertEqual(summary["network_token_count"], row["token_count"])
            self.assertEqual(row["state_dim"], natural_dim(level))
            self.assertEqual(row["token_count"], natural_token_count(level))
            self.assertEqual([part["id"] for part in summary["state_layout"]], row["features"])
            self.assertEqual([part["shape"] for part in summary["state_layout"]],
                             list(natural_shapes(level).values()))
            missing = copy.deepcopy(config)
            missing["state"]["features"] = missing["state"]["features"][:-1]
            with self.assertRaises(ValueError):
                validate_config(missing)
            for section, key, value in (("network", "dropout", 0.1),
                                        ("reward", "preset", "fused"),
                                        ("training", "scene_order", "sequential")):
                changed = copy.deepcopy(config)
                changed[section][key] = value
                with self.assertRaisesRegex(ValueError, "自然状态对照固定"):
                    validate_config(changed)

    def test_visible_values_match_old_c_and_trajectory(self):
        for level in NATURAL_CONTEXT_LEVELS:
            with self.subTest(level=level):
                natural = WorkbenchEnv(make_config(level))
                old = WorkbenchEnv(make_config(level, legacy=True)) if level != "none" else None
                natural.reset(seed=103)
                if old is not None:
                    old.reset(scene=natural.scene())
                for turn in range(3):
                    state = natural._state()
                    self.assertEqual(state.shape, (natural_dim(level),))
                    if old is not None:
                        reference = old._state()
                        np.testing.assert_array_equal(state[:204], reference[:204])
                        offset = 204
                        global_width = natural_shapes(level)["natural_global_context"][0]
                        np.testing.assert_array_equal(state[offset:offset + global_width], reference[204:204 + global_width])
                        offset += global_width
                        if level in ("layout", "full"):
                            width = natural_shapes(level)["natural_beam_context"][-1]
                            expected = reference[214:1314].reshape(100, 11)[:, :width].reshape(-1)
                            np.testing.assert_array_equal(state[offset:offset + expected.size], expected)
                            offset += expected.size
                        if level in ("summary", "layout", "full"):
                            width = natural_shapes(level)["natural_resource_context"][-1]
                            expected = reference[1314:2946].reshape(8, 204)[:, :width].reshape(-1)
                            np.testing.assert_array_equal(state[offset:offset + expected.size], expected)
                    action = [turn * 2, 1, 1]
                    next_state, reward, done, info = natural.step(action)
                    self.assertEqual(next_state.shape, (natural_dim(level),))
                    if old is not None:
                        _, old_reward, old_done, old_info = old.step(action)
                        self.assertEqual(reward, old_reward)
                        self.assertEqual(done, old_done)
                        self.assertEqual(natural.raw_env.total_power, old.raw_env.total_power)
                        np.testing.assert_array_equal(natural.raw_env.satellite_freq_state.freq_pool_real,
                                                      old.raw_env.satellite_freq_state.freq_pool_real)
                    self.assertFalse(done)

    def test_network_shapes_initialization_and_real_update(self):
        base = make_config("none")
        torch.manual_seed(51001)
        original = MultiHeadNetwork(base, 204, [100, 10, 10])
        for level in NATURAL_CONTEXT_LEVELS:
            with self.subTest(level=level):
                config = make_config(level)
                torch.manual_seed(51001)
                network = MultiHeadNetwork(config, natural_dim(level), [100, 10, 10])
                self.assertEqual(network.encoder(torch.zeros(2, natural_dim(level))).shape, (2, 128))
                for key, value in original.state_dict().items():
                    self.assertTrue(torch.equal(value, network.state_dict()[key]), key)
                self.assertEqual(hasattr(network.encoder, "beam_embed"), level in ("layout", "full"))
                self.assertEqual(hasattr(network.encoder, "resource_embed"), level in ("summary", "layout", "full"))
                description = model_description(config, natural_dim(level), [100, 10, 10])
                token = next(layer for layer in description["branches"]["策略"]
                             if layer["type"] == "TokenAssembly")
                self.assertEqual(token["shape"], [natural_token_count(level), 128])
                with self.assertRaises(ValueError):
                    network(torch.zeros(1, natural_dim(level) + 1))
        config = make_config("summary")
        agent = build_agent(config, 246, [100, 10, 10], "cpu")
        before = agent.actor.encoder.global_embed.weight.detach().clone()
        for i in range(2):
            state = np.full(246, 0.01 * (i + 1), dtype=np.float32)
            agent.observe(state, (i, 1, 1), float(i), state + 0.01, False)
        metrics = agent.update()
        self.assertTrue(metrics)
        self.assertFalse(torch.equal(before, agent.actor.encoder.global_embed.weight))

    def test_full_terminal_and_availability_match_old_c(self):
        natural_config = make_config("full")
        old_config = make_config("full", legacy=True)
        natural_config["environment"]["action_mask"] = "joint_feasible"
        old_config["environment"]["action_mask"] = "joint_feasible"
        natural = WorkbenchEnv(validate_config(natural_config))
        natural.reset(seed=103)
        old = WorkbenchEnv(validate_config(old_config))
        old.reset(scene=natural.scene())
        for index in range(100):
            natural_mask = natural.action_mask()
            old_mask = old.action_mask()
            for key in natural_mask:
                np.testing.assert_array_equal(natural_mask[key], old_mask[key])
            frequency, slots = np.argwhere(natural_mask["frequency_slots"])[0]
            action = [int(frequency), int(slots), 0]
            state, reward, done, _ = natural.step(action)
            _, old_reward, old_done, _ = old.step(action)
            self.assertEqual((reward, done), (old_reward, old_done))
            self.assertEqual(done, index == 99)
        self.assertEqual(state.shape, (2946,))
        np.testing.assert_array_equal(state, old._state())
        self.assertEqual(natural.metrics(), old.metrics())

    def test_policy_roundtrip_and_old_layout_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            for level in ("none", "summary"):
                config = make_config(level)
                summary = config_summary(config)
                agent = build_agent(config, summary["state_dim"], summary["action_dims"], "cpu")
                path = Path(tmp) / level
                export_policy(path, agent, config, {"kind": "unit_test"})
                loaded = load_policy(path, config=config)
                self.assertEqual(loaded.metadata["state_dim"], natural_dim(level))
                self.assertEqual(loaded.metadata["network_token_count"], natural_token_count(level))
                self.assertGreater(loaded.metadata["parameter_count"], 0)
            old_config = make_config("compact", legacy=True)
            old_summary = config_summary(old_config)
            old_agent = build_agent(old_config, old_summary["state_dim"], old_summary["action_dims"], "cpu")
            path = Path(tmp) / "legacy"
            export_policy(path, old_agent, old_config, {"kind": "unit_test"})
            self.assertEqual(load_policy(path, config=old_config).metadata["variant"],
                             "workbench_sac_transformer_tsac_32_context_v1")


if __name__ == "__main__":
    unittest.main()
