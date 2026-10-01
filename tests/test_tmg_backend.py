"""Fixed temperature, temporal causality and full artifact recovery."""

import copy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from satellite_rl.agents.workbench_algorithms import build_agent
from satellite_rl.agents.workbench_networks import model_description
from satellite_rl.state_layouts import natural_shapes
from satellite_rl.workbench.checkpoints import (config_hash, export_policy, load_checkpoint,
                                                 load_policy, save_checkpoint)
from satellite_rl.workbench.config import default_config, validate_config
from satellite_rl.workbench.environment import WorkbenchEnv
from satellite_rl.workbench.learning_snapshots import (TEMPORAL_FORMAT,
                                                        save_learning_snapshot)
from satellite_rl.envs.regions import profile_metadata
from satellite_rl.workbench.runner import _evaluation_episode
from tests.test_tmg_sequence import config as temporal_config


def fixed_config(critic):
    value = default_config()
    value["network"].update(kind="transformer", transformer_layout="tsac_natural_context",
                            sac_critic_structure=critic, embedding_dim=16,
                            num_heads=2, num_layers=2, feedforward_dim=32, dropout=0)
    value["state"].update(features=list(natural_shapes("compact")),
                          context_level="compact", demand_visibility="known")
    value["environment"].update(profile="europe_v1", action_mask="joint_feasible")
    value["reward"]["preset"] = "legacy_repaired"
    value["training"].update(scene_order="shuffle_each_cycle", batch_size=2,
                              warmup_steps=0, alpha_mode="fixed", fixed_alpha=.01,
                              initial_alpha=.01, target_entropy=6.9)
    return validate_config(value)


class FixedAlphaTest(unittest.TestCase):
    def test_fixed_temperature_both_critics_and_recovery(self):
        for critic in ("additive", "joint"):
            with self.subTest(critic=critic):
                torch.manual_seed(61)
                cfg = fixed_config(critic)
                agent = build_agent(cfg, 207, (100, 10, 10), "cpu")
                env = WorkbenchEnv(cfg)
                state = env.reset(seed=901)
                for step in range(2):
                    mask = env.action_mask()
                    action = agent.act(state, action_mask=mask)
                    next_state, reward, done, _ = env.step(action)
                    agent.observe(state, action, reward, next_state, done,
                                  mask, env.action_mask())
                    state = next_state
                before = [p.detach().clone() for p in agent.actor.parameters()]
                result = agent.update()
                self.assertEqual(agent.update_step, 1)
                self.assertAlmostEqual(result["alpha"], .01, places=7)
                self.assertEqual(result["alpha_loss"], 0.0)
                self.assertEqual(result["alpha_update_applied"], 0)
                self.assertAlmostEqual(float(agent.log_alpha.exp()), .01, places=7)
                self.assertFalse(agent.log_alpha.requires_grad)
                self.assertTrue(all(math.isfinite(float(value)) for value in result.values()))
                self.assertTrue(any(not torch.equal(x, y) for x, y in
                                    zip(before, agent.actor.parameters())))
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "checkpoint.pt"
                    save_checkpoint(checkpoint, agent, {"env_step": 2}, cfg, {"boundary": "next_episode"})
                    restored = build_agent(cfg, 207, (100, 10, 10), "cpu")
                    progress, _ = load_checkpoint(checkpoint, restored, cfg)
                    self.assertEqual(progress["env_step"], 2)
                    self.assertEqual(restored.update_step, 1)
                    self.assertAlmostEqual(float(restored.log_alpha.exp()), .01, places=7)
                    self.assertTrue(all(torch.equal(x, y) for x, y in
                                        zip(agent.actor.parameters(), restored.actor.parameters())))

    def test_default_adaptive_hash_and_saved_mode_guard(self):
        cfg = fixed_config("additive")
        adaptive = copy.deepcopy(cfg)
        adaptive["training"]["alpha_mode"] = "adaptive"
        without_defaults = copy.deepcopy(adaptive)
        del without_defaults["training"]["alpha_mode"]
        del without_defaults["training"]["fixed_alpha"]
        del without_defaults["network"]["temporal_mode"]
        del without_defaults["network"]["temporal_window"]
        del without_defaults["network"]["graph_mode"]
        self.assertEqual(config_hash(adaptive), config_hash(without_defaults))
        self.assertNotEqual(config_hash(cfg), config_hash(adaptive))
        agent = build_agent(cfg, 207, (100, 10, 10))
        with self.assertRaisesRegex(ValueError, "alpha mode differs"):
            build_agent(adaptive, 207, (100, 10, 10)).load_state_dict(agent.state_dict())


class TemporalArtifactTest(unittest.TestCase):
    def test_terminal_targets_do_not_bootstrap_without_next_mask(self):
        for mode in ("gru", "causal_transformer"):
            with self.subTest(mode=mode):
                cfg = temporal_config(mode)
                agent = build_agent(cfg, 211, (100, 10, 10))
                mask = {"frequency_slots": np.zeros((100, 10), dtype=bool),
                        "power": np.zeros(10, dtype=bool)}
                mask["frequency_slots"][0, 0] = True
                mask["power"][0] = True
                for reward in (1., 3.):
                    state = np.zeros(211, dtype=np.float32)
                    following = state.copy()
                    following[-4:] = (0, .1, .1, 1)
                    agent.observe(state, (0, 0, 0), reward, following, True,
                                  mask, None, {"freq": 0, "slots": 1, "power_w": 5})
                result = agent.update()
                self.assertAlmostEqual(result["target_mean"], 2., places=5)
                self.assertEqual(result["effective_td_targets"], 2)

    def test_padding_causality_policy_and_checkpoint(self):
        for mode in ("gru", "causal_transformer"):
            with self.subTest(mode=mode):
                torch.manual_seed(97)
                cfg = temporal_config(mode)
                agent = build_agent(cfg, 211, (100, 10, 10), "cpu")
                sequence = torch.randn(1, 4, 211)
                lengths = torch.tensor([2])
                changed_future = sequence.clone()
                changed_future[:, 2:] += 100
                with torch.no_grad():
                    before = agent.actor((sequence, lengths))
                    after = agent.actor((changed_future, lengths))
                self.assertTrue(all(torch.allclose(x, y, atol=1e-6) for x, y in zip(before, after)))
                self.assertEqual(model_description(cfg, 211, (100, 10, 10))["parameter_count"],
                                 sum(p.numel() for network in
                                     (agent.actor, agent.critic1, agent.critic2, agent.target1, agent.target2)
                                     for p in network.parameters()) + 1)
                branch = model_description(cfg, 211, (100, 10, 10))["branches"]["策略"]
                spatial = next(row for row in branch if row["label"].endswith("encoder.spatial.tokens"))
                self.assertEqual(spatial["shape"], [15, 16])
                if mode == "gru":
                    temporal = next(row for row in branch if row["label"].endswith("encoder.temporal_gru"))
                    self.assertEqual(temporal["input_shape"], [16, 16])
                    self.assertEqual(agent.actor.encoder.memory.num_layers, 2)
                else:
                    temporal = next(row for row in branch if "encoder.memory.layers.0.self_attn" in row["label"])
                    self.assertEqual(temporal["shape"], [16, 16])
                with tempfile.TemporaryDirectory() as directory:
                    directory = Path(directory)
                    policy = directory / "policy"
                    export_policy(policy, agent, cfg, {"env_step": 0})
                    loaded = load_policy(policy, cfg)
                    loaded.reset_episode()
                    self.assertEqual(len(loaded.act(np.zeros(211, dtype=np.float32),
                                                    action_mask={"frequency_slots": np.ones((100, 10), dtype=bool),
                                                                 "power": np.ones(10, dtype=bool)})), 3)
                    checkpoint = directory / "checkpoint.pt"
                    save_checkpoint(checkpoint, agent, {"env_step": 0}, cfg,
                                    {"boundary": "next_episode"})
                    restored = build_agent(cfg, 211, (100, 10, 10))
                    load_checkpoint(checkpoint, restored, cfg)
                    self.assertEqual(restored.temporal_mode, mode)

    def test_snapshot_carries_executed_action_and_no_training_rng_consumption(self):
        cfg = temporal_config("gru")
        agent = build_agent(cfg, 211, (100, 10, 10))
        env = WorkbenchEnv(cfg)
        state = env.reset(seed=902)
        for _ in range(2):
            mask = env.action_mask()
            action = agent.act(state, action_mask=mask)
            next_state, reward, done, info = env.step(action)
            agent.observe(state, action, reward, next_state, done, mask,
                          env.action_mask(), info["actual_action"])
            state = next_state
        before_torch = torch.get_rng_state().clone()
        before_replay_rng = agent.replay.rng.getstate()
        provenance = {"environment_profile": profile_metadata("europe_v1"),
                      "git_commit": "test", "source_root": "test",
                      "source_manifest_sha256": "test"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(save_learning_snapshot(directory, agent, cfg, provenance, 2))
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["format"], TEMPORAL_FORMAT)
            self.assertEqual(payload["temporal_window"], 16)
            self.assertEqual(len(payload["replay_sample"]["transitions"]), 2)
            for row in payload["replay_sample"]["transitions"]:
                self.assertIsNotNone(row["action_mask"])
                self.assertIsNotNone(row["actual_action"])
                self.assertIn("episode_step", row)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_torch))
        self.assertEqual(agent.replay.rng.getstate(), before_replay_rng)

    def test_sparse_temporal_snapshot_keeps_exact_context_after_ring_wrap(self):
        cfg = temporal_config("gru")
        cfg["network"]["temporal_window"] = 4
        cfg["training"]["replay_capacity"] = 100
        cfg = validate_config(cfg)
        agent = build_agent(cfg, 211, (100, 10, 10))
        mask = {"frequency_slots": np.ones((100, 10), dtype=bool),
                "power": np.ones(10, dtype=bool)}
        previous = np.zeros(4, dtype=np.float32)
        for step in range(150):
            state = np.zeros(211, dtype=np.float32)
            state[0] = step
            state[-4:] = previous
            actual = {"freq": step % 100, "slots": 1, "power_w": 5.}
            previous = np.asarray([actual["freq"] / 99, .1, .1, 1], dtype=np.float32)
            following = state.copy()
            following[0] = step + 1
            following[-4:] = previous
            agent.observe(state, (step % 100, 0, 0), 1., following, False,
                          mask, mask, actual)
        provenance = {"environment_profile": profile_metadata("europe_v1"),
                      "git_commit": "test", "source_root": "test",
                      "source_manifest_sha256": "test"}
        with tempfile.TemporaryDirectory() as directory:
            path = save_learning_snapshot(directory, agent, cfg, provenance, 150)
            saved = torch.load(path, map_location="cpu", weights_only=True)
        samples = saved["replay_sample"]["transitions"]
        self.assertEqual(len(samples), 64)  # sparse among 100 eligible targets
        for row in samples:
            step = int(row["state"][0])
            self.assertEqual(row["context_length"], 4)
            self.assertEqual(row["next_context_length"], 4)
            self.assertEqual(row["context_states"][:, 0].tolist(),
                             [float(x) for x in range(step - 3, step + 1)])
            self.assertEqual(row["next_context_states"][:, 0].tolist(),
                             [float(x) for x in range(step - 2, step + 2)])
            self.assertTrue(torch.equal(row["next_context_states"][-1], row["next_state"]))

    def test_online_and_replay_windows_match_and_evaluation_resets(self):
        cfg = temporal_config("gru")
        cfg["network"]["temporal_window"] = 4
        cfg = validate_config(cfg)
        agent = build_agent(cfg, 211, (100, 10, 10))
        env = WorkbenchEnv(cfg)
        state = env.reset(seed=903)
        fixed_scene = env.scene()
        agent.reset_episode()
        for step in range(20):
            mask = env.action_mask()
            action = agent.act(state, action_mask=mask)
            next_state, reward, done, info = env.step(action)
            agent.observe(state, action, reward, next_state, done, mask,
                          env.action_mask(), info["actual_action"])
            context = agent.replay._context(len(agent.replay.rows) - 1)
            self.assertEqual(len(context), min(4, step + 1))
            self.assertTrue(all(np.array_equal(a, b) for a, b in
                                zip(context, agent.online_history)))
            state = next_state
        before = _evaluation_episode(WorkbenchEnv(cfg), agent, fixed_scene,
                                     "first", 903, "deterministic")
        after = _evaluation_episode(WorkbenchEnv(cfg), agent, fixed_scene,
                                    "second", 903, "deterministic")
        self.assertEqual(before["metrics"], after["metrics"])
        self.assertEqual([row["requested_action"] for row in before["trace"]],
                         [row["requested_action"] for row in after["trace"]])


class GraphWorkbenchTest(unittest.TestCase):
    def test_two_graph_modes_update_on_real_environment(self):
        for mode in ("beam_tokens", "interference_bias"):
            with self.subTest(mode=mode):
                cfg = default_config()
                cfg["network"].update(kind="transformer", transformer_layout="tsac_natural_context",
                                      graph_mode=mode, embedding_dim=16, num_heads=2,
                                      num_layers=1, feedforward_dim=32, dropout=0)
                cfg["state"].update(features=list(natural_shapes("full")),
                                    context_level="full", demand_visibility="known")
                cfg["environment"].update(profile="europe_v1", action_mask="joint_feasible")
                cfg["reward"]["preset"] = "legacy_repaired"
                cfg["training"].update(scene_order="shuffle_each_cycle", batch_size=2,
                                       warmup_steps=0, target_entropy=6.9)
                cfg = validate_config(cfg)
                env = WorkbenchEnv(cfg)
                state = env.reset(seed=904)
                self.assertEqual(len(state), 2946)
                agent = build_agent(cfg, 2946, (100, 10, 10))
                for _ in range(2):
                    mask = env.action_mask()
                    action = agent.act(state, action_mask=mask)
                    next_state, reward, done, _ = env.step(action)
                    agent.observe(state, action, reward, next_state, done,
                                  mask, env.action_mask())
                    state = next_state
                result = agent.update()
                self.assertEqual(agent.update_step, 1)
                self.assertTrue(math.isfinite(result["actor_grad_norm"]))
                description = model_description(cfg, 2946, (100, 10, 10))
                self.assertIn("_graph_" + mode, description["variant"])
                branch = description["branches"]["策略"]
                self.assertEqual(sum(row["parameters"] for row in branch),
                                 sum(parameter.numel() for parameter in agent.actor.parameters()))
                token = next(row for row in branch if row["label"].endswith("encoder.graph.tokens"))
                self.assertEqual(token["shape"], [122, 16])
                if mode == "interference_bias":
                    self.assertTrue(any(row["label"].endswith("encoder.graph.edge_weight")
                                        for row in branch))


if __name__ == "__main__":
    unittest.main()
