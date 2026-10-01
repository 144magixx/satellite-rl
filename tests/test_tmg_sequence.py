"""Finite history, executed-action timing and SAC temporal updates."""

import copy
import unittest

import numpy as np
import torch

from satellite_rl.agents.sequence_replay import SequenceReplay
from satellite_rl.agents.workbench_algorithms import build_agent
from satellite_rl.state_layouts import natural_shapes
from satellite_rl.workbench.config import default_config, validate_config
from satellite_rl.workbench.environment import WorkbenchEnv


def config(mode):
    value = default_config()
    value["network"].update(kind="transformer", transformer_layout="tsac_natural_context",
                            temporal_mode=mode, temporal_window=16, embedding_dim=16,
                            num_heads=2, num_layers=2, feedforward_dim=32, dropout=0)
    value["state"].update(features=list(natural_shapes("compact")) + ["previous_actual_action"],
                          context_level="compact", demand_visibility="known")
    value["environment"].update(profile="europe_v1", action_mask="joint_feasible")
    value["reward"]["preset"] = "legacy_repaired"
    value["training"].update(scene_order="shuffle_each_cycle", batch_size=2,
                              warmup_steps=0, target_entropy=6.9)
    return validate_config(value)


class SequenceReplayTest(unittest.TestCase):
    def test_over_capacity_preserves_exact_online_window_and_resume_sample(self):
        replay = SequenceReplay(capacity=5, seed=41, window=4)
        mask = {"frequency_slots": np.ones((2, 2), dtype=bool),
                "power": np.ones(2, dtype=bool)}
        prior = np.zeros(4, dtype=np.float32)
        for step in range(20):
            state = np.concatenate(([step], prior)).astype(np.float32)
            actual = {"freq": step % 2, "slots": 1, "power_w": 5.0}
            current_actual = np.asarray([actual["freq"] / 99, .1, .1, 1], dtype=np.float32)
            next_state = np.concatenate(([step + 1], current_actual)).astype(np.float32)
            replay.add(state, (step % 2, 0, 0), 1., next_state, False,
                       mask, mask, actual)
            prior = current_actual
            if step >= 7:
                self.assertEqual(len(replay), 5)
                for logical in range(len(replay.rows) - len(replay), len(replay.rows)):
                    row = replay._row(logical)
                    expected = list(range(row[8] - 3, row[8] + 1))
                    self.assertEqual([int(x[0]) for x in replay._context(logical)], expected)
        self.assertEqual(len(replay.rows), 8)
        self.assertEqual([row[8] for row in replay.eligible_rows()], [15, 16, 17, 18, 19])
        saved = copy.deepcopy(replay.state_dict())
        first = replay.sample(5, "cpu", (2, 2, 2))
        restored = SequenceReplay(capacity=5, seed=999, window=4)
        restored.load_state_dict(saved)
        second = restored.sample(5, "cpu", (2, 2, 2))
        for a, b in zip(first, second):
            if isinstance(a, tuple):
                self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
            else:
                self.assertTrue(torch.equal(a, b))
        for step in range(2):
            state = np.concatenate(([100 + step], prior)).astype(np.float32)
            actual = {"freq": 0, "slots": 1, "power_w": 5.0}
            prior = np.asarray([0, .1, .1, 1], dtype=np.float32)
            next_state = np.concatenate(([101 + step], prior)).astype(np.float32)
            replay.add(state, (0, 0, 0), 0., next_state, step == 1,
                       mask, None if step == 1 else mask, actual)
        # A terminal transition closes this episode; the next must reset history.
        initial = np.asarray([200, 0, 0, 0, 0], dtype=np.float32)
        following = np.asarray([201, 0, .1, .1, 1], dtype=np.float32)
        replay.add(initial, (0, 0, 0), 0., following, False, mask, mask,
                   {"freq": 0, "slots": 1, "power_w": 5.0})
        self.assertEqual([int(x[0]) for x in replay._context(len(replay.rows) - 1)], [200])


class TemporalSACIntegrationTest(unittest.TestCase):
    def test_actual_action_fields_and_real_update(self):
        for mode in ("gru", "causal_transformer"):
            with self.subTest(mode=mode):
                torch.manual_seed(13)
                cfg = config(mode)
                env = WorkbenchEnv(cfg)
                state = env.reset(seed=901)
                self.assertEqual(len(state), 211)
                np.testing.assert_array_equal(state[-4:], np.zeros(4))
                agent = build_agent(cfg, 211, (100, 10, 10), "cpu")
                agent.reset_episode()
                for step in range(2):
                    mask = env.action_mask()
                    action = agent.act(state, action_mask=mask)
                    next_state, reward, done, info = env.step(action)
                    self.assertFalse(done)
                    actual = info["actual_action"]
                    np.testing.assert_allclose(next_state[-4:],
                                               [actual["freq"] / 99, actual["slots"] / 10,
                                                actual["power_w"] / 50, 1], atol=2e-6)
                    agent.observe(state, action, reward, next_state, done, mask,
                                  env.action_mask(), actual)
                    state = next_state
                before = [p.detach().clone() for p in agent.actor.parameters()]
                metrics = agent.update()
                self.assertEqual(metrics["effective_td_targets"], 2)
                self.assertGreater(metrics["actor_grad_norm"], 0)
                self.assertTrue(any(not torch.equal(a, b) for a, b in
                                    zip(before, agent.actor.parameters())))
                saved = copy.deepcopy(agent.state_dict())
                recovered = build_agent(cfg, 211, (100, 10, 10), "cpu")
                recovered.load_state_dict(saved)
                self.assertEqual(len(recovered.online_history), 2)
                self.assertEqual(recovered.update_step, 1)


if __name__ == "__main__":
    unittest.main()
