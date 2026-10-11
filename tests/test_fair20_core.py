"""Engineering checks only: no algorithm performance inference."""
import unittest
import io
import pickle
from unittest.mock import patch
import numpy as np
import torch
from satellite_rl.fair_comparison import Fair20Env, make_agent
from satellite_rl.fair_comparison.networks import masked_distribution, parameter_sha
from satellite_rl.fair_comparison.algorithms import (pack_transition, unpack_transition, bellman_target,
    double_dqn_continuation, gae_advantages)


def scene(n=50):
    return {"beam_location": np.column_stack((np.full(n, 45.), np.linspace(2, 25, n))),
            "beam_G_Tx_max": np.full((n, 1), 45.), "UE_G_Rx_max": np.full((n, 1), 35.),
            "distances": np.full(n, 38000.), "rate": np.linspace(1e8, 5e8, n)[:, None],
            "noise_temperature": np.full((n, 1), 290.), "original_indices": np.arange(n)}


class EnvironmentTests(unittest.TestCase):
    def test_complete_reserve_power_telescope(self):
        for n in (50, 200):
            env = Fair20Env(scene(n))
            total = 0.
            for i in range(n):
                mask = env.action_mask()
                self.assertTrue(mask.any())
                # Select longest possible actions to exercise future-slot reserve.
                legal = np.flatnonzero(mask)
                action = int(legal[np.argmax(legal % 10)])
                _, reward, terminal, info = env.step(action)
                self.assertEqual(info["requested_action"], info["actual_action"])
                self.assertEqual(terminal, i == n-1)
                self.assertLessEqual(env.power.sum(), 3000+1e-8)
                total += reward
            self.assertAlmostEqual(total, 100*env.si, places=10)
            self.assertTrue(np.all(env.slots > 0))
            self.assertAlmostEqual(env.power.sum(), 3000)
            self.assertEqual(env.action_mask().sum(), 0)
            self.assertEqual(env.observation()["beams"].shape, (200, 24))
            if n == 50:
                self.assertTrue(np.all(env.power == 60))

    def test_no_hidden_repair_and_restore(self):
        env = Fair20Env(scene())
        with self.assertRaises(ValueError):
            env.step(999)
        env.step(0)
        state = env.state_dict()
        other = Fair20Env(scene())
        other.load_state_dict(state)
        np.testing.assert_array_equal(env.action_mask(), other.action_mask())
        action = int(np.flatnonzero(env.action_mask())[3])
        a, b = env.step(action), other.step(action)
        self.assertEqual(a[1], b[1])
        np.testing.assert_array_equal(a[0]["beams"], b[0]["beams"])

    def test_mask_all_candidates(self):
        env = Fair20Env(scene(200))
        for _ in range(17):
            env.step(int(np.flatnonzero(env.action_mask())[-1]))
        mask = env.action_mask()
        group = env.index % 8
        future = sum(env.groups[env.index+1:] == group)
        for action in range(1000):
            f, remainder = divmod(action, 10)
            length = remainder+1
            legal = f+length <= 100 and not env.occupancy[group, f:f+length].any() and 100-env.occupancy[group].sum()-length >= future
            self.assertEqual(bool(mask[action]), legal)

    def test_later_cross_group_interference_revises_prior_capacity(self):
        data = scene()
        data["beam_location"][:] = [45., 13.]
        env = Fair20Env(data)
        env.step(0)
        initial = env.capacity[0]
        env.step(0)  # Opposite polarization, no same-polarization interference.
        self.assertEqual(env.capacity[0], initial)
        env.step(0)  # Group2 shares polarization with group0 and slot0.
        self.assertLess(env.capacity[0], initial)


class LearningTests(unittest.TestCase):
    def test_deterministic_ppo_uses_action_network_only(self):
        agent = make_agent("ppo", device="cpu")
        obs = Fair20Env(scene()).observation()
        mask = np.zeros(1000, dtype=bool)
        mask[[3, 9]] = True
        logits = torch.zeros((1, 1000))
        logits[0, 12] = 1000.  # Illegal maximum must never be selected.
        with patch.object(agent.policy, "forward", return_value=logits), patch.object(
                agent.value, "forward", side_effect=AssertionError("Deployment must not execute V")):
            action, metadata = agent.act(obs, mask, deterministic=True)
        self.assertEqual(action, 3)  # Equal legal values break ties by smallest index.
        self.assertEqual(metadata["value"], 0.)
        with patch.object(agent.policy, "forward", return_value=logits), patch.object(
                agent.value, "forward", return_value=torch.tensor([[7.]])) as value_forward:
            action, metadata = agent.act(obs, mask, deterministic=False)
        self.assertIn(action, (3, 9))
        self.assertEqual(metadata["value"], 7.)
        self.assertAlmostEqual(metadata["log_prob"], -np.log(2), places=6)
        value_forward.assert_called_once()
    def test_hand_calculable_double_dqn_terminal_and_truncation(self):
        online = torch.tensor([[1., 9., 1000.], [3., 2., 1000.]])
        target = torch.tensor([[20., 7., 2000.], [4., 30., 2000.]])
        mask = torch.tensor([[True, True, False], [True, True, False]])
        continuation = double_dqn_continuation(online, target, mask)
        torch.testing.assert_close(continuation, torch.tensor([7., 4.]))
        # First row is a nonterminal budget truncation; second true terminal.
        torch.testing.assert_close(bellman_target(torch.tensor([2., 3.]), continuation,
                                                 torch.tensor([False, True])), torch.tensor([9., 3.]))

    def test_hand_calculable_gae_bootstrap_and_episode_boundary(self):
        batch = [{"reward": 1., "terminated": False, "truncated": False},
                 {"reward": 2., "terminated": False, "truncated": True},
                 {"reward": 3., "terminated": True, "truncated": False}]
        result = gae_advantages(batch, np.array([.5, 1., 1.5]), np.array([1., 4., 999.]))
        np.testing.assert_allclose(result, [1.5+.95*5., 5., 1.5], rtol=0, atol=1e-14)

    def test_lossless_packed_replay_large_checkpoint(self):
        env = Fair20Env(scene())
        obs, mask = env.observation(), env.action_mask()
        nxt, reward, terminal, _ = env.step(0)
        transition = dict(obs=obs, mask=mask, action=0, reward=reward, next_obs=nxt,
                          next_mask=env.action_mask(), terminated=terminal, truncated=False, metadata={})
        packed = pack_transition(transition)
        restored = unpack_transition(packed)
        self.assertEqual(pickle.dumps(transition, protocol=5), pickle.dumps(restored, protocol=5))
        agent = make_agent("dqn", device="cpu")
        # 100000 references to one immutable engineering sample exercise full
        # capacity and serialization without adding environment interactions.
        agent.replay = [packed]*100000
        state = agent.state_dict()
        self.assertIs(state["replay"][12345], packed)
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, map_location="cpu")
        other = make_agent("dqn", device="cpu")
        other.load_state_dict(loaded)
        self.assertEqual(len(other.replay), 100000)
        self.assertEqual(other.replay[99999], packed)
        self.assertLess(len(packed), len(pickle.dumps(transition, protocol=5))/4)
    def test_mask_gradient_and_singleton(self):
        logits = torch.randn(2, 1000, requires_grad=True)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[0, [4, 8]] = True
        mask[1, 19] = True
        p, lp = masked_distribution(logits, mask)
        loss = (p*(lp-torch.arange(1000))).sum()
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[~mask], torch.zeros_like(logits.grad[~mask])))
        self.assertEqual(float(-(p[1]*lp[1]).sum()), 0.)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA engineering update requires DQN CUDA")
    def test_real_cuda_updates_and_strict_restore(self):
        torch.set_num_threads(2)
        hashes = []
        for algorithm in ("sac", "ppo", "dqn"):
            env = Fair20Env(scene())
            agent = make_agent(algorithm, device="cuda")
            hashes.append(agent.initialization_metadata["policy"]["sha256"])
            before = parameter_sha(agent.policy)
            transitions = []
            # 8 real environment interactions per arm, 24 total; short tail is
            # genuinely bootstrapped and PPO is updated with stored old masks.
            for i in range(8):
                obs, mask = env.observation(), env.action_mask()
                action, metadata = agent.act(obs, mask)
                nxt, reward, terminal, _ = env.step(action)
                row = dict(obs=obs, mask=mask, action=action, reward=reward,
                           next_obs=nxt, next_mask=env.action_mask(), terminated=terminal,
                           truncated=i == 7, metadata=metadata)
                transitions.append(row)
                agent.observe(**row)
            if algorithm == "ppo":
                agent.flush()
            else:
                agent.update_batch(transitions)
            self.assertNotEqual(before, parameter_sha(agent.policy))
            buffer = io.BytesIO()
            torch.save(agent.state_dict(), buffer)
            buffer.seek(0)
            checkpoint = torch.load(buffer, map_location="cuda")
            restored = make_agent(algorithm, device="cuda")
            restored.load_state_dict(checkpoint)
            self.assertEqual(parameter_sha(agent.policy), parameter_sha(restored.policy))
            self.assertEqual(agent.act(env.observation(), env.action_mask()), restored.act(env.observation(), env.action_mask()))
            if algorithm != "ppo":
                agent.update_batch(transitions)
                restored.update_batch(transitions)
                self.assertEqual(parameter_sha(agent.policy), parameter_sha(restored.policy))
            with self.assertRaises(ValueError):
                bad = checkpoint.copy()
                bad["candidate"] = "B"
                restored.load_state_dict(bad)
        self.assertEqual(len(set(hashes)), 1)


if __name__ == "__main__":
    unittest.main()
