"""Behavioral checks for masked finite-horizon coverage PPO."""

import unittest

import numpy as np
import torch

from satellite_rl.coverage.ppo import CoveragePolicy, Rollout, ppo_update


class CoveragePPOTests(unittest.TestCase):
    def test_gae_terminal_cuts_value_and_rollout_boundary_bootstraps(self):
        # The first environment ends at t=0 and starts a new episode at t=1.
        # The second runs through the end of this rollout and must bootstrap.
        rollout = Rollout(
            observations=torch.zeros(2, 2, 3), masks=torch.ones(2, 2, 9, dtype=torch.bool),
            actions=torch.zeros(2, 2, dtype=torch.long), old_logp=torch.zeros(2, 2),
            old_values=torch.tensor([[4., 4.], [2., 2.]]),
            rewards=torch.tensor([[3., 3.], [5., 5.]]),
            dones=torch.tensor([[True, False], [False, False]]),
            bootstrap_value=torch.tensor([7., 7.]))
        advantage, returns = rollout.advantages(gamma=1., gae_lambda=1.)
        self.assertEqual(advantage[0, 0].item(), -1.)
        self.assertEqual(returns[0, 0].item(), 3.)
        self.assertEqual(returns[1, 1].item(), 12.)
        self.assertEqual(returns[0, 1].item(), 15.)

    def test_masked_actions_never_sample_illegal_and_old_logp_is_reused(self):
        torch.manual_seed(73)
        policy = CoveragePolicy(4, hidden_sizes=(16,))
        obs = torch.randn(32, 4)
        mask = torch.zeros(32, 9, dtype=torch.bool)
        mask[:, 0] = True
        mask[::2, 3] = True
        action, old_logp, value = policy.act(obs, mask)
        self.assertTrue(mask.gather(1, action[:, None]).all().item())
        self.assertTrue(torch.allclose(old_logp, policy.distribution(obs, mask).log_prob(action)))
        before = [tensor.detach().clone() for tensor in policy.parameters()]
        rollout = Rollout(obs[None], mask[None], action[None], old_logp[None],
                          value[None], torch.ones(1, 32), torch.ones(1, 32, dtype=torch.bool),
                          torch.full((32,), 1000.))
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        result = ppo_update(policy, optimizer, rollout, epochs=2, minibatch_size=16)
        self.assertEqual(result["updates"], 4)
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(before, policy.parameters())))
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result.values()))

    def test_zero_dropout_and_actor_critic_are_independent(self):
        policy = CoveragePolicy(5, hidden_sizes=(8, 8))
        self.assertFalse(any(isinstance(module, torch.nn.Dropout) for module in policy.modules()))
        actor_ids = {id(p) for p in policy.actor.parameters()}
        critic_ids = {id(p) for p in policy.critic.parameters()}
        self.assertFalse(actor_ids & critic_ids)

    def test_real_coverage_steps_update_policy_without_illegal_actions(self):
        from satellite_rl.coverage.environment import CoverageBatchEnv
        from satellite_rl.coverage.scenes import CoverageDataset, CoverageScene

        centers = np.array([[45.0 + i // 10, float(i % 10)] for i in range(100)])
        scene = CoverageScene("ppo-update", centers.copy(), np.full(100, 1e6),
                              centers.copy(), [str(i) for i in range(100)], {}, "toy-sha")
        env = CoverageBatchEnv(CoverageDataset([scene], {}), {}, num_envs=2,
                               device="cpu", seed=51001)
        obs, mask = env.reset()
        policy = CoveragePolicy(env.obs_dim, hidden_sizes=(32,))
        samples = [[] for _ in range(7)]
        for _ in range(3):
            action, logp, value = policy.act(obs, mask)
            next_obs, reward, done, _ = env.step(action)
            for items, value_to_save in zip(samples, (obs, mask, action, logp,
                                                       value, reward, done)):
                items.append(value_to_save)
            obs, mask = next_obs, env.action_mask()
        rollout = Rollout(*(torch.stack(items) for items in samples), policy.value(obs).detach())
        before = [parameter.detach().clone() for parameter in policy.parameters()]
        diagnostics = ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=1e-3),
                                 rollout, minibatch_size=4, epochs=2)
        self.assertGreater(diagnostics["updates"], 0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, policy.parameters())))

    def test_validation_rng_isolation_preserves_future_training_draws(self):
        import random
        from satellite_rl.coverage.runner import _isolated_rng, _rng_state, _restore_rng

        random.seed(51001)
        np.random.seed(51001)
        torch.manual_seed(51001)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(51001)
        saved = _rng_state()
        expected = (random.random(), float(np.random.random()), torch.rand(1).item())
        _restore_rng(saved)
        with _isolated_rng(61004):
            random.random()
            np.random.random()
            torch.rand(1)
            if torch.cuda.is_available():
                torch.rand(1, device="cuda")
        actual = (random.random(), float(np.random.random()), torch.rand(1).item())
        self.assertEqual(actual, expected)

    def test_summary_keeps_failed_initializer_in_coverage_denominator(self):
        from satellite_rl.coverage.runner import _summary

        rows = [
            {"input": {"demands_bps": [40.0, 60.0],
                       "metadata": {"initialization_valid": True}},
             "metrics": {"coverage_demand": 0.8, "coverage_users": 0.5,
                         "uncovered_demand_bps": 20.0, "uncovered_users": 1,
                         "feasible": 1.0, "load_cv": 0.7}},
            {"input": {"demands_bps": [100.0, 100.0, 100.0],
                       "metadata": {"initialization_valid": False}},
             "metrics": None},
        ]
        summary = _summary(rows)
        self.assertEqual(summary["scene_count"], 2)
        self.assertEqual(summary["valid_metric_scene_count"], 1)
        self.assertEqual(summary["failed_scene_count"], 1)
        self.assertAlmostEqual(summary["coverage_demand"], 0.4)
        self.assertAlmostEqual(summary["coverage_users"], 0.25)
        self.assertAlmostEqual(summary["pooled_demand_coverage"], 0.2)
        self.assertAlmostEqual(summary["uncovered_demand_bps"], 160.0)
        self.assertAlmostEqual(summary["feasible"], 0.5)
        self.assertAlmostEqual(summary["load_cv"], 0.7)

        partial = [{"input": {"demands_bps": [10.0, 90.0],
                              "metadata": {"initialization_valid": False}},
                    "metrics": {"coverage_demand": 0.9, "coverage_users": 0.5,
                                "uncovered_demand_bps": 10.0, "uncovered_users": 1,
                                "feasible": 0.0, "load_cv": 0.8}}]
        partial_summary = _summary(partial)
        self.assertEqual(partial_summary["failed_scene_count"], 1)
        self.assertEqual(partial_summary["valid_metric_scene_count"], 1)
        self.assertEqual(partial_summary["coverage_demand"], 0.0)
        self.assertEqual(partial_summary["uncovered_demand_bps"], 100.0)
        self.assertEqual(partial_summary["load_cv"], 0.8)


if __name__ == "__main__":
    unittest.main()
