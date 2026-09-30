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
            candidate_features=torch.zeros(2, 2, 9, 4),
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
        candidates = torch.randn(32, 9, 4)
        action, old_logp, value = policy.act(obs, mask, candidates)
        self.assertTrue(mask.gather(1, action[:, None]).all().item())
        self.assertTrue(torch.allclose(
            old_logp, policy.distribution(obs, mask, candidates).log_prob(action)))
        before = [tensor.detach().clone() for tensor in policy.parameters()]
        rollout = Rollout(obs[None], mask[None], candidates[None], action[None], old_logp[None],
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
        actor_ids = {id(p) for module in (policy.actor_context,
                     policy.candidate_encoder, policy.candidate_bias) for p in module.parameters()}
        critic_ids = {id(p) for p in policy.critic.parameters()}
        self.assertFalse(actor_ids & critic_ids)

    def test_candidate_scorer_has_same_parameters_for_nine_and_2048_actions(self):
        small = CoveragePolicy(964, action_dim=9, hidden_sizes=(256, 256))
        large = CoveragePolicy(964, action_dim=2048, hidden_sizes=(256, 256))
        self.assertEqual([(name, tuple(weight.shape)) for name, weight in small.state_dict().items()],
                         [(name, tuple(weight.shape)) for name, weight in large.state_dict().items()])
        obs = torch.zeros(2, 964)
        mask = torch.zeros(2, 2048, dtype=torch.bool)
        mask[:, 17] = True
        candidates = torch.randn(2, 2048, 4)
        action, logp, _ = large.act(obs, mask, candidates)
        self.assertTrue(torch.equal(action, torch.full((2,), 17)))
        self.assertTrue(torch.isfinite(logp).all())

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
        samples = [[] for _ in range(8)]
        candidates = env.candidate_features()
        for _ in range(3):
            action, logp, value = policy.act(obs, mask, candidates)
            next_obs, reward, done, _ = env.step(action)
            for items, value_to_save in zip(samples, (obs, mask, candidates, action, logp,
                                                       value, reward, done)):
                items.append(value_to_save)
            obs, mask = next_obs, env.action_mask()
            candidates = env.candidate_features()
        rollout = Rollout(*(torch.stack(items) for items in samples), policy.value(obs).detach())
        before = [parameter.detach().clone() for parameter in policy.parameters()]
        diagnostics = ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=1e-3),
                                 rollout, minibatch_size=4, epochs=2)
        self.assertGreater(diagnostics["updates"], 0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, policy.parameters())))

    def test_empty_start_policy_selects_a_unique_candidate(self):
        from satellite_rl.coverage.environment import CoverageBatchEnv
        from satellite_rl.coverage.scenes import CoverageDataset, CoverageScene, _empty_centers

        positions = np.array([[45.0 + i // 10, float(i % 10)] for i in range(100)])
        demands = np.full(100, 1e6)
        parking, metadata = _empty_centers(positions, demands, 2048)
        scene = CoverageScene("empty-update", positions, demands, parking,
                              [str(i) for i in range(100)], metadata, "empty-sha")
        env = CoverageBatchEnv(CoverageDataset([scene], {"initialization": "empty"}),
                               {"initialization": "empty", "sweeps": 1},
                               num_envs=1, device="cpu", seed=51001)
        obs, mask = env.reset(scene_indices=[0])
        candidates = env.candidate_features()
        policy = CoveragePolicy(env.obs_dim, action_dim=env.action_dim, hidden_sizes=(32,))
        action, logp, _ = policy.act(obs, mask, candidates)
        self.assertEqual(env.obs_dim, 964)
        self.assertEqual(env.action_dim, 2048)
        self.assertEqual(tuple(candidates.shape), (1, 2048, 4))
        self.assertTrue(torch.isfinite(logp).all())
        self.assertTrue(mask[0, action[0]])
        env.step(action)
        self.assertEqual(int(env.active_masks().sum().item()), 1)
        self.assertFalse(env.action_mask()[0, action[0]])

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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA checkpoint mapping test")
    def test_checkpoint_rng_restores_after_cuda_map_location(self):
        import random
        import tempfile
        from pathlib import Path
        from satellite_rl.coverage.runner import (_load_torch, _restore_rng,
                                                  _rng_state, _save_torch)

        random.seed(51001)
        np.random.seed(51001)
        torch.manual_seed(51001)
        torch.cuda.manual_seed_all(51001)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            _save_torch(checkpoint, {"rng": _rng_state()})
            expected = (random.random(), np.random.random(),
                        torch.rand(4), torch.rand(4, device="cuda"))
            random.random()
            np.random.random()
            torch.rand(4)
            torch.rand(4, device="cuda")
            loaded = _load_torch(checkpoint, "cuda")["rng"]
            self.assertEqual(loaded["torch"].device.type, "cuda")
            self.assertTrue(all(state.device.type == "cuda" for state in loaded["cuda"]))
            _restore_rng(loaded)
            actual = (random.random(), np.random.random(),
                      torch.rand(4), torch.rand(4, device="cuda"))
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])
            self.assertTrue(torch.equal(actual[2], expected[2]))
            self.assertTrue(torch.equal(actual[3], expected[3]))

    def test_checkpoint_preserves_sampled_candidate_features(self):
        import tempfile
        from pathlib import Path
        from satellite_rl.coverage.config import default_config
        from satellite_rl.coverage.runner import (_checkpoint, _load_torch, _restore,
                                                  _save_torch)

        class CandidateEnv:
            def __init__(self, values):
                self.values = values.clone()

            def state_dict(self):
                return {"values": self.values.clone()}

            def load_state_dict(self, state):
                self.values = state["values"].clone()

            def candidate_features(self):
                return self.values.clone()

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = default_config()
            job = {"run_id": "candidate_test", "git_commit": "test_commit",
                   "source_root": str(directory)}
            policy = CoveragePolicy(964, action_dim=9, hidden_sizes=(16,))
            optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
            features = torch.randn(2, 9, 4)
            env = CandidateEnv(features)
            _checkpoint(directory, job, config, policy, optimizer, env,
                        torch.zeros(2, 964), torch.ones(2, 9, dtype=torch.bool),
                        features, 32, 4, 0, torch.zeros(2), -1, 0.1, {})
            env.values.zero_()
            restored = _restore(directory, job, config, policy, optimizer, env, {}, "cpu")
            self.assertTrue(torch.equal(restored["candidate_features"], features))
            self.assertTrue(torch.equal(env.candidate_features(), features))
            altered = _load_torch(directory / "checkpoint.pt", "cpu")
            altered["candidate_features"] = features + 1
            _save_torch(directory / "checkpoint.pt", altered)
            with self.assertRaisesRegex(ValueError, "candidate features"):
                _restore(directory, job, config, policy, optimizer, env, {}, "cpu")

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

        pure = [{"input": {"demands_bps": [10.0],
                            "metadata": {"initialization_valid": True}},
                 "metrics": {"coverage_demand": 1.0, "coverage_users": 1.0,
                             "uncovered_demand_bps": 0.0, "uncovered_users": 0,
                             "feasible": 1.0, "load_cv": 0.0,
                             "movement_applicable": 0.0,
                             "movement_mean_deg": 0.0,
                             "movement_max_deg": 0.0}}]
        pure_summary = _summary(pure)
        self.assertEqual(pure_summary["movement_metric_scene_count"], 0)
        self.assertNotIn("movement_mean_deg", pure_summary)
        self.assertNotIn("movement_max_deg", pure_summary)


if __name__ == "__main__":
    unittest.main()
