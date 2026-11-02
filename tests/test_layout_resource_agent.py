"""Bounded synthetic engineering checks, not scientific training/evaluation."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from satellite_rl.layout_resource.agent import AgentConfig, LayoutSAC, collate_observations, joint_distribution, exact_soft_value
from satellite_rl.layout_resource.schema import observation_semantics


def observation(n):
    rng = np.random.default_rng(100 + n)
    mask = np.zeros(1001, bool); mask[[0, 1, 14]] = True
    return {"tokens": rng.normal(size=(n, 16)).astype(np.float32),
            "relations": rng.normal(size=(n, n, 4)).astype(np.float32),
            "global_features": rng.normal(size=12).astype(np.float32),
            "current_index": n - 1, "padding_mask": np.zeros(n, bool), "action_mask": mask,
            "semantics": observation_semantics()}


class LayoutAgentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(51001)
        self.agent = LayoutSAC(AgentConfig(hidden_dim=16, heads=4, layers=1))

    def test_mixed_n_padding_and_relations(self):
        a, b = observation(1), observation(200)
        alone = self.agent.actor(self.agent.batch([a]))
        batch = self.agent.batch([a, b])
        together = self.agent.actor(batch)[:1]
        torch.testing.assert_close(alone, together, atol=2e-6, rtol=2e-6)
        # Even extreme padded token/relation values cannot affect real queries.
        batch["tokens"][0, 1:] = 1e7
        batch["relations"][0, 1:, :] = -1e7
        batch["relations"][0, :, 1:] = 1e7
        torch.testing.assert_close(alone, self.agent.actor(batch)[:1], atol=2e-6, rtol=2e-6)
        batch["tokens"][0, 1:] = torch.nan
        batch["relations"][0, 1:, :] = torch.inf
        batch["relations"][0, :, 1:] = torch.nan
        torch.testing.assert_close(alone, self.agent.actor(batch)[:1], atol=2e-6, rtol=2e-6)
        c = observation(3); changed = observation(3)
        changed["relations"][:] = 0
        self.assertGreater(float((self.agent.actor(self.agent.batch([c])) - self.agent.actor(self.agent.batch([changed]))).abs().max()), 1e-6)

    def test_mask_and_real_updates_terminal_without_next_state(self):
        obs = observation(2)
        p, logs, entropy = joint_distribution(self.agent.actor(self.agent.batch([obs])), self.agent.batch([obs])["action_mask"])
        self.assertEqual(float(p[0, 2]), 0)
        self.assertTrue(torch.isfinite(logs).all())
        torch.testing.assert_close(p.sum(1), torch.ones(1))
        before = [p.detach().clone() for p in self.agent.actor.parameters()]
        critics = [p.detach().clone() for p in self.agent.critic1.parameters()]
        metrics = self.agent.update([(obs, 1, .1, None, True), (observation(3), 14, .2, observation(4), False)])
        self.assertTrue(all(np.isfinite(metrics[k]) for k in ("actor_loss", "critic_loss", "entropy")))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, self.agent.actor.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(critics, self.agent.critic1.parameters())))
        # An all-terminal batch must never consult next-state networks.
        def forbidden(_):
            raise AssertionError("terminal target attempted bootstrap")
        self.agent.target1.forward = forbidden
        self.agent.target2.forward = forbidden
        self.agent.update([(obs, 0, 0., None, True)])

    def test_invalid_padding_and_mask(self):
        obs = observation(2); obs["padding_mask"][1] = True
        with self.assertRaises(ValueError): self.agent.batch([obs])
        obs = observation(2); obs["action_mask"][:] = False
        with self.assertRaises(ValueError): self.agent.batch([obs])
        with self.assertRaises(ValueError): self.agent.batch([observation(201)])
        with self.assertRaises(ValueError): self.agent.update([(observation(2), 2, 0, None, True)])
        for action, done in ((1.5, True), (True, True), (1, 1)):
            with self.assertRaises(ValueError): self.agent.update([(observation(2), action, 0, None, done)])
        obs = observation(2); obs['current_index'] = True
        with self.assertRaises(ValueError): self.agent.batch([obs])
        obs = observation(2); obs['current_index'] = 2
        with self.assertRaisesRegex(ValueError, 'terminal'): self.agent.act(obs)

    def test_semantic_config_and_checkpoint_failures_are_atomic(self):
        for config in ({'token_dim': 15}, {'heads': True}, {'heads': 0}, {'alpha': float('nan')},
                       {'learning_rate': float('inf')}, {'state_schema': 'wrong'}, {'physics_features': 1}):
            with self.assertRaises(ValueError): AgentConfig(**config)
        obs = observation(2); obs['semantics']['physics_SHA'] = 'wrong'
        with self.assertRaises(ValueError): self.agent.batch([obs])
        obs = observation(2); obs['semantics']['token_features'] = tuple(reversed(obs['semantics']['token_features']))
        with self.assertRaises(ValueError): self.agent.batch([obs])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.pt'
            self.agent.update([(observation(2), 1, .1, None, True)])
            self.agent.save(path)
            good = torch.load(path, weights_only=True)
            before = {k: v.clone() for k, v in self.agent.actor.state_dict().items()}
            from copy import deepcopy
            for kind in ('weights', 'optimizer', 'updates'):
                bad = deepcopy(good)
                if kind == 'weights': next(iter(bad['networks']['target2'].values())).fill_(float('nan'))
                if kind == 'optimizer': next(iter(bad['critic_optimizer']['state'].values()))['exp_avg'].fill_(float('inf'))
                if kind == 'updates': bad['updates'] = -1
                torch.save(bad, path)
                with self.assertRaises(ValueError): self.agent.load(path, resume=True)
                for key, value in self.agent.actor.state_dict().items(): torch.testing.assert_close(value, before[key])

    def test_nonfinite_gradient_refuses_update(self):
        before = {k: v.clone() for k, v in self.agent.critic1.state_dict().items()}
        handle = next(self.agent.actor.parameters()).register_hook(lambda gradient: gradient * float('nan'))
        with self.assertRaises(FloatingPointError):
            self.agent.update([(observation(2), 1, .1, None, True)])
        handle.remove()
        self.assertEqual(self.agent.updates, 0)
        for key, value in self.agent.critic1.state_dict().items(): torch.testing.assert_close(value, before[key])

    def test_exact_joint_math_against_manual_enumeration(self):
        import math
        logits = torch.tensor([[math.log(2), 999., math.log(3), math.log(5)]], dtype=torch.float64)
        mask = torch.tensor([[True, False, True, True]])
        probability, safe_logs, entropy = joint_distribution(logits, mask)
        expected = torch.tensor([[.2, 0., .3, .5]], dtype=torch.float64)
        torch.testing.assert_close(probability, expected)
        self.assertEqual(safe_logs[0, 1].item(), 0.)
        expected_entropy = -sum(p * math.log(p) for p in (.2, .3, .5))
        self.assertAlmostEqual(entropy.item(), expected_entropy, places=12)
        q1 = torch.tensor([[4., 999., 1., 7.]], dtype=torch.float64)
        q2 = torch.tensor([[1., -999., 6., 2.]], dtype=torch.float64)
        value = exact_soft_value(probability, safe_logs, q1, q2, .01)
        manual = .2 * 1 + .3 * 1 + .5 * 2 + .01 * expected_entropy
        self.assertAlmostEqual(value.item(), manual, places=12)
        # A minimum taken after expectation would produce a different answer.
        wrong = min((expected*q1).sum().item(), (expected*q2).sum().item()) + .01 * expected_entropy
        self.assertGreater(abs(value.item()-wrong), .1)
        rewards = torch.tensor([.7, .7], dtype=torch.float64)
        done = torch.tensor([False, True])
        target = rewards + (~done) * .99 * value
        torch.testing.assert_close(target, torch.tensor([.7+.99*manual, .7], dtype=torch.float64))

    def test_optimizer_resume_next_update_is_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'continuation.pt'
            first = [(observation(2), 1, .1, None, True)]
            self.agent.update(first); self.agent.save(path)
            resumed = LayoutSAC(self.agent.config); resumed.load(path, resume=True)
            second = [(observation(3), 14, .2, observation(4), False)]
            a, b = self.agent.update(second), resumed.update(second)
            self.assertEqual(a, b)
            for name in ('actor', 'critic1', 'critic2', 'target1', 'target2'):
                for key, value in getattr(self.agent, name).state_dict().items():
                    torch.testing.assert_close(value, getattr(resumed, name).state_dict()[key], atol=0, rtol=0)

    def test_zero_update_checkpoint_rejects_optimizer_configuration_drift(self):
        from copy import deepcopy
        self.assertEqual(self.agent.updates, 0)
        before = {key: value.clone() for key, value in self.agent.actor.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'zero_update.pt'
            self.agent.save(path)
            good = torch.load(path, weights_only=True)
            # No forward, backward, update or optimizer step is executed here.
            self.agent.load(path, resume=True)
            changes = {'lr': .9, 'betas': (.8, .99), 'eps': .1,
                       'weight_decay': .1, 'amsgrad': True, 'maximize': True,
                       'capturable': True, 'foreach': True, 'differentiable': True,
                       'fused': True}
            for optimizer_key in ('actor_optimizer', 'critic_optimizer'):
                for option, replacement in changes.items():
                    with self.subTest(optimizer=optimizer_key, option=option):
                        bad = deepcopy(good)
                        bad[optimizer_key]['param_groups'][0][option] = replacement
                        torch.save(bad, path)
                        with self.assertRaisesRegex(ValueError, 'differ from AgentConfig'):
                            self.agent.load(path, resume=True)
                        self.assertEqual(self.agent.updates, 0)
                        self.assertEqual(self.agent.actor_optimizer.param_groups[0]['lr'], self.agent.config.learning_rate)
                        for key, value in self.agent.actor.state_dict().items():
                            torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_checkpoint_strict_schema_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new.pt"
            self.agent.update([(observation(2), 1, .1, None, True)])
            self.agent.save(path, {"engineering_only": True})
            other = LayoutSAC(self.agent.config)
            self.assertTrue(other.load(path, resume=True)["engineering_only"])
            self.assertEqual(other.updates, 1)
            torch.testing.assert_close(self.agent.actor(self.agent.batch([observation(2)])), other.actor(other.batch([observation(2)])))
            checkpoint = torch.load(path, weights_only=True)
            checkpoint["config"]["physics_schema"] = "legacy_fixed_05"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "Incompatible"): other.load(path)
            torch.save(self.agent.actor.state_dict(), path)
            with self.assertRaisesRegex(ValueError, "legacy 207"): other.load(path)

    def test_input_only_physics_ablation(self):
        agent = LayoutSAC(AgentConfig(hidden_dim=16, heads=4, layers=1, physics_features=False))
        obs = observation(2)
        original = obs["tokens"].copy()
        batch = agent.batch([obs])
        self.assertEqual(float(batch["tokens"][..., 2:4].abs().sum()), 0)
        self.assertEqual(float(batch["relations"][..., 1].abs().sum()), 0)
        np.testing.assert_array_equal(original, obs["tokens"])
        torch.testing.assert_close(batch["tokens"][..., 5], torch.tensor(original[:, 5])[None])

    def test_real_legacy_actor_remains_loadable_and_new_model_rejects(self):
        from satellite_rl.agents.sac_transformer import PolicyNet
        from satellite_rl.evaluation.checkpoints import load_network_checkpoint
        path = Path(__file__).resolve().parents[1] / "models/pretrained/sac_transformer/actor.pth"
        if not path.is_file():
            self.skipTest("historical asset not present in this checkout")
        network = PolicyNet(207, 128, 100, 10, 10)
        self.assertTrue(load_network_checkpoint(network, path, algorithm="legacy T-SAC", evaluation_mode=True))
        with self.assertRaisesRegex(ValueError, "legacy 207"):
            self.agent.load(path)


if __name__ == "__main__":
    unittest.main()
