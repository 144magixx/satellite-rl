"""Small exact checks for conditional masked SAC and complete-action critics."""

import copy
import math
import unittest

import numpy as np
import torch

from satellite_rl.agents.action_masking import conditional_masked_policy
from satellite_rl.agents.workbench_algorithms import build_agent
from satellite_rl.agents.workbench_networks import JointQNetwork, model_description
from satellite_rl.workbench.config import default_config


class ConditionalPolicyTest(unittest.TestCase):
    def test_enumerated_joint_probabilities_entropy_and_gradients(self):
        fl_logits = torch.tensor([[[0.2, 0.7], [-0.4, 1.1]]],
                                 dtype=torch.float64, requires_grad=True)
        power_logits = torch.tensor([[[[0.1, 1.2], [-0.2, 0.4]],
                                      [[0.8, -0.3], [0.5, -0.1]]]],
                                    dtype=torch.float64, requires_grad=True)
        legal_fl = torch.tensor([[[True, False], [True, True]]])
        legal_power = torch.tensor([[True, True]])
        (fl, conditional_power, log_fl, log_power, hfl, hp, hf, hs) = (
            conditional_masked_policy((fl_logits, power_logits), legal_fl, legal_power))
        joint = fl.unsqueeze(-1) * conditional_power
        legal_pairs = [(0, 0), (1, 0), (1, 1)]
        fl_weights = np.exp([float(fl_logits[0, f, l]) for f, l in legal_pairs])
        fl_expected = fl_weights / fl_weights.sum()
        expected = np.zeros((2, 2, 2))
        for (f, l), pair_probability in zip(legal_pairs, fl_expected):
            weights = np.exp([float(power_logits[0, f, l, p]) for p in range(2)])
            expected[f, l] = pair_probability * weights / weights.sum()
        np.testing.assert_allclose(joint.detach().numpy()[0], expected, atol=1e-12)
        self.assertAlmostEqual(float(joint.sum()), 1.0, places=12)
        self.assertEqual(float(joint[0, 0, 1].sum()), 0.0)
        expected_entropy = -sum(p * math.log(p) for p in expected.ravel() if p > 0)
        self.assertAlmostEqual(float(hfl + hp), expected_entropy, places=12)
        self.assertAlmostEqual(float(hf + hs), float(hfl), places=12)
        self.assertEqual(float(log_fl[0, 0, 1]), 0.0)
        q = torch.tensor([[[[1., -2.], [50., 50.]], [[3., 4.], [-1., 5.]]]],
                         dtype=torch.float64)
        objective = (joint * q).sum() + 0.2 * (hfl + hp).sum()
        objective.backward()
        self.assertAlmostEqual(float((joint * q).sum()), float((expected * q.numpy()[0]).sum()), places=12)
        self.assertTrue(torch.isfinite(fl_logits.grad).all())
        self.assertTrue(torch.isfinite(power_logits.grad).all())
        self.assertGreater(float(fl_logits.grad.abs().sum()), 0)
        self.assertGreater(float(power_logits.grad.abs().sum()), 0)
        self.assertEqual(float(fl_logits.grad[0, 0, 1]), 0.0)

    def test_power_mask_zero_probability_and_no_nan_gradient(self):
        fl = torch.zeros((1, 2, 2), requires_grad=True)
        power = torch.zeros((1, 2, 2, 3), requires_grad=True)
        parts = conditional_masked_policy((fl, power),
                                          torch.tensor([[[True, False], [False, True]]]),
                                          torch.tensor([[False, True, False]]))
        self.assertEqual(float(parts[1][..., 0].sum() + parts[1][..., 2].sum()), 0.0)
        self.assertAlmostEqual(float(parts[4] + parts[5]), math.log(2), places=6)
        (parts[4] + parts[5]).sum().backward()
        self.assertTrue(torch.isfinite(fl.grad).all())
        self.assertTrue(torch.isfinite(power.grad).all())


class JointActionAgentTest(unittest.TestCase):
    def setUp(self):
        self.config = default_config()
        self.config["algorithm"] = "sac"
        self.config["environment"]["action_mask"] = "joint_feasible"
        self.config["network"].update(kind="mlp", hidden_sizes=[8], dropout=0)
        self.config["training"].update(batch_size=2, warmup_steps=0, target_entropy=2.0)
        self.dims = (100, 10, 10)
        self.state = np.zeros(4, dtype=np.float32)
        self.mask = {"frequency_slots": np.zeros((100, 10), dtype=bool),
                     "power": np.zeros(10, dtype=bool)}
        self.mask["frequency_slots"][0, 0] = True
        self.mask["frequency_slots"][1, 1] = True
        self.mask["power"][0] = True
        self.mask["power"][1] = True

    def _agent(self, policy, critic):
        config = copy.deepcopy(self.config)
        config["network"].update(sac_policy_structure=policy, sac_critic_structure=critic)
        return build_agent(config, 4, self.dims)

    def test_complete_q_has_cross_interaction_and_true_parameter_count(self):
        agent = self._agent("conditional", "joint")
        q = agent.critic1(torch.zeros(1, 4))
        self.assertEqual(tuple(q.shape), (1, 100, 10, 10))
        with torch.no_grad():
            agent.critic1.joint_q.weight.zero_()
            agent.critic1.joint_q.bias.zero_()
            agent.critic1.joint_q.bias[0] = 1
            agent.critic1.joint_q.bias[10] = 2
            agent.critic1.joint_q.bias[100] = 3
            agent.critic1.joint_q.bias[110] = 8
        q = agent.critic1(torch.zeros(1, 4))
        self.assertEqual(float(q[0, 1, 1, 0] + q[0, 0, 0, 0] -
                               q[0, 1, 0, 0] - q[0, 0, 1, 0]), 4.0)
        description = model_description(agent.config, 4, self.dims)
        networks = (agent.actor, agent.critic1, agent.critic2, agent.target1, agent.target2)
        expected = sum(sum(p.numel() for p in net.parameters()) for net in networks) + 1
        self.assertEqual(description["parameter_count"], expected)
        self.assertNotEqual(sum(p.numel() for p in agent.actor.parameters()),
                            sum(p.numel() for p in agent.critic1.parameters()))

    def test_all_four_combinations_update_and_terminal_no_bootstrap(self):
        for policy in ("factorized", "conditional"):
            for critic in ("additive", "joint"):
                with self.subTest(policy=policy, critic=critic):
                    torch.manual_seed(31)
                    agent = self._agent(policy, critic)
                    for reward in (1., 3.):
                        agent.observe(self.state, (0, 0, 0), reward, self.state,
                                      True, action_mask=self.mask)
                    before = [p.detach().clone() for p in agent.actor.parameters()]
                    result = agent.update()
                    self.assertAlmostEqual(result["target_mean"], 2., places=6)
                    self.assertTrue(all(math.isfinite(float(value)) for value in result.values()))
                    self.assertTrue(any(not torch.equal(p, old) for p, old in
                                        zip(agent.actor.parameters(), before)))

    def test_joint_argmax_and_structure_recovery_guard(self):
        agent = self._agent("conditional", "joint")
        with torch.no_grad():
            agent.actor.frequency_slots.weight.zero_()
            agent.actor.frequency_slots.bias.fill_(-100.)
            agent.actor.frequency_slots.bias[0] = 0.
            agent.actor.frequency_slots.bias[11] = -0.1
            agent.actor.conditional_power.weight.zero_()
            agent.actor.conditional_power.bias.zero_()
            agent.actor.conditional_power.bias[0:10] = 0.
            agent.actor.conditional_power.bias[110] = 10.
        self.assertEqual(agent.act(self.state, deterministic=True, action_mask=self.mask), (1, 1, 0))
        for _ in range(20):
            action = agent.act(self.state, action_mask=self.mask)
            self.assertTrue(self.mask["frequency_slots"][action[:2]])
            self.assertTrue(self.mask["power"][action[2]])
        saved = agent.state_dict()
        self.assertEqual(saved["sac_policy_structure"], "conditional")
        other = self._agent("factorized", "joint")
        with self.assertRaisesRegex(ValueError, "structures differ"):
            other.load_state_dict(saved)

    def test_bootstrap_minimum_is_taken_for_each_complete_action(self):
        agent = self._agent("conditional", "joint")
        with torch.no_grad():
            for layer in (agent.actor.frequency_slots, agent.actor.conditional_power,
                          agent.target1.joint_q, agent.target2.joint_q):
                layer.weight.zero_()
                layer.bias.zero_()
            for index, first, second in ((0, 1., 5.), (1, 5., 1.),
                                         (110, 5., 1.), (111, 1., 5.)):
                agent.target1.joint_q.bias[index] = first
                agent.target2.joint_q.bias[index] = second
        agent.observe(self.state, (0, 0, 0), 2., self.state, False,
                      action_mask=self.mask, next_action_mask=self.mask)
        agent.observe(self.state, (0, 0, 0), 2., self.state, False,
                      action_mask=self.mask, next_action_mask=self.mask)
        expected = 2 + agent.train["gamma"] * (
            1 + float(agent.log_alpha.exp()) * math.log(4))
        self.assertAlmostEqual(agent.update()["target_mean"], expected, places=5)

    def test_j0_loads_historical_checkpoint_without_structure_fields(self):
        original = self._agent("factorized", "additive")
        checkpoint = original.state_dict()
        checkpoint.pop("sac_policy_structure")
        checkpoint.pop("sac_critic_structure")
        restored = self._agent("factorized", "additive")
        restored.load_state_dict(checkpoint)
        self.assertTrue(all(torch.equal(a, b) for a, b in
                            zip(original.actor.parameters(), restored.actor.parameters())))
        with self.assertRaisesRegex(ValueError, "structures differ"):
            self._agent("conditional", "additive").load_state_dict(checkpoint)

    def test_temperature_gradient_moves_toward_feasible_target(self):
        for target_entropy, expected_direction in ((0., -1), (math.log(10000), 1)):
            with self.subTest(target_entropy=target_entropy):
                config = copy.deepcopy(self.config)
                config["network"].update(sac_policy_structure="conditional",
                                         sac_critic_structure="joint")
                config["training"]["target_entropy"] = target_entropy
                agent = build_agent(config, 4, self.dims)
                with torch.no_grad():
                    agent.actor.frequency_slots.weight.zero_()
                    agent.actor.frequency_slots.bias.fill_(-100.)
                    agent.actor.frequency_slots.bias[0] = 20.
                    agent.actor.conditional_power.weight.zero_()
                    agent.actor.conditional_power.bias.zero_()
                    agent.actor.conditional_power.bias[0] = 20.
                for _ in range(2):
                    agent.observe(self.state, (0, 0, 0), 1., self.state,
                                  True, action_mask=self.mask)
                before = float(agent.log_alpha)
                metrics = agent.update()
                self.assertEqual(int(np.sign(float(agent.log_alpha) - before)), expected_direction)
                self.assertAlmostEqual(metrics["target_entropy"],
                                       target_entropy / math.log(10000) * math.log(4), places=5)


if __name__ == "__main__":
    unittest.main()
