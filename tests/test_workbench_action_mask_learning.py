"""Exact masked SAC probabilities and real CPU update/recovery checks."""

import copy
import math
import unittest

import numpy as np
import torch

from satellite_rl.agents.action_masking import (copy_action_mask, entropy_ratio,
                                                feasible_target_entropy, mask_batch,
                                                masked_policy)
from satellite_rl.agents.workbench_algorithms import build_agent
from satellite_rl.workbench.config import default_config


class MaskedPolicyTest(unittest.TestCase):
    def test_small_enumeration_entropy_q_and_gradients(self):
        frequency = torch.tensor([[0.2, -0.4]], dtype=torch.float64, requires_grad=True)
        slots = torch.tensor([[0.7, -0.1, 0.3]], dtype=torch.float64, requires_grad=True)
        power = torch.tensor([[0.1, -0.8]], dtype=torch.float64, requires_grad=True)
        legal_fl = torch.tensor([[[True, False, True], [False, True, False]]])
        legal_power = torch.tensor([[True, False]])
        (fl, pp, log_fl, log_p, hfl, hp, hf, hslots) = masked_policy(
            (frequency, slots, power), legal_fl, legal_power)
        legal = [(0, 0), (0, 2), (1, 1)]
        weights = np.array([math.exp(float(frequency[0, f] + slots[0, l])) for f, l in legal])
        expected = weights / weights.sum()
        for (f, l), probability in zip(legal, expected):
            self.assertAlmostEqual(float(fl[0, f, l]), probability, places=12)
            self.assertAlmostEqual(float(log_fl[0, f, l]), math.log(probability), places=12)
        self.assertEqual(float(fl[0, 0, 1]), 0.0)
        self.assertEqual(float(pp[0, 1]), 0.0)
        self.assertEqual(float(log_fl[0, 0, 1]), 0.0)
        self.assertEqual(float(log_p[0, 1]), 0.0)
        self.assertAlmostEqual(float(hfl), -sum(expected * np.log(expected)), places=12)
        self.assertEqual(float(hp), 0.0)
        self.assertAlmostEqual(float(hf + hslots), float(hfl), places=12)
        q = torch.tensor([[[[2., 100.], [100., 100.], [5., 100.]],
                           [[100., 100.], [-1., 100.], [100., 100.]]]], dtype=torch.float64)
        value = torch.einsum("bfl,bp,bflp->b", fl, pp, q)
        self.assertAlmostEqual(float(value), sum(expected * [2, 5, -1]), places=12)
        objective = value + 0.3 * (hfl + hp)
        objective.backward()
        for head in (frequency, slots, power):
            self.assertTrue(torch.isfinite(head.grad).all())
        self.assertGreater(float(frequency.grad.abs().sum()), 0)
        self.assertGreater(float(slots.grad.abs().sum()), 0)
        self.assertEqual(float(power.grad.abs().sum()), 0.0)

    def test_mask_validation_and_dynamic_entropy(self):
        dims = (4, 3, 2)
        fl = np.zeros((4, 3), dtype=np.bool_)
        fl[3, 1] = True
        fl[0, 0] = True
        power = np.array([True, False])
        mask = copy_action_mask({"frequency_slots": fl, "power": power}, dims)
        fl[3, 1] = False
        self.assertTrue(mask["frequency_slots"][3, 1])
        batched_fl, batched_power = mask_batch([mask], dims, "cpu")
        ratio = entropy_ratio(math.log(24) / 2, dims)
        self.assertAlmostEqual(float(feasible_target_entropy(batched_fl, batched_power, ratio)),
                               math.log(2) / 2, places=6)
        with self.assertRaises(ValueError):
            entropy_ratio(math.log(24) + 0.01, dims)
        with self.assertRaises(ValueError):
            copy_action_mask({"frequency_slots": np.zeros((4, 3), dtype=bool),
                              "power": power}, dims)
        with self.assertRaises(ValueError):
            copy_action_mask({"frequency_slots": mask["frequency_slots"],
                              "power": np.array([1, 0])}, dims)


class MaskedSacTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2101)
        self.config = default_config()
        self.config["algorithm"] = "sac"
        self.config["environment"]["action_mask"] = "joint_feasible"
        self.config["network"]["hidden_sizes"] = [8]
        self.config["training"].update(batch_size=2, warmup_steps=0, target_entropy=1.5,
                                       gradient_clip=1)
        self.dims = (4, 3, 2)
        self.state = np.zeros(4, dtype=np.float32)
        self.mask = {"frequency_slots": np.zeros((4, 3), dtype=np.bool_),
                     "power": np.array([False, True], dtype=np.bool_)}
        self.mask["frequency_slots"][3, 1] = True
        self.mask["frequency_slots"][0, 0] = True

    def test_act_and_observe_fail_closed_and_replay_copies_masks(self):
        agent = build_agent(self.config, 4, self.dims, "cpu")
        with self.assertRaises(ValueError):
            agent.act(self.state)
        self.assertEqual(agent.act(self.state, deterministic=True, action_mask=self.mask)[2], 1)
        for _ in range(30):
            action = agent.act(self.state, action_mask=self.mask)
            self.assertIn(action[:2], ((3, 1), (0, 0)))
            self.assertEqual(action[2], 1)
        with self.assertRaises(ValueError):
            agent.observe(self.state, (3, 2, 1), 1, self.state, False,
                          action_mask=self.mask, next_action_mask=self.mask)
        with self.assertRaises(ValueError):
            agent.observe(self.state, (3, 1, 1), 1, self.state, False, action_mask=self.mask)
        with self.assertRaises(ValueError):
            agent.observe(self.state, (3, 1, 1), 1, self.state, True,
                          action_mask=self.mask, next_action_mask=self.mask)
        agent.observe(self.state, (3, 1, 1), 1, self.state, True, action_mask=self.mask)
        self.mask["frequency_slots"][3, 1] = False
        self.assertTrue(agent.replay.rows[0][5]["frequency_slots"][3, 1])
        self.assertIsNone(agent.replay.rows[0][6])

    def test_real_update_terminal_zero_bootstrap_and_checkpoint_continuation(self):
        agent = build_agent(self.config, 4, self.dims, "cpu")
        for reward in (1., 2.):
            agent.observe(self.state, (3, 1, 1), reward, self.state + 0.1,
                          True, action_mask=self.mask)
        checkpoint = copy.deepcopy(agent.state_dict())
        self.assertEqual(checkpoint["policy_probability_mode"], "eval_v1")
        before = [p.detach().clone() for p in agent.actor.parameters()]
        torch_rng = torch.get_rng_state()
        action_a = agent.act(self.state, action_mask=self.mask)
        metrics_a = agent.update()
        self.assertAlmostEqual(metrics_a["target_mean"], 1.5, places=6)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, agent.actor.parameters())))
        self.assertTrue(all(math.isfinite(value) for value in metrics_a.values()))
        self.assertAlmostEqual(metrics_a["entropy"], metrics_a["entropy_frequency"] +
                               metrics_a["entropy_slots"] + metrics_a["entropy_power"], places=5)
        restored = build_agent(self.config, 4, self.dims, "cpu")
        restored.load_state_dict(checkpoint)
        torch.set_rng_state(torch_rng)
        action_b = restored.act(self.state, action_mask=self.mask)
        metrics_b = restored.update()
        self.assertEqual(action_a, action_b)
        self.assertEqual(metrics_a, metrics_b)
        self.assertTrue(all(torch.equal(a, b) for a, b in
                            zip(agent.actor.parameters(), restored.actor.parameters())))

    def test_nonterminal_target_uses_next_mask(self):
        agent = build_agent(self.config, 4, self.dims, "cpu")
        next_mask = {"frequency_slots": np.zeros((4, 3), dtype=np.bool_),
                     "power": np.array([True, False], dtype=np.bool_)}
        next_mask["frequency_slots"][1, 2] = True
        next_state = self.state + 0.3
        with torch.no_grad():
            ns = torch.as_tensor(np.stack([next_state, next_state]))
            next_fl, next_power = mask_batch([next_mask, next_mask], self.dims, "cpu")
            _, _, _, _, entropy_fl, entropy_power, _, _ = masked_policy(
                agent.actor(ns), next_fl, next_power)
            q1 = agent.target1(ns)
            q2 = agent.target2(ns)
            selected = torch.minimum(
                q1[0][0, 1] + q1[1][0, 2] + q1[2][0, 0],
                q2[0][0, 1] + q2[1][0, 2] + q2[2][0, 0])
            expected_target = (1 + self.config["training"]["gamma"] *
                               (selected + agent.log_alpha.exp() * (entropy_fl[0] + entropy_power[0])))
        for _ in range(2):
            agent.observe(self.state, (3, 1, 1), 1, next_state, False,
                          action_mask=self.mask, next_action_mask=next_mask)
        self.assertAlmostEqual(agent.update()["target_mean"], float(expected_target), places=5)

    def test_disabled_mode_keeps_legacy_replay_shape(self):
        config = copy.deepcopy(self.config)
        config["environment"].pop("action_mask")
        agent = build_agent(config, 4, self.dims, "cpu")
        self.assertEqual(len(agent.act(self.state)), 3)
        agent.observe(self.state, (3, 2, 1), 0., self.state, True)
        self.assertEqual(len(agent.replay.rows[0]), 5)

    def test_dropout_actor_probabilities_match_sampling_in_both_modes(self):
        for masked in (False, True):
            with self.subTest(masked=masked):
                config = copy.deepcopy(self.config)
                config["network"]["dropout"] = 0.4
                if not masked:
                    config["environment"].pop("action_mask")
                agent = build_agent(config, 4, self.dims, "cpu")
                state = torch.as_tensor(self.state).reshape(1, -1)
                first = agent._sac_actor_logits(state)
                second = agent._sac_actor_logits(state)
                self.assertTrue(agent.actor.training)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
                agent.act(self.state, deterministic=True,
                          action_mask=self.mask if masked else None)
                for _ in range(2):
                    agent.observe(self.state, (3, 1, 1), 1., self.state, True,
                                  action_mask=self.mask if masked else None)
                metrics = agent.update()
                self.assertTrue(math.isfinite(metrics["actor_grad_norm"]))
                self.assertGreater(metrics["actor_grad_norm"], 0)

    def test_checkpoint_mode_guard_runs_before_parameter_mutation(self):
        config = copy.deepcopy(self.config)
        config["environment"].pop("action_mask")
        config["network"]["dropout"] = 0.4
        agent = build_agent(config, 4, self.dims, "cpu")
        before = [p.detach().clone() for p in agent.actor.parameters()]
        historical = copy.deepcopy(agent.state_dict())
        historical.pop("policy_probability_mode")
        next(iter(historical["networks"]["actor"].values())).fill_(99)
        with self.assertRaisesRegex(ValueError, "original frozen source"):
            agent.load_state_dict(historical)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, agent.actor.parameters())))
        future = copy.deepcopy(agent.state_dict())
        future["policy_probability_mode"] = "future_v2"
        with self.assertRaisesRegex(ValueError, "Unsupported SAC policy_probability_mode"):
            agent.load_state_dict(future)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, agent.actor.parameters())))
        zero_dropout = copy.deepcopy(config)
        zero_dropout["network"]["dropout"] = 0
        restored = build_agent(zero_dropout, 4, self.dims, "cpu")
        restored.load_state_dict(historical)
        self.assertEqual(float(next(restored.actor.parameters()).flatten()[0]), 99.)

    def test_masked_checkpoint_requires_marker_and_masked_replay(self):
        agent = build_agent(self.config, 4, self.dims, "cpu")
        agent.observe(self.state, (3, 1, 1), 1., self.state, True, action_mask=self.mask)
        before = [p.detach().clone() for p in agent.actor.parameters()]
        missing_mode = copy.deepcopy(agent.state_dict())
        missing_mode.pop("policy_probability_mode")
        with self.assertRaisesRegex(ValueError, "lacks policy_probability_mode"):
            agent.load_state_dict(missing_mode)
        missing_mask = copy.deepcopy(agent.state_dict())
        missing_mask["replay"]["rows"][0] = missing_mask["replay"]["rows"][0][:5]
        with self.assertRaisesRegex(ValueError, "lacks current/next action masks"):
            agent.load_state_dict(missing_mask)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, agent.actor.parameters())))


if __name__ == "__main__":
    unittest.main()
