from copy import deepcopy
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from satellite_rl.agents.action_masking import masked_policy
from satellite_rl.agents.workbench_algorithms import _joint_q
from satellite_rl.t0_width.agent import WidthAgent, build_config as width_config
from satellite_rl.t0_width.runner import seed_all
from satellite_rl.tsac_search27.agent import (SearchAgent, build_config, lp_masked_policy, lp_joint_q,
                                             lp_soft_value, lp_expectation, rng_sha)


def tiny_config(variant):
    config = build_config(variant, phase='correctness')
    config['training'].update(batch_size=2, warmup_steps=2, replay_capacity=16)
    return config


def fixture_replay(agent, terminal_only=False):
    mask = dict(frequency_slots=np.ones((100, 10), bool), power=np.ones(10, bool))
    mask['frequency_slots'][7:20, 5:] = False
    mask['power'][3] = False
    for i in range(2):
        state = np.linspace(-.2, .7, 208, dtype=np.float32) + i * .01
        terminal = terminal_only or i == 1
        agent.observe(state, (i, 1, 2), float(i - .5), state + .01, terminal, mask, None if terminal else mask)
    return state, mask


class SearchAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_B_matches_old_S10_parameters_and_one_real_update(self):
        old_config = width_config('S10', phase='correctness')
        old_config['training'].update(batch_size=2, warmup_steps=2, replay_capacity=16)
        seed_all(51001)
        old = WidthAgent(old_config)
        old_rng = rng_sha()
        seed_all(51001)
        new = SearchAgent(tiny_config('B'))
        self.assertEqual(old_rng, rng_sha())
        for name in ('actor', 'critic1', 'critic2', 'target1', 'target2'):
            for key, value in getattr(old, name).state_dict().items():
                torch.testing.assert_close(value, getattr(new, name).state_dict()[key], rtol=0, atol=0)
        fixture_replay(old)
        fixture_replay(new)
        seed_all(123)
        expected = old.update()
        seed_all(123)
        actual = new.update()
        self.assertEqual(expected, actual)
        for key, value in old.actor.state_dict().items():
            torch.testing.assert_close(value, new.actor.state_dict()[key], rtol=0, atol=0)

    def test_E_changes_only_temperature_and_engineering_seed_is_distinct(self):
        seed_all(51001)
        base = SearchAgent(tiny_config('B'))
        seed_all(51001)
        candidate = SearchAgent(tiny_config('E'))
        for name in ('actor', 'critic1', 'critic2'):
            for key, value in getattr(base, name).state_dict().items():
                torch.testing.assert_close(value, getattr(candidate, name).state_dict()[key], rtol=0, atol=0)
        self.assertAlmostEqual(candidate.log_alpha.exp().item(), .003, places=8)
        self.assertEqual(candidate.config['reward'], base.config['reward'])
        self.assertEqual(build_config('E', phase='engineering')['training']['seed'], 73027)
        with self.assertRaises(ValueError):
            build_config('B', total_steps=50000)

    def test_LP_initial_probability_Q_entropy_target_equivalence_and_rng(self):
        seed_all(51001)
        base = SearchAgent(tiny_config('B'))
        after_base = rng_sha()
        seed_all(51001)
        lp = SearchAgent(tiny_config('LP'))
        self.assertEqual(after_base, rng_sha())
        self.assertTrue(lp.initialization_rng['unchanged'])
        states = torch.linspace(-.5, .9, 208).repeat(2, 1)
        fl_mask = torch.ones((2, 100, 10), dtype=torch.bool)
        fl_mask[:, 5:50, 4:] = False
        p_mask = torch.ones((2, 10), dtype=torch.bool)
        p_mask[:, 7:] = False
        with torch.no_grad():
            old_parts = masked_policy(base.actor(states), fl_mask, p_mask)
            parts = lp_masked_policy(lp.actor(states), fl_mask, p_mask)
            old_joint = old_parts[0].unsqueeze(-1) * old_parts[1][:, None, None, :]
            joint = parts[0].unsqueeze(-1) * parts[1][:, None, :, :]
            torch.testing.assert_close(old_joint, joint, rtol=3e-6, atol=2e-8)
            torch.testing.assert_close(old_parts[4] + old_parts[5], parts[4] + parts[5], rtol=2e-6, atol=2e-6)
            for name in ('critic1', 'critic2', 'target1', 'target2'):
                # Merged Linear changes FP32 sum order; tolerance is arithmetic,
                # not a claim of bitwise identity or later optimizer equivalence.
                torch.testing.assert_close(_joint_q(getattr(base, name)(states)), lp_joint_q(getattr(lp, name)(states)), rtol=3e-6, atol=2e-6)
            old_q = torch.minimum(_joint_q(base.target1(states)), _joint_q(base.target2(states)))
            old_soft = (old_joint * old_q).sum((1, 2, 3)) + .01 * (old_parts[4] + old_parts[5])
            soft = lp_soft_value(parts, lp_joint_q(lp.target1(states)), lp_joint_q(lp.target2(states)), .01)
            torch.testing.assert_close(old_soft, soft, rtol=3e-6, atol=3e-7)
        for name, value in lp.parameter_counts().items():
            self.assertEqual(value - base.parameter_counts()[name], 10320)

    def test_exact_masked_distribution_entropy_and_gradient_vs_exhaustive(self):
        f = torch.tensor([[.3, -.7, .5]], dtype=torch.float64, requires_grad=True)
        lp = torch.tensor([[[.1, -.4, .6], [.8, -.9, .2]]], dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[[True, False], [True, True], [False, True]]])
        power = torch.tensor([[True, False, True]])
        parts = lp_masked_policy((f, lp), mask, power)
        valid = mask.unsqueeze(-1) & power[:, None, None, :]
        logits = f[:, :, None, None] + lp[:, None, :, :]
        direct_logs = logits.masked_fill(~valid, -torch.inf).flatten(1).log_softmax(-1).reshape_as(logits)
        direct = direct_logs.exp()
        joint = parts[0].unsqueeze(-1) * parts[1][:, None, :, :]
        torch.testing.assert_close(direct, joint, rtol=1e-13, atol=1e-14)
        safe_logs = torch.where(valid, direct_logs, torch.zeros_like(direct_logs))
        entropy = -(direct * safe_logs).sum((1, 2, 3))
        torch.testing.assert_close(entropy, parts[4] + parts[5], rtol=1e-13, atol=1e-14)
        q = torch.arange(18, dtype=torch.float64).reshape(1, 3, 2, 3)
        direct_value = (direct * q).sum((1, 2, 3)) + .01 * entropy
        value = lp_expectation(parts, q) + .01 * (parts[4] + parts[5])
        actual_grad = torch.autograd.grad(value.sum(), (f, lp), retain_graph=True)
        expected_grad = torch.autograd.grad(direct_value.sum(), (f, lp))
        for a, b in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-13)
        with self.assertRaises(ValueError):
            lp_masked_policy((f, lp), torch.zeros_like(mask), power)

    def test_nonseparable_slot_power_and_complete_Q_min_order(self):
        interaction = torch.tensor([[[0., 0.], [0., 1.]]])
        self.assertEqual(float(interaction[0, 0, 0] + interaction[0, 1, 1] - interaction[0, 0, 1] - interaction[0, 1, 0]), 1.)
        first = (torch.tensor([[0., 10.]]), torch.full((1, 2, 2), 10.))
        second = (torch.tensor([[10., 0.]]), torch.zeros((1, 2, 2)))
        correct = torch.minimum(lp_joint_q(first), lp_joint_q(second))
        wrong = lp_joint_q((torch.minimum(first[0], second[0]), torch.minimum(first[1], second[1])))
        self.assertFalse(torch.equal(correct, wrong))
        parts = lp_masked_policy((torch.zeros((1, 2)), interaction), torch.ones((1, 2, 2), dtype=torch.bool), torch.ones((1, 2), dtype=torch.bool))
        expected = (parts[0].unsqueeze(-1) * parts[1][:, None, :, :] * correct).sum((1, 2, 3)) + .01 * (parts[4] + parts[5])
        torch.testing.assert_close(expected, lp_soft_value(parts, lp_joint_q(first), lp_joint_q(second), .01))

    def test_LP_sampling_has_two_draws_and_uses_correct_conditional(self):
        seed_all(51001)
        lp = SearchAgent(tiny_config('LP'))
        state = np.zeros(208, np.float32)
        mask = dict(frequency_slots=np.ones((100, 10), bool), power=np.ones(10, bool))
        calls = []
        class FixedCategorical:
            def __init__(self, probs):
                calls.append(probs.detach().clone())
            def sample(self):
                return torch.tensor([23 if len(calls) == 1 else 4])
        with patch('torch.distributions.Categorical', FixedCategorical):
            action = lp.act(state, action_mask=mask)
        self.assertEqual(action, (2, 3, 4))
        self.assertEqual(len(calls), 2)
        with torch.no_grad():
            parts = lp_masked_policy(lp.actor(torch.tensor(state)[None]), torch.tensor(mask['frequency_slots'])[None], torch.tensor(mask['power'])[None])
        torch.testing.assert_close(calls[0], parts[0].flatten(1))
        torch.testing.assert_close(calls[1], parts[1][:, 3, :])

    def test_terminal_no_bootstrap_real_update_and_target_interpolation(self):
        seed_all(51001)
        lp = SearchAgent(tiny_config('LP'))
        fixture_replay(lp, terminal_only=True)
        before = {k: v.clone() for k, v in lp.target1.state_dict().items()}
        actor_before = lp.actor.slot_power.weight.detach().clone()
        with patch.object(lp.target1, 'forward', side_effect=AssertionError('terminal target must not run')):
            diagnostics = lp.update()
        self.assertEqual(lp.update_step, 1)
        self.assertEqual(diagnostics['target_mean'], 0.)
        self.assertFalse(torch.equal(actor_before, lp.actor.slot_power.weight))
        self.assertTrue(all(np.isfinite(v) for v in diagnostics.values() if isinstance(v, (float, int))))
        for key, value in lp.target1.state_dict().items():
            expected = before[key].lerp(lp.critic1.state_dict()[key], lp.train['tau'])
            torch.testing.assert_close(value, expected, rtol=0, atol=0)

    def test_full_recovery_next_action_update_and_strict_variant_rejection(self):
        for variant in ('B', 'E', 'LP'):
            seed_all(51001)
            agent = SearchAgent(tiny_config(variant))
            state, mask = fixture_replay(agent)
            agent.update()
            buffer = io.BytesIO()
            torch.save(agent.state_dict(), buffer)
            buffer.seek(0)
            saved = torch.load(buffer, map_location='cpu', weights_only=False)
            expected_action = agent.act(state, action_mask=mask)
            expected_update = agent.update()
            restored = SearchAgent(tiny_config(variant))
            restored.load_state_dict(saved)
            self.assertEqual(restored.act(state, action_mask=mask), expected_action)
            self.assertEqual(restored.update(), expected_update)
            for name in ('actor', 'critic1', 'critic2', 'target1', 'target2'):
                for key, value in getattr(agent, name).state_dict().items():
                    torch.testing.assert_close(value, getattr(restored, name).state_dict()[key], rtol=0, atol=0)
            wrong = SearchAgent(tiny_config('LP' if variant != 'LP' else 'B'))
            with self.assertRaisesRegex(ValueError, 'schema'):
                wrong.load_state_dict(saved)


if __name__ == '__main__':
    unittest.main()
