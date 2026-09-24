"""完整动作双 Q 与温度更新方向的独立小例子。"""
import importlib
import itertools
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from satellite_rl.agents.sac_math import expected_min_joint_q
from satellite_rl.training.experiment_runner import TARGET_ENTROPY_BY_FORMULA, _config, build_parser


class JointFormulaTests(unittest.TestCase):
    def test_joint_min_matches_enumeration_and_policy_gradient(self):
        torch.manual_seed(42)
        logits = [torch.randn(2, n, dtype=torch.float64, requires_grad=True) for n in (3, 2, 2)]
        probabilities = [x.softmax(-1) for x in logits]
        q1 = [torch.randn(2, n, dtype=torch.float64) for n in (3, 2, 2)]
        q2 = [torch.randn(2, n, dtype=torch.float64) for n in (3, 2, 2)]
        actual = expected_min_joint_q(probabilities, q1, q2, frequency_chunk=2)
        expected = torch.zeros(2, dtype=torch.float64)
        for f, n, p in itertools.product(range(3), range(2), range(2)):
            qa = q1[0][:, f] + q1[1][:, n] + q1[2][:, p]
            qb = q2[0][:, f] + q2[1][:, n] + q2[2][:, p]
            expected = expected + probabilities[0][:, f] * probabilities[1][:, n] * probabilities[2][:, p] * torch.minimum(qa, qb)
        torch.testing.assert_close(actual[:, 0], expected, rtol=1e-12, atol=1e-12)
        gradient = torch.autograd.grad(actual.sum(), logits, retain_graph=True)
        reference = torch.autograd.grad(expected.sum(), logits)
        for a, b in zip(gradient, reference):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_all_variants_compare_complete_action_not_separate_heads(self):
        class Outputs:
            def __init__(self, values): self.values = values
            def __call__(self, state):
                return tuple(torch.full((len(state), n), v) for n, v in zip((100, 10, 10), self.values))
        for name in ('sac_transformer', 'sac_cnn', 'sac_mlp', 'sac_cnn_matched', 'sac_mlp_matched'):
            with self.subTest(model=name):
                cls = importlib.import_module('satellite_rl.agents.' + name).SAC
                agent = cls.__new__(cls)
                agent.actor = Outputs((.01, .1, .1))
                agent.target_critic_1 = Outputs((5., 0., 0.))
                agent.target_critic_2 = Outputs((0., 5., 0.))
                agent.log_alpha = torch.tensor(float('-inf'))
                agent.gamma = 1.
                for formula, expected in [('legacy', 0.), ('corrected', 5.),
                                          ('negative_joint', 5.), ('positive_per_head', 0.)]:
                    agent.formula_variant = formula
                    target = agent.calc_target(torch.zeros(1, 1), torch.zeros(1, 204), torch.zeros(1, 1))
                    self.assertAlmostEqual(target.item(), expected, places=5)

    def test_all_four_runner_formulas_record_the_requested_entropy_and_q_rule(self):
        expectations = {
            'legacy': (-6.9, None), 'corrected': (6.9, None),
            'negative_joint': (-6.9, 'joint'),
            'positive_per_head': (6.9, 'per_head'),
        }
        parser = build_parser()
        for formula, (target, q_rule) in expectations.items():
            with self.subTest(formula=formula):
                args = parser.parse_args(['--model', 'transformer', '--formula', formula,
                                          '--seed', '51001', '--run-id', 'probe'])
                args.experiment_id = 'probe'
                config = _config(args, {})
                self.assertEqual(config['target_entropy'], target)
                self.assertEqual(TARGET_ENTROPY_BY_FORMULA[formula], target)
                self.assertEqual(config.get('q_min_variant'), q_rule)
                self.assertEqual(config['formula'], formula)

    def test_real_update_uses_each_q_path_and_updates_parameters(self):
        from satellite_rl.agents import sac_mlp
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)
        states = np.random.default_rng(51001).normal(size=(8, 204))
        transitions = dict(states=states, next_states=states.copy(),
                           actions=[[0, 0, 0]] * 8, rewards=[0.] * 8, dones=[0.] * 8)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'SATELLITE_RL_OUTPUT_ROOT': directory}):
            for formula in ('legacy', 'corrected', 'negative_joint', 'positive_per_head'):
                with self.subTest(formula=formula):
                    torch.manual_seed(51001)
                    agent = sac_mlp.SAC(204, 128, 100, 10, 10, 1e-5, 1e-4, 1e-4,
                                        TARGET_ENTROPY_BY_FORMULA[formula], .005, .99,
                                        torch.device('cpu'), formula_variant=formula)
                    before_actor = next(agent.actor.parameters()).detach().clone()
                    before_critic = next(agent.critic_1.parameters()).detach().clone()
                    with patch.object(sac_mlp, 'expected_min_joint_q',
                                      wraps=expected_min_joint_q) as joint_q:
                        agent.update(transitions)
                    self.assertEqual(joint_q.call_count, 2 if formula in ('corrected', 'negative_joint') else 0)
                    self.assertFalse(torch.equal(next(agent.actor.parameters()), before_actor))
                    self.assertFalse(torch.equal(next(agent.critic_1.parameters()), before_critic))

    def test_real_temperature_update_moves_towards_positive_target(self):
        from satellite_rl.agents.sac_mlp import SAC
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'SATELLITE_RL_OUTPUT_ROOT': directory}):
            for concentrated in (False, True):
                agent = SAC(204, 128, 100, 10, 10, 1e-5, 1e-4, 1e-4, 6.9, .005, .99, torch.device('cpu'))
                with torch.no_grad():
                    for p in agent.actor.parameters(): p.zero_()
                    if concentrated:
                        for head in (agent.actor.freq_head, agent.actor.slots_head, agent.actor.power_head):
                            head.bias[0] = 80
                before = agent.log_alpha.item()
                agent.update(dict(states=np.zeros((8, 204)), next_states=np.zeros((8, 204)),
                                  actions=[[0, 0, 0]] * 8, rewards=[0.] * 8, dones=[0.] * 8))
                if concentrated: self.assertGreater(agent.log_alpha.item(), before)
                else: self.assertLess(agent.log_alpha.item(), before)


if __name__ == '__main__': unittest.main()
