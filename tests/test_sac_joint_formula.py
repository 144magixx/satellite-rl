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
                for formula, expected in [('legacy', 0.), ('corrected', 5.)]:
                    agent.formula_variant = formula
                    target = agent.calc_target(torch.zeros(1, 1), torch.zeros(1, 204), torch.zeros(1, 1))
                    self.assertAlmostEqual(target.item(), expected, places=5)

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
