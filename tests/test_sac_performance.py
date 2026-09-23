"""SAC 提速后仍需保留策略梯度、模型模式、参数接口和输入检查。"""
import importlib
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


VARIANTS = (
    ("sac_transformer", 128),
    ("sac_cnn", 128),
    ("sac_mlp", 128),
    ("sac_cnn_matched", 128),
    ("sac_mlp_matched", 512),
)
NETWORKS = ("actor", "critic_1", "critic_2", "target_critic_1", "target_critic_2")


def build_agent(module, hidden):
    return module.SAC(204, hidden, 100, 10, 10, 1e-5, 1e-4, 1e-4,
                      -6.9, 0.005, 0.99, torch.device("cpu"))


class SACPerformanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="satellite_sac_perf_")
        self.addCleanup(temporary.cleanup)
        override = mock.patch.dict(os.environ, {"SATELLITE_RL_OUTPUT_ROOT": temporary.name})
        override.start()
        self.addCleanup(override.stop)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.manual_seed(20260924)
        # 固定、非退化的小 batch，用实际 backward/optimizer 检查梯度路径。
        states = np.linspace(0.01, 0.99, 8 * 204, dtype=np.float32).reshape(8, 204)
        self.batch = {
            "states": states,
            "next_states": states[:, ::-1].copy(),
            "actions": [[i * 11, i % 10, (i + 2) % 10] for i in range(8)],
            "rewards": [1.0, -0.2, 0.7, 2.0, -0.5, 0.4, 1.2, 0.3],
            "dones": [False] * 7 + [True],
        }

    def test_update_retains_learning_and_only_required_backward_paths(self):
        for name, hidden in VARIANTS:
            with self.subTest(variant=name):
                agent = build_agent(importlib.import_module("satellite_rl.agents." + name), hidden)
                networks = {role: getattr(agent, role) for role in NETWORKS}
                previous = {role: [p.detach().clone() for p in net.parameters()]
                            for role, net in networks.items()}
                previous_alpha = agent.log_alpha.detach().clone()
                backward_calls = dict.fromkeys(NETWORKS + ("alpha",), 0)

                def count(role):
                    def hook(gradient):
                        backward_calls[role] += 1
                    return hook

                handles = [next(net.parameters()).register_hook(count(role))
                           for role, net in networks.items()]
                handles.append(agent.log_alpha.register_hook(count("alpha")))
                try:
                    agent.update(self.batch)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(backward_calls, {
                    "actor": 1, "critic_1": 1, "critic_2": 1,
                    "target_critic_1": 0, "target_critic_2": 0, "alpha": 1,
                })
                for role, net in networks.items():
                    self.assertTrue(net.training)
                    self.assertTrue(all(p.requires_grad for p in net.parameters()))
                    self.assertTrue(any(not torch.equal(old, new)
                                        for old, new in zip(previous[role], net.parameters())), role)
                self.assertFalse(torch.equal(previous_alpha, agent.log_alpha))

    def test_sampling_and_targets_do_not_build_graphs_or_change_training_mode(self):
        for name, hidden in VARIANTS:
            with self.subTest(variant=name):
                agent = build_agent(importlib.import_module("satellite_rl.agents." + name), hidden)
                observed = []

                def capture(net, inputs, outputs):
                    observed.append((net.training, torch.is_grad_enabled(),
                                     any(tensor.requires_grad for tensor in outputs)))

                handle = agent.actor.register_forward_hook(capture)
                try:
                    with torch.enable_grad():
                        action = agent.take_action(self.batch["states"][0])
                        self.assertTrue(torch.is_grad_enabled())
                        target = agent.calc_target(
                            torch.tensor(self.batch["rewards"]).unsqueeze(1),
                            torch.tensor(self.batch["next_states"]),
                            torch.tensor(self.batch["dones"], dtype=torch.float32).unsqueeze(1),
                        )
                        self.assertTrue(torch.is_grad_enabled())
                    self.assertEqual(observed, [(True, False, False), (True, False, False)])
                    self.assertFalse(target.requires_grad)
                    self.assertIsNone(target.grad_fn)
                    self.assertEqual(len(action), 3)
                    for value, limit in zip(action, (100, 10, 10)):
                        self.assertIsInstance(value, int)
                        self.assertTrue(0 <= value < limit)
                    agent.actor.eval()
                    agent.take_action(self.batch["states"][0])
                    self.assertEqual(observed[-1], (False, False, False))
                finally:
                    handle.remove()

    def test_invalid_probabilities_still_raise(self):
        for name, hidden in VARIANTS:
            with self.subTest(variant=name):
                agent = build_agent(importlib.import_module("satellite_rl.agents." + name), hidden)
                with self.assertRaises(ValueError):
                    agent.take_action(np.full(204, np.nan, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
