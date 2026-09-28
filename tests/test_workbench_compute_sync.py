"""Finite checks retain all parameter, gradient and loss safety gates."""

import copy
import unittest

import torch
from torch import nn

from satellite_rl.agents.workbench_algorithms import (
    _diagnostics, _finite_loss, _grad_norm, ensure_finite_modules, finite_tensor_check,
)
from satellite_rl.agents.workbench_networks import MultiHeadNetwork


def tiny_network(kind, device):
    config = {"network": {"kind": kind, "transformer_layout": "scalar_tokens",
                          "hidden_sizes": [8], "activation": "relu", "dropout": 0,
                          "cnn_channels": [4], "kernel_size": 3,
                          "embedding_dim": 8, "num_heads": 2, "num_layers": 1,
                          "feedforward_dim": 16}}
    return MultiHeadNetwork(config, 4, [3, 2, 2]).to(device)


class ComputeSynchronizationTests(unittest.TestCase):
    def check_device(self, device):
        for kind in ("mlp", "cnn", "transformer"):
            with self.subTest(device=device, network=kind):
                module = tiny_network(kind, device)
                others = {"actor": module, "critic1": copy.deepcopy(module),
                          "critic2": copy.deepcopy(module)}
                alpha = torch.tensor(0.0, device=device)
                self.assertTrue(finite_tensor_check(
                    parameter for network in others.values() for parameter in network.parameters()))
                ensure_finite_modules(others, alpha)
                for bad in (float("nan"), float("inf"), float("-inf")):
                    with self.subTest(value=bad):
                        parameter = next(others["critic2"].parameters())
                        original = parameter.detach().clone()
                        parameter.data.view(-1)[0] = bad
                        self.assertFalse(finite_tensor_check(
                            p for network in others.values() for p in network.parameters()))
                        with self.assertRaisesRegex(FloatingPointError, "critic2 parameter"):
                            ensure_finite_modules(others, alpha)
                        parameter.data.copy_(original)
                        alpha.fill_(bad)
                        with self.assertRaisesRegex(FloatingPointError, "temperature"):
                            ensure_finite_modules(others, alpha)
                        alpha.zero_()
                        for p in module.parameters():
                            p.grad = torch.ones_like(p)
                        target = next(module.parameters())
                        target.grad.view(-1)[0] = bad
                        with self.assertRaisesRegex(FloatingPointError, "gradient before optimizer"):
                            _grad_norm(module, 1.0)
                        self.assertTrue(torch.isinf(target.grad).any() or torch.isnan(target.grad).any())
                for p in module.parameters():
                    p.grad = torch.ones_like(p)
                actual = _grad_norm(module, 1.0)
                self.assertGreater(actual, 1.0)
                clipped = torch.sqrt(sum((p.grad.square().sum() for p in module.parameters())))
                self.assertLessEqual(float(clipped), 1.000001)

    def test_cpu_finite_checks_and_clipping(self):
        self.check_device("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "DQN CUDA unavailable")
    def test_gpu_finite_checks_and_clipping(self):
        self.check_device("cuda")

    def test_loss_and_diagnostic_semantics(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(FloatingPointError, "learning loss"):
                _finite_loss(torch.tensor(bad))
        values = {"loss": torch.tensor(0.125), "alpha": torch.tensor(0.01),
                  "gradient": 2.5, "target_updated": 1}
        self.assertEqual(_diagnostics(values),
                         {"loss": 0.125, "alpha": float(torch.tensor(0.01)),
                          "gradient": 2.5, "target_updated": 1})
        self.assertEqual(values["loss"].item(), 0.125)


if __name__ == "__main__":
    unittest.main()
