"""参数规模对照变体：结构、更新、保存和训练入口验证。"""

import ast
from contextlib import redirect_stdout
import importlib
import inspect
import io
import os
from pathlib import Path
import runpy
import tempfile
import textwrap
import unittest
from unittest import mock

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from satellite_rl.training import sac_runner


CASES = (
    ("sac_mlp_matched", 512, 429176),
    ("sac_cnn_matched", 128, 447176),
)


def parameter_count(network):
    return sum(parameter.numel() for parameter in network.parameters())


def build_agent(module, hidden):
    return module.SAC(
        n_states=204, n_hiddens=hidden,
        freq_dim=100, slots_dim=10, power_dim=10,
        actor_lr=1e-5, critic_lr=1e-4, alpha_lr=1e-4,
        target_entropy=-6.9, tau=0.005, gamma=0.99,
        device=torch.device("cpu"),
    )


class MatchedSACTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="satellite_rl_matched_")
        self.addCleanup(self.temporary.cleanup)
        self.output_root = Path(self.temporary.name)
        override = mock.patch.dict(os.environ, {"SATELLITE_RL_OUTPUT_ROOT": self.temporary.name})
        override.start()
        self.addCleanup(override.stop)
        self.addCleanup(plt.close, "all")
        torch.manual_seed(42)

    def test_network_sizes_and_outputs(self):
        for name, hidden, expected in CASES:
            with self.subTest(variant=name):
                module = importlib.import_module("satellite_rl.agents." + name)
                actor = module.PolicyNet(204, hidden, 100, 10, 10)
                critic = module.ValueNet(204, hidden, 100, 10, 10)
                self.assertEqual(parameter_count(actor), expected)
                self.assertEqual(parameter_count(critic), expected)
                inputs = torch.randn(3, 204)
                with torch.no_grad():
                    probabilities = actor(inputs)
                    q_values = critic(inputs)
                for probability, q_value, width in zip(probabilities, q_values, (100, 10, 10)):
                    self.assertEqual(tuple(probability.shape), (3, width))
                    self.assertEqual(tuple(q_value.shape), (3, width))
                    self.assertTrue(torch.isfinite(probability).all())
                    self.assertTrue(torch.isfinite(q_value).all())
                    self.assertTrue((probability >= 0).all())
                    torch.testing.assert_close(probability.sum(dim=1), torch.ones(3))

    def test_mlp_critic_has_nonlinearity(self):
        from satellite_rl.agents.sac_mlp_matched import ValueNet

        critic = ValueNet(204, 512, 100, 10, 10)
        with torch.no_grad():
            for parameter in critic.parameters():
                parameter.zero_()
            critic.fc1.weight[0, 0] = 1
            critic.fc2.weight[0, 0] = 1
            critic.q_freq.weight[0, 0] = 1
            inputs = torch.zeros(3, 204)
            inputs[:, 0] = torch.tensor([-1.0, 0.0, 1.0])
            values = critic(inputs)[0][:, 0]
        torch.testing.assert_close(values, torch.tensor([0.0, 0.0, 1.0]))

    def test_update_and_saved_parameters(self):
        rng = np.random.default_rng(42)
        transitions = {
            "states": rng.normal(size=(8, 204)).astype(np.float32),
            "next_states": rng.normal(size=(8, 204)).astype(np.float32),
            "actions": np.column_stack([
                rng.integers(0, 100, 8), rng.integers(0, 10, 8), rng.integers(0, 10, 8),
            ]),
            "rewards": rng.normal(size=8).astype(np.float32),
            "dones": np.array([0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32),
        }
        for name, hidden, expected in CASES:
            with self.subTest(variant=name):
                module = importlib.import_module("satellite_rl.agents." + name)
                agent = build_agent(module, hidden)
                networks = [agent.actor, agent.critic_1, agent.critic_2,
                            agent.target_critic_1, agent.target_critic_2]
                self.assertEqual(sum(parameter_count(net) for net in networks) + agent.log_alpha.numel(),
                                 5 * expected + 1)
                before = [[p.detach().clone() for p in net.parameters()] for net in networks]
                agent.update(transitions)
                for network, previous in zip(networks, before):
                    self.assertTrue(all(torch.isfinite(p).all() for p in network.parameters()))
                    self.assertTrue(any(not torch.equal(old, new) for old, new in zip(previous, network.parameters())))
                self.assertTrue(torch.isfinite(agent.log_alpha))
                self.assertEqual(Path(agent.log_file).parent,
                                 self.output_root / "outputs" / "logs" / name)
                with redirect_stdout(io.StringIO()):
                    agent.save_model()
                model_dir = self.output_root / "models" / "runs" / name
                files = list(model_dir.glob("*.pth"))
                self.assertEqual(len(files), 6)
                for role in ("actor", "critic_1", "critic_2", "target_critic_1", "target_critic_2"):
                    path = next(model_dir.glob(f"{name}_{role}_*.pth"))
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    for key, tensor in getattr(agent, role).state_dict().items():
                        torch.testing.assert_close(saved[key], tensor, rtol=0, atol=0)
                actor_path = next(model_dir.glob(f"{name}_actor_*.pth"))
                restored = module.PolicyNet(204, hidden, 100, 10, 10)
                restored.load_state_dict(torch.load(actor_path, map_location="cpu", weights_only=True), strict=True)
                with torch.no_grad():
                    probe = torch.zeros(1, 204)
                    for actual, expected_head in zip(restored(probe), agent.actor(probe)):
                        torch.testing.assert_close(actual, expected_head, rtol=0, atol=0)

    def test_training_entry_configuration(self):
        for name, hidden, _ in CASES:
            with self.subTest(variant=name):
                module = importlib.import_module("satellite_rl.agents." + name)
                with mock.patch.object(sac_runner, "run_training") as train:
                    runpy.run_module("satellite_rl.training.train_" + name, run_name="__main__")
                train.assert_called_once_with(module.SAC, module.ReplayBuffer,
                                              n_hiddens=hidden, experiment_name=name)
        signature = inspect.signature(sac_runner.run_training)
        self.assertEqual(signature.parameters["n_hiddens"].default, 128)
        self.assertIsNone(signature.parameters["experiment_name"].default)
        for name in ("sac_transformer", "sac_cnn", "sac_mlp"):
            module = importlib.import_module("satellite_rl.agents." + name)
            with mock.patch.object(sac_runner, "run_training") as train:
                runpy.run_module("satellite_rl.training.train_" + name, run_name="__main__")
            train.assert_called_once_with(module.SAC, module.ReplayBuffer)

    def test_one_episode_training_outputs(self):
        # 只在测试进程内把回合数临时限制为 1；生产源码仍为 4000。
        tree = ast.parse(textwrap.dedent(inspect.getsource(sac_runner.run_training)))
        episode_assignments = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "num_epochs" for target in node.targets)
        ]
        self.assertEqual(len(episode_assignments), 1)
        self.assertEqual(episode_assignments[0].value.value, 4000)
        episode_assignments[0].value = ast.Constant(value=1)
        namespace = dict(sac_runner.__dict__)
        exec(compile(ast.fix_missing_locations(tree), "<one-episode-test>", "exec"), namespace)
        short_training = namespace["run_training"]
        for name, _, _ in CASES:
            with self.subTest(variant=name):
                np.random.seed(42)
                torch.manual_seed(42)
                with mock.patch.object(sac_runner, "run_training", short_training), \
                        mock.patch.object(plt, "show"), redirect_stdout(io.StringIO()):
                    runpy.run_module("satellite_rl.training.train_" + name, run_name="__main__")
                metrics_dir = self.output_root / "outputs" / "metrics" / name
                metrics_path = next(metrics_dir.glob("*.npz"))
                with np.load(metrics_path) as metrics:
                    self.assertEqual(metrics["returns"].shape, (1,))
                    self.assertTrue(np.isfinite(metrics["returns"]).all())
                    self.assertTrue(np.isfinite(metrics["satisfactions"]).all())
                self.assertEqual(len(list((self.output_root / "outputs" / "figures" / name).glob("*.png"))), 2)
                self.assertEqual(len(list((self.output_root / "models" / "runs" / name).glob("*.pth"))), 6)
                self.assertTrue(any((self.output_root / "outputs" / "logs" / name).glob("*.csv")))
        self.assertFalse((self.output_root / "outputs" / "metrics" / "sac").exists())
        self.assertFalse((self.output_root / "outputs" / "figures" / "sac").exists())


if __name__ == "__main__":
    unittest.main()
