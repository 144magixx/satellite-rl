"""验证 PPO 终止目标及三种 SAC 的独立 Critic 存档。"""

import contextlib
import importlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


class AgentFixTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="satellite_agent_fixes_")
        self.addCleanup(self.temporary.cleanup)
        self.environment = mock.patch.dict(
            os.environ, {"SATELLITE_RL_OUTPUT_ROOT": self.temporary.name}
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.manual_seed(42)
        np.random.seed(42)

    def check_ppo_targets(self, name):
        module = importlib.import_module("satellite_rl.agents." + name)
        cpu = torch.device("cpu")
        with mock.patch.object(module, "device", cpu), contextlib.redirect_stdout(io.StringIO()):
            agent = module.PPO(
                state_dim=204, actor_lr=1e-3, critic_lr=1e-3,
                lmbda=0.95, epochs=1, eps=0.2, gamma=0.9, device=cpu,
            )
            # 下一状态有明确的非零价值，终止和非终止样本应产生不同目标。
            with torch.no_grad():
                for parameter in agent.critic.parameters():
                    parameter.zero_()
                agent.critic.net[-1].bias.fill_(5.0)

            transition = {
                "states": np.zeros((2, 204), dtype=np.float32),
                "actions": [[0, 0, 0], [1, 1, 1]],
                "rewards": [1.0, 2.0],
                "next_states": np.ones((2, 204), dtype=np.float32),
                "dones": [True, False],
                "old_log_probs": [0.0, 0.0],
                "masks": np.ones((2, 10), dtype=np.float32),
            }
            captured_targets = []
            mse_loss = module.F.mse_loss

            def capture_critic_target(prediction, target, *args, **kwargs):
                captured_targets.append(target.detach().clone())
                return mse_loss(prediction, target, *args, **kwargs)

            # 运行真实 update、损失与反向传播，仅观察传给价值损失的目标。
            with mock.patch.object(module.F, "mse_loss", side_effect=capture_critic_target):
                actor_loss, critic_loss = agent.update(transition, entropy_coef=0.01)

            self.assertEqual(len(captured_targets), 1)
            torch.testing.assert_close(
                captured_targets[0], torch.tensor([[1.0], [6.5]])
            )
            self.assertTrue(torch.isfinite(actor_loss).all())
            self.assertTrue(torch.isfinite(critic_loss).all())
            self.assertNotEqual(agent.critic.net[-1].bias.item(), 5.0)

    def test_ppo_direct_masks_terminal_bootstrap(self):
        self.check_ppo_targets("ppo_direct")

    def test_ppo_grouped_masks_terminal_bootstrap(self):
        self.check_ppo_targets("ppo_grouped")

    def assert_state_dict_equal(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for key, value in expected.items():
            self.assertTrue(torch.equal(value.cpu(), actual[key].cpu()), key)

    def check_sac_checkpoints(self, name, prefix):
        module = importlib.import_module("satellite_rl.agents." + name)
        with contextlib.redirect_stdout(io.StringIO()):
            agent = module.SAC(
                n_states=204, n_hiddens=128, freq_dim=100, slots_dim=10,
                power_dim=10, actor_lr=1e-3, critic_lr=1e-3, alpha_lr=1e-3,
                target_entropy=-1, tau=0.005, gamma=0.9,
                device=torch.device("cpu"),
            )
            # 显式赋予两个 Critic 不同参数，确保覆盖故障无法偶然通过。
            with torch.no_grad():
                for parameter in agent.critic_1.parameters():
                    parameter.fill_(0.125)
                for parameter in agent.critic_2.parameters():
                    parameter.fill_(-0.25)
            agent.save_model()

        model_root = Path(self.temporary.name) / "models" / "runs" / name
        files = list(model_root.glob("*.pth"))
        self.assertEqual(len(files), 6)
        full_paths = list(model_root.glob(prefix + "_full_model_*.pth"))
        self.assertEqual(len(full_paths), 1)
        full = torch.load(full_paths[0], map_location="cpu", weights_only=True)
        components = {
            "actor_state_dict": ("actor", agent.actor),
            "critic_1": ("critic_1", agent.critic_1),
            "critic_2": ("critic_2", agent.critic_2),
            "target_critic_1": ("target_critic_1", agent.target_critic_1),
            "target_critic_2": ("target_critic_2", agent.target_critic_2),
        }
        self.assertEqual(set(full), set(components))
        saved_paths = []
        for key, (suffix, network) in components.items():
            paths = list(model_root.glob(f"{prefix}_{suffix}_*.pth"))
            self.assertEqual(len(paths), 1, suffix)
            saved_paths.append(paths[0])
            saved = torch.load(paths[0], map_location="cpu", weights_only=True)
            self.assert_state_dict_equal(network.state_dict(), saved)
            self.assert_state_dict_equal(network.state_dict(), full[key])
        self.assertEqual(len(set(saved_paths)), 5)

    def test_sac_transformer_saves_distinct_critics(self):
        self.check_sac_checkpoints("sac_transformer", "sac_transformer")

    def test_sac_cnn_saves_distinct_critics(self):
        self.check_sac_checkpoints("sac_cnn", "sac_cnn")

    def test_sac_mlp_saves_distinct_critics(self):
        self.check_sac_checkpoints("sac_mlp", "sac_fc")


if __name__ == "__main__":
    unittest.main()
