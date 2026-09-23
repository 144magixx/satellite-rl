"""验证实际训练/评估动作映射，以及权重加载失败时立即停止评估。"""

from contextlib import ExitStack, redirect_stdout
import importlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch


class RecordingEnvironment:
    """记录送入环境的动作，跑完全部十个功率档位后结束。"""

    def __init__(self):
        self.beam_idx = 0
        self.actions = []
        self.beam_info = {"rate": np.full((100, 1), 1e8)}
        self.beta = 1e8
        self.beam_bps = []
        self.history_cinr = []
        self.satellite_freq_state = SimpleNamespace(
            freq_pool_real=np.zeros((8, 100)),
        )
        self.interfere_num = 0
        self.interfere_in_range_num = 0

    def reset(self, beam_info):
        self.beam_idx = 0
        return np.zeros(204, dtype=np.float32)

    def step(self, action):
        self.actions.append(list(action))
        self.beam_bps.append(1e8)
        self.beam_idx += 1
        info = {
            "total_satisfaction": 1.0,
            "current_satisfaction": 1.0,
            "spectral_efficiency_avg": 1.0,
            "sgm_rate": 1.0,
        }
        return np.zeros(204, dtype=np.float32), 1.0, self.beam_idx == 10, info

    def get_occupy(self):
        return 0.0


class SequentialPolicy(torch.nn.Module):
    """可加载的小型网络；确定性地产生 0..9 的功率索引。"""

    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1))
        self.action_index = 0

    def next_action(self):
        action = [17, 2, self.action_index]
        self.action_index += 1
        return action

    def forward(self, state):
        action = self.next_action()
        heads = [torch.zeros(1, size) for size in (100, 10, 10)]
        for head, index in zip(heads, action):
            head[0, index] = 1.0
        return tuple(heads)


class SequentialAgent:
    def __init__(self, is_ppo=False):
        self.actor = SequentialPolicy()
        self.online = SequentialPolicy()
        self.is_ppo = is_ppo
        self.log_step = Mock()
        self.update = Mock(return_value=(0.0, 0.0))

    def take_action(self, state):
        action = self.actor.next_action()
        return (action, torch.tensor(0.0)) if self.is_ppo else action

    def choose_action(self, state, env, eps):
        return self.online.next_action()


class EvaluationFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="satellite_eval_fixes_")
        cls.root = Path(cls.temporary.name)
        cls.environment = patch.dict(os.environ, {
            "SATELLITE_RL_OUTPUT_ROOT": str(cls.root / "outputs"),
            "MPLBACKEND": "Agg",
        })
        cls.environment.start()
        cls.modules = [
            importlib.import_module("satellite_rl.evaluation." + name)
            for name in (
                "compare_metrics", "demand_sweep", "compare_beam_satisfaction",
            )
        ]
        cls.training = importlib.import_module(
            "satellite_rl.training.ppo_direct_utils"
        )
        cls.checkpoints = importlib.import_module(
            "satellite_rl.evaluation.checkpoints"
        )

    @classmethod
    def tearDownClass(cls):
        cls.environment.stop()
        cls.temporary.cleanup()

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

    def run_evaluation(self, module, kind, env, checkpoint_root=None):
        """执行原评估入口，仅替换环境与网络体积；保留动作转换和分支。"""
        beams = SimpleNamespace(beam_info={})
        agent = SequentialAgent(is_ppo=kind == "ppo")
        is_beam_comparison = module.__name__.endswith("compare_beam_satisfaction")
        aliases = {
            "sac_trans": "t-sac", "sac_cnn": "cnn-sac", "sac_fc": "mlp-sac",
        }
        with ExitStack() as stack:
            for name in ("SAC", "SAC_Transformer", "SAC_CNN", "SAC_FC", "PPO", "DQNAgent"):
                if hasattr(module, name):
                    stack.enter_context(patch.object(module, name, return_value=agent))
            stack.enter_context(patch.object(module, "Env", return_value=env))
            stack.enter_context(patch.object(module, "BeamInfo", return_value=beams))
            stack.enter_context(patch.object(module, "SatelliteFreqState"))
            stack.enter_context(patch.object(module, "CurrentBeamState"))
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
            if hasattr(module, "device"):
                stack.enter_context(patch.object(module, "device", torch.device("cpu")))
            if checkpoint_root is None:
                for loader in ("load_actor_only", "load_network_checkpoint"):
                    if hasattr(module, loader):
                        stack.enter_context(patch.object(module, loader, return_value=True))
            else:
                def checkpoint_path(original):
                    original = Path(original)
                    if original.suffix in (".pth", ".pt"):
                        return str(checkpoint_root / original.name)
                    return str(checkpoint_root)

                stack.enter_context(patch.object(module, "project_path", side_effect=checkpoint_path))
            stack.enter_context(redirect_stdout(io.StringIO()))
            if is_beam_comparison:
                return module.eval_once(aliases.get(kind, kind))
            if module.__name__.endswith("demand_sweep"):
                return module.run_episode_per_beam(beams, kind)
            return module.run_episode_per_beam(lambda: (None, beams), kind)

    def test_ppo_training_and_all_evaluations_use_same_ten_power_levels(self):
        train_env = RecordingEnvironment()
        agent = SequentialAgent(is_ppo=True)
        with patch.object(self.training, "BeamInfo"), redirect_stdout(io.StringIO()):
            self.training.train_on_policy_agent(train_env, agent, num_episodes=1)
        agent.update.assert_called_once()
        self.assertEqual(len(train_env.actions), 10)
        train_power = 10.0 ** (np.asarray(train_env.actions)[:, 3] / 10.0)
        np.testing.assert_allclose(train_power, [3, 6, 9, 12, 15, 18, 21, 24, 27, 30])
        for module in self.modules:
            with self.subTest(evaluation=module.__name__):
                env = RecordingEnvironment()
                self.run_evaluation(module, "ppo", env)
                self.assertEqual(len(env.actions), 10)
                np.testing.assert_allclose(env.actions, train_env.actions)

    def test_sac_and_dqn_evaluation_power_levels_remain_unchanged(self):
        for module in self.modules:
            for kind in ("sac_trans", "sac_cnn", "sac_fc", "dqn"):
                with self.subTest(evaluation=module.__name__, algorithm=kind):
                    env = RecordingEnvironment()
                    self.run_evaluation(module, kind, env)
                    self.assertEqual(len(env.actions), 10)
                    power = 10.0 ** (np.asarray(env.actions)[:, 3] / 10.0)
                    np.testing.assert_allclose(power, [5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

    def test_missing_actor_directory_or_file_reports_algorithm_and_path(self):
        empty = self.root / "empty"
        empty.mkdir(exist_ok=True)
        for directory in (self.root / "does-not-exist", empty):
            with self.subTest(directory=directory):
                with self.assertRaises(self.checkpoints.CheckpointLoadError) as caught:
                    self.checkpoints.load_actor_only(
                        SequentialAgent(), directory, algorithm="PPO-test",
                    )
                self.assertIn("PPO-test", str(caught.exception))
                self.assertIn(str(directory), str(caught.exception))

    def test_actor_and_network_loaders_accept_raw_and_nested_weights(self):
        original = torch.nn.Linear(2, 1)
        with torch.no_grad():
            original.weight.fill_(0.75)
            original.bias.fill_(0.25)
        for filename, nested in (("actor.pth", False), ("checkpoint.pt", True)):
            with self.subTest(filename=filename):
                directory = self.root / filename.replace(".", "_")
                directory.mkdir(exist_ok=True)
                state = original.state_dict()
                torch.save({"actor": state} if nested else state, directory / filename)
                loaded = torch.nn.Linear(2, 1)
                self.checkpoints.load_actor_only(
                    SimpleNamespace(actor=loaded), directory, algorithm="PPO-test",
                )
                self.assertFalse(loaded.training)
                for expected, actual in zip(original.parameters(), loaded.parameters()):
                    torch.testing.assert_close(actual, expected)
        # The original checkpoint.pt fallback requires the explicit actor key.
        raw_fallback = self.root / "raw_fallback"
        raw_fallback.mkdir()
        torch.save(original.state_dict(), raw_fallback / "checkpoint.pt")
        with self.assertRaises(self.checkpoints.CheckpointLoadError):
            self.checkpoints.load_actor_only(
                SimpleNamespace(actor=torch.nn.Linear(2, 1)), raw_fallback,
                algorithm="PPO-test",
            )
        for nested in (False, True):
            with self.subTest(dqn_nested=nested):
                path = self.root / f"network_{nested}.pth"
                state = original.state_dict()
                torch.save({"online": state} if nested else state, path)
                loaded = torch.nn.Linear(2, 1)
                self.checkpoints.load_network_checkpoint(
                    loaded, path, algorithm="DQN-test", state_dict_key="online",
                    evaluation_mode=True,
                )
                self.assertFalse(loaded.training)
                for expected, actual in zip(original.parameters(), loaded.parameters()):
                    torch.testing.assert_close(actual, expected)

    def test_network_loader_errors_include_algorithm_and_checkpoint_path(self):
        for failure in ("missing", "corrupt", "incompatible"):
            path = self.root / f"network_{failure}.pth"
            if failure == "corrupt":
                path.write_bytes(b"this is not a checkpoint")
            elif failure == "incompatible":
                torch.save({"marker": torch.zeros(2)}, path)
            with self.subTest(failure=failure):
                with self.assertRaises(self.checkpoints.CheckpointLoadError) as caught:
                    self.checkpoints.load_network_checkpoint(
                        SequentialPolicy(), path, algorithm="DQN-test",
                    )
                self.assertIn("DQN-test", str(caught.exception))
                self.assertIn(str(path), str(caught.exception))

    def test_failed_weights_stop_every_evaluation_before_first_step(self):
        for failure in ("missing", "corrupt", "incompatible"):
            directory = self.root / ("evaluation_" + failure)
            directory.mkdir(exist_ok=True)
            if failure != "missing":
                for filename in ("actor.pth", "dqn_online.pth"):
                    path = directory / filename
                    if failure == "corrupt":
                        path.write_bytes(b"this is not a checkpoint")
                    else:
                        torch.save({"marker": torch.zeros(2)}, path)
            for module in self.modules:
                for kind in ("sac_trans", "sac_cnn", "sac_fc", "ppo", "dqn"):
                    with self.subTest(failure=failure, evaluation=module.__name__, algorithm=kind):
                        env = RecordingEnvironment()
                        with self.assertRaises(self.checkpoints.CheckpointLoadError) as caught:
                            self.run_evaluation(module, kind, env, directory)
                        self.assertIn(str(directory), str(caught.exception))
                        self.assertEqual(env.actions, [], "权重失败后不得运行环境或返回评估指标")


if __name__ == "__main__":
    unittest.main()
