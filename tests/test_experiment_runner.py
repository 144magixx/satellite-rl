"""Recovery and record integrity for the long SAC experiment worker."""

from collections import deque
from argparse import Namespace
import csv
import gzip
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from satellite_rl.training.experiment_runner import (
    _evaluation_action, _json_safe, _scalar_metrics, evaluate, load_checkpoint, run, run_id_for,
    save_actor_weights, save_checkpoint,
)
from satellite_rl.training.run_monitor import RunMonitor, atomic_json


class _Replay:
    def __init__(self):
        self.buffer = deque(maxlen=8)


class _Agent:
    def __init__(self):
        for name in ("actor", "critic_1", "critic_2", "target_critic_1", "target_critic_2"):
            setattr(self, name, torch.nn.Linear(2, 2))
        self.log_alpha = torch.nn.Parameter(torch.tensor(-2.0))
        for name in ("actor", "critic_1", "critic_2"):
            setattr(self, name + "_optimizer", torch.optim.Adam(getattr(self, name).parameters()))
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha])
        self.target_entropy = 6.9


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_run_ids_and_atomic_status_do_not_collide(self):
        left = run_id_for("transformer", "legacy", 51001)
        right = run_id_for("transformer", "legacy", 51001)
        self.assertNotEqual(left, right)
        first = RunMonitor(self.root / left, left)
        second = RunMonitor(self.root / right, right)
        try:
            first.row("episodes", {"episode": 1, "env_step": 2, "return": 0.996})
            first.write_status(stage="training", env_step=2)
            second.write_status(stage="training", env_step=7)
        finally:
            first.close()
            second.close()
        self.assertEqual(json.loads((self.root / left / "status.json").read_text())["env_step"], 2)
        self.assertEqual(json.loads((self.root / right / "status.json").read_text())["env_step"], 7)
        with (self.root / left / "episodes.csv").open(newline="") as stream:
            self.assertEqual(float(next(csv.DictReader(stream))["return"]), 0.996)
        self.assertFalse(list((self.root / left).glob("*.tmp")))

    def test_checkpoint_restores_network_optimizer_replay_and_rng(self):
        random.seed(17)
        np.random.seed(18)
        torch.manual_seed(19)
        agent = _Agent()
        replay = _Replay()
        replay.buffer.extend([("a", 1), ("b", 2)])
        for name in ("actor", "critic_1", "critic_2"):
            network = getattr(agent, name)
            optimizer = getattr(agent, name + "_optimizer")
            optimizer.zero_grad()
            network(torch.ones(1, 2)).sum().backward()
            optimizer.step()
        agent.log_alpha_optimizer.zero_grad()
        agent.log_alpha.backward()
        agent.log_alpha_optimizer.step()
        expected_weight = agent.actor.weight.detach().clone()
        expected_optimizer_step = agent.actor_optimizer.state[agent.actor.weight]["step"].clone()
        path = self.root / "latest.pt"
        progress = {"env_step": 31, "episode": 2, "env": {"in_progress": True}}
        save_checkpoint(path, agent, replay, progress, "frozen-config")
        expected_rng = (random.random(), np.random.random(), torch.rand(1).item())
        agent.actor.weight.data.zero_()
        replay.buffer.clear()
        random.random()
        np.random.random()
        torch.rand(1)
        actual_progress = load_checkpoint(path, agent, replay, "frozen-config")
        self.assertEqual(actual_progress, progress)
        torch.testing.assert_close(agent.actor.weight, expected_weight)
        torch.testing.assert_close(agent.actor_optimizer.state[agent.actor.weight]["step"],
                                   expected_optimizer_step)
        self.assertEqual(list(replay.buffer), [("a", 1), ("b", 2)])
        self.assertEqual((random.random(), np.random.random(), torch.rand(1).item()), expected_rng)
        with self.assertRaisesRegex(RuntimeError, "配置不匹配"):
            load_checkpoint(path, agent, replay, "another-config")

    def test_best_actor_file_can_be_loaded_independently(self):
        actor = torch.nn.Linear(2, 1)
        expected = actor.weight.detach().clone()
        path = self.root / "best_actor.pt"
        digest = save_actor_weights(path, actor)
        self.assertEqual(len(digest), 64)
        actor.weight.data.zero_()
        actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        torch.testing.assert_close(actor.weight, expected)

    def test_recorded_rows_not_duplicated_after_resume(self):
        run_id = "resume-test"
        monitor = RunMonitor(self.root / run_id, run_id)
        row = {"checkpoint": "final_test", "env_step": 400000,
               "scene_id": "81000", "action_seed": 91001, "si_mean": 0.7}
        monitor.row("evaluations", row)
        monitor.close()
        resumed = RunMonitor(self.root / run_id, run_id, resume=True)
        resumed.row("evaluations", row)
        resumed.close()
        with (self.root / run_id / "evaluations.csv").open(newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 1)

    def test_low_frequency_metrics_keep_full_precision(self):
        values = _scalar_metrics({"critic1_loss": torch.tensor(1.23456789),
                                  "alpha": torch.tensor(0.01)})
        self.assertIn("critic_1_loss", values)
        self.assertAlmostEqual(values["critic_1_loss"], 1.23456789, places=6)
        self.assertAlmostEqual(values["alpha"], 0.01, places=6)
        with self.assertRaises(FloatingPointError):
            _scalar_metrics({"actor_loss": torch.tensor(float("nan"))})
        self.assertEqual(_json_safe({"bad": [float("inf"), np.float64(float("nan"))]}),
                         {"bad": ["inf", "nan"]})

    def test_bad_network_cannot_replace_latest_checkpoint(self):
        agent = _Agent()
        replay = _Replay()
        path = self.root / "latest.pt"
        save_checkpoint(path, agent, replay, {"env_step": 1}, "config")
        original = path.read_bytes()
        agent.actor.weight.data[0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            save_checkpoint(path, agent, replay, {"env_step": 2}, "config")
        self.assertEqual(path.read_bytes(), original)

    def test_argmax_uses_each_actor_head(self):
        class Actor(torch.nn.Module):
            def forward(self, _state):
                return (torch.tensor([[0.1, 0.9]]),
                        torch.tensor([[0.8, 0.2]]),
                        torch.tensor([[0.3, 0.7]]))

        agent = type("Agent", (), {"device": torch.device("cpu"), "actor": Actor()})()
        self.assertEqual(_evaluation_action(agent, np.zeros(204), "argmax"), [1, 0, 1])

    def test_evaluation_restores_training_rng_and_actor_mode(self):
        class Env:
            beam_idx = 0

            def step(self, action):
                return None, 0.0, True, {}

        class Agent:
            def __init__(self):
                self.actor = torch.nn.Dropout()

            def take_action(self, _state):
                torch.rand(1)
                np.random.random()
                random.random()
                return [0, 0, 0]

        random.seed(23)
        np.random.seed(24)
        torch.manual_seed(25)
        agent = Agent()
        monitor = RunMonitor(self.root / "evaluation-run", "evaluation-run")
        try:
            with mock.patch("satellite_rl.training.experiment_runner._load_scene", return_value=object()), \
                 mock.patch("satellite_rl.training.experiment_runner._new_environment",
                            return_value=(Env(), None)), \
                 mock.patch("satellite_rl.training.experiment_runner._terminal_metrics",
                            return_value=({"si_mean": 0.3}, {"rates_bps": np.array([1.0])})):
                result = evaluate(agent, {"scene": {"path": "unused", "sha256": "unused"}},
                                  (91001,), "final_test", 10000, monitor,
                                  self.root / "evaluation-data")
            with gzip.open(result["trace_path"], "rt", encoding="utf-8") as stream:
                trace = json.loads(stream.readline())
            self.assertEqual(trace["requested"][1:3], [0, 1])
            self.assertEqual(trace["actual"], trace["requested"])
            self.assertTrue(agent.actor.training)
            observed = (random.random(), np.random.random(), torch.rand(1).item())
            random.seed(23)
            np.random.seed(24)
            torch.manual_seed(25)
            expected = (random.random(), np.random.random(), torch.rand(1).item())
            self.assertEqual(observed, expected)
        finally:
            monitor.close()

    def test_worker_stops_at_exact_interaction_budget_and_resumes(self):
        class Agent(_Agent):
            def take_action(self, _state):
                return [0, 0, 0]

        class Replay(_Replay):
            def add(self, *transition):
                self.buffer.append(transition)

            def size(self):
                return len(self.buffer)

        args = Namespace(model="mlp_matched", formula="legacy", seed=51001,
                         steps=2, experiment_id="smoke", archive_root=str(self.root),
                         validation_interval=0, checkpoint_interval=10000,
                         validation_scene_count=0, test_scene_count=0,
                         skip_final_evaluation=True,
                         run_id="two-steps", resume=False, stop_after_steps=1)
        with mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch("satellite_rl.training.experiment_runner.strict_cuda_preflight",
                        return_value={"git_commit": "test"}), \
             mock.patch("satellite_rl.training.experiment_runner.prepare_fixed_scenes",
                        return_value={}), \
             mock.patch("satellite_rl.training.experiment_runner._create_agent",
                        side_effect=lambda *_: (Agent(), Replay())), \
             mock.patch("satellite_rl.training.experiment_runner._tensorboard_writer",
                        return_value=None), \
             mock.patch("satellite_rl.training.experiment_runner.evaluate",
                        return_value={"si_mean": 0.1}):
            self.assertEqual(run(args), 0)
            status_path = self.root / "outputs" / "experiments" / "smoke" / "two-steps" / "status.json"
            self.assertEqual(json.loads(status_path.read_text())["env_step"], 1)
            self.assertEqual(json.loads(status_path.read_text())["stage"], "paused")
            args.resume = True
            args.stop_after_steps = None
            self.assertEqual(run(args), 0)
            self.assertEqual(json.loads(status_path.read_text())["env_step"], 2)


if __name__ == "__main__":
    unittest.main()
