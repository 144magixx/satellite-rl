"""Early evaluation must neither shift periodic nodes nor duplicate resumes."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from satellite_rl.workbench.config import default_config
from satellite_rl.workbench.runner import _training, _validation_due


class ValidationScheduleTests(unittest.TestCase):
    def test_early_nodes_keep_forty_periodic_nodes(self):
        evaluation = dict(interval_steps=10000, extra_steps=[0, 2000, 5000])
        progress = {"last_validation_step": -1}
        nodes = []
        for step in range(0, 400001, 100):
            if _validation_due(evaluation, progress, step):
                nodes.append(step)
                progress["last_validation_step"] = step
        self.assertEqual(nodes, [0, 2000, 5000] + list(range(10000, 400001, 10000)))

    def test_legacy_off_grid_cadence_is_unchanged(self):
        evaluation = dict(interval_steps=3)
        progress = {"last_validation_step": -1}
        nodes = []
        for step in range(0, 11, 2):
            if _validation_due(evaluation, progress, step):
                nodes.append(step)
                progress["last_validation_step"] = step
        self.assertEqual(nodes, [2, 6, 10])

    def test_duplicate_and_requested_off_boundary_nodes(self):
        ev = dict(interval_steps=100, extra_steps=[0, 25, 30])
        self.assertFalse(_validation_due(ev, {"last_validation_step": 0}, 0))
        self.assertFalse(_validation_due(ev, {"last_validation_step": 0}, 20))
        self.assertTrue(_validation_due(ev, {"last_validation_step": 0}, 40))
        self.assertFalse(_validation_due(ev, {"last_validation_step": 40}, 40))
        self.assertTrue(_validation_due(ev, {"last_validation_step": 40}, 100))

    def test_initial_checkpoint_and_resume_keep_completed_nodes(self):
        class TinyEnv:
            def __init__(self, _config):
                pass

            def reset(self, **_kwargs):
                return np.array([0.0])

            def scene(self):
                return {"source": np.array([1])}

            def action_mask(self):
                return None

            def step(self, _action):
                return np.array([0.0]), 0.0, True, {}

            def metrics(self):
                return {"final_si_mean": 0.0}

        class TinyAgent:
            update_step = 0
            last_action_info = {}

            def act(self, *_args, **_kwargs):
                return [0, 0, 0]

            def observe(self, *_args, **_kwargs):
                pass

            def ready(self, *_args):
                return False

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            config = default_config()
            config["training"].update(total_steps=9, checkpoint_interval=1, log_interval=100)
            config["evaluation"].update(interval_steps=4, extra_steps=[0, 2], test_count=0)
            job = {"run_id": "early", "git_commit": "f" * 40, "source_root": temp,
                   "model_dir": str(directory / "model")}
            ledger = Mock(status={"preflight": {"gpu": "fixture"}})
            saved, checkpoints, nodes = {}, [], []

            def save(path, _agent, progress, _config, environment):
                Path(path).touch()
                saved.update(progress=deepcopy(progress), environment=deepcopy(environment))
                checkpoints.append(deepcopy(progress))

            def evaluate(_config, _agent, _directory, group, _count, _seed, step, _ledger, **_kwargs):
                if group == "validation":
                    nodes.append(step)
                return [], {}

            def record(group, step, _values):
                if group == "training" and step == 4:
                    (directory / "stop.request").touch()

            ledger.record.side_effect = record
            with patch("satellite_rl.workbench.environment.WorkbenchEnv", TinyEnv), \
                 patch("satellite_rl.workbench.runner.build_agent", side_effect=lambda *_a: TinyAgent()), \
                 patch("satellite_rl.workbench.runner.save_checkpoint", side_effect=save), \
                 patch("satellite_rl.workbench.runner.load_checkpoint", side_effect=lambda *_a: (deepcopy(saved["progress"]), deepcopy(saved["environment"]))), \
                 patch("satellite_rl.workbench.runner.save_learning_snapshot", return_value=None), \
                 patch("satellite_rl.workbench.runner.export_policy"), \
                 patch("satellite_rl.workbench.runner._ensure_finite_agent"), \
                 patch("satellite_rl.workbench.runner._fixed_evaluate", side_effect=evaluate):
                _training(directory, job, config, ledger, resume=False)
                self.assertEqual(checkpoints[0]["env_step"], 0)
                self.assertEqual(checkpoints[0]["last_validation_step"], 0)
                self.assertEqual(nodes, [0, 2, 4])
                (directory / "stop.request").unlink()
                ledger.record.side_effect = None
                _training(directory, job, config, ledger, resume=True)
                self.assertEqual(nodes, [0, 2, 4, 8])


if __name__ == "__main__":
    unittest.main()
