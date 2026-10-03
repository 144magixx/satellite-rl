"""Bounded fixture for S2-P17 same-job evaluation recovery, without real inference."""

import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from satellite_rl.workbench.config import default_config
from satellite_rl.workbench.runner import _evaluate_variable_job


class _Ledger:
    def __init__(self):
        self.status = {"preflight": {"gpu": "fixture"}}

    def save_status(self, **fields):
        self.status.update(fields)

    def event(self, *_args, **_kwargs):
        pass


class _Policy:
    def __init__(self, config):
        self.config = config
        self.agent = type("Agent", (), {"rng": random.Random(1)})()
        self.metadata = {"state_dim": 210, "provenance": {"run_id": "fixture"}}


class VariableEvaluationResumeTests(unittest.TestCase):
    def test_completed_cases_reused_and_corruption_or_contract_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = default_config()
            config["environment"]["profile"] = "europe_v1"
            config["environment"]["beam_count_max"] = 200
            records = [{"scene_id": f"s{i}", "beam_count": 100,
                        "mother_scene_id": f"m{i // 22}", "load_multiplier": 0.75,
                        "partition_role": "test", "sha256": "a" * 64}
                       for i in range(440)]
            scene_set = {"id": "fixture", "count": 440, "files": records}
            job = {"experiment_id": "S2-P17", "run_id": "fixture", "git_commit": "commit-a",
                   "model_path": str(root / "model"), "model_files": [],
                   "scene_sets": {"test": scene_set}, "action_seed": 53001,
                   "repeats": 3, "action_mode": "stochastic", "baselines": []}
            calls = []

            def episode(_env, _policy, _scene, scene_id, seed, _mode, *_args, **_kwargs):
                calls.append((scene_id, seed))
                if len(calls) in (2, 3):
                    (root / "stop.request").write_text("stop", encoding="utf-8")
                return {"scene_id": scene_id, "metrics": {"final_si_mean": 0.5},
                        "beam_count": 100, "scene_metadata": records[0],
                        "inference_seconds": 0.01, "initial_allocation_seconds": 0.02,
                        "beams": [], "trace": [], "service_timing": "fixture"}

            files = [(root / f"s{i}.npz", "a" * 64) for i in range(440)]
            with patch("satellite_rl.workbench.runner.load_policy", return_value=_Policy(config)), \
                    patch("satellite_rl.workbench.datasets.assert_scene_profile"), \
                    patch("satellite_rl.workbench.datasets.scene_set_files", return_value=files), \
                    patch("satellite_rl.workbench.environment.WorkbenchEnv", return_value=object()), \
                    patch("satellite_rl.workbench.runner._sha256", return_value="a" * 64), \
                    patch("satellite_rl.workbench.runner._load_scene",
                          return_value={"rate": np.ones((100, 1))}), \
                    patch("satellite_rl.workbench.runner._save_scene",
                          return_value={"path": str(root / "archived.npz"), "sha256": "a" * 64}), \
                    patch("satellite_rl.workbench.runner._evaluation_episode", side_effect=episode):
                _evaluate_variable_job(root, job, config, _Ledger(), device="cpu")
                self.assertEqual(len(calls), 2)
                (root / "stop.request").unlink()
                second = _Ledger()
                _evaluate_variable_job(root, job, config, second, device="cpu")
                self.assertEqual(len(calls), 3)
                self.assertEqual(second.status["reused_completed_cases"], 2)
                self.assertEqual(second.status["newly_completed_cases"], 1)
                (root / "stop.request").unlink()
                with self.assertRaises(ValueError):
                    _evaluate_variable_job(root, {**job, "git_commit": "commit-b"},
                                           config, _Ledger(), device="cpu")
                case_path = root / "cases" / "model_00000_00.json"
                case = json.loads(case_path.read_text(encoding="utf-8"))
                case["body"]["result"]["metrics"]["final_si_mean"] = 0.9
                case_path.write_text(json.dumps(case), encoding="utf-8")
                with self.assertRaises(ValueError):
                    _evaluate_variable_job(root, job, config, _Ledger(), device="cpu")


if __name__ == "__main__":
    unittest.main()
