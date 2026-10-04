"""S2-P17 node-zero cases resume without repeating committed allocations."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from satellite_rl.agents.workbench_algorithms import Replay
from satellite_rl.workbench.runner import _fixed_evaluate, _reconcile, _training


class _Env:
    def reset(self, seed=None, scene=None):
        self._scene = scene
        return np.zeros(1, dtype=np.float32)

    def scene(self):
        return self._scene


class _Ledger:
    def __init__(self):
        self.records = []
        self.status = {"preflight": {"gpu": "fixture"}}

    def record(self, group, step, values):
        self.records.append((group, step, values))

    def save_status(self, **fields):
        self.status.update(fields)

    def event(self, *_args, **_kwargs):
        pass

    def flush(self):
        pass

    def open_tensorboard(self, **_kwargs):
        pass


class ZeroValidationResumeTests(unittest.TestCase):
    def test_reconcile_preserves_training_metrics_but_rolls_back_zero_validation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = [
                {"step": 0, "group": "learning", "values": {"loss": 1.0}},
                {"step": 0, "group": "action", "values": {"slots": 1.0}},
                {"step": 0, "group": "reward", "values": {"si": 0.5}},
                {"step": 0, "group": "validation", "values": {"si": 0.5}},
            ]
            (root / "metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in metrics), encoding="utf-8")
            (root / "count_metrics.jsonl").write_text(
                json.dumps({"env_step": 0, "group": "validation"}) + "\n", encoding="utf-8")
            (root / "traces.jsonl").write_text(
                "".join(json.dumps({"env_step": 0, "group": group}) + "\n"
                        for group in ("training", "validation")), encoding="utf-8")
            _reconcile(root, 0, -1)
            remaining = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["group"] for row in remaining],
                             ["learning", "action", "reward"])
            self.assertEqual((root / "count_metrics.jsonl").read_text(), "")
            traces = [json.loads(line) for line in (root / "traces.jsonl").read_text().splitlines()]
            self.assertEqual([row["group"] for row in traces], ["training"])

    def test_initial_checkpoint_precedes_zero_validation_and_resume_commits_marker(self):
        from satellite_rl.workbench.config import default_config

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = default_config()
            config["environment"].update(profile="europe_v1", beam_count_max=200)
            config["training"].update(beam_count_mode="fixed_100",
                                      scene_order="variable_beams", total_steps=0)
            config["evaluation"].update(validation_count=3, test_count=0,
                                        extra_steps=[0], interval_steps=20000, repeats=1)
            train = {"id": "train", "files": [
                {"path": str(root / f"train_{i}.npz"), "sha256": str(i) * 64,
                 "beam_count": 100, "mother_scene_id": "m0",
                 "load_multiplier": load}
                for i, load in enumerate((0.75, 1.25), 1)]}
            validation = {"id": "validation", "files": [
                {"path": str(root / f"val_{i}.npz"), "sha256": str(i) * 64}
                for i in range(3)]}
            job = {"run_id": "fixture", "git_commit": "a" * 40,
                   "model_dir": str(root / "model"), "source_root": str(root),
                   "scene_sets": {"train": train, "validation": validation}}
            agent = type("Agent", (), {"rng": random.Random(1), "update_step": 0,
                                         "replay": Replay(20000, 51042, "fixed_100")})()
            saved = {}
            calls = []

            def save_checkpoint(path, _agent, progress, _config, environment):
                saved["progress"] = deepcopy(progress)
                saved["environment"] = deepcopy(environment)
                Path(path).write_text(json.dumps({"last_validation_step":
                    progress["last_validation_step"]}), encoding="utf-8")

            def evaluate(*_args, **_kwargs):
                self.assertTrue((root / "checkpoint.pt").is_file())
                calls.append(1)
                if len(calls) == 1:
                    raise RuntimeError("interrupted node zero")
                return [], {"final_si_mean": 0.5}

            def scene_files(record):
                return [(Path(item["path"]), item["sha256"]) for item in record["files"]]

            with patch("satellite_rl.workbench.runner.build_agent", return_value=agent), \
                    patch("satellite_rl.workbench.config.config_summary",
                          return_value={"state_dim": 210, "action_dims": (100, 10, 10)}), \
                    patch("satellite_rl.workbench.runner.save_checkpoint", side_effect=save_checkpoint), \
                    patch("satellite_rl.workbench.runner.load_checkpoint",
                          side_effect=lambda *_: (deepcopy(saved["progress"]),
                                                   deepcopy(saved["environment"]))), \
                    patch("satellite_rl.workbench.runner.save_learning_snapshot", return_value=None), \
                    patch("satellite_rl.workbench.runner.export_policy"), \
                    patch("satellite_rl.workbench.runner._fixed_evaluate", side_effect=evaluate), \
                    patch("satellite_rl.workbench.datasets.assert_scene_profile"), \
                    patch("satellite_rl.workbench.datasets.scene_set_files", side_effect=scene_files):
                with self.assertRaisesRegex(RuntimeError, "interrupted node zero"):
                    _training(root, job, config, _Ledger(), resume=False)
                self.assertEqual(saved["progress"]["last_validation_step"], -1)
                _training(root, job, config, _Ledger(), resume=True)
                self.assertEqual(calls, [1, 1])
                self.assertEqual(saved["progress"]["last_validation_step"], 0)

    def test_interrupted_node_zero_reuses_cases_and_rebuilds_logs_once(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"frozen-initial-weights-and-rng")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            files = [(root / f"source_{i}.npz", "a" * 63 + str(i)) for i in range(3)]
            scene_set = {"id": "validation", "files": [
                {"path": str(path), "sha256": digest, "beam_count": 1}
                for path, digest in files]}
            config = {"training": {"beam_count_mode": "fixed_100"},
                      "environment": {"profile": "europe_v1"},
                      "evaluation": {"repeats": 1, "action_mode": "stochastic"}}
            cache = {"checkpoint_sha256": checkpoint_sha, "config_hash": "b" * 64}
            agent = type("Agent", (), {"rng": random.Random(1)})()
            calls = []

            def save_scene(path, scene):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"archived-scene")
                return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

            def episode(_env, _agent, _scene, scene_id, seed, _mode, *_args, **_kwargs):
                calls.append(scene_id)
                if len(calls) == 3:
                    raise RuntimeError("fixture interruption before third case commits")
                return {"scene_id": scene_id,
                        "metrics": {"final_si_mean": 0.5, "final_sgm_mean": 0.6},
                        "beams": [{"beam_id": 0}], "trace": [
                            {"env_step": 0, "group": "validation", "step": 0,
                             "scene_id": scene_id}], "service_timing": "fixture",
                        "beam_count": 1, "inference_seconds": 0.01,
                        "initial_preparation_seconds": 0.01,
                        "initial_verification_seconds": 0.01,
                        "initial_allocation_seconds": 0.03}

            with patch("satellite_rl.workbench.datasets.assert_scene_profile"), \
                    patch("satellite_rl.workbench.datasets.scene_set_files", return_value=files), \
                    patch("satellite_rl.workbench.environment.WorkbenchEnv", return_value=_Env()), \
                    patch("satellite_rl.workbench.runner._load_scene",
                          return_value={"rate": np.ones((1, 1))}), \
                    patch("satellite_rl.workbench.runner._save_scene", side_effect=save_scene), \
                    patch("satellite_rl.workbench.runner._evaluation_episode", side_effect=episode):
                with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
                    _fixed_evaluate(config, agent, root, "validation", 3, 53001, 0,
                                    _Ledger(), scene_set=scene_set, zero_cache=cache)
                self.assertEqual(len(list((root / "validation_zero_cases").glob("*.json"))), 2)
                _reconcile(root, 0, -1)
                _, summary = _fixed_evaluate(config, agent, root, "validation", 3, 53001,
                                              0, _Ledger(), scene_set=scene_set, zero_cache=cache)
                self.assertEqual(summary["final_si_mean"], 0.5)
                self.assertEqual(len(calls), 4)  # Only the incomplete third case reruns.
                for name in ("traces", "evaluations"):
                    rows = [json.loads(line) for line in (root / f"{name}.jsonl").read_text().splitlines()]
                    self.assertEqual(len(rows), 3)
                bad = dict(cache, checkpoint_sha256="c" * 64)
                with self.assertRaisesRegex(ValueError, "缓存损坏"):
                    _fixed_evaluate(config, agent, root, "validation", 3, 53001,
                                    0, _Ledger(), scene_set=scene_set, zero_cache=bad)


if __name__ == "__main__":
    unittest.main()
