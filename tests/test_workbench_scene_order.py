"""Training scene order and episode-boundary recovery checks."""

from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from satellite_rl.workbench.config import default_config, validate_config
from satellite_rl.workbench.runner import _training
from satellite_rl.workbench.scene_order import SceneOrder


class SceneOrderTests(unittest.TestCase):
    def test_default_matches_historical_modulo_exactly(self):
        config = default_config()
        self.assertEqual(config["training"]["scene_order"], "sequential")
        order = SceneOrder("sequential", 51001, [str(i) for i in range(7)])
        self.assertEqual([order.selection(i)["source_index"] for i in range(24)],
                         [i % 7 for i in range(24)])
        old = deepcopy(config)
        old["training"].pop("scene_order")
        old["training"].pop("scene_order_seed")
        self.assertEqual(validate_config(old), config)

    def test_each_cycle_is_a_reproducible_permutation_without_global_rng_use(self):
        sources = [f"sha-{i}" for i in range(17)]
        first = SceneOrder("shuffle_each_cycle", 51001, sources)
        before = random.getstate()
        cycles = [[first.selection(cycle * len(sources) + position)["source_index"]
                   for position in range(len(sources))] for cycle in range(3)]
        self.assertEqual(random.getstate(), before)
        self.assertTrue(all(sorted(cycle) == list(range(len(sources))) for cycle in cycles))
        self.assertNotEqual(cycles[0], cycles[1])
        restarted = SceneOrder("shuffle_each_cycle", 51001, sources)
        self.assertEqual(cycles, [[restarted.selection(cycle * len(sources) + position)["source_index"]
                                   for position in range(len(sources))] for cycle in range(3)])

    def test_checkpoint_binds_next_position_and_ordered_source_hashes(self):
        sources = ["a" * 64, "b" * 64, "c" * 64]
        order = SceneOrder("shuffle_each_cycle", 51001, sources)
        saved = order.checkpoint_state(5)
        SceneOrder("shuffle_each_cycle", 51001, sources).verify_checkpoint(saved, 5)
        with self.assertRaisesRegex(ValueError, "场景顺序"):
            order.verify_checkpoint(saved, 4)
        with self.assertRaisesRegex(ValueError, "场景顺序"):
            SceneOrder("shuffle_each_cycle", 51001, list(reversed(sources))).verify_checkpoint(saved, 5)
        with self.assertRaisesRegex(ValueError, "场景顺序"):
            SceneOrder("shuffle_each_cycle", 51002, sources).verify_checkpoint(saved, 5)
        with self.assertRaisesRegex(ValueError, "缺少"):
            order.verify_checkpoint(None, 5)
        SceneOrder("sequential", 51001, sources).verify_checkpoint(None, 5)

    def test_runner_paused_sequence_matches_uninterrupted_and_logs_source(self):
        class TinyEnv:
            def __init__(self, _config):
                self.data = None

            def reset(self, *, seed=None, scene=None):
                self.data = scene
                return np.array([0.0])

            def scene(self):
                return self.data

            def action_mask(self):
                return None

            def step(self, _action):
                return np.array([0.0]), 0.0, True, {}

            def metrics(self):
                return {"final_si_mean": 0.0}

        class TinyAgent:
            update_step = 0
            last_action_info = {}

            def act(self, _state, **_kwargs):
                return [0, 0, 0]

            def observe(self, *_args, **_kwargs):
                pass

            def ready(self, *_args):
                return False

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = root / "sources"
            sources.mkdir()
            for index in range(3):
                np.savez_compressed(sources / f"scene_{index}.npz", source=np.array([index]))
            config = default_config()
            config["training"].update(scene_order="shuffle_each_cycle", total_steps=9,
                                      checkpoint_interval=1, log_interval=100)
            config["evaluation"].update(interval_steps=0, test_count=0)

            def run(name, pause, tamper=False):
                directory = root / name
                directory.mkdir()
                job = {"run_id": name, "git_commit": "f" * 40, "model_dir": str(directory / "model"),
                       "source_root": str(root), "training_scene_path": str(sources)}
                ledger = Mock(status={"preflight": {"gpu": "fixture"}})
                checkpoint = {}
                snapshot_steps = []

                def save(path, _agent, progress, _config, environment_state):
                    Path(path).touch()
                    checkpoint["progress"] = deepcopy(progress)
                    checkpoint["environment_state"] = deepcopy(environment_state)

                def load(_path, _agent, _config):
                    return deepcopy(checkpoint["progress"]), deepcopy(checkpoint["environment_state"])

                def record(group, step, _values):
                    if pause and group == "training" and step == 4:
                        (directory / "stop.request").touch()

                ledger.record.side_effect = record
                def snapshot(_directory, _agent, _config, _provenance, step, source_history=None):
                    snapshot_steps.append(step)
                    return str(_directory / "learning_snapshots" / f"step_{step:09d}.pt")

                with patch("satellite_rl.workbench.environment.WorkbenchEnv", TinyEnv), \
                     patch("satellite_rl.workbench.runner.build_agent", side_effect=lambda *_args: TinyAgent()), \
                     patch("satellite_rl.workbench.runner.save_checkpoint", side_effect=save), \
                     patch("satellite_rl.workbench.runner.load_checkpoint", side_effect=load), \
                     patch("satellite_rl.workbench.runner.save_learning_snapshot", side_effect=snapshot), \
                     patch("satellite_rl.workbench.runner.export_policy"), \
                     patch("satellite_rl.workbench.runner._ensure_finite_agent"), \
                     patch("satellite_rl.workbench.runner._fixed_evaluate", return_value=([], {})):
                    _training(directory, job, config, ledger, resume=False)
                    if pause:
                        self.assertEqual(checkpoint["progress"]["episode"], 4)
                        self.assertEqual(checkpoint["environment_state"]["scene_order"]["next_episode"], 4)
                        (directory / "stop.request").unlink()
                        if tamper:
                            np.savez_compressed(sources / "scene_0.npz", source=np.array([999]))
                            with self.assertRaisesRegex(ValueError, "场景顺序"):
                                _training(directory, job, config, ledger, resume=True)
                        else:
                            _training(directory, job, config, ledger, resume=True)
                self.assertEqual(snapshot_steps, ([0, 1, 2, 3, 4] if tamper else
                                                   [0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9] if pause else
                                                   list(range(10))))
                artifacts = json.loads((directory / "artifacts.json").read_text(encoding="utf-8"))
                self.assertEqual(artifacts["learning_snapshots"]["latest"],
                                 str(directory / "learning_snapshots" /
                                     f"step_{4 if tamper else 9:09d}.pt"))
                return [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]

            continuous = run("continuous", False)
            resumed = run("resumed", True)
            self.assertEqual([row["scene_order"] for row in continuous],
                             [row["scene_order"] for row in resumed])
            self.assertEqual([row["source_scene"]["index"] for row in continuous],
                             [row["source_scene"]["index"] for row in resumed])
            self.assertEqual([row["episode"] for row in resumed], list(range(9)))
            for row in resumed:
                self.assertEqual(row["scene_order"]["source_index"], row["source_scene"]["index"])
                self.assertEqual(row["scene_order"]["cycle"], row["episode"] // 3)
                self.assertEqual(row["scene_order"]["cycle_position"], row["episode"] % 3)
            run("tampered", True, tamper=True)


if __name__ == "__main__":
    unittest.main()
