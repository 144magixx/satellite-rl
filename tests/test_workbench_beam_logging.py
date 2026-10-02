"""Per-beam service evidence is complete and keeps step/final meanings separate."""
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from satellite_rl.workbench.config import default_config
from satellite_rl.workbench.environment import WorkbenchEnv, beam_service_fields
from satellite_rl.workbench.runner import _fixed_evaluate, _reconcile, _training


class BeamLoggingTests(unittest.TestCase):
    def test_oversupply_ratio_is_not_capped_with_si(self):
        value = beam_service_fields(250.0, 100.0, 0.4)
        self.assertEqual(value["capacity_bps"], 250.0)
        self.assertEqual(value["effective_served_bps"], 100.0)
        self.assertEqual(value["capacity_demand_ratio"], 2.5)
        self.assertEqual(value["si"], 1.0)

    def test_later_real_interference_changes_first_beam_final_value(self):
        config = default_config()
        env = WorkbenchEnv(config)
        env.reset(seed=103)
        first = None
        for step in range(9):
            _, _, _, info = env.step([0, 0, 0])
            if step == 0:
                first = info["beam_service_step"]
        final = env.beam_results()
        self.assertEqual(len(final), 100)
        self.assertNotEqual(first["capacity_bps"], final[0]["capacity_bps"])
        self.assertAlmostEqual(final[0]["capacity_bps"], env.raw_env.beam_bps[0])
        self.assertTrue(all(row["capacity_bps"] == 0 and row["sgm"] == 0
                            for row in final[9:]))
        self.assertEqual(sum(row["si"] for row in final),
                         sum(row["si"] for row in env.beam_results()))

    def test_same_scene_actions_and_rng_preserve_transition(self):
        config = default_config()
        a = WorkbenchEnv(config)
        scene_state = a.reset(seed=103)
        scene = a.scene()
        b = WorkbenchEnv(config)
        np.testing.assert_array_equal(scene_state, b.reset(seed=999, scene=scene))
        for _ in range(9):
            state_a, reward_a, done_a, info_a = a.step([0, 0, 0])
            state_b, reward_b, done_b, info_b = b.step([0, 0, 0])
            np.testing.assert_array_equal(state_a, state_b)
            self.assertEqual((reward_a, done_a), (reward_b, done_b))
            self.assertEqual(info_a["beam_service_step"], info_b["beam_service_step"])

    def test_training_trace_has_every_action_and_all_final_beams(self):
        class Agent:
            update_step = 0
            last_action_info = {}

            def act(self, _state, **_kwargs):
                return [0, 0, 0]

            def observe(self, *_args, **_kwargs):
                pass

            def ready(self, *_args):
                return False

        config = default_config()
        config["training"].update(total_steps=9, log_interval=100, checkpoint_interval=9)
        config["evaluation"].update(interval_steps=0, test_count=0)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            job = {"run_id": "short_logging_check", "git_commit": "f" * 40,
                   "model_dir": str(directory / "model"), "source_root": str(directory)}
            ledger = Mock(status={"preflight": {"gpu": "short_check"}})
            with patch("satellite_rl.workbench.runner.build_agent", return_value=Agent()), \
                 patch("satellite_rl.workbench.runner.save_checkpoint",
                       side_effect=lambda path, *_args: Path(path).touch()), \
                 patch("satellite_rl.workbench.runner.save_learning_snapshot", return_value=None), \
                 patch("satellite_rl.workbench.runner.export_policy"), \
                 patch("satellite_rl.workbench.runner._ensure_finite_agent"), \
                 patch("satellite_rl.workbench.runner._fixed_evaluate", return_value=([], {})):
                _training(directory, job, config, ledger, resume=False)
            traces = [json.loads(line) for line in (directory / "traces.jsonl").read_text().splitlines()]
            self.assertEqual([row["env_step"] for row in traces], list(range(1, 10)))
            self.assertEqual([row["step"] for row in traces], list(range(9)))
            self.assertTrue(all(row["beam_service_step"]["required_bps"] > 0 for row in traces))
            episodes = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]
            self.assertEqual(len(episodes), 1)
            self.assertEqual(len(episodes[0]["beams_final"]), 100)
            self.assertEqual(episodes[0]["beams_final"][99]["capacity_bps"], 0)

    def test_resume_trim_discards_uncheckpointed_beam_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "traces.jsonl"
            path.write_text("".join(json.dumps({"env_step": step, "beam_service_step": {"capacity_bps": step}}) + "\n"
                                    for step in (1, 2, 3)), encoding="utf-8")
            _reconcile(temp, 2)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"env_step": 3, "beam_service_step": {"capacity_bps": 3}}) + "\n")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["env_step"] for row in rows], [1, 2, 3])

    def test_resume_removes_partial_validation_at_checkpoint_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            rows = [
                {"env_step": 2, "group": "training"},
                {"env_step": 2, "group": "validation", "scene_id": "completed"},
                {"env_step": 3, "group": "training"},
                {"env_step": 3, "group": "validation", "scene_id": "partial"},
                {"env_step": 3, "group": "test", "scene_id": "partial_final"},
            ]
            for name in ("traces", "evaluations"):
                (directory / f"{name}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            _reconcile(directory, checkpoint_step=3, last_validation_step=2)
            for name in ("traces", "evaluations"):
                kept = [json.loads(line) for line in
                        (directory / f"{name}.jsonl").read_text().splitlines()]
                self.assertEqual(kept, rows[:3])

    def test_validation_keeps_each_step_and_all_final_beams(self):
        class Policy:
            rng = random.Random(51001)
            last_action_info = {}

            def act(self, _state, **_kwargs):
                return [0, 0, 0]

        config = default_config()
        config["evaluation"]["repeats"] = 1
        with tempfile.TemporaryDirectory() as temp:
            results, _ = _fixed_evaluate(config, Policy(), temp, "validation", 1,
                                          94000, 17, Mock())
            self.assertEqual(len(results), 1)
            result = results[0]
            trace = [json.loads(line) for line in (Path(temp) / "traces.jsonl").read_text().splitlines()]
            record = json.loads((Path(temp) / "evaluations.jsonl").read_text().splitlines()[0])
            self.assertEqual(len(trace), len(result["trace"]))
            self.assertEqual([row["step"] for row in trace], list(range(len(trace))))
            self.assertTrue(all(row["beam_service_step"]["required_bps"] > 0 and
                                row["scene_sha256"] == result["scene"]["sha256"] for row in trace))
            self.assertEqual(len(record["beams_final"]), 100)
            self.assertEqual(record["beams_final"], result["beams"])


if __name__ == "__main__":
    unittest.main()
