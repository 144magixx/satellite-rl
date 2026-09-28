import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from satellite_rl.workbench.monitoring import MetricsReader, reduce_points


class MonitoringTests(unittest.TestCase):
    def test_reward_components_keep_group_and_record_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = dict(step=50, wall_time="2026-09-25T00:00:00+00:00", group="reward",
                       values={"si_delta/raw": 2.0, "si_delta/weighted": 0.4})
            (root / "metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            series = {item["tag"]: item for item in MetricsReader().read(root)["series"]}
            self.assertEqual(set(series), {"reward/si_delta/raw", "reward/si_delta/weighted"})
            self.assertEqual(series["reward/si_delta/raw"]["group"], "reward")
            self.assertGreater(series["reward/si_delta/raw"]["points"][0]["wall_time"], 1e9)

    def test_downsampling_keeps_extreme_and_endpoints(self):
        points = [{"step": i, "value": 99 if i == 481 else -88 if i == 722 else 0,
                   "wall_time": i} for i in range(1000)]
        reduced = reduce_points(points, 50)
        self.assertLessEqual(len(reduced), 50)
        self.assertEqual(reduced[0], points[0])
        self.assertEqual(reduced[-1], points[-1])
        self.assertIn(points[481], reduced)
        self.assertIn(points[722], reduced)

    def test_nonfinite_is_diagnostic_not_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [dict(step=1, group="learning", values={"loss": 2.0}),
                       dict(step=2, group="learning", values={"loss": float("nan")}),
                       dict(step=3, group="learning", values={"loss": float("inf")})]
            (root / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
            result = MetricsReader().read(root)
            points = next(item for item in result["series"] if item["tag"] == "learning/loss")["points"]
            self.assertEqual([point["value"] for point in points], [2.0])
            self.assertTrue(any(item["level"] == "error" and len(item["details"]) == 2
                                for item in result["diagnostics"]))

    def test_historical_evaluations_aggregate_scene_by_checkpoint_and_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                {"checkpoint": "first", "action_mode": "sample", "env_step": 10, "scene_id": "a", "si_mean": 0.1},
                {"checkpoint": "first", "action_mode": "sample", "env_step": 10, "scene_id": "b", "si_mean": 0.9},
                {"checkpoint": "first", "action_mode": "greedy", "env_step": 10, "scene_id": "a", "si_mean": 0.8},
                {"checkpoint": "second", "action_mode": "sample", "env_step": 20, "scene_id": "a", "si_mean": 0.7},
            ]
            with (root / "evaluations.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            series = {item["tag"]: item["points"] for item in MetricsReader().read(root)["series"]}
            self.assertEqual(series["evaluation/first/sample/si_mean"][0]["value"], 0.5)
            self.assertEqual(series["evaluation/first/greedy/si_mean"][0]["value"], 0.8)
            self.assertEqual(series["evaluation/second/sample/si_mean"][0]["value"], 0.7)
            self.assertIsNone(series["evaluation/first/sample/si_mean"][0]["wall_time"])

    def test_episode_time_is_accumulated_not_per_episode_duration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "episodes.csv").write_text(
                "env_step,episode,wall_seconds,si_mean\n100,0,2,0.4\n200,1,3,0.5\n", encoding="utf-8")
            series = MetricsReader().read(root)["series"]
            values = next(item for item in series if item["tag"] == "training/si_mean")
            self.assertEqual([point["wall_time"] for point in values["points"]], [2, 5])
            self.assertIn("交互耗时", values["time_basis"])

    def test_incremental_jsonl_keeps_partial_line_and_last_step_wins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "metrics.jsonl"
            path.write_bytes(b'{"step":1,"group":"learning","values":{"loss":2}}\n'
                             b'{"step":2,"group":"learning","values":{"loss":')
            reader = MetricsReader()
            first = reader.read(root)
            self.assertEqual(first["series"][0]["raw_count"], 1)
            with path.open("ab") as stream:
                stream.write(b'3}}\n{"step":1,"group":"learning","values":{"loss":4}}\n')
            second = reader.read(root)
            self.assertEqual([point["value"] for point in second["series"][0]["points"]], [4.0, 3.0])
            self.assertEqual(second["series"][0]["raw_count"], 2)
            self.assertEqual(reader.read(root), second)

    def test_replacement_and_resume_trim_rebuild_cached_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "metrics.jsonl"
            def write(target, values):
                target.write_text("".join(json.dumps({"step": step, "group": "learning",
                                                   "values": {"loss": value}}) + "\n"
                                          for step, value in values), encoding="utf-8")
            write(path, [(1, 1), (2, 2), (3, 3)])
            reader = MetricsReader()
            reader.read(root)
            # Resume reconciliation atomically replaces the log, then appends
            # a new value at a previously observed step.
            replacement = root / "metrics.tmp"
            write(replacement, [(1, 1), (2, 20)])
            os.replace(replacement, path)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": 3, "group": "learning", "values": {"loss": 30}}) + "\n")
            points = reader.read(root)["series"][0]["points"]
            self.assertEqual([(p["step"], p["value"]) for p in points],
                             [(1.0, 1.0), (2.0, 20.0), (3.0, 30.0)])

    def test_unterminated_valid_record_is_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "metrics.jsonl"
            path.write_text('{"step":1,"group":"learning","values":{"loss":NaN}}',
                            encoding="utf-8")
            reader = MetricsReader()
            first = reader.read(root)
            self.assertEqual(len(first["diagnostics"][0]["details"]), 1)
            self.assertEqual(reader.read(root), first)
            with path.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            second = reader.read(root)
            self.assertEqual(len(second["diagnostics"][0]["details"]), 1)

    def test_tensorboard_reuses_accumulator_and_preserves_purge(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            self.skipTest("TensorBoard unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tensorboard"
            with SummaryWriter(log_dir=str(path)) as writer:
                writer.add_scalar("learning/loss", 1.0, 1)
                writer.add_scalar("learning/loss", 2.0, 2)
            reader = MetricsReader()
            first = reader.read(root)
            self.assertEqual([p["value"] for p in first["series"][0]["points"]], [1.0, 2.0])
            accumulator = reader.cache[str(root)]["tb_accumulator"]
            with patch.object(accumulator, "Reload", side_effect=AssertionError("unexpected reload")):
                self.assertEqual(reader.read(root), first)
            with SummaryWriter(log_dir=str(path), purge_step=2) as writer:
                writer.add_scalar("learning/loss", 9.0, 2)
            second = reader.read(root)
            self.assertEqual([(p["step"], p["value"]) for p in second["series"][0]["points"]],
                             [(1.0, 1.0), (2.0, 9.0)])
            self.assertIn("TensorBoard", second["sources"][0])

    def test_reader_cache_is_bounded_across_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader = MetricsReader()
            reader.MAX_RUNS = 2
            for index in range(3):
                directory = root / str(index)
                directory.mkdir()
                (directory / "metrics.jsonl").write_text(
                    json.dumps({"step": index, "group": "training", "values": {"si": index}}) + "\n",
                    encoding="utf-8")
                self.assertEqual(reader.read(directory)["series"][0]["points"][0]["value"], index)
            self.assertEqual(len(reader.cache), 2)
            self.assertNotIn(str(root / "0"), reader.cache)
            self.assertEqual(reader.read(root / "0")["series"][0]["points"][0]["value"], 0)


if __name__ == "__main__":
    unittest.main()
