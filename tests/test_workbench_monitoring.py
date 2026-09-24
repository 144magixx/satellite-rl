import csv
import json
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
