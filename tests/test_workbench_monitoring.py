import csv
import json
from pathlib import Path
import tempfile
import unittest

from satellite_rl.workbench.monitoring import MetricsReader, reduce_points


class MonitoringTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
