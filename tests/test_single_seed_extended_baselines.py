"""Check that four-color and random results share scenes and distinct repeats."""

import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import unittest

from satellite_rl.evaluation.single_seed_extended_baselines import (
    FIELDS, load_extended_baselines,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ExtendedBaselineTests(unittest.TestCase):
    def test_full_scope_and_scene_level_random_aggregation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "results"
            directory.mkdir()
            script = root / "evaluator.py"
            script.write_text("# saved evaluator\n", encoding="utf-8")
            trace = directory / "steps.jsonl.gz"
            trace.write_bytes(b"saved trace")
            scenes = {}
            scene_origins = {}
            for scene_id in range(81000, 81100):
                path = root / f"scene_{scene_id}.npz"
                path.write_bytes(str(scene_id).encode())
                scenes[str(scene_id)] = {"path": str(path), "sha256": digest(path)}
                scene_origins[str(scene_id)] = scenes[str(scene_id)]
            seeds = (91001, 91002, 91003)
            rows = []
            for scene_id in range(81000, 81100):
                for policy, action_seed in (("four_color", None),
                                            *(("random", seed) for seed in seeds)):
                    si = (0.5 if policy == "four_color"
                          else {91001: 0.4, 91002: 0.5, 91003: 0.6}[action_seed])
                    rate = 2.0 * si
                    row = {"policy": policy, "scene_id": scene_id,
                           "scene_path": scenes[str(scene_id)]["path"],
                           "scene_sha256": scenes[str(scene_id)]["sha256"],
                           "action_seed": action_seed,
                           "final_rate_bps_per_beam": [rate] * 100,
                           "required_rate_bps_per_beam": [2.0] * 100,
                           "satisfaction_per_beam": [si] * 100,
                           "sgm_per_beam": [si] * 100,
                           "si_sum": si * 100, "si_mean": si, "sgm_mean": si,
                           "throughput_bps": rate * 100,
                           "total_power_w": 2000, "served_beams": 100,
                           "total_beams": 100, "power_exceeded": False,
                           "completed": True, "occupancy_fraction": 0.5,
                           "zero_slot_rate": 0.0, "allocated_slots": 500,
                           "spectral_efficiency": 1.0,
                           "interference_beams": 0,
                           "in_range_interference_beams": 0,
                           "requested_actual_history": [[0, 0, 5, 20]] * 100,
                           "trace_path": str(trace), "trace_sha256": digest(trace)}
                    rows.append(row)
            records = directory / "per_scene_results.jsonl"
            records.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                               encoding="utf-8")
            summary = {}
            for policy in ("four_color", "random"):
                policy_rows = [row for row in rows if row["policy"] == policy]
                summary[policy] = {}
                for field in FIELDS:
                    scene_values = [statistics.mean(row[field] for row in policy_rows
                                                    if row["scene_id"] == scene_id)
                                    for scene_id in range(81000, 81100)]
                    summary[policy][field] = {
                        "mean": statistics.mean(scene_values),
                        "std": statistics.stdev(scene_values),
                    }
            config = {"training_commit": "36c14d74cbe6b9cf7410ee0595ee6d73083afdc8",
                      "training_root": str(root), "evaluation_commit": "synthetic-check",
                      "scene_ids": list(range(81000, 81100)),
                      "scene_count": 100, "rollout_count": 400,
                      "scene_sha256": scene_origins,
                      "action_seeds": list(seeds),
                      "policies": {"four_color": "four_color_policy",
                                   "random": "random_action_rule"},
                      "summary": summary,
                      "per_scene_results_path": str(records),
                      "per_scene_results_sha256": digest(records),
                      "evaluation_script": str(script),
                      "evaluation_script_sha256": digest(script),
                      "source_sha256": {str(script): digest(script)}}
            config_path = directory / "config_and_summary.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            plan = {"source": {"git_commit": config["training_commit"]},
                    "test_scenes": scenes, "action_seeds": list(seeds)}
            previous = {"config": {"policies": {"greedy": "g", "fixed": "f"},
                                   "summary": {"greedy": {}, "fixed": {}}},
                        "by_policy": {policy: {scene: {"si_mean": 0.5}
                                               for scene in scenes}
                                      for policy in ("greedy", "fixed")}}
            joined = load_extended_baselines(plan, root, previous, directory)
            self.assertEqual(set(joined["by_policy"]),
                             {"greedy", "fixed", "four_color", "random"})
            self.assertEqual(len(joined["extended"]["trace_path_sha256"]), 1)
            self.assertAlmostEqual(joined["by_policy"]["random"]["81000"]["si_mean"],
                                   0.5)
            self.assertEqual(joined["config"]["summary"]["random"]["si_mean"]["std"],
                             0.0)  # Three repeats are not 300 independent scenes.

            rows[0]["final_rate_bps_per_beam"][0] = 0.1
            records.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                               encoding="utf-8")
            config["per_scene_results_sha256"] = digest(records)
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "终态逐束服务"):
                load_extended_baselines(plan, root, previous, directory)


if __name__ == "__main__":
    unittest.main()
