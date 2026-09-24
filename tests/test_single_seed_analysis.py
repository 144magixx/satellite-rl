"""CPU-only tests for one-seed metrics, curves, and saved baselines."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from satellite_rl.evaluation.experiment_report import MAIN_FIELDS
from satellite_rl.evaluation.single_seed_analysis import (
    _plot_training_episodes, _plot_validation, _report_text,
    compare_with_baselines, generate_single_seed_analysis,
    learning_curve_for_run, load_same_scene_baselines, paired_formula_differences,
    summarize_runs, validation_curve_for_run,
)
from satellite_rl.training import experiment_runner as runner


class SingleSeedAnalysisTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="single_seed_analysis_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_incomplete_six_run_gate_stops_before_analysis_output(self):
        with mock.patch("satellite_rl.evaluation.single_seed_analysis.source_guard",
                        return_value={"post_commit": "test"}), \
             mock.patch("satellite_rl.evaluation.single_seed_analysis.require_completed",
                        side_effect=RuntimeError("six not complete")), \
             mock.patch("satellite_rl.evaluation.single_seed_analysis.load_same_scene_baselines") as baseline:
            with self.assertRaisesRegex(RuntimeError, "six not complete"):
                generate_single_seed_analysis(self.root, self.root)
            baseline.assert_not_called()

    def test_single_seed_curve_keeps_forty_actual_steps_and_deduplicates_final(self):
        scenes = list(range(71000, 71020))
        seeds = list(runner.ACTION_SEEDS)
        rows = []
        for index in range(1, 41):
            for scene in scenes:
                for seed in seeds:
                    rows.append({"checkpoint": "validation", "action_mode": "sample",
                                 "env_step": index * 10000, "scene_id": scene,
                                 "action_seed": seed, "si_mean": index / 100,
                                 "sgm_mean": 0.5, "throughput_bps": 1e9,
                                 "return": float(index)})
        rows.extend({**row, "checkpoint": "final_validation"} for row in rows[-60:])
        run = {"job": {"run_id": "transformer_legacy_s51001"},
               "manifest": {"config": {"validation_scene_seeds": scenes,
                                       "action_seeds": seeds}},
               "evaluations": rows}
        plan = {"validation_interval": 10000}
        curve = validation_curve_for_run(run, plan)
        self.assertEqual(len(curve), 40)
        self.assertEqual((curve[0]["env_step"], curve[-1]["env_step"]), (10000, 400000))
        self.assertAlmostEqual(curve[-1]["si_mean"], 0.4)
        run["evaluations"] = rows[:-1]
        with self.assertRaisesRegex(RuntimeError, "缺失或重复"):
            validation_curve_for_run(run, plan)

    def test_learning_curve_keeps_all_update_points_and_head_entropies(self):
        path = self.root / "updates.csv"
        fields = ("env_step", "update_step", "alpha", "entropy", "entropy_freq",
                  "entropy_slots", "entropy_power", "actor_loss", "critic_1_loss",
                  "critic_2_loss", "alpha_loss", "q_mean", "target_mean")
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for step in range(100, 398001, 100):
                writer.writerow({"env_step": step + 2000, "update_step": step,
                                 **{name: 1.0 for name in fields[2:]}})
        curve = learning_curve_for_run({"run_dir": self.root,
                                        "job": {"run_id": "transformer_legacy_s51001"}})
        self.assertEqual(len(curve), 3980)
        self.assertEqual(curve[-1]["update_step"], 398000)
        self.assertEqual(curve[-1]["entropy_power"], 1.0)

    def _paired_runs(self):
        runs = []
        for model in runner.MODELS:
            for formula, si in (("legacy", 0.5), ("corrected", 0.6)):
                rows = []
                for scene in ("81000", "81001"):
                    for action in (91001, 91002):
                        row = {"scene_id": scene, "action_seed": action,
                               **{field: 1.0 for field in MAIN_FIELDS}}
                        row.update(si_mean=si, sgm_mean=si, throughput_bps=si * 1e9,
                                   total_power_w=2000 + si * 100)
                        rows.append(row)
                runs.append({"job": {"run_id": f"{model}_{formula}_s51001",
                                     "model": model, "formula": formula, "seed": 51001},
                             "rows": rows,
                             "argmax_rows": rows[:2],
                             "best_test": {"rows": [{**row, "action_mode": "sample"}
                                                    for row in rows]},
                             "by_key": {(str(row["scene_id"]), str(row["action_seed"])): row
                                        for row in rows}})
        return runs

    def test_one_seed_summary_and_paired_difference_have_no_seed_std(self):
        runs = self._paired_runs()
        summary = summarize_runs(runs)
        self.assertEqual(len(summary), 6)
        self.assertNotIn("std_across_train_seeds", summary[0])
        pairs = paired_formula_differences(runs, {"test_scenes": {"81000": {}, "81001": {}}})
        self.assertEqual(len(pairs), 3)
        self.assertAlmostEqual(pairs[0]["difference"]["si_mean"]["mean_over_paired_actions"], 0.1)
        self.assertEqual(pairs[0]["higher_scenes"]["si_mean"], 2)
        baseline = {"by_policy": {"greedy": {scene: {"si_mean": 0.55}
                                             for scene in ("81000", "81001")},
                                  "fixed": {scene: {"si_mean": 0.65}
                                            for scene in ("81000", "81001")}}}
        comparisons = compare_with_baselines(runs, baseline)
        self.assertEqual(len(comparisons), 12)

    def test_plot_uses_saved_single_seed_points_without_std_band(self):
        curves = {f"{model}_{formula}": [{"env_step": step,
                                           "si_mean": 0.5, "sgm_mean": 0.4,
                                           "throughput_bps": 1e9, "return": 10.0}
                                          for step in (10000, 20000)]
                  for model in runner.MODELS for formula in ("legacy", "corrected")}
        path = _plot_validation(curves, self.root)
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 1000)
        runs = [{"job": {"model": model, "formula": formula},
                 "episodes": [{"env_step": step, "return": step / 1000,
                               "si_mean": 0.5} for step in range(100, 601, 100)]}
                for model in runner.MODELS for formula in ("legacy", "corrected")]
        training_figures = _plot_training_episodes(runs, self.root)
        self.assertEqual(len(training_figures), 3)
        self.assertTrue(all(path.stat().st_size > 1000 for path in training_figures))

    def test_report_states_one_seed_and_keeps_math_evidence_separate(self):
        runs = self._paired_runs()
        summary = summarize_runs(runs)
        for row in summary:
            row["parameters"] = {"actor": 100, "critic_each": 100}
        paired = paired_formula_differences(runs, {"test_scenes": {"81000": {}, "81001": {}}})
        baseline = {"config": {"summary": {policy: {field: {"mean": value}
                                             for field, value in (("si_mean", 0.6),
                                                                  ("sgm_mean", 0.6),
                                                                  ("total_power_w", 2500),
                                                                  ("power_exceeded", 0),
                                                                  ("completed", 1),
                                                                  ("allocated_slots", 500))}
                                          for policy in ("greedy", "fixed")}}}
        timing = [{"run_id": run["job"]["run_id"],
                   "calendar_seconds_including_retries": 3600,
                   "recorded_active_seconds": 3300} for run in runs]
        learning = {f"{run['job']['model']}_{run['job']['formula']}":
                    [{"alpha": 0.1, "critic_1_loss": 1.0, "q_mean": 0.2}]
                    for run in runs}
        text = _report_text(summary, paired, baseline, [], timing, [], learning, None,
                            {"post_commit": "analysis-sha"})
        self.assertIn("不能估计训练种子间波动", text)
        self.assertIn("数学与环境计算的独立证据", text)
        self.assertIn("贪心", text)
        self.assertNotIn("训练种子间标准差：", text)

    def test_saved_baseline_source_and_scenarios_are_checked(self):
        baseline_root = self.root / "baseline"
        baseline_root.mkdir()
        scenes = {}
        for scene_id in (81000, 81001):
            path = self.root / f"scene_{scene_id}.npz"
            path.write_bytes(str(scene_id).encode())
            scenes[str(scene_id)] = {"path": str(path), "sha256": runner.sha256_file(path)}
        records = []
        for policy, si in (("greedy", 0.6), ("fixed", 0.7)):
            for scene_id in (81000, 81001):
                records.append({"policy": policy, "scene_id": scene_id,
                                "scene_path": scenes[str(scene_id)]["path"],
                                "scene_sha256": scenes[str(scene_id)]["sha256"],
                                "total_beams": 100,
                                "satisfaction_per_beam": [si] * 100,
                                "sgm_per_beam": [si] * 100,
                                "si_sum": si * 100, "si_mean": si,
                                "sgm_mean": si, "throughput_bps": 1e9,
                                "total_power_w": 2000})
        (baseline_root / "per_scene_results.jsonl").write_text(
            "\n".join(json.dumps(row) for row in records), encoding="utf-8")
        config = {"git_commit": "36c14d74cbe6b9cf7410ee0595ee6d73083afdc8",
                  "scene_ids": [81000, 81001], "scene_count": 2,
                  "code_root": str(self.root),
                  "policies": {"greedy": "greedy", "fixed": "fixed"},
                  "source_sha256": {},
                  "summary": {policy: {field: {"mean": value}
                                       for field, value in (("si_mean", si), ("sgm_mean", si),
                                                            ("throughput_bps", 1e9),
                                                            ("total_power_w", 2000))}
                              for policy, si in (("greedy", 0.6), ("fixed", 0.7))}}
        (baseline_root / "config_and_summary.json").write_text(json.dumps(config), encoding="utf-8")
        loaded = load_same_scene_baselines({"test_scenes": scenes}, self.root, baseline_root)
        self.assertEqual({key: len(value) for key, value in loaded["by_policy"].items()},
                         {"greedy": 2, "fixed": 2})
        records[0]["scene_sha256"] = "wrong"
        (baseline_root / "per_scene_results.jsonl").write_text(
            "\n".join(json.dumps(row) for row in records), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "终态需求满足率不一致"):
            load_same_scene_baselines({"test_scenes": scenes}, self.root, baseline_root)


if __name__ == "__main__":
    unittest.main()
