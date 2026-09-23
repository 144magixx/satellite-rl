"""Deterministic pairing and refusal checks; no GPU work."""

import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest import mock

from satellite_rl.evaluation import post_results_summary as summary
from satellite_rl.evaluation.post_training_checks import ACTION_SEEDS, CONDITIONS, SCENE_IDS, TRAIN_COMMIT
from satellite_rl.training.experiment_runner import sha256_file
from satellite_rl.training.run_monitor import atomic_json


class PostResultsSummaryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def rows(self, si_shift=0.0):
        return [{"scene_id": str(scene), "action_seed": seed, "action_mode": "sample",
                 "si_mean": 0.4 + si_shift, "sgm_mean": 0.3 + si_shift,
                 "throughput_bps": 1e9 * (scene - 80999) + si_shift * 1e9,
                 "total_power_w": 2000 + si_shift * 100,
                 "power_exceeded": False, "completed": True}
                for scene in SCENE_IDS for seed in ACTION_SEEDS]

    def test_conditions_pair_exact_scene_and_action_before_seed_aggregation(self):
        original = summary._sample_rows({"rows": self.rows()})
        changed = summary._sample_rows({"rows": self.rows(0.1)})
        differences = summary.paired_means(original, changed)
        self.assertAlmostEqual(differences["si_mean"], 0.1)
        self.assertAlmostEqual(differences["sgm_mean"], 0.1)
        self.assertAlmostEqual(differences["throughput_bps"], 1e8)
        self.assertAlmostEqual(differences["total_power_w"], 10)
        self.assertEqual(differences["power_exceeded"], 0)
        del changed[("81000", "91001")]
        with self.assertRaisesRegex(RuntimeError, "完整配对"):
            summary.paired_means(original, changed)
        with self.assertRaisesRegex(RuntimeError, "必须恰好"):
            summary._sample_rows({"rows": self.rows()[:-1]})

    def test_original_hundred_scenes_select_only_predeclared_twenty(self):
        original = [{"scene_id": str(scene), "action_seed": seed,
                     "action_mode": "sample", "si_mean": scene / 1e5}
                    for scene in range(81000, 81100) for seed in ACTION_SEEDS]
        selected = summary._original_sample_rows({"rows": original})
        self.assertEqual(len(selected), 60)
        self.assertEqual({scene for scene, _ in selected}, {str(scene) for scene in SCENE_IDS})
        # The formal result also contains one argmax evaluation per scene.
        argmax = [{"scene_id": str(scene), "action_seed": "argmax",
                   "action_mode": "argmax"} for scene in range(81000, 81100)]
        self.assertEqual(summary._original_sample_rows({"rows": original + argmax}), selected)
        with self.assertRaisesRegex(RuntimeError, "100场景"):
            summary._original_sample_rows({"rows": original[:-1]})
        with self.assertRaisesRegex(RuntimeError, "必须恰好"):
            summary._sample_rows({"rows": self.rows() + argmax[:1]})

    def test_old_weight_variation_uses_twenty_scene_means(self):
        trials = [{"scene_id": str(scene), "action_seed": seed,
                   **{field: (scene - 81000) + offset for field in summary.FIELDS}}
                  for scene in SCENE_IDS
                  for seed, offset in zip(ACTION_SEEDS, (-2, 0, 2))]
        measured = summary._std_across_scene_means(trials)
        self.assertAlmostEqual(measured["si_mean"], statistics.stdev(range(20)))
        with self.assertRaisesRegex(RuntimeError, "缺少20场景"):
            summary._std_across_scene_means(trials[:-1])

    def test_three_train_seeds_define_the_uncertainty(self):
        rows = []
        for condition in CONDITIONS:
            for model in ("transformer", "cnn_matched", "mlp_matched"):
                for formula in ("legacy", "corrected"):
                    for seed, si in zip((51001, 51002, 51003), (0.1, 0.2, 0.3)):
                        rows.append({"condition": condition, "model": model,
                                     "formula": formula, "train_seed": seed,
                                     **{field: si for field in summary.FIELDS}})
        result = summary.summarize_three_seeds(rows)
        self.assertEqual(len(result), 18)
        self.assertAlmostEqual(result[0]["si_mean_difference_mean"], 0.2)
        self.assertAlmostEqual(result[0]["si_mean_difference_std_across_train_seeds"], 0.1)
        with self.assertRaisesRegex(RuntimeError, "三个独立训练种子"):
            summary.summarize_three_seeds(rows[:-1])

    def test_timing_index_must_include_and_verify_all_eighteen(self):
        post_root = self.root / "post_training_checks"
        timing_dir = post_root / "timing"
        timing_dir.mkdir(parents=True)
        state_path = timing_dir / "states.npz"
        state_path.write_bytes(b"states")
        eval_commit = "fixed-evaluation-code"
        timing_manifest = {"training_commit": TRAIN_COMMIT, "post_commit": eval_commit,
                           "state_path": str(state_path), "state_sha256": sha256_file(state_path),
                           "warmup_per_scope": 40, "repeats_per_scope": 240,
                           "runtime": {"torch_cuda": "12.1"},
                           "scope_forward": "GPU actor; synchronized before/after",
                           "scope_take_action": "host to action; synchronized before/after"}
        manifest_path = timing_dir / "manifest.json"
        atomic_json(manifest_path, timing_manifest)
        run_ids = {f"{model}_{formula}_s{seed}"
                   for seed in (51001, 51002, 51003)
                   for model in ("transformer", "cnn_matched", "mlp_matched")
                   for formula in ("legacy", "corrected")}
        final = {}
        indexed = {}
        values = {"mean_ms": 1.0, "median_ms": 0.9, "p90_ms": 1.2,
                  "sample_std_ms": 0.1, "min_ms": 0.8, "max_ms": 1.4}
        for run_id in run_ids:
            # CNN/MLP identifiers contain an underscore; recover via suffixes.
            model = run_id.split("_legacy_")[0] if "_legacy_" in run_id else run_id.split("_corrected_")[0]
            formula = "legacy" if "_legacy_" in run_id else "corrected"
            seed = int(run_id.rsplit("s", 1)[-1])
            actor = self.root / (run_id + ".pt")
            actor.write_bytes(run_id.encode())
            final[run_id] = {"actor": str(actor), "actor_sha256": sha256_file(actor),
                             "job": {"model": model, "formula": formula, "seed": seed}}
            path = timing_dir / (run_id + ".json")
            atomic_json(path, {"config": {"run_id": run_id, "post_commit": eval_commit,
                                         "actor_sha256": final[run_id]["actor_sha256"],
                                         "state_sha256": timing_manifest["state_sha256"],
                                         "warmup": 40, "repeats": 240},
                               "parameters": 400000,
                               "gpu_before": {"utilization_percent": 5,
                                              "compute_process_query": "own-process"},
                               "gpu_after": {"utilization_percent": 8,
                                             "compute_process_query": "own-process"},
                               "timing": {"forward_device_tensor": values,
                                          "take_action_host_to_action": values}})
            indexed[run_id] = {"path": str(path), "sha256": sha256_file(path)}
        atomic_json(timing_dir / "result_index.json", {
            "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
            "runs": indexed})
        self.assertEqual(len(summary._read_timing(post_root, final, eval_commit, {})), 18)
        indexed.pop(next(iter(indexed)))
        atomic_json(timing_dir / "result_index.json", {
            "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
            "runs": indexed})
        with self.assertRaisesRegex(RuntimeError, "不完整"):
            summary._read_timing(post_root, final, eval_commit, {})

    def test_incomplete_formal_training_refuses_report_and_output(self):
        output = self.root / "summary"
        with mock.patch.object(summary, "source_guard", return_value={"post_commit": "report"}), \
             mock.patch.object(summary, "require_completed",
                               side_effect=RuntimeError("18组尚未全部完成")):
            with self.assertRaisesRegex(RuntimeError, "尚未全部完成"):
                summary.generate_summary(self.root, self.root, output)
        self.assertFalse(output.exists())

    def test_report_keeps_training_evaluation_and_report_versions_distinct(self):
        exp_dir = self.root / "outputs" / "experiments" / "sac_comparison_20260924_r2"
        post_root = exp_dir / "post_training_checks"
        post_root.mkdir(parents=True)
        atomic_json(exp_dir / "suite_plan.json", {"jobs": []})
        atomic_json(exp_dir / "suite_status.json", {"all_completed": True})
        atomic_json(post_root / "runtime.json", {
            "training_commit": TRAIN_COMMIT, "post_commit": "evaluation-commit",
            "runtime": {"torch_cuda": "12.1"}, "source_hashes": {"env.py": "saved"}})
        by_seed = []
        timing = []
        for model in ("transformer", "cnn_matched", "mlp_matched"):
            for formula in ("legacy", "corrected"):
                for seed in (51001, 51002, 51003):
                    for condition in CONDITIONS:
                        by_seed.append({"condition": condition, "model": model,
                                        "formula": formula, "train_seed": seed,
                                        **{field: 0.1 for field in summary.FIELDS}})
                    timing.append({"model": model, "formula": formula,
                                   "train_seed": seed, "actor_parameters": 417272,
                                   "gpu_before": {"utilization_percent": 2,
                                                  "compute_process_query": "own"},
                                   "gpu_after": {"utilization_percent": 3,
                                                 "compute_process_query": "own"},
                                   **{scope + "_" + field: 1.0
                                      for scope in ("forward_device_tensor",
                                                    "take_action_host_to_action")
                                      for field in ("mean_ms", "median_ms", "p90_ms")}})
        environment = {"mean_differences": {field: 0.1 for field in summary.FIELDS},
                       "paired_trials": [{"scene_id": "81000", "action_seed": 91001,
                                          **{field: 0.1 for field in summary.FIELDS}}]}
        output = self.root / "independent_summary"
        with mock.patch.object(summary, "source_guard", return_value={
                "post_commit": "report-commit", "post_source_root": str(self.root),
                "matched_file_sha256": {"env.py": "report"}}), \
             mock.patch.object(summary, "require_completed", return_value=(exp_dir, {}, {})), \
             mock.patch.object(summary, "_read_condition_results",
                               return_value=(by_seed, summary.summarize_three_seeds(by_seed))), \
             mock.patch.object(summary, "_read_environment_pair", return_value=environment), \
             mock.patch.object(summary, "_read_timing", return_value=timing):
            report = summary.generate_summary(self.root, self.root, output)
        manifest = json.loads((output / "summary_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["training_commit"], TRAIN_COMMIT)
        self.assertEqual(manifest["evaluation_generation_commit"], "evaluation-commit")
        self.assertEqual(manifest["report_code_commit"], "report-commit")
        self.assertIn("unknown", report.read_text(encoding="utf-8"))
        self.assertTrue((output / "conditions_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
