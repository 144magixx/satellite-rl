"""Small deterministic checks for the offline four-cell comparison gate."""

import csv
import tempfile
from pathlib import Path
import unittest

from satellite_rl.evaluation import formula_factorial_analysis as analysis


def _rows(value):
    return {(str(scene), str(seed)): {"si_mean": value}
            for scene in (81000, 81001) for seed in analysis.ACTION_SEEDS}


class FactorialAnalysisTests(unittest.TestCase):
    def test_paired_effects_use_four_cells_and_scene_pairing(self):
        cells = {"legacy": _rows(0.2), "negative_joint": _rows(0.3),
                 "positive_per_head": _rows(0.5), "corrected": _rows(0.8)}
        result = analysis.paired_factorial(cells, "si_mean")
        self.assertAlmostEqual(result["entropy_at_per_head"]["mean"], 0.3)
        self.assertAlmostEqual(result["entropy_at_joint"]["mean"], 0.5)
        self.assertAlmostEqual(result["joint_q_at_negative"]["mean"], 0.1)
        self.assertAlmostEqual(result["joint_q_at_positive"]["mean"], 0.3)
        self.assertAlmostEqual(result["interaction"]["mean"], 0.2)
        self.assertEqual(result["interaction"]["independent_scenes"], 2)
        self.assertEqual(result["interaction"]["paired_action_trials"], 6)

    def test_paired_effects_reject_missing_action_and_formula(self):
        cells = {"legacy": _rows(0.2), "negative_joint": _rows(0.3),
                 "positive_per_head": _rows(0.5), "corrected": _rows(0.8)}
        del cells["negative_joint"][("81000", "91003")]
        with self.assertRaisesRegex(RuntimeError, "场景和动作种子键"):
            analysis.paired_factorial(cells, "si_mean")
        del cells["negative_joint"]
        with self.assertRaisesRegex(RuntimeError, "四组合不完整"):
            analysis.paired_factorial(cells, "si_mean")

    def test_new_plan_rejects_missing_run(self):
        jobs = [{"run_id": f"{model}_{formula}_s51001", "seed": 51001,
                 "model": model, "formula": formula}
                for model in analysis.MODELS for formula in analysis.NEW_FORMULAS]
        plan = {"experiment_id": analysis.NEW_EXPERIMENT, "steps": 400000,
                "action_seeds": list(analysis.ACTION_SEEDS), "jobs": jobs[:-1]}
        with self.assertRaisesRegex(RuntimeError, "精确包含"):
            analysis._check_plan(plan, analysis.NEW_EXPERIMENT,
                                 analysis.NEW_FORMULAS, ".")

    def test_new_plan_rejects_mismatched_scene_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary) / "scene.npz"
            scene.write_bytes(b"actual scene data")
            plan = {"test_scenes": {str(i): {"path": str(scene), "sha256": "0" * 64}
                                     for i in range(81000, 81100)}}
            with self.assertRaisesRegex(RuntimeError, "SHA变化"):
                analysis._scene_hashes(plan)

    def test_config_matrix_rejects_non_formula_budget_change(self):
        runs = {}
        for model in analysis.MODELS:
            for formula in analysis.FORMULAS:
                runs[model, formula] = {"manifest": {"config": {
                    "formula": formula, "train_seed": 51001,
                    "budget_env_steps": 400000,
                    "target_entropy": -6.9 if formula in
                    ("legacy", "negative_joint") else 6.9,
                    "run_id": f"{model}_{formula}_s51001",
                    "experiment_id": "fixture", "runtime": {"git_commit": "fixture"},
                }}}
        source = {"source": {"cities_data": {"sha256": "same"}}}
        analysis._check_config_matrix(runs, source, source)
        runs["transformer", "negative_joint"]["manifest"]["config"]["budget_env_steps"] = 300000
        with self.assertRaisesRegex(RuntimeError, "除公式外训练配置"):
            analysis._check_config_matrix(runs, source, source)

    def test_validation_curve_rejects_missing_round(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evaluations.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("checkpoint", "action_mode",
                                        "env_step", "scene_id", "action_seed",
                                        "si_mean", "sgm_mean"))
                writer.writeheader()
                for scene in range(71000, 71020):
                    for seed in analysis.ACTION_SEEDS:
                        writer.writerow({"checkpoint": "validation", "action_mode": "sample",
                                         "env_step": 10013, "scene_id": scene,
                                         "action_seed": seed, "si_mean": 0.5,
                                         "sgm_mean": 0.4})
            plan = {"validation_scenes": {str(i): {} for i in range(71000, 71020)},
                    "action_seeds": list(analysis.ACTION_SEEDS),
                    "validation_interval": 10000}
            with self.assertRaisesRegex(RuntimeError, "缺少完整40轮"):
                analysis._validation_curve(Path(temporary), "fixture", plan)


if __name__ == "__main__":
    unittest.main()
