"""Deterministic checks for grouping repeated validation observations."""

import unittest

from satellite_rl.evaluation.experiment_curves import build_comparison_curves


def example_runs():
    runs = []
    for model in ("transformer", "cnn_matched", "mlp_matched"):
        for formula in ("legacy", "corrected"):
            for seed in (1, 2, 3):
                rows = []
                for step in (9 + seed, 20):
                    for action in (11, 12, 13):
                        value = 0.2 + seed * 0.1 + (action - 11) * 0.1
                        rows.append({"checkpoint": "validation", "env_step": step,
                                     "scene_id": "100", "action_seed": action,
                                     "action_mode": "sample", "si_mean": value,
                                     "sgm_mean": value, "throughput_bps": value * 1e9,
                                     "return": value})
                runs.append({"job": {"run_id": f"{model}_{formula}_{seed}",
                                     "model": model, "formula": formula, "seed": seed},
                             "manifest": {"config": {"validation_scene_seeds": [100],
                                                       "action_seeds": [11, 12, 13]}},
                             "evaluations": rows})
    return runs


class ExperimentCurvesTests(unittest.TestCase):
    def test_seed_aggregation_retains_real_validation_step_counts(self):
        result = build_comparison_curves(example_runs(), 10, 20)
        point = result["curves"]["transformer_legacy"][0]
        self.assertEqual(point["env_step_mean"], 11)
        self.assertEqual((point["env_step_min"], point["env_step_max"]), (10, 12))
        self.assertEqual(len(point["by_train_seed"]), 3)
        self.assertAlmostEqual(point["metrics"]["si_mean"]["mean"], 0.5)
        self.assertAlmostEqual(point["metrics"]["si_mean"]["std"], 0.1)

    def test_final_validation_duplicate_is_not_an_additional_training_repeat(self):
        runs = example_runs()
        for run in runs:
            extra = [dict(row, checkpoint="final_validation") for row in run["evaluations"]
                     if row["env_step"] == 20]
            run["evaluations"].extend(extra)
        self.assertEqual(build_comparison_curves(runs, 10, 20),
                         build_comparison_curves(example_runs(), 10, 20))

    def test_missing_observation_or_seed_is_rejected(self):
        runs = example_runs()
        runs[0]["evaluations"].pop()
        with self.assertRaisesRegex(RuntimeError, "缺少或重复验证场景"):
            build_comparison_curves(runs, 10, 20)
        runs = example_runs()[:-1]
        with self.assertRaisesRegex(RuntimeError, "三个不同训练种子"):
            build_comparison_curves(runs, 10, 20)

    def test_conflicting_final_validation_is_rejected(self):
        runs = example_runs()
        extra = [dict(row, checkpoint="final_validation", si_mean=0.9)
                 for row in runs[0]["evaluations"] if row["env_step"] == 20]
        runs[0]["evaluations"].extend(extra)
        with self.assertRaisesRegex(RuntimeError, "重复验证数据不一致"):
            build_comparison_curves(runs, 10, 20)


if __name__ == "__main__":
    unittest.main()
