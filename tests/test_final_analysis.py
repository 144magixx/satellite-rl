"""Small deterministic checks for offline timing and beam denominators."""

import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from satellite_rl.evaluation import final_analysis
from satellite_rl.training.experiment_runner import sha256_file
from satellite_rl.training.run_monitor import atomic_json


class FinalAnalysisTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_unserved_targets_stay_in_si_denominator_and_actions_do_not(self):
        required = np.array([100, 150, 250, 300, 100, 250], dtype=float) * 1e6
        satisfaction = np.array([1.0, 0.5, 0.25, 0, 0, 0])
        specs = [(2, 2, 10), (4, 0, 20), (3, 2, 30), (1, 1, 40)]
        actions = [{"beam_index": index,
                    "requested": [index % 8, 20, requested_slots, 10 * np.log10(power)],
                    "actual": [index % 8, 20, actual_slots, 10 * np.log10(power)]}
                   for index, (requested_slots, actual_slots, power) in enumerate(specs)]
        raw = final_analysis.defaultdict(final_analysis._new_accumulator)
        final_analysis._accumulate_trial(raw, required, satisfaction, actions)
        low = final_analysis._finish_accumulator(raw[("demand", final_analysis.DEMAND_BANDS[0])])
        medium = final_analysis._finish_accumulator(raw[("demand", final_analysis.DEMAND_BANDS[1])])
        high = final_analysis._finish_accumulator(raw[("demand", final_analysis.DEMAND_BANDS[2])])
        late = final_analysis._finish_accumulator(raw[("phase", final_analysis.PHASES[2])])
        self.assertEqual((low["target_beams"], low["attempted_beams"], low["unserved_beams"]),
                         (2, 1, 1))
        self.assertAlmostEqual(low["final_si_mean"], 0.5)
        self.assertAlmostEqual(low["requested_power_mean_w"], 10)
        self.assertEqual(low["requested_start_counts"], {"20": 1})
        self.assertEqual(low["requested_power_w_counts"], {"10.0": 1})
        self.assertEqual(medium["zero_slot_rate"], 1)
        self.assertEqual(medium["clipped_rate"], 1)
        self.assertEqual((high["target_beams"], high["attempted_beams"]), (3, 2))
        self.assertAlmostEqual(high["final_si_mean"], 0.25 / 3)
        self.assertAlmostEqual(high["unserved_rate"], 1 / 3)
        self.assertIsNone(late["requested_power_mean_w"])
        self.assertEqual(late["final_si_mean"], 0)
        self.assertEqual([final_analysis.phase(index, 100) for index in (0, 33, 34, 66, 67, 99)],
                         ["前段", "前段", "中段", "中段", "后段", "后段"])

    def test_three_training_seeds_are_the_aggregation_units(self):
        rows = []
        for seed, si in ((51001, 0.4), (51002, 0.5), (51003, 0.6)):
            row = {field: 0.0 for field in final_analysis.METRICS}
            row.update(model="transformer", formula="corrected", axis="phase", group="前段",
                       train_seed=seed, final_si_mean=si)
            rows.append(row)
            old = {field: 0.0 for field in final_analysis.METRICS}
            old.update(model="transformer", formula="legacy", axis="phase", group="前段",
                       train_seed=seed, final_si_mean=si - 0.1)
            rows.append(old)
        summary = final_analysis._aggregate_seeds(rows)
        corrected = next(row for row in summary if row["formula"] == "corrected")
        self.assertEqual(corrected["train_seeds"], [51001, 51002, 51003])
        self.assertAlmostEqual(corrected["final_si_mean_mean"], 0.5)
        self.assertAlmostEqual(corrected["final_si_mean_std_across_seeds"], 0.1)
        differences = final_analysis._formula_differences(rows)
        self.assertEqual(len(differences), 1)
        self.assertAlmostEqual(differences[0]["final_si_mean_mean"], 0.1)
        self.assertAlmostEqual(differences[0]["final_si_mean_std_across_seeds"], 0)

    def test_saved_scene_trace_and_terminal_arrays_join_by_scene_and_action_seed(self):
        required = np.array([100, 150, 250, 300, 100, 250], dtype=float) * 1e6
        satisfaction = np.array([1.0, 0.5, 0.25, 0, 0, 0])
        scene_path = self.root / "scene_81000.npz"
        np.savez(scene_path, rate=required[:, None])
        array_path = self.root / "final_test_400000.npz"
        rates = satisfaction * required
        np.savez(array_path, scene_81000_action_91001_required_bps=required,
                 scene_81000_action_91001_rates_bps=rates,
                 scene_81000_action_91001_satisfaction=satisfaction)
        trace_path = self.root / "final_test_400000_steps.jsonl.gz"
        with gzip.open(trace_path, "wt", encoding="utf-8") as stream:
            for index in range(4):
                stream.write(json.dumps({"scene_id": "81000", "action_seed": 91001,
                                         "action_mode": "sample", "beam_index": index,
                                         "requested": [index % 8, 20, 2, 10.0],
                                         "actual": [index % 8, 20, 2, 10.0]}) + "\n")
            stream.write(json.dumps({"scene_id": "81000", "action_seed": "argmax",
                                     "action_mode": "argmax", "beam_index": 0,
                                     "requested": [0, 20, 2, 10.0],
                                     "actual": [0, 20, 2, 10.0]}) + "\n")
        run = {"job": {"run_id": "run", "model": "transformer", "formula": "legacy",
                       "seed": 51001},
               "result": {"trace_path": str(trace_path), "arrays_path": str(array_path)},
               "by_key": {("81000", "91001"): {
                   "si_sum": satisfaction.sum(), "si_mean": satisfaction.mean(),
                   "served_beams": 4, "throughput_bps": rates.sum()}}}
        grouped = final_analysis.behavior_for_run(
            run, {"test_scenes": {"81000": {"path": str(scene_path)}}})
        late = next(row for row in grouped if row["axis"] == "phase" and row["group"] == "后段")
        self.assertEqual((late["target_beams"], late["unserved_beams"]), (2, 2))
        self.assertIsNone(late["requested_power_mean_w"])
        distributions = final_analysis._distribution_rows(grouped)
        self.assertTrue(any(row["dimension"] == "requested_start" and row["value"] == "20"
                            and row["count"] == 2 for row in distributions))
        # A corrupt satisfaction array must be rejected even if the summary row matches it.
        np.savez(array_path, scene_81000_action_91001_required_bps=required,
                 scene_81000_action_91001_rates_bps=rates,
                 scene_81000_action_91001_satisfaction=satisfaction + 0.01)
        with self.assertRaisesRegex(RuntimeError, "逐束速率/需求"):
            final_analysis.behavior_for_run(
                run, {"test_scenes": {"81000": {"path": str(scene_path)}}})

    def test_timing_event_windows_are_labeled_as_blocks(self):
        run_dir = self.root / "run"
        run_dir.mkdir()
        checkpoint = self.root / "final.pt"
        checkpoint.write_bytes(b"final checkpoint")
        checkpoint.with_name("final_actor.pt").write_bytes(b"actor")
        events = [
            {"at": "2026-09-24T00:00:00+00:00", "kind": "started"},
            {"at": "2026-09-24T00:00:10+00:00", "kind": "checkpoint", "name": "latest", "env_step": 10000},
            {"at": "2026-09-24T00:00:14+00:00", "kind": "validation", "env_step": 10000},
            {"at": "2026-09-24T00:00:40+00:00", "kind": "checkpoint", "name": "final",
             "env_step": 400000, "path": str(checkpoint), "sha256": sha256_file(checkpoint)},
            {"at": "2026-09-24T00:00:50+00:00", "kind": "completed"},
        ]
        (run_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
        with (run_dir / "updates.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("update_step", "wall_seconds"))
            writer.writeheader()
            writer.writerows(({"update_step": 100, "wall_seconds": 10},
                              {"update_step": 200, "wall_seconds": 20}))
        run = {"run_dir": run_dir, "job": {"run_id": "run", "model": "transformer",
                                            "formula": "legacy", "seed": 51001, "attempts": 1},
               "status": {"completed_at": "2026-09-24T00:00:50+00:00",
                          "elapsed_seconds": 50, "env_step": 400000, "update_step": 398000}}
        timing = final_analysis.timing_for_run(run, {"steps": 400000})
        self.assertEqual(timing["calendar_seconds_including_retries"], 50)
        self.assertEqual(timing["periodic_validation_block_seconds_approx"], 4)
        self.assertEqual(timing["periodic_validation_count_with_timing"], 1)
        self.assertEqual(timing["final_assessment_block_seconds"], 10)
        self.assertEqual(timing["post_warmup_update_steps_per_second"], 10)

    def test_missing_jobs_refuse_formal_analysis_without_creating_output(self):
        suite = self.root / "suite"
        suite.mkdir()
        atomic_json(suite / "suite_plan.json", {"jobs": [{"state": "running"}]})
        atomic_json(suite / "suite_status.json", {"all_completed": False,
                                                   "counts": {"completed": 0}})
        output = self.root / "analysis"
        with self.assertRaisesRegex(RuntimeError, "尚未全部完成"):
            final_analysis.generate_final_analysis(suite, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
