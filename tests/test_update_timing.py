"""CPU-only checks for the fixed SAC update-cost protocol."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from satellite_rl.evaluation.update_timing import (
    BATCH_SIZE, REPEATS, WARMUP, _batch_dicts, _experiment_gpu_children,
    _finish_reports, _immutable_json, _validate_cached_result, _verify_existing_index,
    fixed_batches, require_ready, summarize_ms, time_updates,
)
from satellite_rl.agents.sac_transformer import ReplayBuffer
from satellite_rl.training import experiment_runner as runner


class UpdateTimingTests(unittest.TestCase):
    def test_saved_batch_selection_is_seeded_disjoint_and_complete(self):
        replay = [(np.full(204, index), [index % 100, (index // 100) % 10, index % 10],
                   float(index), np.full(204, index + 1), bool(index % 2))
                  for index in range(1024)]
        first = fixed_batches(replay, seed=93002, batch_count=2, batch_size=BATCH_SIZE)
        second = fixed_batches(replay, seed=93002, batch_count=2, batch_size=BATCH_SIZE)
        self.assertEqual(first["indices"].shape, (2, 256))
        self.assertEqual(len(set(first["indices"].flat)), 512)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(first["states"][:, :, 0], first["indices"])
        np.testing.assert_array_equal(first["next_states"][:, :, 0], first["indices"] + 1)
        self.assertEqual(first["actions"].shape, (2, 256, 3))

    def test_rebuilt_batches_preserve_real_replay_sample_containers(self):
        replay = [(np.full(204, index, dtype=np.float64), [index % 100, index % 10, 0],
                   float(index), np.full(204, index + 1, dtype=np.float64), bool(index % 2))
                  for index in range(512)]
        arrays = fixed_batches(replay, batch_count=1, batch_size=256)
        restored = _batch_dicts(arrays)[0]
        buffer = ReplayBuffer(capacity=512)
        buffer.buffer.extend(replay)
        actual = buffer.sample(256)
        self.assertIsInstance(restored["states"], np.ndarray)
        self.assertIsInstance(restored["next_states"], np.ndarray)
        self.assertEqual(type(restored["actions"]), type(actual[1]))
        self.assertEqual(type(restored["actions"][0]), type(actual[1][0]))
        self.assertEqual(type(restored["rewards"]), type(actual[2]))
        self.assertEqual(type(restored["dones"]), type(actual[4]))
        self.assertIsInstance(restored["actions"][0][0], int)
        self.assertIsInstance(restored["rewards"][0], float)
        self.assertIsInstance(restored["dones"][0], bool)
        chosen = int(arrays["indices"][0, 0])
        self.assertEqual(restored["actions"][0], replay[chosen][1])
        self.assertEqual(restored["rewards"][0], replay[chosen][2])
        self.assertEqual(restored["dones"][0], replay[chosen][4])

    def test_update_timer_excludes_batch_selection_and_syncs_each_boundary(self):
        class Agent:
            def __init__(self):
                self.calls = []

            def update(self, batch):
                self.calls.append(batch)

            def save_model(self):
                raise AssertionError("计时副本不得写正式权重")

        agent = Agent()
        clock_values = iter(value * 0.002 for value in range(REPEATS * 2))
        sync = mock.Mock()
        batches = [object() for _ in range(8)]
        measurements = time_updates(agent, batches, warmup=WARMUP, repeats=REPEATS,
                                    synchronize=sync, clock=lambda: next(clock_values))
        self.assertEqual(len(agent.calls), WARMUP + REPEATS)
        self.assertEqual(sync.call_count, 1 + 2 * REPEATS)
        self.assertEqual(len(measurements), 100)
        self.assertTrue(all(abs(value - 2.0) < 1e-9 for value in measurements))
        self.assertIs(agent.calls[WARMUP], batches[0])

    def test_millisecond_statistics_and_update_rate_units(self):
        summary = summarize_ms([10.0, 20.0, 30.0, 40.0])
        self.assertAlmostEqual(summary["mean_ms"], 25.0)
        self.assertAlmostEqual(summary["median_ms"], 25.0)
        self.assertAlmostEqual(summary["p90_ms"], 37.0)
        self.assertAlmostEqual(summary["updates_per_second"], 40.0)

    def test_incomplete_eighteen_runs_refuse_before_cuda_preflight(self):
        with mock.patch("satellite_rl.evaluation.update_timing.source_guard",
                        return_value={"post_commit": "test"}), \
             mock.patch("satellite_rl.evaluation.update_timing.require_completed",
                        side_effect=RuntimeError("18 not complete")), \
             mock.patch("satellite_rl.evaluation.update_timing._experiment_gpu_children") as gpu_children:
            with self.assertRaisesRegex(RuntimeError, "18 not complete"):
                require_ready("unused", "unused")
            gpu_children.assert_not_called()

    def test_active_experiment_gpu_child_refuses_timing(self):
        with tempfile.TemporaryDirectory() as folder:
            exp = Path(folder)
            handoff = exp / "post_training_checks" / "handoff"
            handoff.mkdir(parents=True)
            (handoff / "state.json").write_text(json.dumps({
                "training_commit": "36c14d74cbe6b9cf7410ee0595ee6d73083afdc8",
                "stage": "awaiting_state_diagnosis",
                "steps": {name: {"state": "completed"} for name in
                          ("conditions", "environment", "timing", "final_analysis")},
            }), encoding="utf-8")
            plan = {"jobs": [{"state": "completed", "pid": None}]}
            with mock.patch("satellite_rl.evaluation.update_timing.source_guard",
                            return_value={"post_commit": "test"}), \
                 mock.patch("satellite_rl.evaluation.update_timing.require_completed",
                            return_value=(exp, plan, {})), \
                 mock.patch("satellite_rl.evaluation.update_timing._experiment_gpu_children",
                            return_value=[{"ProcessId": 111}]):
                with self.assertRaisesRegex(RuntimeError, "GPU子进程"):
                    require_ready(folder, folder)

    def test_incomplete_post_checks_refuse_before_process_scan(self):
        with tempfile.TemporaryDirectory() as folder:
            exp = Path(folder)
            handoff = exp / "post_training_checks" / "handoff"
            handoff.mkdir(parents=True)
            (handoff / "state.json").write_text(json.dumps({
                "training_commit": "36c14d74cbe6b9cf7410ee0595ee6d73083afdc8",
                "stage": "waiting_training",
                "steps": {"conditions": {"state": "pending"}},
            }), encoding="utf-8")
            with mock.patch("satellite_rl.evaluation.update_timing.source_guard",
                            return_value={"post_commit": "test"}), \
                 mock.patch("satellite_rl.evaluation.update_timing.require_completed",
                            return_value=(exp, {"jobs": []}, {})), \
                 mock.patch("satellite_rl.evaluation.update_timing._experiment_gpu_children") as children:
                with self.assertRaisesRegex(RuntimeError, "四步后处理"):
                    require_ready(folder, folder)
                children.assert_not_called()

    def test_immutable_manifest_rejects_different_config(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            _immutable_json(path, {"seed": 93002})
            with self.assertRaisesRegex(RuntimeError, "拒绝覆盖"):
                _immutable_json(path, {"seed": 93003})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"seed": 93002})

    def test_process_scan_parses_one_live_worker(self):
        with mock.patch("satellite_rl.evaluation.update_timing.subprocess.run") as run:
            run.return_value.stdout = json.dumps({"ProcessId": 111,
                                                  "CommandLine": "experiment_runner"})
            self.assertEqual(_experiment_gpu_children()[0]["ProcessId"], 111)

    def test_process_scan_includes_peer_update_timing_but_excludes_self(self):
        with mock.patch("satellite_rl.evaluation.update_timing.subprocess.run") as run:
            run.return_value.stdout = json.dumps([
                {"ProcessId": os.getpid(), "CommandLine": "update_timing self"},
                {"ProcessId": 99999, "CommandLine": "update_timing peer"},
            ])
            self.assertEqual([item["ProcessId"] for item in _experiment_gpu_children()], [99999])
            command = run.call_args.args[0][-1]
            self.assertIn("update_timing", command)
            self.assertIn(str(os.getpid()), command)

    def test_bad_cached_statistics_and_metadata_are_rejected(self):
        config = {"run_id": "one"}
        raw = [10.0] * REPEATS
        valid = {"config": config, "raw_milliseconds": raw,
                 "statistics": summarize_ms(raw), "parameters_actor": 10,
                 "temporary_log_file": "log.csv", "completed_at": "now",
                 "gpu_before": {"at": "start", "used_mib": 1, "free_mib": 2,
                                "utilization_percent": 3, "compute_process_query": ""},
                 "gpu_after": {"at": "end", "used_mib": 1, "free_mib": 2,
                               "utilization_percent": 3, "compute_process_query": ""}}
        self.assertIs(_validate_cached_result(valid, config), valid)
        bad = json.loads(json.dumps(valid))
        bad["statistics"]["mean_ms"] = 11.0
        with self.assertRaisesRegex(RuntimeError, "不一致"):
            _validate_cached_result(bad, config)
        bad = json.loads(json.dumps(valid))
        del bad["gpu_after"]["free_mib"]
        with self.assertRaisesRegex(RuntimeError, "元数据"):
            _validate_cached_result(bad, config)

    def test_existing_result_index_checks_all_input_hashes_before_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            manifest = directory / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            runs = directory / "runs"
            runs.mkdir()
            refs = {}
            from satellite_rl.evaluation.post_training_checks import EXPECTED_RUN_IDS
            for run_id in EXPECTED_RUN_IDS:
                result = runs / (run_id + ".json")
                result.write_text(run_id, encoding="utf-8")
                refs[run_id] = {"path": str(result), "sha256": runner.sha256_file(result)}
            (directory / "result_index.json").write_text(json.dumps({
                "manifest_sha256": runner.sha256_file(manifest), "runs": refs,
            }), encoding="utf-8")
            _verify_existing_index(directory)
            first = next(iter(refs))
            (runs / (first + ".json")).write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "损坏"):
                _verify_existing_index(directory)

    def test_three_seed_summary_and_formula_difference_keep_units(self):
        jobs, results = [], {}
        for model in runner.MODELS:
            for formula, elapsed in (("legacy", 10.0), ("corrected", 15.0)):
                for seed in runner.TRAIN_SEEDS:
                    run_id = f"{model}_{formula}_s{seed}"
                    jobs.append({"run_id": run_id, "model": model,
                                 "formula": formula, "seed": seed})
                    milliseconds = [elapsed] * 100
                    results[run_id] = {"raw_milliseconds": milliseconds,
                                       "statistics": summarize_ms(milliseconds)}
        with tempfile.TemporaryDirectory() as folder:
            _finish_reports(Path(folder), {"jobs": jobs}, results)
            with (Path(folder) / "formula_differences.csv").open(newline="", encoding="utf-8") as stream:
                import csv
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(float(row["corrected_minus_legacy_mean_ms"]) == 5.0
                                for row in rows))
            self.assertIn("修正公式减原公式", (Path(folder) / "中文更新耗时.md").read_text(encoding="utf-8"))
            report = (Path(folder) / "中文更新耗时.md").read_text(encoding="utf-8")
            self.assertIn("Transformer", report)
            self.assertIn("CNN", report)
            self.assertIn("原公式", report)


if __name__ == "__main__":
    unittest.main()
