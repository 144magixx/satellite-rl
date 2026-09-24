"""CPU-only gates and output-shape checks for six-run post-training commands."""

import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from satellite_rl.evaluation import single_seed_postchecks as checks
from satellite_rl.evaluation import single_seed_update_timing as updates
from satellite_rl.evaluation.update_timing import summarize_ms


def six_jobs():
    return [{"run_id": f"{model}_{formula}_s51001", "model": model,
             "formula": formula, "seed": 51001}
            for model in ("transformer", "cnn_matched", "mlp_matched")
            for formula in ("legacy", "corrected")]


class SingleSeedPostEvalTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_six_run_gate_stops_before_any_gpu_check_if_incomplete(self):
        for module in (checks, updates):
            with self.subTest(module=module.__name__), \
                 mock.patch.object(module.common, "source_guard", return_value={"post_commit": "test"}), \
                 mock.patch.object(module, "require_completed",
                                   side_effect=RuntimeError("six runs incomplete")), \
                 mock.patch.object(module.runner, "strict_cuda_preflight") as cuda:
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    module.require_cuda_ready(self.root, self.root) if module is checks else \
                        module.require_ready(self.root, self.root)
                cuda.assert_not_called()

    def test_completed_status_still_rejects_a_live_worker(self):
        plan = {"jobs": six_jobs()}
        final = {job["run_id"]: {} for job in plan["jobs"]}
        snapshot = {"counts": {"completed": 6},
                    "runs": [{"worker_identity_alive": True}]}
        for module in (checks, updates):
            with self.subTest(module=module.__name__), \
                 mock.patch.object(module.common, "source_guard", return_value={}), \
                 mock.patch.object(module, "require_completed",
                                   return_value=(self.root, plan, final)), \
                 mock.patch.object(module, "current_six_status", return_value=snapshot), \
                 mock.patch.object(module.runner, "strict_cuda_preflight") as cuda:
                with self.assertRaisesRegex(RuntimeError, "尚未退出"):
                    module.require_cuda_ready(self.root, self.root) if module is checks else \
                        module.require_ready(self.root, self.root)
                cuda.assert_not_called()

    def test_condition_entry_writes_exactly_three_times_six_scoped_results(self):
        plan = {"jobs": six_jobs()}
        final = {job["run_id"]: {"actor": str(self.root / "actor.pt"),
                                 "original_test": str(self.root / "original.json"),
                                 "original_test_sha256": "saved"}
                 for job in plan["jobs"]}
        self.assertEqual(checks.scope_root(self.root), self.root / "single_seed_51001")

        def scenes(post_root, *_):
            result = {}
            for condition in checks.common.CONDITIONS:
                path = post_root / "scenes" / condition / "manifest.json"
                path.parent.mkdir(parents=True)
                path.write_text("{}", encoding="utf-8")
                result[condition] = {}
            return result

        def evaluate(task_dir, *_):
            path = task_dir / "evaluation" / "final_test_400000.json"
            path.parent.mkdir(parents=True)
            path.write_text("{}", encoding="utf-8")

        def save_summary(post_root, *_):
            numeric = post_root / "conditions" / "paired_differences.json"
            report = post_root / "conditions" / "中文条件对照.md"
            numeric.write_text("{}", encoding="utf-8")
            report.write_text("仅一训练种子", encoding="utf-8")
            return {"numeric": str(numeric), "numeric_sha256": checks.runner.sha256_file(numeric),
                    "report": str(report), "report_sha256": checks.runner.sha256_file(report)}

        with mock.patch.object(checks, "require_cuda_ready", return_value=(
                {"post_commit": "report", "matched_file_sha256": {}}, self.root,
                plan, final, {"python": "DQN"})), \
             mock.patch.object(checks.common, "prepare_condition_scenes", side_effect=scenes), \
             mock.patch.object(checks.common, "load_final_agent", return_value=object()), \
             mock.patch.object(checks.common, "_task_config", return_value={}), \
             mock.patch.object(checks, "_save_condition_summary", side_effect=save_summary), \
             mock.patch.object(checks.common, "_run_evaluation_task", side_effect=evaluate) as called, \
             mock.patch.object(checks.torch.cuda, "empty_cache"):
            checks.run_conditions(self.root, self.root)
        self.assertEqual(called.call_count, 18)
        path = self.root / "single_seed_51001" / "conditions" / "result_index.json"
        index = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(index["conditions"]), 3)
        self.assertTrue(all(len(item["runs"]) == 6 for item in index["conditions"].values()))
        checks._verify_existing_index(path, "conditions", set(final), "report")
        one = next(iter(index["conditions"].values()))
        changed_path = Path(next(iter(one["runs"].values()))["path"])
        changed_path.write_text('{"changed":true}', encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "校验值不符"):
            checks._verify_existing_index(path, "conditions", set(final), "report")

    def test_inference_cached_result_requires_positive_stats_and_gpu_snapshot(self):
        values = {"mean_ms": 1., "median_ms": 1., "p90_ms": 1.2,
                  "min_ms": .8, "max_ms": 1.4}
        result = {"config": {"run_id": "a"}, "parameters": 417272,
                  "completed_at": "time", "gpu_before": {"at": "time", "compute_process_query": "own"},
                  "gpu_after": {"at": "time", "compute_process_query": "own"},
                  "timing": {"forward_device_tensor": values,
                             "take_action_host_to_action": values}}
        self.assertIs(checks._validate_timing_result(result, result["config"]), result)
        result["timing"]["forward_device_tensor"] = {**values, "mean_ms": -1}
        with self.assertRaisesRegex(RuntimeError, "数值无效"):
            checks._validate_timing_result(result, result["config"])

    def test_exclusive_timing_refuses_another_gpu_compute_process(self):
        with mock.patch.object(checks.common, "_gpu_snapshot", return_value={
                "compute_process_query": f"{os.getpid()}, 100\n999999, 200"}):
            with self.assertRaisesRegex(RuntimeError, "其他计算进程"):
                checks.exclusive_gpu_snapshot()
        with mock.patch.object(checks.common, "_gpu_snapshot", return_value={
                "compute_process_query": f"{os.getpid()}, 100"}):
            checks.exclusive_gpu_snapshot()

    def test_saved_condition_and_old_environment_summaries_are_single_seed(self):
        from satellite_rl.evaluation.post_training_checks import ACTION_SEEDS, SCENE_IDS
        post_root = self.root / "single_seed_51001"
        sample = [{"scene_id": str(scene), "action_seed": seed,
                   "action_mode": "sample", "si_mean": 0.4, "sgm_mean": 0.3,
                   "throughput_bps": 1e9, "total_power_w": 2000,
                   "power_exceeded": False, "completed": True}
                  for scene in SCENE_IDS for seed in ACTION_SEEDS]
        original = [{**row, "scene_id": str(scene)}
                    for scene in range(81000, 81100)
                    for row in sample[:3]]
        changed = [{**row, "si_mean": 0.5, "sgm_mean": 0.4,
                    "throughput_bps": 1.1e9} for row in sample]
        final = {}
        plan = {"jobs": six_jobs()}
        for job in plan["jobs"]:
            run_id = job["run_id"]
            original_path = self.root / f"{run_id}.json"
            original_path.write_text(json.dumps({"rows": original}), encoding="utf-8")
            final[run_id] = {"original_test": str(original_path)}
            for condition in checks.common.CONDITIONS:
                path = post_root / "conditions" / condition / run_id / "evaluation" / "final_test_400000.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"rows": changed}), encoding="utf-8")
        checks._save_condition_summary(post_root, plan, final)
        numeric = json.loads((post_root / "conditions" / "paired_differences.json").read_text(encoding="utf-8"))
        self.assertEqual(len(numeric["rows"]), 18)
        self.assertTrue(all(abs(row["si_mean"] - 0.1) < 1e-12 for row in numeric["rows"]))
        self.assertIn("仅训练种子51001",
                      (post_root / "conditions" / "中文条件对照.md").read_text(encoding="utf-8"))
        for label, rows in (("before_c4963ba", sample), ("corrected_r2", changed)):
            path = post_root / "environment_comparison" / label / "evaluation" / "final_test_unknown.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
        checks._save_environment_summary(post_root)
        environment = json.loads((post_root / "environment_comparison" / "paired_differences.json").read_text(encoding="utf-8"))
        self.assertEqual(environment["independent_training_repeats"], 1)
        self.assertAlmostEqual(environment["mean_differences"]["si_mean"], 0.1)

    def test_environment_entry_writes_only_two_single_weight_results(self):
        plan = {"jobs": six_jobs(), "test_scenes": {
            str(scene): {"path": "saved", "sha256": "saved"}
            for scene in checks.common.SCENE_IDS}}
        frozen = self.root / "frozen"
        actor = frozen / "models" / "pretrained" / "sac_transformer" / "actor.pth"
        actor.parent.mkdir(parents=True)
        actor.write_bytes(b"historical actor")
        old_source = self.root / "old_env.py"
        old_source.write_text("# saved", encoding="utf-8")

        def evaluate(task_dir, _task_id, _config, _agent, _refs, **_kwargs):
            path = task_dir / "evaluation" / "final_test_unknown.json"
            path.parent.mkdir(parents=True)
            path.write_text("{}", encoding="utf-8")

        with mock.patch.object(checks, "require_cuda_ready", return_value=(
                {"post_commit": "report", "matched_file_sha256": {}}, self.root,
                plan, {}, {"python": "DQN"})), \
             mock.patch.object(checks.common, "legacy_environment_source",
                               return_value=(object(), old_source, checks.runner.sha256_file(old_source))), \
             mock.patch.object(checks.common, "EvaluationAgent", return_value=mock.Mock(actor=object())), \
             mock.patch.object(checks, "load_network_checkpoint"), \
             mock.patch.object(checks.common, "_run_evaluation_task", side_effect=evaluate) as called, \
             mock.patch.object(checks, "_save_environment_summary", return_value={}), \
             mock.patch.object(checks.torch.cuda, "empty_cache"):
            checks.run_environment_pair(self.root, frozen)
        self.assertEqual(called.call_count, 2)
        self.assertTrue(all(call.kwargs["step"] == "unknown" for call in called.call_args_list))
        post_root = self.root / "single_seed_51001"
        index = json.loads((post_root / "environment_comparison" / "result_index.json").read_text(encoding="utf-8"))
        self.assertEqual(set(index["runs"]), {"before_c4963ba", "corrected_r2"})
        source = json.loads((post_root / "environment_comparison" / "source_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(source["scenes"]), 20)
        self.assertEqual(source["action_seeds"], list(checks.common.ACTION_SEEDS))
        self.assertFalse((self.root / "post_training_checks").exists())

    def test_inference_timing_entry_indexes_six_actual_state_runs(self):
        plan = {"jobs": six_jobs()}
        final = {job["run_id"]: {"actor_sha256": "actor"} for job in plan["jobs"]}
        state_path = self.root / "states.npz"
        state_path.write_bytes(b"fixed states")
        snapshot = {"at": "time", "compute_process_query": "own", "used_mib": 100,
                    "free_mib": 10000, "utilization_percent": 2}
        distribution = {"mean_ms": 1., "median_ms": .9, "p90_ms": 1.2,
                        "min_ms": .8, "max_ms": 1.4, "sample_std_ms": .1}
        actor = torch.nn.Linear(2, 1)
        fake_agent = mock.Mock(actor=actor)
        with mock.patch.object(checks, "require_cuda_ready", return_value=(
                {"post_commit": "report", "matched_file_sha256": {}}, self.root,
                plan, final, {"python": "DQN"})), \
             mock.patch.object(checks, "exclusive_gpu_snapshot", return_value=snapshot), \
             mock.patch.object(checks.common, "make_timing_states",
                               return_value=(state_path, checks.runner.sha256_file(state_path),
                                             np.zeros((80, 204), dtype=np.float32))), \
             mock.patch.object(checks.common, "load_final_agent", return_value=fake_agent), \
             mock.patch.object(checks.common, "_time_actor", return_value={
                 "forward_device_tensor": distribution,
                 "take_action_host_to_action": distribution}), \
             mock.patch.object(checks.common, "_gpu_snapshot", return_value=snapshot), \
             mock.patch.object(checks.torch.cuda, "empty_cache"):
            checks.run_inference_timing(self.root, self.root)
        timing = self.root / "single_seed_51001" / "timing"
        index = json.loads((timing / "result_index.json").read_text(encoding="utf-8"))
        self.assertEqual(set(index["runs"]), set(final))
        self.assertIn("仅来自训练种子51001", (timing / "中文推理计时.md").read_text(encoding="utf-8"))

    def test_update_report_uses_six_runs_without_false_seed_variance(self):
        plan = {"jobs": six_jobs()}
        raw = [10.0] * 100
        results = {job["run_id"]: {"raw_milliseconds": raw,
                                   "statistics": summarize_ms(raw),
                                   "parameters_actor": 417272}
                   for job in plan["jobs"]}
        updates._finish_reports(self.root, plan, results)
        with (self.root / "summary.csv").open(newline="", encoding="utf-8") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 6)
        with (self.root / "formula_differences.csv").open(newline="", encoding="utf-8") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 3)
        text = (self.root / "中文更新耗时.md").read_text(encoding="utf-8")
        self.assertIn("同一个训练种子", text)
        self.assertNotIn("三种子样本标准差", text)

    def test_full_update_entry_indexes_only_six_copied_agents(self):
        plan = {"jobs": six_jobs()}
        final = {}
        for job in plan["jobs"]:
            run_id = job["run_id"]
            final[run_id] = {"checkpoint": str(self.root / (run_id + ".pt")),
                             "checkpoint_sha256": "checkpoint-hash",
                             "actor": str(self.root / (run_id + ".actor.pt")),
                             "actor_sha256": "actor-hash"}
            manifest = self.root / run_id / "manifest.json"
            manifest.parent.mkdir()
            manifest.write_text('{"config_hash":"fixed"}', encoding="utf-8")
        batch = self.root / "fixed_batches.npz"
        batch.write_bytes(b"saved fixed batches")
        arrays = {"indices": np.zeros((8, 256), dtype=np.int64)}
        actor = torch.nn.Linear(2, 1)
        agent = mock.Mock(actor=actor, log_file="temporary log")
        gpu = {"at": "time", "compute_process_query": "own", "used_mib": 100,
               "free_mib": 10000, "utilization_percent": 2}
        with mock.patch.object(updates, "require_ready", return_value=(
                {"post_commit": "report", "matched_file_sha256": {}}, self.root,
                plan, final, {"python": "DQN"})), \
             mock.patch.object(updates, "exclusive_gpu_snapshot", return_value=gpu), \
             mock.patch.object(updates.shared, "_load_or_save_batches",
                               return_value=(batch, "batch-hash", arrays, 20000)), \
             mock.patch.object(updates.shared, "_batch_dicts", return_value=[{}] * 8), \
             mock.patch.object(updates.runner, "_create_agent", return_value=(agent, object())), \
             mock.patch.object(updates.runner, "load_checkpoint",
                               return_value={"env_step": 400000, "update_step": 398000}), \
             mock.patch.object(updates.shared, "_gpu_snapshot", return_value=gpu), \
             mock.patch.object(updates.shared, "time_updates", return_value=[10.] * 100), \
             mock.patch.object(updates.runner, "_ensure_finite_agent"), \
             mock.patch.object(updates.torch.cuda, "empty_cache"):
            updates.run_update_timing(self.root, self.root)
        output = self.root / "single_seed_51001" / "update_timing"
        index = json.loads((output / "result_index.json").read_text(encoding="utf-8"))
        self.assertEqual(set(index["runs"]), set(final))
        self.assertIn("同一个训练种子", (output / "中文更新耗时.md").read_text(encoding="utf-8"))
        self.assertFalse((self.root / "post_training_checks").exists())


if __name__ == "__main__":
    unittest.main()
