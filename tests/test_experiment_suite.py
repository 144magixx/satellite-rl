"""Queue safety, recovery and paired-report checks without starting training."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from satellite_rl.evaluation import experiment_report
from satellite_rl.training import experiment_suite
from satellite_rl.training.run_monitor import atomic_json


class _ExitedProcess:
    pid = 12345

    def __init__(self, code=0):
        self.code = code

    def poll(self):
        return self.code


class ExperimentSuiteTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.directory = experiment_suite.suite_dir(self.root, "test_suite")
        self.directory.mkdir(parents=True)
        self.source = {"python": os.path.abspath(__file__),
                       "source_root": str(self.root), "git_commit": "frozen",
                       "cities_data": {"sha256": "fixed"}}
        self.plan = {"experiment_id": "test_suite", "archive_root": str(self.root),
                     "source": self.source, "steps": 400000,
                     "validation_interval": 10000, "checkpoint_interval": 10000,
                     "validation_scenes": {}, "test_scenes": {},
                     "action_seeds": [91001, 91002, 91003],
                     "jobs": experiment_suite.deterministic_jobs()}

    def test_plan_has_exact_interleaved_pairs_and_source_guard(self):
        token = experiment_suite._process_token(os.getpid())
        self.assertTrue(experiment_suite.process_alive(os.getpid(), token))
        if token is not None:
            self.assertFalse(experiment_suite.process_alive(os.getpid(), token + 1))
        jobs = self.plan["jobs"]
        self.assertEqual(len(jobs), 18)
        self.assertEqual(len({job["run_id"] for job in jobs}), 18)
        self.assertEqual([(job["model"], job["formula"]) for job in jobs[:6]], [
            ("transformer", "legacy"), ("transformer", "corrected"),
            ("cnn_matched", "legacy"), ("cnn_matched", "corrected"),
            ("mlp_matched", "legacy"), ("mlp_matched", "corrected")])
        experiment_suite._validate_plan(self.plan)
        with mock.patch.object(experiment_suite, "strict_cuda_preflight",
                               return_value={**self.source, "git_commit": "different"}):
            with self.assertRaisesRegex(RuntimeError, "冻结源码提交"):
                experiment_suite._verify_source(self.plan)

    def test_launch_uses_frozen_dqn_python_and_only_resumes_with_checkpoint(self):
        job = self.plan["jobs"][0]
        run_dir = self.directory / job["run_id"]
        with mock.patch.object(experiment_suite.subprocess, "Popen",
                               return_value=_ExitedProcess()) as launch:
            experiment_suite._launch(self.directory, self.plan, job)
        self.assertFalse(run_dir.exists())  # The worker creates its own run directory.
        command = launch.call_args.args[0]
        self.assertEqual(command[:4], [self.source["python"], "-u", "-m",
                                       "satellite_rl.training.experiment_runner"])
        self.assertNotIn("--resume", command)
        self.assertEqual(launch.call_args.kwargs["env"]["PYTHONPATH"],
                         str(self.root / "src"))
        self.assertEqual(launch.call_args.kwargs["env"]["OMP_NUM_THREADS"], "1")
        self.assertEqual(launch.call_args.kwargs["env"]["MKL_NUM_THREADS"], "1")
        self.assertEqual(launch.call_args.kwargs["cwd"], str(self.root))
        self.assertEqual(job["attempts"], 1)
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
        checkpoint = self.root / "models" / "runs" / "test_suite" / job["run_id"] / "latest.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        with mock.patch.object(experiment_suite.subprocess, "Popen",
                               return_value=_ExitedProcess()) as launch:
            experiment_suite._launch(self.directory, self.plan, job)
        self.assertIn("--resume", launch.call_args.args[0])

    def test_completed_status_requires_final_artifact_and_matching_commit(self):
        job = self.plan["jobs"][0]
        run_dir = self.directory / job["run_id"]
        (run_dir / "evaluation").mkdir(parents=True)
        atomic_json(run_dir / "status.json", {"stage": "completed", "env_step": 400000})
        atomic_json(run_dir / "manifest.json", {"config": {
            "model": job["model"], "formula": job["formula"],
            "train_seed": job["seed"], "runtime": {"git_commit": "frozen"}}})
        self.assertFalse(experiment_suite._worker_completed(self.directory, job, self.plan))
        atomic_json(run_dir / "evaluation" / "final_test_400000.json", {"rows": []})
        self.assertTrue(experiment_suite._worker_completed(self.directory, job, self.plan))
        self.plan["source"]["git_commit"] = "changed"
        self.assertFalse(experiment_suite._worker_completed(self.directory, job, self.plan))

    def test_supervisor_waits_for_all_jobs_before_reporting(self):
        atomic_json(self.directory / "suite_plan.json", self.plan)

        def launch(_directory, _plan, job):
            job["state"] = "running"
            job["attempts"] += 1
            job["pid"] = 12345
            return _ExitedProcess()

        with mock.patch.object(experiment_suite, "_verify_source"), \
             mock.patch.object(experiment_suite, "_launch", side_effect=launch) as launched, \
             mock.patch.object(experiment_suite, "_worker_completed", return_value=True), \
             mock.patch.object(experiment_suite.time, "sleep"), \
             mock.patch.object(experiment_report, "generate_progress_report"), \
             mock.patch.object(experiment_report, "generate_report",
                               return_value=self.directory / "report.md") as report:
            self.assertEqual(experiment_suite.run_queue(
                self.root, "test_suite", max_concurrent=3), 0)
        self.assertEqual(launched.call_count, 18)
        report.assert_called_once()
        summary = json.loads((self.directory / "suite_status.json").read_text())
        self.assertEqual(summary["counts"]["completed"], 18)
        self.assertTrue(summary["all_completed"])

    def test_exhausted_retries_keep_suite_incomplete(self):
        atomic_json(self.directory / "suite_plan.json", self.plan)
        with mock.patch.object(experiment_suite, "_verify_source"), \
             mock.patch.object(experiment_suite, "_launch",
                               side_effect=RuntimeError("simulated worker launch failure")) as launched, \
             mock.patch.object(experiment_suite.time, "sleep"), \
             mock.patch.object(experiment_report, "generate_progress_report"), \
             mock.patch.object(experiment_report, "generate_report") as report:
            self.assertEqual(experiment_suite.run_queue(
                self.root, "test_suite", max_concurrent=3, retries=1), 1)
        self.assertEqual(launched.call_count, 36)
        report.assert_not_called()
        summary = json.loads((self.directory / "suite_status.json").read_text())
        self.assertEqual(summary["counts"]["failed"], 18)
        self.assertFalse(summary["all_completed"])

    def test_report_pairs_same_seed_scene_action_and_refuses_partial_suite(self):
        atomic_json(self.directory / "suite_plan.json", self.plan)
        with self.assertRaisesRegex(RuntimeError, "尚未全部完成"):
            experiment_report.generate_report(self.directory)
        runs = []
        for job in self.plan["jobs"]:
            rows = []
            for scene in ("81000", "81001"):
                row = {field: 1.0 for field in experiment_report.FINAL_FIELDS}
                row.update(scene_id=scene, action_seed=91001,
                           si_mean=0.5 + (job["formula"] == "corrected") * 0.1,
                           power_exceeded=job["seed"] == 51001)
                rows.append(row)
            runs.append({"job": job, "rows": rows,
                         "by_key": {(row["scene_id"], str(row["action_seed"])): row
                                    for row in rows},
                         "argmax_rows": rows,
                         "best_test": {"rows": [dict(row, si_mean=0.9) for row in rows]}})
        summary = experiment_report.summarize(runs)
        for model in experiment_report.MODELS:
            paired = summary["paired_corrected_minus_legacy"][model]["si_mean"]
            self.assertAlmostEqual(paired["mean"], 0.1)
            self.assertEqual(paired["improved_scenes_by_train_seed"], [2, 2, 2])
            extra = summary["variants"][f"{model}_corrected"]["supplemental"]
            self.assertAlmostEqual(extra["argmax_power_exceeded"]["mean"], 1 / 3)
            self.assertAlmostEqual(extra["best_validation_selected_test_power_exceeded"]["mean"], 1 / 3)
            self.assertAlmostEqual(extra["best_validation_selected_test_si_mean"]["mean"], 0.9)
        for job in self.plan["jobs"]:
            job["state"] = "completed"
        atomic_json(self.directory / "suite_plan.json", self.plan)
        behavior = {f"{model}_{formula}": {
            "requested_power_w": {20: 1}, "requested_slots": {2: 1},
            "clipped_rate": 0.0}
            for model in experiment_report.MODELS for formula in experiment_report.FORMULAS}
        with mock.patch.object(experiment_report, "_load_run", side_effect=runs), \
             mock.patch.object(experiment_report, "_behavior_from_traces",
                               return_value=behavior), \
             mock.patch.object(experiment_report, "_plot_curves", return_value=[]):
            report_path = experiment_report.generate_report(self.directory)
        report_text = report_path.read_text(encoding="utf-8")
        self.assertIn("终态 SI", report_text)
        self.assertIn("SGM", report_text)
        self.assertIn("Transformer SAC", report_text)
        self.assertIn("10.00", report_text)


if __name__ == "__main__":
    unittest.main()
