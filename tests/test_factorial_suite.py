"""Safety checks for the single-seed formula queue; no training is started."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from satellite_rl.training import factorial_suite
from satellite_rl.training.run_monitor import atomic_json


class _ExitedProcess:
    pid = 12345

    def poll(self):
        return 0


class FactorialSuiteTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = factorial_suite.suite_dir(
            self.root, "sac_formula_factorial_test")
        self.directory.mkdir(parents=True)
        self.source = {"python": os.path.abspath(__file__),
                       "source_root": str(self.root), "git_commit": "frozen",
                       "cities_data": {"sha256": "fixed"}}
        self.plan = {
            "experiment_id": "sac_formula_factorial_test",
            "archive_root": str(self.root), "source": self.source,
            "steps": 400000, "validation_interval": 10000,
            "checkpoint_interval": 10000,
            "validation_scenes": {str(seed): {} for seed in factorial_suite.VALIDATION_SCENE_SEEDS},
            "test_scenes": {str(seed): {} for seed in factorial_suite.TEST_SCENE_SEEDS},
            "action_seeds": list(factorial_suite.ACTION_SEEDS),
            "jobs": factorial_suite.deterministic_jobs(),
        }

    def test_plan_exact_six_single_seed_and_safe_namespace(self):
        factorial_suite._validate_plan(self.plan)
        self.assertEqual(len(self.plan["jobs"]), 6)
        self.assertEqual({job["seed"] for job in self.plan["jobs"]}, {51001})
        self.assertEqual({job["formula"] for job in self.plan["jobs"]},
                         {"negative_joint", "positive_per_head"})
        with self.assertRaises(ValueError):
            factorial_suite.suite_dir(self.root, "sac_comparison_20260924_r2")
        self.plan["action_seeds"] = [0]
        with self.assertRaisesRegex(RuntimeError, "固定场景"):
            factorial_suite._validate_plan(self.plan)

    def test_frozen_source_guard(self):
        with mock.patch.object(factorial_suite, "strict_cuda_preflight",
                               return_value={**self.source, "git_commit": "changed"}):
            with self.assertRaisesRegex(RuntimeError, "冻结源码提交"):
                factorial_suite._verify_source(self.plan)

    def test_lock_rejects_second_supervisor_and_recovers_stale_pid(self):
        lock_path = self.directory / "suite.lock"
        with factorial_suite.SuiteLock(lock_path):
            with self.assertRaisesRegex(RuntimeError, "监督进程仍在运行"):
                with factorial_suite.SuiteLock(lock_path):
                    pass
        atomic_json(lock_path, {"pid": -1, "process_token": None})
        with factorial_suite.SuiteLock(lock_path):
            self.assertTrue(lock_path.exists())
        self.assertFalse(lock_path.exists())

    def test_launch_uses_frozen_python_and_requires_real_checkpoint(self):
        job = self.plan["jobs"][0]
        with mock.patch.object(factorial_suite.subprocess, "Popen",
                               return_value=_ExitedProcess()) as launcher:
            factorial_suite._launch(self.directory, self.plan, job)
        command = launcher.call_args.args[0]
        self.assertEqual(command[:4], [self.source["python"], "-u", "-m",
                                       "satellite_rl.training.experiment_runner"])
        self.assertIn(job["formula"], command)
        self.assertNotIn("--resume", command)
        self.assertEqual(launcher.call_args.kwargs["env"]["PYTHONPATH"],
                         str(self.root / "src"))
        run_dir = self.directory / job["run_id"]
        run_dir.mkdir()
        atomic_json(run_dir / "manifest.json", {})
        with self.assertRaisesRegex(RuntimeError, "缺少可恢复断点"):
            factorial_suite._launch(self.directory, self.plan, job)
        checkpoint = (self.root / "models" / "runs" / self.plan["experiment_id"] /
                      job["run_id"] / "latest.pt")
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        with mock.patch.object(factorial_suite.subprocess, "Popen",
                               return_value=_ExitedProcess()) as launcher:
            factorial_suite._launch(self.directory, self.plan, job)
        self.assertIn("--resume", launcher.call_args.args[0])

    def test_completion_requires_formal_test_and_matching_manifest(self):
        job = self.plan["jobs"][0]
        run_dir = self.directory / job["run_id"]
        (run_dir / "evaluation").mkdir(parents=True)
        atomic_json(run_dir / "status.json", {"stage": "completed", "env_step": 400000})
        config = {"model": job["model"], "formula": job["formula"],
                  "train_seed": 51001, "budget_env_steps": 400000,
                  "diagnostic_not_formal_comparison": False,
                  "runtime": {"git_commit": "frozen"}}
        atomic_json(run_dir / "manifest.json", {"config": config})
        self.assertFalse(factorial_suite._worker_completed(self.directory, job, self.plan))
        atomic_json(run_dir / "evaluation" / "final_test_400000.json", {"rows": []})
        self.assertTrue(factorial_suite._worker_completed(self.directory, job, self.plan))
        config["diagnostic_not_formal_comparison"] = True
        atomic_json(run_dir / "manifest.json", {"config": config})
        self.assertFalse(factorial_suite._worker_completed(self.directory, job, self.plan))

    def test_queue_finishes_six_without_old_eighteen_report(self):
        atomic_json(self.directory / "suite_plan.json", self.plan)

        def launch(_directory, _plan, job):
            job["state"] = "running"
            job["attempts"] += 1
            job["pid"] = 12345
            return _ExitedProcess()

        with mock.patch.object(factorial_suite, "_verify_source"), \
             mock.patch.object(factorial_suite, "_verify_fixed_scenes"), \
             mock.patch.object(factorial_suite, "_launch", side_effect=launch) as launched, \
             mock.patch.object(factorial_suite, "_worker_completed", return_value=True), \
             mock.patch.object(factorial_suite.time, "sleep"):
            self.assertEqual(factorial_suite.run_queue(
                self.root, "sac_formula_factorial_test", max_concurrent=3), 0)
        self.assertEqual(launched.call_count, 6)
        summary = factorial_suite._load_json(self.directory / "suite_status.json")
        self.assertEqual(summary["total_jobs"], 6)
        self.assertTrue(summary["all_completed"])

    def test_retry_limit_marks_actual_launch_failures(self):
        atomic_json(self.directory / "suite_plan.json", self.plan)
        with mock.patch.object(factorial_suite, "_verify_source"), \
             mock.patch.object(factorial_suite, "_verify_fixed_scenes"), \
             mock.patch.object(factorial_suite, "_launch",
                               side_effect=RuntimeError("worker launch failed")) as launched, \
             mock.patch.object(factorial_suite.time, "sleep"):
            self.assertEqual(factorial_suite.run_queue(
                self.root, "sac_formula_factorial_test", retries=1), 1)
        self.assertEqual(launched.call_count, 12)
        summary = factorial_suite._load_json(self.directory / "suite_status.json")
        self.assertEqual(summary["counts"]["failed"], 6)
        self.assertFalse(summary["all_completed"])

    def test_reference_scenes_are_copied_byte_for_byte(self):
        reference = self.root / "old"
        (reference / "scenes" / "validation").mkdir(parents=True)
        (reference / "scenes" / "test").mkdir(parents=True)
        old_plan = {"experiment_id": factorial_suite.REFERENCE_EXPERIMENT_ID}
        for group, seed in (("validation", 71000), ("test", 81000)):
            source = reference / "scenes" / group / f"scene_{seed}.npz"
            source.write_bytes(f"fixed scene {seed}".encode())
            old_plan[f"{group}_scenes"] = {
                str(seed): {"path": str(source),
                            "sha256": factorial_suite.sha256_file(source)}}
        atomic_json(reference / "suite_plan.json", old_plan)
        with mock.patch.object(factorial_suite, "VALIDATION_SCENE_SEEDS", (71000,)), \
             mock.patch.object(factorial_suite, "TEST_SCENE_SEEDS", (81000,)):
            refs = factorial_suite._copy_reference_scenes(self.directory, reference)
        self.assertEqual(refs["test"]["81000"]["sha256"],
                         old_plan["test_scenes"]["81000"]["sha256"])
        target = Path(refs["test"]["81000"]["path"])
        target.write_bytes(b"changed")
        with mock.patch.object(factorial_suite, "VALIDATION_SCENE_SEEDS", (71000,)), \
             mock.patch.object(factorial_suite, "TEST_SCENE_SEEDS", (81000,)):
            with self.assertRaisesRegex(RuntimeError, "新实验场景内容"):
                factorial_suite._copy_reference_scenes(self.directory, reference)


if __name__ == "__main__":
    unittest.main()
