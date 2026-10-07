"""Preparation errors must leave durable evidence or a nonzero exit for Scheduler."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from satellite_rl.training.run_monitor import atomic_json as actual_atomic_json
from satellite_rl.workbench.s2p19_preparation import run_job


class PreparationFailureTests(unittest.TestCase):
    def fixture(self, directory, with_config=True):
        root = Path(directory)
        (root / "job.json").write_text(json.dumps({"kind": "prepare",
            "experiment_id": "S2-P19", "run_id": "prepare_smoke", "git_commit": "smoke"}),
            encoding="utf-8")
        if with_config:
            (root / "config.json").write_text(json.dumps({
                "preparation": {"preparation_type": "demonstrations"},
                "training": {"total_steps": 576}}), encoding="utf-8")
        (root / "status.json").write_text(json.dumps({"run_id": "prepare_smoke",
                                                   "stage": "starting"}), encoding="utf-8")
        return root

    def test_missing_config_before_preflight_records_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp, with_config=False)
            with self.assertRaises(ValueError):
                run_job(root)
            self.assertEqual(json.loads((root / "status.json").read_text(encoding="utf-8"))["stage"], "failed")
            self.assertIn("缺少冻结job/config", (root / "failure.json").read_text(encoding="utf-8"))

    def test_config_drift_failure_does_not_call_broken_progress_again(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp)

            def drift(_directory, _job, _prep, progress, _stopped):
                (root / "config.json").write_text('{"changed":true}', encoding="utf-8")
                progress({"phase": "drift"})

            with patch("satellite_rl.workbench.runner.strict_cuda_preflight",
                       return_value={"smoke": True}), \
                    patch("satellite_rl.workbench.s2p19_preparation._teacher", side_effect=drift):
                with self.assertRaisesRegex(ValueError, "冻结配置被改变"):
                    run_job(root)
            status = json.loads((root / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["stage"], "failed")
            self.assertIn("冻结配置被改变", (root / "failure.json").read_text(encoding="utf-8"))

    def test_status_write_failure_retries_with_original_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp)
            calls = {"status": 0}

            def fail_first_status(path, data):
                if Path(path).name == "status.json":
                    calls["status"] += 1
                    if calls["status"] == 1:
                        raise OSError("status write failed")
                return actual_atomic_json(path, data)

            with patch("satellite_rl.workbench.runner.strict_cuda_preflight",
                       return_value={"smoke": True}), \
                    patch("satellite_rl.workbench.s2p19_preparation.atomic_json",
                          side_effect=fail_first_status):
                with self.assertRaisesRegex(OSError, "status write failed"):
                    run_job(root)
            self.assertEqual(json.loads((root / "status.json").read_text(encoding="utf-8"))["stage"], "failed")
            self.assertIn("status write failed", (root / "failure.json").read_text(encoding="utf-8"))

    def test_failure_record_write_failure_marks_interrupted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp)

            def fail_failure(path, data):
                if Path(path).name == "failure.json":
                    raise OSError("failure record write failed")
                return actual_atomic_json(path, data)

            with patch("satellite_rl.workbench.runner.strict_cuda_preflight",
                       return_value={"smoke": True}), \
                    patch("satellite_rl.workbench.s2p19_preparation._teacher",
                          side_effect=RuntimeError("teacher failed")), \
                    patch("satellite_rl.workbench.s2p19_preparation.atomic_json",
                          side_effect=fail_failure):
                with self.assertRaisesRegex(OSError, "failure record write failed"):
                    run_job(root)
            status = json.loads((root / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["stage"], "interrupted")
            self.assertIn("teacher failed", status["traceback"])
            self.assertFalse((root / "failure.json").exists())

    def test_stop_status_write_failure_is_interrupted_not_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp)
            calls = {"status": 0}

            def fail_stop_status(path, data):
                if Path(path).name == "status.json":
                    calls["status"] += 1
                    if calls["status"] == 2:
                        raise OSError("stop status write failed")
                return actual_atomic_json(path, data)

            with patch("satellite_rl.workbench.runner.strict_cuda_preflight",
                       return_value={"smoke": True}), \
                    patch("satellite_rl.workbench.s2p19_preparation._teacher",
                          side_effect=InterruptedError("user stop")), \
                    patch("satellite_rl.workbench.s2p19_preparation.atomic_json",
                          side_effect=fail_stop_status):
                with self.assertRaisesRegex(OSError, "stop status write failed"):
                    run_job(root)
            status = json.loads((root / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["stage"], "interrupted")
            self.assertEqual(status["stop_reason"], "user stop")


if __name__ == "__main__":
    unittest.main()
