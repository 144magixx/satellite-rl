"""The D12-A 65,536-step pilot is a normal, auditable worker exit."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench import drl_r04_gate, runtime
from satellite_rl.workbench.storage import digest


RUN_ID = "train_ppo_20261010_034306_03e7f00f"
COMMIT = drl_r04_gate.FROZEN_PILOT_COMMIT


class _Store:
    def __init__(self, root, run_id=RUN_ID):
        self.root = root
        self.archive_root = root
        self.run_id = run_id

    def job_dir(self, run_id):
        assert run_id == self.run_id
        return self.root / run_id

    def jobs(self):
        directory = self.job_dir(self.run_id)
        return [(json.loads((directory / "job.json").read_text(encoding="utf-8")),
                 json.loads((directory / "status.json").read_text(encoding="utf-8")))]


class PilotReconcileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = _Store(Path(self.temp.name))
        self.directory = self.store.job_dir(RUN_ID)
        self.directory.mkdir()
        (self.directory / "checkpoint.pt").write_bytes(b"fixed-pilot-checkpoint")
        self.checkpoint_sha = digest(self.directory / "checkpoint.pt")
        self.job = {"run_id": RUN_ID, "experiment_id": "S1-D12A",
                    "kind": "train", "train_mode": "P0", "experiment_role": "formal",
                    "git_commit": COMMIT}
        self.status = {"run_id": RUN_ID, "stage": "interrupted",
                       "stop_reason": "pilot_gate", "env_step": 65536,
                       "checkpoint_sha256": self.checkpoint_sha,
                       "pid": 12345, "exit_code": 0}
        atomic_json(self.directory / "job.json", self.job)
        atomic_json(self.directory / "status.json", self.status)
        atomic_json(self.directory / "config.json", {"schema_version": 6})
        atomic_json(self.directory / "launch.json", {"pid": 12345,
                    "process_start_time": 1000.0})
        atomic_json(self.directory / "failure.json", {"run_id": RUN_ID,
                    "pid": 12345, "exit_code": 0, "git_commit": COMMIT,
                    "error": "进程在没有Python异常记录的情况下结束；原生退出应结合Windows事件和驱动日志排查"})
        self.application = type("App", (), {"store": self.store})()

    def body(self):
        return {"expected_status_sha256": digest(self.directory / "status.json"),
                "expected_checkpoint_sha256": self.checkpoint_sha,
                "expected_failure_sha256": digest(self.directory / "failure.json")}

    def test_scheduler_keeps_normal_pilot_pending_and_does_not_write_failure(self):
        status = {**self.status, "stage": "pilot_pending"}
        atomic_json(self.directory / "status.json", status)
        (self.directory / "failure.json").unlink()
        scheduler = runtime.Scheduler(self.store)
        child = Mock()
        child.poll.return_value = 0
        child.returncode = 0
        scheduler.children[RUN_ID] = child
        scheduler.tick()
        actual = json.loads((self.directory / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(actual["stage"], "pilot_pending")
        self.assertFalse((self.directory / "failure.json").exists())
        self.assertNotIn(RUN_ID, scheduler.children)

    def test_scheduler_does_not_exempt_invalid_pilot(self):
        status = {**self.status, "stage": "pilot_pending", "env_step": 65535}
        atomic_json(self.directory / "status.json", status)
        scheduler = runtime.Scheduler(self.store)
        child = Mock()
        child.poll.return_value = 0
        child.returncode = 0
        scheduler.children[RUN_ID] = child
        scheduler.tick()
        actual = json.loads((self.directory / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(actual["stage"], "interrupted")

    def test_reconcile_requires_exact_evidence_and_preserves_failure(self):
        audit_path = self.store.root / "pilot_audit.json"
        atomic_json(audit_path, {"run_id": RUN_ID, "passed": False})
        body = self.body()
        with (patch.object(drl_r04_gate, "deployment", return_value="20418"),
              patch.object(drl_r04_gate, "pid_alive", return_value=False),
              patch.object(drl_r04_gate, "_audit", return_value=(
                  self.directory, self.job, {**self.status, "stage": "pilot_pending"},
                  audit_path, {"progress_pass": False, "zero_progress_stop": False})) as audit):
            with self.assertRaisesRegex(ValueError, "exact status"):
                drl_r04_gate.reconcile_misclassified_pilot(
                    self.application, RUN_ID, {**body, "expected_status_sha256": "0" * 64})
            self.assertEqual(json.loads((self.directory / "status.json").read_text(encoding="utf-8"))["stage"],
                             "interrupted")
            result = drl_r04_gate.reconcile_misclassified_pilot(self.application, RUN_ID, body)
            self.assertEqual(result["status"]["stage"], "pilot_pending")
            self.assertEqual(result["receipt"]["previous_status_sha256"], body["expected_status_sha256"])
            self.assertEqual(result["receipt"]["pilot_audit_sha256"], digest(audit_path))
            self.assertTrue((self.directory / "failure.json").is_file())
            self.assertEqual(audit.call_count, 1)

    def test_reconcile_rejects_live_worker_or_nonzero_exit(self):
        body = self.body()
        with patch.object(drl_r04_gate, "deployment", return_value="20418"):
            with patch.object(drl_r04_gate, "pid_alive", return_value=True):
                with self.assertRaisesRegex(ValueError, "exact normal gate exit"):
                    drl_r04_gate.reconcile_misclassified_pilot(self.application, RUN_ID, body)
            self.status["exit_code"] = 1
            atomic_json(self.directory / "status.json", self.status)
            with patch.object(drl_r04_gate, "pid_alive", return_value=False):
                with self.assertRaisesRegex(ValueError, "exact normal gate exit"):
                    drl_r04_gate.reconcile_misclassified_pilot(self.application, RUN_ID, self.body())


if __name__ == "__main__":
    unittest.main()
