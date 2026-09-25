"""A stopped job may rebind source only to the narrowly verified I/O repair."""

import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from satellite_rl.workbench.checkpoints import config_hash
from satellite_rl.workbench.config import default_config
from satellite_rl.workbench.recovery import REPAIR_ID, WORKER_FILES, verify_io_retry_repair
from satellite_rl.workbench.service import Application
from satellite_rl.workbench.storage import atomic_json, digest, read_json


class IoRecoveryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.app = Application(self.root)
        self.old_commit, self.new_commit = "a" * 40, "b" * 40
        snapshots = self.app.store.root / "source_snapshots"
        self.old = snapshots / self.old_commit
        self.new = snapshots / self.new_commit
        self.contents = {name: 'import os\ndef save(a, b):\n    os.replace(a, b)\n'
                         for name in WORKER_FILES}
        self.contents["src/satellite_rl/agents/workbench_algorithms.py"] = "ALGORITHM = 1\n"
        self.write_snapshot(self.old, self.old_commit, self.contents)
        fixed = {name: ('from satellite_rl.training.run_monitor import replace_with_retry\n'
                        + text.replace("os.replace(a, b)", "replace_with_retry(a, b)"))
                 if name in WORKER_FILES else text for name, text in self.contents.items()}
        self.fixed = fixed
        self.write_snapshot(self.new, self.new_commit, fixed)
        self.config = default_config()
        with patch.object(self.app.store, "freeze_source", return_value=(self.old, self.old_commit)):
            self.job = self.app.store.create_job(self.config, request_id="io_test")
        self.directory = self.app.store.job_dir(self.job["run_id"])
        self.checkpoint = self.directory / "checkpoint.pt"
        torch.save({"format": "satellite_rl_workbench_checkpoint_v1", "config_hash": config_hash(self.config),
                    "progress": {"env_step": 100, "episode": 1},
                    "rng": {"python": random.getstate()},
                    "agent": {"rng": random.getstate(), "update_step": 1}}, self.checkpoint)
        atomic_json(self.directory / "status.json", {"stage": "failed", "env_step": 150,
                                                    "pid": None, "error": "file lock"})
        atomic_json(self.directory / "failure.json", {
            "error": "[WinError 5] file is in use", "traceback": "os.replace(temp, path)",
            "last_normal_checkpoint": {"sha256": digest(self.checkpoint), "env_step": 100}})

    def write_snapshot(self, root, commit, contents):
        for name, text in contents.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        atomic_json(root / "source_manifest.json", {"git_commit": commit,
                    "files": {name: digest(root / name) for name in contents}})

    def resume(self):
        return self.app.api("POST", ["runs", self.job["run_id"], "resume"], {}, {"source_repair": REPAIR_ID})

    def test_only_atomic_retry_change_is_accepted(self):
        proof = verify_io_retry_repair(self.old, self.old_commit, self.new, self.new_commit)
        self.assertTrue(proof["worker_ast_equivalent_except_atomic_retry"])
        self.assertEqual(set(proof["changed_files"]), WORKER_FILES)

    def test_learning_change_or_worker_semantic_change_is_rejected(self):
        for name, addition in (("src/satellite_rl/agents/workbench_algorithms.py", "ALGORITHM = 2\n"),
                               ("src/satellite_rl/workbench/runner.py", "gamma = 0.5\n")):
            with self.subTest(name=name):
                changed = dict(self.fixed)
                changed[name] += addition
                self.write_snapshot(self.new, self.new_commit, changed)
                with self.assertRaises(ValueError):
                    verify_io_retry_repair(self.old, self.old_commit, self.new, self.new_commit)
        self.write_snapshot(self.new, self.new_commit, self.fixed)

    def test_corrupt_source_and_checkpoint_are_rejected_before_job_change(self):
        (self.new / "src/satellite_rl/workbench/runner.py").write_text("tampered", encoding="utf-8")
        with patch.object(self.app.store, "freeze_source", return_value=(self.new, self.new_commit)):
            with self.assertRaises(ValueError):
                self.resume()
        self.assertEqual(read_json(self.directory / "job.json"), self.job)

        self.checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "断点校验值"):
            self.resume()
        self.assertEqual(read_json(self.directory / "job.json"), self.job)

    def test_control_changes_are_limited_to_resume_and_option_forwarding(self):
        old = dict(self.contents)
        new = dict(self.fixed)
        runtime = "src/satellite_rl/workbench/runtime.py"
        service = "src/satellite_rl/workbench/service.py"
        old[runtime] = "class Scheduler:\n def resume(self, run_id): return run_id\n def launch(self): return 1\n"
        new[runtime] = "class Scheduler:\n def resume(self, run_id, source_repair=None): return run_id\n def launch(self): return 1\n"
        old[service] = "class Application:\n def api(self, identifier, body): return self.scheduler.resume(identifier)\n"
        new[service] = 'class Application:\n def api(self, identifier, body): return self.scheduler.resume(identifier, source_repair=body.get("source_repair"))\n'
        self.write_snapshot(self.old, self.old_commit, old)
        self.write_snapshot(self.new, self.new_commit, new)
        verify_io_retry_repair(self.old, self.old_commit, self.new, self.new_commit)
        new[runtime] = new[runtime].replace("return 1", "return 2")
        self.write_snapshot(self.new, self.new_commit, new)
        with self.assertRaisesRegex(ValueError, "启动/服务逻辑"):
            verify_io_retry_repair(self.old, self.old_commit, self.new, self.new_commit)
    def test_api_keeps_job_identity_and_archives_checkpoint_and_failure(self):
        original_sha = digest(self.checkpoint)
        with patch.object(self.app.store, "freeze_source", return_value=(self.new, self.new_commit)):
            result = self.resume()
        updated = read_json(self.directory / "job.json")
        self.assertEqual(result["stage"], "queued")
        self.assertEqual(result["env_step"], 100)
        self.assertIsNone(result["error"])
        for key in ("run_id", "created_at", "request_id", "request_hash", "model_dir"):
            self.assertEqual(updated[key], self.job[key])
        self.assertEqual(updated["git_commit"], self.new_commit)
        self.assertEqual(updated["original_git_commit"], self.old_commit)
        archive = Path(updated["io_repair"]["archive_path"])
        self.assertEqual(digest(archive / "checkpoint.pt"), original_sha)
        self.assertEqual(digest(self.checkpoint), original_sha)
        self.assertEqual(read_json(archive / "job.json"), self.job)
        self.assertTrue((archive / "failure.json").is_file())
        self.assertEqual(read_json(self.directory / "config.json"), self.config)
        verify_io_retry_repair(self.old, self.old_commit, self.new, self.new_commit)

    def test_failed_queue_write_can_retry_without_new_migration(self):
        with patch.object(self.app.store, "freeze_source", return_value=(self.new, self.new_commit)) as freeze:
            with patch.object(self.app.scheduler, "_status", side_effect=OSError("disk")):
                with self.assertRaises(OSError):
                    self.resume()
            result = self.resume()
        self.assertEqual(freeze.call_count, 1)
        self.assertEqual(result["stage"], "queued")
        job = read_json(self.directory / "job.json")
        self.assertEqual(len(job["source_history"]), 1)

    def test_live_worker_and_unknown_repair_are_rejected(self):
        with patch("satellite_rl.workbench.runtime.pid_alive", return_value=True):
            with self.assertRaisesRegex(ValueError, "仍在运行"):
                self.resume()
        with self.assertRaisesRegex(ValueError, "不支持"):
            self.app.api("POST", ["runs", self.job["run_id"], "resume"], {}, {"source_repair": "arbitrary"})
        self.assertEqual(read_json(self.directory / "job.json"), self.job)


if __name__ == "__main__":
    unittest.main()
