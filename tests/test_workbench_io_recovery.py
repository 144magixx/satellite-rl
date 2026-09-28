"""A stopped job may rebind source only to the narrowly verified I/O repair."""

import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from satellite_rl.agents.workbench_algorithms import build_agent
from satellite_rl.workbench.checkpoints import config_hash
from satellite_rl.workbench.config import config_summary, default_config
from satellite_rl.workbench.learning_snapshots import save_learning_snapshot
from satellite_rl.workbench.recovery import REPAIR_ID, WORKER_FILES, verify_io_retry_repair
from satellite_rl.workbench.runner import run_job
from satellite_rl.workbench.service import Application
from satellite_rl.workbench.storage import atomic_json, digest, read_json
from satellite_rl.envs.regions import profile_metadata


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

    def test_learning_snapshot_accepts_only_verified_source_repair_and_keeps_original(self):
        config = default_config()
        config["network"]["hidden_sizes"] = [8]
        summary = config_summary(config)
        agent = build_agent(config, summary["state_dim"], summary["action_dims"], "cpu")
        zeros = np.zeros(summary["state_dim"], dtype=np.float32)
        agent.replay.add(zeros, (0, 0, 0), 0.25, zeros, True)
        old_source = {"run_id": "snapshot_repair", "git_commit": self.old_commit,
                      "source_root": str(self.old),
                      "source_manifest_sha256": digest(self.old / "source_manifest.json"),
                      "environment_profile": profile_metadata("china_legacy"), "scene_sets": {}}
        new_source = dict(old_source, git_commit=self.new_commit, source_root=str(self.new),
                          source_manifest_sha256=digest(self.new / "source_manifest.json"))
        snapshots = self.root / "learning_test"
        original = Path(save_learning_snapshot(snapshots, agent, config, old_source, 100))
        original_sha = digest(original)
        proof = verify_io_retry_repair(self.old, self.old_commit, self.new, self.new_commit)
        self.assertEqual(save_learning_snapshot(snapshots, agent, config, new_source, 100,
                                                source_history=[proof]), str(original))
        self.assertEqual(digest(original), original_sha)
        audit = original.with_name(f"{original.stem}_recovery_{self.new_commit[:12]}.json")
        self.assertTrue(audit.is_file())
        self.assertEqual(read_json(audit)["verified_repair_chain"], [proof])

        unrelated = dict(old_source, git_commit="c" * 40,
                         source_root=str(self.root / "unrelated"),
                         source_manifest_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "absent from"):
            save_learning_snapshot(snapshots, agent, config, unrelated, 100,
                                   source_history=[proof])
        with torch.no_grad():
            next(agent.critic1.parameters()).add_(1)
        with self.assertRaisesRegex(ValueError, "semantics differ"):
            save_learning_snapshot(snapshots, agent, config, new_source, 100,
                                   source_history=[proof])
        self.assertEqual(digest(original), original_sha)

    def test_snapshot_io_failure_after_new_checkpoint_records_recoverable_step(self):
        class TinyEnv:
            def __init__(self, _config):
                self.data = {"value": np.array([1], dtype=np.int64)}

            def reset(self, *, seed=None, scene=None):
                return np.array([0.0], dtype=np.float32)

            def scene(self):
                return self.data

            def action_mask(self):
                return None

            def step(self, _action):
                return np.array([0.0], dtype=np.float32), 0.0, True, {}

            def metrics(self):
                return {"final_si_mean": 0.0}

        class TinyAgent:
            update_step = 0
            last_action_info = {}

            def act(self, _state, **_kwargs):
                return [0, 0, 0]

            def observe(self, *_args, **_kwargs):
                pass

            def ready(self, *_args):
                return False

            def state_dict(self):
                return {"rng": random.getstate(), "update_step": 0}

        config = default_config()
        config["training"].update(total_steps=2, checkpoint_interval=1, log_interval=100)
        config["evaluation"]["interval_steps"] = 0
        with patch.object(self.app.store, "freeze_source", return_value=(self.old, self.old_commit)):
            job = self.app.store.create_job(config, request_id="snapshot_io_failure")
        directory = self.app.store.job_dir(job["run_id"])

        def snapshot(_directory, _agent, _config, _provenance, step, source_history=None):
            if step == 1:
                raise PermissionError("[WinError 5] replace failed for learning snapshot")
            return None

        with patch("satellite_rl.workbench.runner.strict_cuda_preflight", return_value={"gpu": "fixture"}), \
             patch("satellite_rl.workbench.environment.WorkbenchEnv", TinyEnv), \
             patch("satellite_rl.workbench.runner.build_agent", return_value=TinyAgent()), \
             patch("satellite_rl.workbench.runner._ensure_finite_agent"), \
             patch("satellite_rl.workbench.runner.save_learning_snapshot", side_effect=snapshot), \
             self.assertRaisesRegex(PermissionError, "replace failed"):
            run_job(directory)
        failure = read_json(directory / "failure.json")
        self.assertEqual(failure["last_normal_checkpoint"]["env_step"], 1)
        self.assertEqual(failure["last_normal_checkpoint"]["sha256"], digest(directory / "checkpoint.pt"))
        self.assertEqual(torch.load(directory / "checkpoint.pt", map_location="cpu",
                                    weights_only=False)["progress"]["env_step"], 1)
        status = read_json(directory / "status.json")
        self.assertEqual(status["checkpoint_step"], 1)
        atomic_json(directory / "status.json", {**status, "pid": None})
        with patch.object(self.app.store, "freeze_source", return_value=(self.new, self.new_commit)), \
             patch("satellite_rl.workbench.runtime.pid_alive", return_value=False):
            resumed = self.app.api("POST", ["runs", job["run_id"], "resume"], {},
                                   {"source_repair": REPAIR_ID})
        self.assertEqual(resumed["stage"], "queued")
        self.assertEqual(resumed["env_step"], 1)


if __name__ == "__main__":
    unittest.main()
