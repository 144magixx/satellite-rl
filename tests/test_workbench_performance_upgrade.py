"""Receipt-gated, same-job G08 performance source migration."""

from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from satellite_rl.agents.workbench_algorithms import build_agent
from satellite_rl.envs.regions import profile_metadata
from satellite_rl.training.run_monitor import atomic_json as real_atomic_json
from satellite_rl.workbench.checkpoints import config_hash
from satellite_rl.workbench.config import config_summary, default_config
from satellite_rl.workbench.learning_snapshots import save_learning_snapshot
from satellite_rl.workbench.performance_upgrade import (
    ALLOWED_CHANGED, BACKGROUND_RUNS, EQUIVALENCE_CHECKS, MEASURED_WORKER_FILES,
    OLD_COMMIT, TELEMETRY, UPGRADE_TYPE, verify_source_history_entry,
)
from satellite_rl.workbench.service import Application
from satellite_rl.workbench.storage import digest, read_json


class PerformanceUpgradeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.app = Application(self.root, source_root=self.root)
        self.new_commit = "c" * 40
        self.measured_commit = "d" * 40
        snapshots = self.app.store.root / "source_snapshots"
        self.old = snapshots / OLD_COMMIT
        self.new = snapshots / self.new_commit
        old_files = {name: b"same-source\n" for name in MEASURED_WORKER_FILES
                     if not name.endswith("/telemetry.py")}
        old_files["src/satellite_rl/workbench/runner.py"] = b"old-runner\n"
        old_files["src/satellite_rl/workbench/environment.py"] = b"same-environment\n"
        new_files = dict(old_files)
        new_files["src/satellite_rl/workbench/runner.py"] = b"new-runner\n"
        new_files["src/satellite_rl/workbench/telemetry.py"] = b"new-telemetry\n"
        new_files["src/satellite_rl/workbench/performance_upgrade.py"] = b"control-module\n"
        self._snapshot(self.old, OLD_COMMIT, old_files)
        self._snapshot(self.new, self.new_commit, new_files)
        self.config = default_config()
        self.config["network"]["hidden_sizes"] = [8]
        self.config["training"]["total_steps"] = 10
        with patch.object(self.app.store, "freeze_source", return_value=(self.old, OLD_COMMIT)):
            self.job = self.app.store.create_job(self.config, request_id="g08-20260929-train-fixture")
        self.directory = self.app.store.job_dir(self.job["run_id"])
        self.folder = self.root / "outputs" / "verification" / "async_telemetry_20260929"
        self.folder.mkdir(parents=True)
        self.receipt = self._receipt()

    @staticmethod
    def _snapshot(root, commit, files):
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        real_atomic_json(root / "source_manifest.json", {"git_commit": commit,
            "files": {name: digest(root / name) for name in files}})

    def _receipt(self):
        old = read_json(self.old / "source_manifest.json")["files"]
        new = read_json(self.new / "source_manifest.json")["files"]
        changed = [{"path": name, "old_sha256": old.get(name), "new_sha256": new.get(name)}
                   for name in sorted(set(old) | set(new)) if old.get(name) != new.get(name)]
        assert set(item["path"] for item in changed).issubset(ALLOWED_CHANGED)
        measured_files = {name: new[name] for name in MEASURED_WORKER_FILES}
        equality = {"type": "async_telemetry_equivalence_v1", "passed": True,
                    "old_commit": OLD_COMMIT, "new_commit": self.measured_commit,
                    "measured_worker_files": measured_files,
                    "checks": {key: True for key in EQUIVALENCE_CHECKS}}
        speed = {"type": "async_telemetry_performance_v1", "passed": True,
                 "old_commit": OLD_COMMIT, "new_commit": self.measured_commit,
                 "measured_worker_files": measured_files,
                 "measurement_mode": "paired_interleaved_with_background",
                 "background_runs": sorted(BACKGROUND_RUNS), "same_workload": True,
                 "layout_results": {name: {"old_steps_per_second": 100.0,
                                           "new_steps_per_second": 106.0, "passed": True}
                                    for name in ("13", "32")}}
        references = {}
        for name, record in (("equivalence", equality), ("performance", speed)):
            path = self.folder / f"{name}.json"
            real_atomic_json(path, record)
            references[name] = {"path": str(path), "sha256": digest(path)}
        receipt = {"type": UPGRADE_TYPE, "status": "passed", "old_commit": OLD_COMMIT,
                   "new_commit": self.new_commit, "measured_worker_commit": self.measured_commit,
                   "measured_worker_files": measured_files,
                   "old_manifest_sha256": digest(self.old / "source_manifest.json"),
                   "new_manifest_sha256": digest(self.new / "source_manifest.json"),
                   "changed_files": changed, **references}
        path = self.folder / "receipt.json"
        real_atomic_json(path, receipt)
        return path

    def _hold(self):
        return self.app.api("POST", ["dispatch", "hold"], {},
                            {"ttl_seconds": 120, "reason": "G08 rolling upgrade"})

    def _upgrade(self):
        with patch.object(self.app.store, "freeze_source", return_value=(self.new, self.new_commit)):
            return self.app.api("POST", ["runs", self.job["run_id"], "upgrade"], {},
                                {"upgrade_type": UPGRADE_TYPE, "receipt_path": str(self.receipt)})

    def test_queued_upgrade_keeps_identity_and_scientific_config(self):
        self._hold()
        status = self._upgrade()
        updated = read_json(self.directory / "job.json")
        config = read_json(self.directory / "config.json")
        self.assertEqual(status["stage"], "queued")
        self.assertEqual(updated["run_id"], self.job["run_id"])
        self.assertEqual(updated["request_hash"], self.job["request_hash"])
        self.assertEqual(updated["git_commit"], self.new_commit)
        self.assertEqual(updated["telemetry"], TELEMETRY)
        self.assertEqual(config_hash(config), config_hash(self.config))
        self.assertEqual(config["telemetry"], TELEMETRY)
        self.assertEqual(len(updated["source_history"]), 1)
        self.assertEqual(updated["source_history"][0]["measured_worker_commit"], self.measured_commit)
        self.assertEqual(verify_source_history_entry(updated["source_history"][0]),
                         updated["source_history"][0])
        # Historical proof stays verifiable after this live job moves on.
        (self.directory / "config.json").write_text("{}", encoding="utf-8")
        (self.directory / "job.json").write_text("{}", encoding="utf-8")
        self.assertEqual(verify_source_history_entry(updated["source_history"][0]),
                         updated["source_history"][0])
        archive = Path(updated["performance_upgrade"]["archive_path"])
        self.assertEqual(read_json(archive / "config.json"), self.config)
        self.assertEqual(read_json(archive / "job.json"), self.job)

    def test_static_history_rejects_evidence_or_archived_config_tampering(self):
        self._hold()
        self._upgrade()
        entry = read_json(self.directory / "job.json")["source_history"][0]
        archive = Path(entry["archive_path"])
        original = (archive / "config.json").read_bytes()
        (archive / "config.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "归档配置"):
            verify_source_history_entry(entry)
        (archive / "config.json").write_bytes(original)
        evidence = self.folder / "equivalence.json"
        evidence.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            verify_source_history_entry(entry)

    def test_snapshot_resume_accepts_only_verified_operational_telemetry_change(self):
        self._hold()
        self._upgrade()
        updated = read_json(self.directory / "job.json")
        history = updated["source_history"]
        old_config = deepcopy(self.config)
        old_config.pop("telemetry")  # Old frozen G08 config predates this operational field.
        new_config = read_json(self.directory / "config.json")
        self.assertEqual(new_config["telemetry"], TELEMETRY)
        self.assertEqual(config_hash(old_config), config_hash(new_config))
        dimensions = config_summary(old_config)
        agent = build_agent(old_config, dimensions["state_dim"], dimensions["action_dims"], "cpu")
        state = np.zeros(dimensions["state_dim"], dtype=np.float32)
        agent.replay.add(state, (0, 0, 0), 0.25, state, True)
        agent.replay.add(state + 1, (1, 1, 1), 0.5, state + 2, False)
        profile = profile_metadata("china_legacy")
        old_provenance = {"run_id": self.job["run_id"], "git_commit": OLD_COMMIT,
                          "source_root": str(self.old),
                          "source_manifest_sha256": digest(self.old / "source_manifest.json"),
                          "environment_profile": profile, "scene_sets": {}}
        new_provenance = {**old_provenance, "git_commit": self.new_commit,
                          "source_root": str(self.new),
                          "source_manifest_sha256": digest(self.new / "source_manifest.json")}
        snapshot = Path(save_learning_snapshot(self.directory, agent, old_config,
                                               old_provenance, 5))
        original_sha = digest(snapshot)
        self.assertNotIn("telemetry", torch.load(snapshot, map_location="cpu",
                                                 weights_only=True)["config"])
        self.assertEqual(save_learning_snapshot(self.directory, agent, new_config,
                                                new_provenance, 5, source_history=history), str(snapshot))
        self.assertEqual(digest(snapshot), original_sha)
        audit = snapshot.with_name(f"{snapshot.stem}_recovery_{self.new_commit[:12]}.json")
        self.assertTrue(audit.is_file())
        self.assertEqual(read_json(audit)["verified_repair_chain"], history)

        scientific_change = deepcopy(new_config)
        scientific_change["training"]["gamma"] = 0.8
        with self.assertRaisesRegex(ValueError, "semantics differ"):
            save_learning_snapshot(self.directory, agent, scientific_change, new_provenance, 5,
                                   source_history=history)
        parameter = next(agent.critic1.parameters())
        original_parameter = parameter.detach().clone()
        with torch.no_grad():
            parameter.add_(1)
        with self.assertRaisesRegex(ValueError, "semantics differ"):
            save_learning_snapshot(self.directory, agent, new_config, new_provenance, 5,
                                   source_history=history)
        with torch.no_grad():
            parameter.copy_(original_parameter)
        agent.replay.add(state + 3, (2, 2, 2), 0.75, state + 4, True)
        with self.assertRaisesRegex(ValueError, "semantics differ"):
            save_learning_snapshot(self.directory, agent, new_config, new_provenance, 5,
                                   source_history=history)
        self.assertEqual(digest(snapshot), original_sha)

        # A forged/changed local receipt cannot retroactively authorize history.
        evidence = self.folder / "performance.json"
        evidence.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            save_learning_snapshot(self.directory, agent, new_config, new_provenance, 5,
                                   source_history=history)

    def test_atomic_write_failure_rolls_job_config_and_status_back(self):
        self._hold()
        before_status = read_json(self.directory / "status.json")
        calls = {"failed": False}

        def fail_once(path, payload):
            if Path(path).name == "job.json" and not calls["failed"]:
                calls["failed"] = True
                raise OSError("fixture interrupted publication")
            real_atomic_json(path, payload)

        with patch("satellite_rl.workbench.performance_upgrade.atomic_json", side_effect=fail_once), \
             self.assertRaisesRegex(OSError, "interrupted publication"):
            self._upgrade()
        self.assertEqual(read_json(self.directory / "config.json"), self.config)
        self.assertEqual(read_json(self.directory / "job.json"), self.job)
        self.assertEqual(read_json(self.directory / "status.json"), before_status)

    def test_stopped_upgrade_backs_up_complete_checkpoint(self):
        checkpoint = self.directory / "checkpoint.pt"
        torch.save({"format": "satellite_rl_workbench_checkpoint_v1",
                    "config_hash": config_hash(self.config),
                    "progress": {"env_step": 5, "episode": 1},
                    "agent": {"update_step": 3, "rng": random.getstate(),
                              "optimizers": {name: {} for name in
                                             ("actor_opt", "critic1_opt", "critic2_opt", "alpha_opt")},
                              "replay": {"rows": []}},
                    "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state(), "cuda": [torch.zeros(1, dtype=torch.uint8)]}},
                   checkpoint)
        original_sha = digest(checkpoint)
        status = read_json(self.directory / "status.json")
        real_atomic_json(self.directory / "status.json", {**status, "stage": "stopped", "env_step": 5,
                                                       "update_step": 3, "pid": None})
        self._hold()
        result = self._upgrade()
        self.assertEqual(result["stage"], "stopped")
        proof = read_json(self.directory / "job.json")["performance_upgrade"]
        self.assertEqual(proof["checkpoint"]["sha256"], original_sha)
        self.assertEqual(proof["checkpoint"]["env_step"], 5)
        self.assertEqual(digest(Path(proof["archive_path"]) / "checkpoint.pt"), original_sha)
        self.assertEqual(digest(checkpoint), original_sha)

    def test_invalid_gain_or_unrelated_source_rejected_without_job_change(self):
        self._hold()
        speed_path = self.folder / "performance.json"
        speed = read_json(speed_path)
        speed["layout_results"]["32"]["new_steps_per_second"] = 104.0
        real_atomic_json(speed_path, speed)
        receipt = read_json(self.receipt)
        receipt["performance"]["sha256"] = digest(speed_path)
        real_atomic_json(self.receipt, receipt)
        with self.assertRaisesRegex(ValueError, "5%"):
            self._upgrade()
        self.assertEqual(read_json(self.directory / "job.json"), self.job)
        self.assertEqual(read_json(self.directory / "config.json"), self.config)
        # Even a matching changed-files receipt cannot authorize environment edits.
        (self.new / "src/satellite_rl/workbench/environment.py").write_bytes(b"changed-physics\n")
        manifest = read_json(self.new / "source_manifest.json")
        manifest["files"]["src/satellite_rl/workbench/environment.py"] = digest(
            self.new / "src/satellite_rl/workbench/environment.py")
        real_atomic_json(self.new / "source_manifest.json", manifest)
        receipt["new_manifest_sha256"] = digest(self.new / "source_manifest.json")
        receipt["changed_files"].append({"path": "src/satellite_rl/workbench/environment.py",
            "old_sha256": digest(self.old / "src/satellite_rl/workbench/environment.py"),
            "new_sha256": manifest["files"]["src/satellite_rl/workbench/environment.py"]})
        real_atomic_json(self.receipt, receipt)
        with self.assertRaisesRegex(ValueError, "未许可"):
            self._upgrade()
        self.assertEqual(read_json(self.directory / "job.json"), self.job)

    def test_measured_worker_fingerprint_must_match_final_source(self):
        self._hold()
        receipt = read_json(self.receipt)
        receipt["measured_worker_files"]["src/satellite_rl/workbench/runner.py"] = "0" * 64
        real_atomic_json(self.receipt, receipt)
        with self.assertRaisesRegex(ValueError, "逐文件指纹不同"):
            self._upgrade()
        self.assertEqual(read_json(self.directory / "job.json"), self.job)

    def test_hold_expires_and_never_stops_active_workers(self):
        hold = self._hold()
        self.assertTrue(hold["active"])
        with patch.object(self.app.scheduler, "_launch") as launch, \
             patch("satellite_rl.workbench.runtime.gpu_snapshot", return_value={
                 "memory_total_mb": 12000, "memory_used_mb": 0, "compute_pids": [],
                 "utilization_percent": 0}):
            self.app.scheduler.tick()
            launch.assert_not_called()
            with patch("satellite_rl.workbench.runtime.time.time",
                       return_value=hold["expires_at_unix"] + 1):
                self.assertFalse(self.app.scheduler.dispatch_hold()["active"])
                self.app.scheduler.tick()
            launch.assert_called_once()
        self.assertFalse((self.app.store.root / "dispatch_hold.json").exists())
        self._hold()
        self.assertEqual(self.app.api("POST", ["dispatch", "hold", "release"], {}, {}),
                         {"active": False})

    def test_upgrade_requires_hold_and_dead_pid(self):
        with self.assertRaisesRegex(ValueError, "派发保留"):
            self._upgrade()
        self._hold()
        with patch("satellite_rl.workbench.runtime.pid_alive", return_value=True), \
             self.assertRaisesRegex(ValueError, "仍在运行"):
            self._upgrade()
        self.assertEqual(read_json(self.directory / "job.json"), self.job)


if __name__ == "__main__":
    unittest.main()
