"""Only the two failed S2-P19 pilot checkpoints may use replay-tail repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from satellite_rl.workbench import learning_snapshots, recovery
from satellite_rl.workbench.runtime import Scheduler
from satellite_rl.workbench.storage import Store, digest


ORIGIN = recovery.S2P19_ORIGIN_COMMIT
REPAIR = "b" * 40
RUN = "train_sac_20261007_152717_479365fb"
OTHER = "train_sac_20261007_153047_1faf6d10"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class ReplayRecoveryTests(unittest.TestCase):
    def fixture(self, temporary):
        root = Path(temporary)
        archive = root / "archive"
        store = Store(archive, source_root=root)
        job_dir = store.job_dir(RUN)
        job_dir.mkdir(parents=True)
        old = store.root / "source_snapshots" / ORIGIN
        new = store.root / "source_snapshots" / REPAIR
        for directory, marker, commit in ((old, "old", ORIGIN), (new, "fixed", REPAIR)):
            manifest = {}
            for name in recovery.S2P19_REPLAY_SOURCE_FILES:
                file = directory / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(f"{name}:{marker}", encoding="utf-8")
                manifest[name] = digest(file)
            write_json(directory / "source_manifest.json", {"git_commit": commit,
                                                               "files": manifest})
        checkpoint = job_dir / "checkpoint.pt"
        checkpoint.write_bytes(b"immutable checkpoint fixture")
        config = {"environment": {"profile": "europe_v1"}, "training": {"total_steps": 400000}}
        write_json(job_dir / "config.json", config)
        checkpoint_sha, config_sha = digest(checkpoint), digest(job_dir / "config.json")
        expected_runs = {
            RUN: {"arm": "C0", "request_id": "s2p19-c0-train-51001",
                  "checkpoint_sha256": checkpoint_sha, "config_sha256": config_sha},
            OTHER: {"arm": "C1", "request_id": "s2p19-c1-train-51001",
                    "checkpoint_sha256": "a" * 64, "config_sha256": "c" * 64}}
        job = {"run_id": RUN, "request_id": expected_runs[RUN]["request_id"],
               "kind": "train", "experiment_id": "S2-P19", "git_commit": ORIGIN,
               "source_root": str(old), "seed": 51001, "total_steps": 400000}
        status = {"stage": "failed", "env_step": 100000, "update_step": 98001,
                  "stop_reason": "s2p19_pilot_gate_pending", "pid": 0}
        failure = {"error": "逐束实体档位的回放缺真实束数记录",
                   "traceback": ('self.replay.load_state_dict(data["replay"])\n'
                                 'workbench_algorithms.py", line 394, in load_state_dict\n'
                                 'ValueError: 逐束实体档位的回放缺真实束数记录'),
                   "last_normal_checkpoint": {"sha256": checkpoint_sha, "env_step": 100000}}
        write_json(job_dir / "job.json", job)
        write_json(job_dir / "status.json", status)
        write_json(job_dir / "failure.json", failure)
        source = archive / "ops" / "audit" / "source.json"
        write_json(source, {"source": "frozen"})
        audit_path = archive / "ops" / "audit" / "C0.json"
        write_json(audit_path, {"experiment_id": "S2-P19", "run_id": RUN,
                                "checkpoint_sha256": checkpoint_sha, "passed": True,
                                "criteria": {"matched": True},
                                "source_files": [{"path": str(source), "sha256": digest(source)}]})
        gate = {"pass": True, "run_id": RUN, "checkpoint_step": 100000,
                "checkpoint_sha256": checkpoint_sha, "audit_path": str(audit_path),
                "audit_sha256": digest(audit_path)}
        write_json(job_dir / "pilot_gate.json", gate)
        pilot = job_dir / "pilot_artifacts"
        pilot.mkdir()
        (pilot / "checkpoint.pt").write_bytes(checkpoint.read_bytes())
        write_json(pilot / "manifest.json", {"run_id": RUN, "env_step": 100000,
             "git_commit": ORIGIN, "config_sha256": config_sha,
             "checkpoint_sha256": checkpoint_sha, "audit_path": str(audit_path),
             "audit_sha256": digest(audit_path)})
        counts = [100] * 20000
        normalized_sha = recovery._canonical_sha(counts)
        validation = {"format": "s2p19_replay_tail_validation_v1",
                      "repair_id": recovery.S2P19_REPLAY_REPAIR_ID,
                      "origin_science_commit": ORIGIN, "repair_commit": REPAIR,
                      "tests_passed": True, "runs": {}}
        for run_id, expected in expected_runs.items():
            validation["runs"][run_id] = {
                "checkpoint_sha256": expected["checkpoint_sha256"],
                "config_sha256": expected["config_sha256"],
                "env_step": 100000, "update_step": 98001, "capacity": 20000,
                "rows": 20000, "saved_counts": 20001, "next_position": 0,
                "unused_tail_zero": True,
                "all_active_counts_match_state_and_bucket": True,
                "normalized_counts_sha256": normalized_sha,
                "network_optimizer_rng_equivalent": True,
                "next_update_equivalent": True}
        validation_path = job_dir / "recoveries" / "validation.json"
        write_json(validation_path, validation)
        payload = {"format": "satellite_rl_workbench_checkpoint_v1",
                   "config_hash": "fixture-config-hash",
                   "progress": {"env_step": 100000}, "rng": {"python": [1]},
                   "agent": {"update_step": 98001, "rng": [1],
                             "replay": {"capacity": 20000, "next_position": 0,
                                        "rows": [None] * 20000,
                                        "slot_entity_counts": counts + [0]}}}
        return store, job, payload, validation_path, expected_runs

    def test_exact_failed_pilot_rebinds_without_touching_checkpoint_or_gate(self):
        with TemporaryDirectory() as temporary:
            store, job, payload, validation_path, expected = self.fixture(temporary)
            directory = store.job_dir(RUN)
            before = {name: digest(directory / name) for name in
                      ("checkpoint.pt", "failure.json", "pilot_gate.json")}
            new_root = store.root / "source_snapshots" / REPAIR
            with (patch.object(recovery, "S2P19_REPLAY_RUNS", expected),
                  patch.object(store, "freeze_source", return_value=(new_root, REPAIR)),
                  patch("satellite_rl.workbench.checkpoints.config_hash",
                        return_value="fixture-config-hash"),
                  patch("torch.load", return_value=payload)):
                updated = recovery.prepare_s2p19_replay_recovery(
                    store, job, str(validation_path), digest(validation_path), REPAIR)
                interrupted_after_archive = recovery.prepare_s2p19_replay_recovery(
                    store, job, str(validation_path), digest(validation_path), REPAIR)
                repeated = recovery.prepare_s2p19_replay_recovery(
                    store, updated, str(validation_path), digest(validation_path), REPAIR)
            self.assertEqual(interrupted_after_archive, updated)
            self.assertEqual(repeated, updated)
            self.assertEqual(updated["git_commit"], REPAIR)
            self.assertEqual(updated["scientific_origin_commit"], ORIGIN)
            self.assertEqual(updated["replay_repair"]["checkpoint_sha256"], before["checkpoint.pt"])
            self.assertEqual(len(updated["source_history"]), 1)
            self.assertEqual(before, {name: digest(directory / name) for name in before})
            archive = Path(updated["replay_repair"]["archive_path"])
            self.assertEqual(digest(archive / "original/checkpoint.pt"), before["checkpoint.pt"])
            saved_source = {"git_commit": ORIGIN, "source_root": job["source_root"],
                            "source_manifest_sha256": digest(Path(job["source_root"]) /
                                                             "source_manifest.json")}
            current_source = {"git_commit": REPAIR, "source_root": updated["source_root"],
                              "source_manifest_sha256": digest(Path(updated["source_root"]) /
                                                               "source_manifest.json")}
            with patch.object(recovery, "S2P19_REPLAY_RUNS", expected):
                chain = learning_snapshots._verified_source_chain(
                    saved_source, current_source, updated["source_history"])
            self.assertEqual(len(chain), 1)
            layer = torch.nn.Linear(1, 1)
            row = (np.array([1.0], dtype=np.float32), (0,), 0.5,
                   np.array([2.0], dtype=np.float32), True)
            agent = SimpleNamespace(actor=layer, critic1=layer, critic2=layer,
                                    target1=layer, target2=layer,
                                    log_alpha=torch.tensor(0.0),
                                    update_step=98001,
                                    replay=SimpleNamespace(rows=[row]))
            snapshot_dir = Path(temporary) / "learning"
            snapshot_config = {"algorithm": "sac", "network": {"kind": "mlp"},
                               "state": {"features": ["x"]}}
            base = {"environment_profile": {"id": "europe_v1"}, "run_id": RUN}
            with (patch.object(recovery, "S2P19_REPLAY_RUNS", expected),
                  patch.object(learning_snapshots, "config_hash",
                               return_value="fixture-config-hash")):
                old_path = Path(learning_snapshots.save_learning_snapshot(
                    snapshot_dir, agent, snapshot_config, {**base, **saved_source}, 100000))
                old_sha = digest(old_path)
                reused = Path(learning_snapshots.save_learning_snapshot(
                    snapshot_dir, agent, snapshot_config, {**base, **current_source}, 100000,
                    source_history=updated["source_history"]))
                fresh = Path(learning_snapshots.save_learning_snapshot(
                    snapshot_dir, agent, snapshot_config, {**base, **current_source}, 110000,
                    source_history=updated["source_history"]))
            self.assertEqual(reused, old_path)
            self.assertEqual(digest(old_path), old_sha)
            self.assertTrue(old_path.with_name(
                "step_000100000_recovery_" + REPAIR[:12] + ".json").is_file())
            self.assertEqual(torch.load(fresh, map_location="cpu", weights_only=True)["provenance"]["git_commit"],
                             REPAIR)
            validation_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "receipt SHA changed"):
                learning_snapshots._verified_source_chain(
                    saved_source, current_source, updated["source_history"])

    def test_nonzero_tail_or_other_run_cannot_use_repair(self):
        with TemporaryDirectory() as temporary:
            store, job, payload, validation_path, expected = self.fixture(temporary)
            new_root = store.root / "source_snapshots" / REPAIR
            payload["agent"]["replay"]["slot_entity_counts"][-1] = 3
            with (patch.object(recovery, "S2P19_REPLAY_RUNS", expected),
                  patch.object(store, "freeze_source", return_value=(new_root, REPAIR)),
                  patch("satellite_rl.workbench.checkpoints.config_hash",
                        return_value="fixture-config-hash"),
                  patch("torch.load", return_value=payload)):
                with self.assertRaisesRegex(ValueError, "唯一未使用零尾位"):
                    recovery.prepare_s2p19_replay_recovery(
                        store, job, str(validation_path), digest(validation_path), REPAIR)
                with self.assertRaisesRegex(ValueError, "只允许两个既有试点"):
                    recovery.prepare_s2p19_replay_recovery(
                        store, {**job, "run_id": "train_sac_i0"},
                        str(validation_path), digest(validation_path), REPAIR)
                payload["agent"]["replay"]["slot_entity_counts"][-1] = 0
                failure_path = store.job_dir(RUN) / "failure.json"
                failure = json.loads(failure_path.read_text(encoding="utf-8"))
                failure["traceback"] = "different replay failure"
                write_json(failure_path, failure)
                with self.assertRaisesRegex(ValueError, "失败签名"):
                    recovery.prepare_s2p19_replay_recovery(
                        store, job, str(validation_path), digest(validation_path), REPAIR)

    def test_scheduler_rejects_plain_retry_of_known_failed_pilot(self):
        with TemporaryDirectory() as temporary:
            store, job, _, _, expected = self.fixture(temporary)
            scheduler = Scheduler(store)
            with patch.object(recovery, "S2P19_REPLAY_RUNS", expected):
                with self.assertRaisesRegex(ValueError, "不能普通恢复"):
                    scheduler.resume(job["run_id"])
            self.assertEqual(json.loads((store.job_dir(RUN) / "status.json").read_text())["stage"],
                             "failed")

    def test_scheduler_routes_exact_same_id_repair_and_is_idempotent_when_queued(self):
        with TemporaryDirectory() as temporary:
            store, job, _, validation_path, expected = self.fixture(temporary)
            scheduler = Scheduler(store)
            proof = {"repair_id": recovery.S2P19_REPLAY_REPAIR_ID,
                     "origin_science_commit": ORIGIN, "repair_commit": REPAIR,
                     "validation_receipt_sha256": digest(validation_path),
                     "checkpoint_step": 100000, "checkpoint_update_step": 98001,
                     "checkpoint_sha256": digest(store.job_dir(RUN) / "checkpoint.pt"),
                     "archive_path": str(store.job_dir(RUN) / "recoveries" / "saved")}
            repaired = {**job, "source_root": str(store.root / "source_snapshots" / REPAIR),
                        "git_commit": REPAIR, "replay_repair": proof,
                        "scientific_origin_commit": ORIGIN}
            args = {"source_repair": recovery.S2P19_REPLAY_REPAIR_ID,
                    "repair_validation_path": str(validation_path),
                    "repair_validation_sha256": digest(validation_path),
                    "expected_origin_commit": ORIGIN, "expected_repair_commit": REPAIR}
            with (patch.object(recovery, "S2P19_REPLAY_RUNS", expected),
                  patch.object(recovery, "prepare_s2p19_replay_recovery",
                               return_value=repaired) as prepare,
                  patch("satellite_rl.workbench.runtime.pid_alive", return_value=False)):
                first = scheduler.resume(RUN, **args)
                second = scheduler.resume(RUN, **args)
            prepare.assert_called_once()
            self.assertEqual(first["stage"], "queued")
            self.assertEqual(second["stage"], "queued")
            self.assertEqual(first["env_step"], 100000)
            self.assertEqual(first["source_repair"], recovery.S2P19_REPLAY_REPAIR_ID)
            saved_job = json.loads((store.job_dir(RUN) / "job.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_job["run_id"], RUN)
            self.assertEqual(saved_job["request_id"], expected[RUN]["request_id"])
            self.assertEqual(saved_job["git_commit"], REPAIR)


if __name__ == "__main__":
    unittest.main()
