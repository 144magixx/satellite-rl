"""Strict S1-V10 same-ID throughput recovery gates with a synthetic checkpoint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from satellite_rl.coverage.config import config_hash
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.coverage_variable import variable_default_config
from satellite_rl.workbench.storage import Store, digest
from satellite_rl.workbench.runtime import Scheduler
from satellite_rl.workbench.variable_throughput_recovery import (
    CRITERIA, ORIGIN_COMMIT, REPAIR_ID, TARGETS,
    TRAIN_SHA256, VALIDATION_SHA256, DATA_CONTRACT_SHA256,
    _checkpoint_state, prepare_variable_throughput_recovery)


NEW_COMMIT = "d" * 40
RUN_ID = "train_ppo_20261009_095314_19a69288"


def _source(root, commit, changed):
    root.mkdir(parents=True)
    content = {}
    for name in ("src/satellite_rl/coverage/variable_design.py",
                 "src/satellite_rl/coverage/variable_runner.py",
                 "src/satellite_rl/workbench/runtime.py",
                 "src/satellite_rl/workbench/coverage_variable.py"):
        payload = ("new" if changed else "old").encode() + name.encode()
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        content[name] = hashlib.sha256(payload).hexdigest()
    if changed:
        name = "src/satellite_rl/workbench/variable_throughput_recovery.py"
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("recovery helper")
        content[name] = digest(target)
    atomic_json(root / "source_manifest.json", {"git_commit": commit, "files": content})


class ThroughputRecoveryTests(unittest.TestCase):
    def fixture(self, root):
        store = Store(root / "20418")
        old = store.root / "source_snapshots" / ORIGIN_COMMIT
        new = store.root / "source_snapshots" / NEW_COMMIT
        _source(old, ORIGIN_COMMIT, False)
        _source(new, NEW_COMMIT, True)
        store.freeze_source = lambda: (new, NEW_COMMIT)
        directory = store.job_dir(RUN_ID)
        directory.mkdir()
        config = variable_default_config()
        config["coverage"]["variable"]["route"] = "R08"
        config["coverage"]["method"] = "ppo"
        config["datasets"] = {
            "train": {"sha256": TRAIN_SHA256, "partition_role": "train",
                      "path": str(directory / "train_inputs.npz")},
            "validation": {"sha256": VALIDATION_SHA256, "partition_role": "validation",
                           "path": str(directory / "validation_inputs.npz")}}
        atomic_json(directory / "config.json", config)
        contract = directory / "input_sources" / "canonical_dataset_contract.json"
        contract.parent.mkdir()
        contract.write_text("synthetic original scene contract")
        job = {"run_id": RUN_ID, "request_id": TARGETS[RUN_ID]["request_id"],
               "kind": "train", "research_stage": "coverage", "experiment_id": "S1-V10",
               "experiment_role": "formal", "variable_route": "R08",
               "execution_target": "u2024110224@10.160.17.102:20418",
               "archive_root": str(store.archive_root), "source_root": str(old),
               "git_commit": ORIGIN_COMMIT,
               "variable_dataset_sha256": {"train": TRAIN_SHA256,
                                           "validation": VALIDATION_SHA256},
               "variable_dataset_contract": {"path": str(contract),
                                             "sha256": DATA_CONTRACT_SHA256}}
        atomic_json(directory / "job.json", job)
        checkpoint = {"format_version": 2, "run_id": RUN_ID,
                      "config_hash": config_hash(config), "source_commit": ORIGIN_COMMIT,
                      "policy": {"weight": torch.tensor([1.0])},
                      "optimizer": {"state": {0: {"exp_avg": torch.tensor([.1])}},
                                    "param_groups": [{"lr": .0003}]},
                      "rng": {"python": (1,), "numpy": (2,),
                              "torch": torch.tensor([3], dtype=torch.uint8),
                              "cuda": [torch.tensor([4], dtype=torch.uint8)]},
                      "streams": [{"sampler_state": {"state": i}, "layout": {},
                                   "scene_index": i, "level": 0, "episode_serial": 0}
                                  for i in range(32)],
                      "step": 2048, "updates": 128, "episode": 10,
                      "last_validation": 0,
                      "ledger_counts": {name: 0 for name in
                          ("actions.jsonl", "episodes.jsonl", "updates.jsonl", "evaluations.jsonl")}}
        torch.save(checkpoint, directory / "checkpoint.pt")
        checkpoint_sha = digest(directory / "checkpoint.pt")
        atomic_json(directory / "status.json", {"run_id": RUN_ID, "stage": "stopped",
                   "stop_reason": "requested", "pid": 999999,
                   "env_step": 2048, "update_step": 128, "episode": 10,
                   "checkpoint_sha256": checkpoint_sha})
        atomic_json(directory / "launch.json", {"pid": 999999,
                   "process_start_time": 1_700_000_000.0})
        (directory / "worker.log").write_text("stopped normally\n")
        evidence = []
        for index in range(5):
            path = store.archive_root / f"proof_{index}.json"
            path.write_text(json.dumps({"index": index}))
            evidence.append({"path": str(path), "sha256": digest(path)})
        receipt = {"format": "s1v10_throughput_repair_validation_v1",
                   "repair_id": REPAIR_ID,
                   "origin_commit": ORIGIN_COMMIT, "repair_commit": NEW_COMMIT,
                   "old_source_root": str(old), "new_source_root": str(new),
                   "run_id": RUN_ID, "request_id": job["request_id"], "route": "R08",
                   "config_sha256": digest(directory / "config.json"),
                   "config_hash": config_hash(config),
                   "dataset_sha256": {"train": TRAIN_SHA256,
                                      "validation": VALIDATION_SHA256},
                   "stream_count": 32,
                   "saved_validation_nodes": [],
                   "actual_checkpoint": {"sha256": checkpoint_sha,
                                         "step": 2048, "updates": 128, "episode": 10},
                   "tests_passed": True, "approved": True,
                   "criteria": {key: True for key in CRITERIA},
                   "evidence_files": evidence}
        receipt_path = directory / "recoveries" / "validation_receipt.json"
        receipt_path.parent.mkdir()
        atomic_json(receipt_path, receipt)
        return store, job, directory, receipt_path, contract

    def test_exact_stopped_job_rebinds_and_archives_without_replacing_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, job, directory, receipt, contract = self.fixture(Path(temporary))
            original_job_sha = digest(directory / "job.json")
            original_checkpoint_sha = digest(directory / "checkpoint.pt")
            real_digest = digest
            def checked_digest(path):
                return DATA_CONTRACT_SHA256 if Path(path) == contract else real_digest(path)
            with patch("satellite_rl.workbench.variable_throughput_recovery.pid_alive",
                       return_value=False), patch(
                           "satellite_rl.workbench.variable_throughput_recovery.digest",
                           side_effect=checked_digest):
                updated = prepare_variable_throughput_recovery(
                    store, job, receipt, real_digest(receipt), NEW_COMMIT)
            self.assertEqual(updated["run_id"], RUN_ID)
            self.assertEqual(updated["request_id"], job["request_id"])
            self.assertEqual(updated["git_commit"], NEW_COMMIT)
            self.assertEqual(updated["variable_throughput_repair"]["checkpoint_sha256"],
                             original_checkpoint_sha)
            archive = Path(updated["variable_throughput_repair"]["archive_path"])
            self.assertEqual(digest(archive / "original/job.json"), original_job_sha)
            self.assertEqual(digest(archive / "original/checkpoint.pt"), original_checkpoint_sha)
            self.assertEqual(digest(directory / "job.json"), original_job_sha)

    def test_rejects_wrong_run_and_unbound_validation_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, job, directory, receipt, contract = self.fixture(Path(temporary))
            bad_job = dict(job, request_id="another-request")
            with self.assertRaises(ValueError):
                prepare_variable_throughput_recovery(store, bad_job, receipt,
                                                     digest(receipt), NEW_COMMIT)
            with patch("satellite_rl.workbench.variable_throughput_recovery.pid_alive",
                       return_value=False):
                with self.assertRaises(ValueError):
                    prepare_variable_throughput_recovery(store, job, receipt,
                                                         "0" * 64, NEW_COMMIT)
            self.assertFalse((directory / "recoveries" /
                f"{REPAIR_ID}_{NEW_COMMIT[:12]}").exists())

    def test_runtime_queues_same_id_once_with_repair_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, job, directory, receipt, contract = self.fixture(Path(temporary))
            real_digest = digest
            def checked_digest(path):
                return DATA_CONTRACT_SHA256 if Path(path) == contract else real_digest(path)
            scheduler = Scheduler(store)
            arguments = {"source_repair": REPAIR_ID,
                         "repair_validation_path": str(receipt),
                         "repair_validation_sha256": real_digest(receipt),
                         "expected_origin_commit": ORIGIN_COMMIT,
                         "expected_repair_commit": NEW_COMMIT}
            with patch("satellite_rl.workbench.runtime.pid_alive", return_value=False), \
                 patch("satellite_rl.workbench.variable_throughput_recovery.pid_alive",
                       return_value=False), \
                 patch("satellite_rl.workbench.variable_throughput_recovery.digest",
                       side_effect=checked_digest):
                queued = scheduler.resume(RUN_ID, **arguments)
                duplicate = scheduler.resume(RUN_ID, **arguments)
            self.assertEqual(queued["stage"], "queued")
            self.assertEqual(duplicate["stage"], "queued")
            saved = json.loads((directory / "job.json").read_text())
            self.assertEqual(saved["run_id"], RUN_ID)
            self.assertEqual(saved["git_commit"], NEW_COMMIT)
            self.assertEqual(saved["variable_throughput_repair"]["repair_id"], REPAIR_ID)
            self.assertEqual(len(saved["source_history"]), 1)

    def test_saved_validation_prefix_is_bound_to_checkpoint_and_summary_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, job, directory, receipt, contract = self.fixture(Path(temporary))
            checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu",
                                    weights_only=False)
            checkpoint.update(step=16384, updates=1024, episode=20,
                              last_validation=16384)
            checkpoint["ledger_counts"]["evaluations.jsonl"] = 1
            torch.save(checkpoint, directory / "checkpoint.pt")
            rows = [{"scene_id": f"mother_{i}", "scene_sha256": f"sha_{i}",
                     "demand_multiplier": multiplier, "route": "R08",
                     "demand_conserved": True}
                    for i in range(20) for multiplier in (.75, 1., 1.25)]
            node = directory / "validation_16384.jsonl"
            node.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            summary = {"env_step": 16384, "scene_count": 60,
                       "rows_sha256": digest(node)}
            (directory / "evaluations.jsonl").write_text(json.dumps(summary) + "\n")
            status = json.loads((directory / "status.json").read_text())
            status.update(env_step=16384, update_step=1024, episode=20,
                          checkpoint_sha256=digest(directory / "checkpoint.pt"))
            atomic_json(directory / "status.json", status)
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            checked = _checkpoint_state(directory, job, config, status)
            self.assertEqual(checked["saved_validation_nodes"], [
                {"env_step": 16384, "rows": 60, "sha256": digest(node)}])
            rows[0]["demand_conserved"] = False
            node.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation row"):
                _checkpoint_state(directory, job, config, status)


if __name__ == "__main__":
    unittest.main()
