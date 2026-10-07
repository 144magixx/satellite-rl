"""Exact same-ID I0 validation-budget source repair, never a new training arm."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from satellite_rl.workbench import recovery
from satellite_rl.workbench.runtime import Scheduler
from satellite_rl.workbench.storage import Store, digest


RUN = recovery.I0_RUN_ID
ORIGIN = recovery.I0_ORIGIN_COMMIT
REPAIR = "b" * 40


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class I0BudgetRecoveryTests(unittest.TestCase):
    def fixture(self, temporary, *, with_200k=False):
        root = Path(temporary)
        archive = root / "archive"
        store = Store(archive, source_root=root)
        directory = store.job_dir(RUN)
        directory.mkdir(parents=True)
        old = store.root / "source_snapshots" / ORIGIN
        new = store.root / "source_snapshots" / REPAIR
        for source, marker, commit in ((old, "old", ORIGIN), (new, "budget", REPAIR)):
            files = {}
            for name in recovery.I0_BUDGET_SOURCE_FILES:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name + marker, encoding="utf-8")
                files[name] = digest(path)
            write(source / "source_manifest.json", {"git_commit": commit, "files": files})
        controls = {"seed": 51001, "total_steps": 400000, "pilot_stop_steps": 100000,
                    "pilot_max_episodes": 3125, "max_episodes": 12500, "device": "cuda"}
        config = {"schema_version": 1, "research_stage": "sac_improvement", "i0": controls,
                  "training": {"seed": 51001, "total_steps": 400000, "device": "cuda"}}
        write(directory / "config.json", config)
        config_sha = digest(directory / "config.json")
        job = {"run_id": RUN, "request_id": recovery.I0_REQUEST_ID, "kind": "train",
               "experiment_id": "S2-P19", "research_stage": "sac_improvement",
               "mode": "train", "git_commit": ORIGIN, "source_root": str(old),
               "seed": 51001, "total_steps": 400000, "i0_config": controls}
        write(directory / "job.json", job)
        checkpoint = directory / "checkpoint.pt"
        checkpoint.write_bytes(b"unchanged stopped I0 checkpoint fixture")
        cp_sha = digest(checkpoint)
        step = 230000 if with_200k else 130000
        episodes, updates = 7000, step - 1999
        status = {"run_id": RUN, "stage": "stopped", "stop_reason": "user_stop",
                  "pilot_gate_passed": True, "env_step": step, "episodes": episodes,
                  "updates": updates, "checkpoint_step": step,
                  "checkpoint_sha256": cp_sha, "pid": 0}
        write(directory / "status.json", status)
        pilot = directory / "pilot_artifacts"
        pilot.mkdir()
        (pilot / "checkpoint.pt").write_bytes(b"approved 97112 checkpoint fixture")
        pilot_sha = digest(pilot / "checkpoint.pt")
        audit_source = archive / "ops" / "audit" / "source.json"
        write(audit_source, {"original": True})
        audit = archive / "ops" / "audit" / "I0.json"
        write(audit, {"run_id": RUN, "checkpoint_sha256": pilot_sha,
                      "checkpoint_step": 97112, "passed": True,
                      "criteria": {"real_72_cases": True},
                      "source_files": [{"path": str(audit_source),
                                        "sha256": digest(audit_source)}]})
        gate = {"pass": True, "run_id": RUN, "checkpoint_step": 97112,
                "checkpoint_sha256": pilot_sha, "audit_path": str(audit),
                "audit_sha256": digest(audit)}
        write(directory / "pilot_gate.json", gate)
        write(pilot / "manifest.json", {"run_id": RUN, "env_step": 97112,
                                         "checkpoint_sha256": pilot_sha,
                                         "config_sha256": config_sha,
                                         "git_commit": ORIGIN, "audit_path": str(audit),
                                         "audit_sha256": digest(audit)})
        nodes = []
        steps = (*recovery.I0_SAVED_VALIDATION_STEPS, *((200000,) if with_200k else ()))
        for node in steps:
            node_dir = directory / "validation" / f"step_{node:09d}"
            cases, hashes = [], []
            for index in range(72):
                path = node_dir / f"scene_{index:05d}.json"
                write(path, {"env_step": node, "checkpoint_sha256": str(node).zfill(64),
                             "scene_id": index})
                case_sha = digest(path)
                cases.append({"path": str(path), "sha256": case_sha})
                hashes.append(case_sha)
            summary = node_dir / "summary.json"
            write(summary, {"env_step": node, "count": 72,
                            "checkpoint_sha256": str(node).zfill(64),
                            "cases": cases})
            nodes.append({"step": node, "summary_sha256": digest(summary),
                          "case_hashes_sha256": recovery._canonical_sha(hashes),
                          "checkpoint_sha256": str(node).zfill(64)})
        eq = archive / "ops" / "equality" / "120k.json"
        write(eq, {"strict_equivalence_passed": True,
                   "copy_sha256_before": recovery.I0_EQUIVALENCE_120K_SHA256,
                   "copy_sha256_after": recovery.I0_EQUIVALENCE_120K_SHA256,
                   "copied_env_step": 120000, "diagnostic_max_abs": 0,
                   "old_source_commit": ORIGIN,
                   "patch_commit": "708c5264f9ebcd2a0b423da946c7042337211d74",
                   "old_new_config_equal": True,
                   "old_new_checkpoint_loader_source_equal": True,
                   "gpu_replay_sample_equal": True,
                   "partial_env_state_equal": True,
                   "simulated_skip": {"rng_unchanged": True,
                                      "no_validation_case_or_summary_created": True,
                                      "300k_checkpoint_due": True},
                   "post_network_max_abs": 0, "post_optimizer_max_abs": 0,
                   "actor_double_q_parameter_changes": {"actor": .01,
                                                       "critic1": .02,
                                                       "critic2": .03}})
        criteria = {key: True for key in (
            "saved_nodes_sha_unchanged", "same_checkpoint_learning_state",
            "one_gpu_update_equal", "training_rng_equal", "validation_rng_restored",
            "future_direct_nodes_at_most_six", "skip_only_future_300k_validation")}
        validation_path = directory / "recoveries" / "budget_validation.json"
        write(validation_path, {"format": "s2p19_i0_validation_budget_repair_v1",
              "repair_id": recovery.I0_BUDGET_REPAIR_ID,
              "origin_commit": ORIGIN, "repair_commit": REPAIR,
              "run_id": RUN, "tests_passed": True, "criteria": criteria,
              "actual_recovery_checkpoint": {"sha256": cp_sha, "env_step": step,
                                             "episodes": episodes, "updates": updates},
              "equivalence_checkpoint": {"sha256": recovery.I0_EQUIVALENCE_120K_SHA256,
                                         "env_step": 120000, "source_commit": ORIGIN,
                                         "proof_path": str(eq), "proof_sha256": digest(eq)},
              "config_sha256": config_sha, "pilot_checkpoint_sha256": pilot_sha,
              "pilot_audit_sha256": digest(audit), "saved_nodes": nodes,
              "evidence_files": [{"path": str(eq), "sha256": digest(eq)}]})
        payload = {"version": 1, "run_id": RUN,
                   "frozen_config_sha256": config_sha, "i0_config": controls,
                   "env_step": step, "episodes": episodes, "scene_index": 7,
                   "pilot_gate_passed": True, "environment": {"version": 1},
                   "agent": {"updates": updates, **{name: {} for name in (
                       "actor", "critic1", "critic2", "target1", "target2",
                       "actor_opt", "critic1_opt", "critic2_opt", "replay")}},
                   "rng": {name: [1] for name in ("python", "numpy", "torch", "cuda")}}
        return store, job, payload, validation_path, new

    def test_exact_i0_same_id_rebinds_and_archives_all_old_nodes(self):
        with TemporaryDirectory() as temporary:
            store, job, payload, validation_path, new_root = self.fixture(temporary)
            directory = store.job_dir(RUN)
            checkpoint_sha = digest(directory / "checkpoint.pt")
            old_node = directory / "validation/step_000097112/scene_00000.json"
            old_case_sha = digest(old_node)
            with (patch.object(recovery, "I0_CONFIG_SHA256", digest(directory / "config.json")),
                  patch.object(store, "freeze_source", return_value=(new_root, REPAIR)),
                  patch("torch.load", return_value=payload)):
                updated = recovery.prepare_i0_validation_budget_recovery(
                    store, job, str(validation_path), digest(validation_path), REPAIR)
                interrupted_after_archive = recovery.prepare_i0_validation_budget_recovery(
                    store, job, str(validation_path), digest(validation_path), REPAIR)
                repeated = recovery.prepare_i0_validation_budget_recovery(
                    store, updated, str(validation_path), digest(validation_path), REPAIR)
            self.assertEqual(interrupted_after_archive, updated)
            self.assertEqual(repeated, updated)
            self.assertEqual(updated["run_id"], RUN)
            self.assertEqual(updated["git_commit"], REPAIR)
            self.assertEqual(updated["scientific_origin_commit"], ORIGIN)
            self.assertEqual(digest(directory / "checkpoint.pt"), checkpoint_sha)
            self.assertEqual(digest(old_node), old_case_sha)
            archive = Path(updated["i0_validation_budget_repair"]["archive_path"])
            self.assertEqual(digest(archive / "original/checkpoint.pt"), checkpoint_sha)
            self.assertEqual(digest(archive / "original/validation/step_000097112/scene_00000.json"),
                             old_case_sha)
            self.assertEqual(len(updated["i0_validation_budget_repair"]["saved_nodes"]), 4)
            self.assertEqual(len(updated["source_history"]), 1)

    def test_optional_200k_node_preserved_but_300k_partial_rejected(self):
        with TemporaryDirectory() as temporary:
            store, job, payload, validation_path, new_root = self.fixture(
                temporary, with_200k=True)
            directory = store.job_dir(RUN)
            with (patch.object(recovery, "I0_CONFIG_SHA256", digest(directory / "config.json")),
                  patch.object(store, "freeze_source", return_value=(new_root, REPAIR)),
                  patch("torch.load", return_value=payload)):
                updated = recovery.prepare_i0_validation_budget_recovery(
                    store, job, str(validation_path), digest(validation_path), REPAIR)
            self.assertEqual(len(updated["i0_validation_budget_repair"]["saved_nodes"]), 5)
        with TemporaryDirectory() as temporary:
            store, job, payload, validation_path, new_root = self.fixture(temporary)
            directory = store.job_dir(RUN)
            partial = directory / "validation/step_000300000/scene_00000.json"
            write(partial, {"partial": True})
            with (patch.object(recovery, "I0_CONFIG_SHA256", digest(directory / "config.json")),
                  patch("torch.load", return_value=payload)):
                with self.assertRaisesRegex(ValueError, "已存验证节点"):
                    recovery.prepare_i0_validation_budget_recovery(
                        store, job, str(validation_path), digest(validation_path), REPAIR)

    def test_plain_resume_and_other_task_are_rejected(self):
        with TemporaryDirectory() as temporary:
            store, job, _, validation_path, _ = self.fixture(temporary)
            scheduler = Scheduler(store)
            with self.assertRaisesRegex(ValueError, "六节点预算修复"):
                scheduler.resume(RUN)
            with self.assertRaisesRegex(ValueError, "原单种子"):
                recovery.prepare_i0_validation_budget_recovery(
                    store, {**job, "run_id": "train_sac_other"},
                    str(validation_path), digest(validation_path), REPAIR)

    def test_scheduler_routes_exact_same_id_and_queued_retry_is_idempotent(self):
        with TemporaryDirectory() as temporary:
            store, job, _, validation_path, new_root = self.fixture(temporary)
            scheduler = Scheduler(store)
            proof = {"repair_id": recovery.I0_BUDGET_REPAIR_ID,
                     "origin_commit": ORIGIN, "repair_commit": REPAIR,
                     "validation_receipt_sha256": digest(validation_path),
                     "checkpoint_step": 130000, "checkpoint_episodes": 7000,
                     "checkpoint_updates": 128001,
                     "checkpoint_sha256": digest(store.job_dir(RUN) / "checkpoint.pt"),
                     "archive_path": str(store.job_dir(RUN) / "recoveries" / "saved")}
            updated = {**job, "git_commit": REPAIR, "source_root": str(new_root),
                       "i0_validation_budget_repair": proof,
                       "scientific_origin_commit": ORIGIN}
            arguments = {"source_repair": recovery.I0_BUDGET_REPAIR_ID,
                         "repair_validation_path": str(validation_path),
                         "repair_validation_sha256": digest(validation_path),
                         "expected_origin_commit": ORIGIN,
                         "expected_repair_commit": REPAIR}
            with (patch.object(recovery, "prepare_i0_validation_budget_recovery",
                               return_value=updated) as prepare,
                  patch("satellite_rl.workbench.runtime.pid_alive", return_value=False)):
                first = scheduler.resume(RUN, **arguments)
                second = scheduler.resume(RUN, **arguments)
            prepare.assert_called_once()
            self.assertEqual(first["stage"], second["stage"])
            self.assertEqual(first["stage"], "queued")
            self.assertEqual(first["env_step"], 130000)
            self.assertEqual(first["updates"], 128001)
            self.assertEqual(first["source_repair"], recovery.I0_BUDGET_REPAIR_ID)
            saved = json.loads((store.job_dir(RUN) / "job.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["run_id"], RUN)
            self.assertEqual(saved["request_id"], recovery.I0_REQUEST_ID)
            self.assertEqual(saved["git_commit"], REPAIR)


if __name__ == "__main__":
    unittest.main()
