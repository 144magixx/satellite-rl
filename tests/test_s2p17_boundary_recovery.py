"""Narrow, filesystem-only proof for the S2-P17 same-ID boundary repair."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from satellite_rl.workbench.recovery import (
    S2P17_BOUNDARY_REPAIR_ID, S2P17_FAILED_REQUEST, S2P17_FAILED_RUN,
    S2P17_ORIGIN_COMMIT, prepare_s2p17_boundary_recovery,
)
from satellite_rl.workbench.runtime import Scheduler


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class _Store:
    def __init__(self, root, new_root, new_commit):
        self.root, self.new_root, self.new_commit = root, new_root, new_commit

    def job_dir(self, run_id):
        return self.root / "jobs" / run_id

    def freeze_source(self):
        return self.new_root, self.new_commit


class BoundaryRecoveryTests(unittest.TestCase):
    def test_scheduler_resume_keeps_run_request_and_exposes_both_commits(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            new_commit = "b" * 40
            directory = root / "jobs" / S2P17_FAILED_RUN
            original = {"run_id": S2P17_FAILED_RUN, "request_id": S2P17_FAILED_REQUEST,
                        "kind": "refine", "experiment_id": "S2-P17",
                        "git_commit": S2P17_ORIGIN_COMMIT}
            _write(directory / "job.json", original)
            _write(directory / "status.json", {"stage": "failed", "pid": None,
                                               "env_step": 299, "budget_env_steps": 3960})
            proof = {"repair_id": S2P17_BOUNDARY_REPAIR_ID,
                     "origin_science_commit": S2P17_ORIGIN_COMMIT,
                     "repair_commit": new_commit,
                     "archive_path": str(directory / "recoveries" / "proof"),
                     "fallback_contract_sha256": "c" * 64,
                     "validation_receipt_sha256": "d" * 64}
            updated = {**original, "git_commit": new_commit,
                       "source_root": str(root / "source_snapshots" / new_commit),
                       "scientific_origin_commit": S2P17_ORIGIN_COMMIT,
                       "boundary_repair": proof}
            scheduler = Scheduler(_Store(root, root / "source_snapshots" / new_commit,
                                         new_commit))
            with patch("satellite_rl.workbench.runtime.pid_alive", return_value=False), \
                    patch("satellite_rl.workbench.recovery.prepare_s2p17_boundary_recovery",
                          return_value=updated) as prepare:
                status = scheduler.resume(
                    S2P17_FAILED_RUN, source_repair=S2P17_BOUNDARY_REPAIR_ID,
                    repair_validation_path=str(directory / "recoveries" / "validation.json"),
                    repair_validation_sha256="d" * 64,
                    expected_origin_commit=S2P17_ORIGIN_COMMIT,
                    expected_repair_commit=new_commit)
                self.assertEqual(status["stage"], "queued")
                self.assertEqual(status["scientific_origin_commit"], S2P17_ORIGIN_COMMIT)
                self.assertEqual(status["repair_commit"], new_commit)
                self.assertEqual(json.loads((directory / "job.json").read_text())["request_id"],
                                 S2P17_FAILED_REQUEST)
                self.assertEqual(prepare.call_count, 1)
                again = scheduler.resume(
                    S2P17_FAILED_RUN, source_repair=S2P17_BOUNDARY_REPAIR_ID,
                    repair_validation_path=str(directory / "recoveries" / "validation.json"),
                    repair_validation_sha256="d" * 64,
                    expected_origin_commit=S2P17_ORIGIN_COMMIT,
                    expected_repair_commit=new_commit)
                self.assertEqual(again["stage"], "queued")
                self.assertEqual(prepare.call_count, 1)

    def test_exact_failed_job_binds_new_snapshot_and_preserves_299(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            new_commit = "b" * 40
            old_root = root / "source_snapshots" / S2P17_ORIGIN_COMMIT
            new_root = root / "source_snapshots" / new_commit
            names = ("src/satellite_rl/power_refinement/priority.py",
                     "src/satellite_rl/power_refinement/runner.py",
                     "src/satellite_rl/workbench/recovery.py")
            for source_root, commit, tag in ((old_root, S2P17_ORIGIN_COMMIT, "old"),
                                             (new_root, new_commit, "new")):
                files = {}
                for name in names:
                    path = source_root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(tag + name, encoding="utf-8")
                    files[name] = _sha(path)
                _write(source_root / "source_manifest.json", {"git_commit": commit,
                                                               "files": files})
            directory = root / "jobs" / S2P17_FAILED_RUN
            cases = directory / "power_cases"
            expected = [f"scene_{index:05d}_r{repeat}_{method}.json"
                        for index in range(440) for repeat in range(3)
                        for method in ("none", "sca", "sca_si_lp")][:299]
            for name in expected:
                _write(cases / name, {"method": name.rsplit("_", 1)[-1]})
            a_path = cases / "scene_00033_r0_sca.json"
            _write(a_path, {"method": "sca", "repeat": 0,
                            "scene_id": "scene33-r0", "source_scene_sha256": "c" * 64,
                            "initial_allocation_sha256": "d" * 64})
            _write(directory / "status.json", {"stage": "failed", "env_step": 299,
                                                "budget_env_steps": 3960})
            _write(directory / "failure.json", {
                "error": "SI优先结果新增严格基本服务跨下波束"})
            _write(directory / "job.json", {"run_id": S2P17_FAILED_RUN})
            _write(directory / "config.json", {"experiment_id": "S2-P17"})
            _write(directory / "refinement_manifest.json", {"identity": "fixture",
                                                          "budget_cases": 3960})
            _write(directory / "refinement_checkpoint.json", {
                "identity": "fixture", "completed_cases": 299,
                "last_case": str(a_path), "last_case_sha256": _sha(a_path)})
            (directory / "worker.log").write_text("failure", encoding="utf-8")
            (directory / "events.jsonl").write_text("", encoding="utf-8")
            (directory / "power_iterations.jsonl").write_text("".join(
                json.dumps({"method": "sca_si_lp", "repeat": 0,
                            "scene_id": "scene33-r0", "evaluation": 684 if i == 48 else i + 1,
                            "elapsed_seconds": 1.0373822273686528 if i == 48 else 0.01 * i}) + "\n"
                for i in range(49)), encoding="utf-8")
            validation_path = directory / "recoveries" / "validation_receipt.json"
            _write(validation_path, {"format": "s2p17_boundary_validation_v1",
                "repair_id": S2P17_BOUNDARY_REPAIR_ID,
                "origin_science_commit": S2P17_ORIGIN_COMMIT,
                "repair_commit": new_commit,
                "original_299_cases_verified": True,
                "failed_B_fallback_to_A_verified": True,
                "strict_final_guard_unchanged": True,
                "candidate_boundary_consistent": True,
                "tests_passed": True})
            job = {"run_id": S2P17_FAILED_RUN, "request_id": S2P17_FAILED_REQUEST,
                   "kind": "refine", "experiment_id": "S2-P17",
                   "git_commit": S2P17_ORIGIN_COMMIT, "source_root": str(old_root),
                   "initial_source": {"group": "model"}}
            store = _Store(root, new_root, new_commit)
            updated = prepare_s2p17_boundary_recovery(store, job, validation_path,
                                                         _sha(validation_path), new_commit)
            self.assertEqual(updated["run_id"], S2P17_FAILED_RUN)
            self.assertEqual(updated["git_commit"], new_commit)
            self.assertEqual(updated["scientific_origin_commit"], S2P17_ORIGIN_COMMIT)
            self.assertEqual(updated["boundary_repair"]["original_299_inventory_sha256"],
                             updated["source_history"][-1]["original_299_inventory_sha256"])
            self.assertEqual(updated["failed_attempt_fallback_path"],
                             str(directory / "failed_attempt_fallback.json"))
            self.assertEqual(prepare_s2p17_boundary_recovery(
                store, job, validation_path, _sha(validation_path), new_commit), updated)
            self.assertEqual(prepare_s2p17_boundary_recovery(
                store, updated, validation_path, _sha(validation_path), new_commit), updated)
            self.assertEqual(len(list((directory / "power_cases").glob("*.json"))), 299)


if __name__ == "__main__":
    unittest.main()
