"""Only the SHA-bound S2-P17 failed B case may use the audited A fallback."""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from satellite_rl.power_refinement import runner


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FailedAttemptContractTests(unittest.TestCase):
    def test_immutable_failure_evidence_and_repair_source_are_required(self):
        with TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "source_manifest.json").write_text("frozen source", encoding="utf-8")
            evidence = root / "recoveries" / "repair" / "original"
            evidence.mkdir(parents=True)
            failure = evidence / "failure.json"
            failure.write_text("original failure", encoding="utf-8")
            iterations = evidence / "power_iterations.jsonl"
            iterations.write_text("original callbacks", encoding="utf-8")
            a_path = root / "power_cases" / "scene_00033_r0_sca.json"
            a_path.parent.mkdir()
            a_path.write_text("committed A case", encoding="utf-8")
            expected = dict(runner._FAILED_B_REPAIR)
            expected.update(A_case_sha256=digest(a_path),
                            failure_sha256=digest(failure),
                            iterations_sha256=digest(iterations))
            stack.enter_context(patch.object(runner, "_FAILED_B_REPAIR", expected))
            contract = {**expected, "A_case_path": str(a_path),
                        "failure_path": str(failure), "iterations_path": str(iterations),
                        "callback_lower_bounds": {
                            "capacity_calls": 684, "elapsed_seconds": 1.0373822273686528,
                            "accepted_callbacks": 49},
                        "unknown_exact_fields": ["original_attempt_capacity_calls",
                                                 "original_attempt_optimizer_wall_seconds"],
                        "missing_reason": "solver result not persisted after final guard failure",
                        "completed_prefix_case_count": 299,
                        "fallback": "failed_attempt_fallback_to_A"}
            contract_path = root / "failed_attempt_fallback.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            repair_commit = "b" * 40
            job = {"run_id": expected["original_run_id"],
                   "request_id": expected["request_id"], "experiment_id": "S2-P17",
                   "scientific_origin_commit": expected["origin_science_commit"],
                   "original_git_commit": expected["origin_science_commit"],
                   "git_commit": repair_commit, "source_root": str(source),
                   "failed_attempt_fallback_path": str(contract_path),
                   "boundary_repair": {"repair_commit": repair_commit,
                                       "origin_science_commit": expected["origin_science_commit"],
                                       "new_source_root": str(source),
                                       "new_manifest_sha256": digest(source / "source_manifest.json"),
                                       "fallback_contract_sha256": digest(contract_path)}}
            verified = runner._failed_attempt_contract(root, job)
            self.assertEqual(verified["sha256"], digest(contract_path))
            failure.write_text("changed failure", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA"):
                runner._failed_attempt_contract(root, job)
            failure.write_text("original failure", encoding="utf-8")
            job["git_commit"] = "c" * 40
            with self.assertRaisesRegex(ValueError, "原工作台任务"):
                runner._failed_attempt_contract(root, job)

    def test_failed_attempt_time_is_missing_not_zero_in_summary(self):
        metrics = {"final_si_mean": 0.5, "final_sgm_mean": 0.6,
                   "basic_met_count": 2, "all_basic_met": True,
                   "power_w": 100.0, "effective_carried_bps": 3e6,
                   "throughput_bps": 4e6}
        rows = [{"method": "sca_si_lp", "source_scene_sha256": "a" * 64,
                 "metrics": metrics, "solver": {"elapsed_seconds": 2.0,
                                                 "fallback_used": False}},
                {"method": "sca_si_lp", "source_scene_sha256": "b" * 64,
                 "metrics": metrics, "solver": {"elapsed_seconds": None,
                                                 "fallback_used": True}}]
        summary = runner._summaries(rows)["sca_si_lp"]
        self.assertEqual(summary["mean_solver_seconds"], 2.0)
        self.assertEqual(summary["missing_solver_seconds_count"], 1)
        self.assertFalse(summary["timing_available"])


if __name__ == "__main__":
    unittest.main()
