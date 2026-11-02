"""Saved-result repair verification: zero optimizer or learning interactions."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from satellite_rl.workbench.fair20_runner import (full_b_case, validation_only_resume_verified,
                                                 validation_only_recovered_clock)
from satellite_rl.workbench.storage import digest


class StrictFallbackTests(unittest.TestCase):
    def test_saved_complete_solver_journal_falls_back_without_new_solver(self):
        evidence_path = Path("D:/paper/outputs/experiments/s2_fair20_20261012/preflight/validation_failure_100000_20261012/remote_failure_evidence.json")
        if not evidence_path.exists():
            self.skipTest("Archived failure evidence absent on this host")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))["files"]
        direct = evidence["validation_100000/direct_00002.json"]["content"]
        saved = evidence["validation_100000/optimizer_00002.json"]["content"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"optimizer.json"
            path.write_text(json.dumps(saved), encoding="utf-8")
            with patch("satellite_rl.power_refinement.optimizer.optimize_power", side_effect=AssertionError("No extra A budget")), patch(
                    "satellite_rl.power_refinement.priority.optimize_priority", side_effect=AssertionError("No extra B budget")):
                result = full_b_case(direct, lambda: False, journal=path)
            acceptance = result["acceptance"]
            self.assertFalse(acceptance["accepted"])
            self.assertTrue(acceptance["fallback"])
            self.assertEqual(acceptance["candidate"]["basic_met_count"], 49)
            self.assertEqual(acceptance["returned_basic_met_count"], 50)
            self.assertEqual(acceptance["candidate"]["basic_floor_shortfall_beam_ids"], [2])
            self.assertAlmostEqual(acceptance["candidate"]["SI"], .9684989899332939, places=14)
            self.assertAlmostEqual(result["metrics"]["SI"], direct["metrics"]["SI"], places=14)
            self.assertEqual(result["solver"]["A"], saved["A"])
            self.assertEqual(result["solver"]["B"], saved["B"])
            self.assertEqual(result["timing"]["A_seconds"], saved["A_seconds"])
            self.assertEqual(result["timing"]["B_seconds"], saved["B_seconds"])
            self.assertTrue(result["timing"]["recovered_A"])
            self.assertTrue(result["timing"]["recovered_B"])
            self.assertEqual([x["power_w"] for x in result["beams"]], [x["power_w"] for x in direct["beams"]])
            final_journal = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(final_journal["phase"], "complete")
            self.assertEqual(final_journal["acceptance"], acceptance)


class ValidationProofTests(unittest.TestCase):
    def test_sha_bound_validation_only_and_invalid_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            scientific = "2fc16884078ad64b97333013737fef5cd9294626"
            job = {"run_id": "one", "git_commit": scientific, "execution_source": {"execution_commit": "repair"}}
            def write(name, value):
                path = directory/name
                path.write_text(json.dumps(value), encoding="utf-8")
                return {"path": str(path), "sha256": digest(path)}
            config = write("config.json", {"contract": "unchanged"})
            checkpoint = write("checkpoint.pt", {"opaque": "checkpoint bytes"})
            clock = {"environment_seconds": 10., "sampling_seconds": 20., "update_seconds": 30.,
                     "validation_seconds": 0., "io_seconds": 4.}
            status = write("origin_status.json", {"run_id": "one", "stage": "failed", "phase": "validation_100000",
                            "env_step": 100000, "checkpoint_sha256": checkpoint["sha256"],
                            "clock": {**clock, "io_seconds": 7.}})
            failure = write("origin_failure.json", {"run_id": "one", "config_sha256": config["sha256"],
                              "traceback": f"{scientific} in validate_node in _evaluate in full_b_case"})
            origin_job = write("origin_job.json", {"run_id": "one", "git_commit": scientific})
            launch = write("origin_launch.json", {"source": scientific})
            logs = {}
            for name in ("traces.jsonl", "episodes.jsonl", "updates.jsonl"):
                record = write(name, {"committed": 100000})
                logs[name] = {**record, "saved_size": Path(record["path"]).stat().st_size}
            sizes = {key: row["saved_size"] for key, row in logs.items()}
            saved = {"step": 100000, "agent": {"interactions": 100000}, "node_done": [], "run_id": "one",
                     "log_sizes": sizes, "config_sha256": config["sha256"], "clock": clock}
            proof = {"version": "fair20_validation_only_resume_v1", "run_id": "one", "scientific_commit": scientific,
                     "execution_commit": "repair", "checkpoint_step": 100000, "agent_interactions": 100000,
                     "failed_phase": "validation_100000", "checkpoint_sha256": checkpoint["sha256"],
                     "failure_sha256": failure["sha256"], "failed_status_sha256": status["sha256"],
                     "training_log_sizes": sizes, "training_logs": logs,
                     "origin_records": {"job": origin_job, "status": status, "config": config,
                                        "launch": launch, "failure": failure, "checkpoint": checkpoint}}
            reference = write("proof.json", proof)
            job["validation_only_resume_proof"] = {"proof_path": reference["path"], "proof_sha256": reference["sha256"]}
            self.assertTrue(validation_only_resume_verified(job, directory, saved, {}))
            restored_clock, timing = validation_only_recovered_clock(job, saved)
            self.assertEqual(restored_clock["io_seconds"], 7.)
            self.assertEqual(timing["post_checkpoint_measured_io_seconds_recovered"], 3.)
            self.assertIsNone(timing["historical_validation_wall_seconds"])
            self.assertIn("unavailable", timing["historical_validation_wall_missing_reason"])
            for key in ("environment_seconds", "sampling_seconds", "update_seconds", "validation_seconds"):
                self.assertEqual(restored_clock[key], clock[key])
            self.assertEqual(saved["clock"]["io_seconds"], 4.)
            bad_clock = deepcopy(saved)
            bad_clock["clock"]["update_seconds"] += 1
            self.assertFalse(validation_only_resume_verified(job, directory, bad_clock, {}))
            bad_io = deepcopy(saved)
            bad_io["clock"]["io_seconds"] = 8.
            self.assertFalse(validation_only_resume_verified(job, directory, bad_io, {}))
            for key, value in (("step", 100001), ("node_done", [100000]), ("agent", {"interactions": 99999})):
                invalid = deepcopy(saved)
                invalid[key] = value
                self.assertFalse(validation_only_resume_verified(job, directory, invalid, {}))
            self.assertFalse(validation_only_resume_verified(job, directory, saved, {"traces.jsonl": {}}))
            bad = deepcopy(job)
            bad["execution_source"]["execution_commit"] = "other"
            self.assertFalse(validation_only_resume_verified(bad, directory, saved, {}))
            (directory/"updates.jsonl").write_text("new uncommitted tail", encoding="utf-8")
            self.assertFalse(validation_only_resume_verified(job, directory, saved, {}))


if __name__ == "__main__":
    unittest.main()
