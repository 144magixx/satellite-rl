"""Small archived evidence fixtures for S2-P19 cost accounting."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.s2p19 import cost_audit


def _package(root, run_id, files):
    root = Path(root)
    archive = root / "part_0001.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for name, payload in files.items():
            output.writestr(name, payload)
    receipt = root / "receipt.json"
    receipt.write_text(json.dumps({
        "schema": "s2p19.archive.v1", "experiment_id": "S2-P19",
        "run_id": run_id, "job_stage": "completed", "port": 20474,
        "inventory_sha256": "a" * 64, "member_count": len(files),
        "members": [{"member": name, "bytes": len(payload),
                     "sha256": hashlib.sha256(payload).hexdigest()}
                    for name, payload in files.items()],
        "volumes": [{"name": archive.name, "bytes": archive.stat().st_size,
                     "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}]}),
        encoding="utf-8")
    return receipt


def _json_bytes(value):
    return json.dumps(value, separators=(",", ":")).encode()


class S2P19CostAuditTests(unittest.TestCase):
    def test_direct_case_uses_saved_trace_and_enforces_both_32_limits(self):
        case = {"method": "model", "scene_sha256": "b" * 64,
                "scene_metadata": {"mother_scene_id": "mother_0"},
                "repeat_index": 0, "beams": [{}] * 50,
                "metrics": {"decisions": 2, "probes": 1, "invalid": 0,
                            "repeats": 1, "proposals_generated": 7},
                "trace": [{"reason": "candidate", "decisions": 1, "probes": 1,
                           "actual_action": {"rejection_reason": None}},
                          {"reason": "repeat", "decisions": 2, "probes": 1,
                           "actual_action": {"rejection_reason": None},
                           "protected_output": True}]}
        name = "job/cases/scene_00000_r0.json"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipt = _package(root, "run_i0", {name: _json_bytes(case)})
            spec = {"run_id": "run_i0", "port": 20474}
            row = cost_audit._case_row("I0_final_direct_20474", spec, receipt, name)
            self.assertEqual((row["decisions"], row["probes"], row["repeats"]),
                             (2, 1, 1))
            self.assertIsNone(row["fallback_used"])
            case["metrics"]["decisions"] = 33
            receipt = _package(root, "run_i0", {name: _json_bytes(case)})
            from scripts.s2p19.archive_sources import receipt_members, _zip_digest
            receipt_members.cache_clear()
            _zip_digest.cache_clear()
            with self.assertRaisesRegex(ValueError, "decisions"):
                cost_audit._case_row("I0_final_direct_20474", spec, receipt, name)

    def test_i0_unique_episode_account_excludes_resume_snapshot(self):
        complete = lambda episode, decisions, probes, beams: {
            "episode": episode, "env_step": decisions,
            "beam_count": beams, "metrics": {"decisions": decisions,
                                                "probes": probes},
            "terminal_snapshot": {"complete_episode": True}}
        one, two = complete(0, 2, 1, 50), complete(1, 1, 0, 60)
        episodes = b"\n".join(map(_json_bytes, (one, two, one))) + b"\n"
        files = {"job/job.json": _json_bytes({"run_id": "run_i0"}),
                 "job/config.json": _json_bytes({"i0": {"max_decisions": 32,
                                                          "max_probes": 32,
                                                          "total_steps": 400000,
                                                          "max_episodes": 12500}}),
                 "job/status.json": _json_bytes({"run_id": "run_i0",
                                                  "stage": "completed",
                                                  "env_step": 397112,
                                                  "budget_env_steps": 400000,
                                                  "updates": 395113,
                                                  "completed_due_to": "episode_cap",
                                                  "episodes": 12500}),
                 "source/source_manifest.json": _json_bytes({"files": {
                     "src/satellite_rl/sac_improvement/environment.py": "c" * 64}}),
                 "job/episodes.jsonl": episodes,
                 "job/episode_snapshots.jsonl": _json_bytes({"episode": 1,
                                                               "metrics": {"decisions": 1}}) + b"\n"}
        with tempfile.TemporaryDirectory() as temp:
            receipt = _package(temp, "run_i0", files)
            spec = {"run_id": "run_i0", "episodes": {
                "sha256": hashlib.sha256(episodes).hexdigest()}}
            result, _, _ = cost_audit._training_rows("I0", spec, receipt)
            self.assertEqual(result["completed_unique_episodes"], 2)
            self.assertEqual(result["duplicate_identical_episode_rows"], 1)
            self.assertEqual(result["resume_snapshot_rows_not_added_to_cost"], 1)
            self.assertEqual(result["macro_decisions_in_unique_episode_rows"], 3)
            self.assertEqual(result["candidate_truth_probes_in_unique_episode_rows_lower_bound"], 1)
            self.assertEqual(result["greedy_initializer_beam_decisions_in_unique_episode_rows_lower_bound"], 110)
            self.assertIsNone(result["unlogged_recovery_repeat_cost"])
            self.assertFalse(result["macro_decisions_match_status"])

    def test_constructive_endpoint_uses_status_and_nested_model_provenance(self):
        episodes = _json_bytes({"episode": 0, "env_step": 400000,
                                "length": 400000, "terminal": True}) + b"\n"
        files = {"job/job.json": _json_bytes({"run_id": "run_c0"}),
                 "job/config.json": _json_bytes({"training": {"total_steps": 400000}}),
                 "job/status.json": _json_bytes({"run_id": "run_c0",
                                                  "stage": "completed",
                                                  "env_step": 400000,
                                                  "budget_env_steps": 400000,
                                                  "update_step": 398001}),
                 "source/source_manifest.json": _json_bytes({"files": {}}),
                 "job/episodes.jsonl": episodes,
                 "weights/model_dir/final/model.json": _json_bytes({
                     "provenance": {"run_id": "run_c0", "env_step": 400000,
                                    "selection": "fixed_endpoint"},
                     "parameter_count": 123})}
        with tempfile.TemporaryDirectory() as temp:
            receipt = _package(temp, "run_c0", files)
            spec = {"run_id": "run_c0", "episodes": {
                "sha256": hashlib.sha256(episodes).hexdigest()}}
            result, _, _ = cost_audit._training_rows("C0", spec, receipt)
            self.assertEqual(result["beam_decisions"], 400000)
            self.assertEqual(result["recorded_update_step"], 398001)
            self.assertTrue(result["beam_decisions_match_status"])
            self.assertEqual(result["parameter_count"], 123)
            self.assertEqual(result["model_selection"], "fixed_endpoint")

    def test_default_check_reports_missing_training_and_teacher_without_reading_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index = root / "index.json"
            manifest = root / "manifest.json"
            plan = root / "plan.json"
            state = root / "state.json"
            index.write_text(json.dumps({"schema": "s2p19.source_index.v1",
                                         "jobs": {}, "training": {}}))
            manifest.write_text(json.dumps({"experiment_id": "S2-P19",
                                            "tasks": {}}))
            plan.write_text(json.dumps({"experiment_id": "S2-P19",
                                        "tasks": [], "completed_legacy": {}}))
            state.write_text(json.dumps({"plan_sha256": cost_audit.digest(plan),
                                         "tasks": {}}))
            result = cost_audit.check_inputs(index, manifest, plan, state)
            self.assertFalse(result["ready"])
            self.assertIn("C0: source_index.training missing", result["missing"])
            self.assertIn("C1 teacher: --teacher-receipt absent", result["missing"])

    def test_teacher_count_requires_c1_job_config_and_archived_manifest_sha(self):
        records = [{"transition_count": 119 if i < 320 else 118}
                   for i in range(576)]
        manifest = _json_bytes({"format": "s2p19_greedy_trajectory_v1",
                                "source_set_sha256": "b" * 64,
                                "records": records})
        teacher_job = _json_bytes({"run_id": "teacher_run"})
        files = {"job/job.json": teacher_job,
                 "job/status.json": _json_bytes({"stage": "completed"}),
                 "job/teacher/manifest.json": manifest}
        with tempfile.TemporaryDirectory() as temp:
            receipt = _package(temp, "teacher_run", files)
            source = {"run_id": "teacher_run",
                      "job_sha256": hashlib.sha256(teacher_job).hexdigest(),
                      "manifest_sha256": hashlib.sha256(manifest).hexdigest()}
            config = {"training": {"demonstrations": {
                "manifest_sha256": source["manifest_sha256"],
                "source_set_sha256": "b" * 64}}}
            result = cost_audit._teacher(receipt,
                                          {"demonstration_source": source}, config)
            self.assertEqual(result["transitions_from_frozen_teacher_manifest"],
                             68288)
            bad = {"demonstration_source": {**source,
                                            "manifest_sha256": "0" * 64}}
            with self.assertRaisesRegex(ValueError, "teacher"):
                cost_audit._teacher(receipt, bad, config)


if __name__ == "__main__":
    unittest.main()
