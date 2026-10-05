"""Narrow S2-P17 repair provenance and missing-time reporting checks."""

import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.variable_beams import report_results
from scripts.variable_beams.report_results import (
    AuditError, LOADS, N_VALUES, REPEATS, ROLES, _check_timing, _timing_curve_values,
    load_interruption_ledger, load_manifest, load_refinements, summarize_timing,
)


class RepairReportTests(unittest.TestCase):
    def test_interruption_ledger_binds_pause_snapshot_and_successful_cases(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = root / "controller_manifest.json"
            snapshot = root / "user_resume_20261004T185523Z"
            snapshot.mkdir()
            run_id = report_results.INTERRUPTED_RUN
            request_id = report_results.INTERRUPTED_REQUEST
            scene_sha = "s" * 64
            initial_sha = "i" * 64
            allocation_sha = "a" * 64
            source = {"case_sha256": initial_sha, "beams_sha256": allocation_sha}
            case_dir = root / "power_cases"
            case_dir.mkdir()
            case_rows = {
                "none": {"source_scene_sha256": scene_sha,
                         "source_initial_case_sha256": initial_sha,
                         "initial_allocation_sha256": allocation_sha},
                "sca": {"method": "sca"},
                "sca_si_lp": {"method": "sca_si_lp"},
            }
            cases = {}
            for method, row in case_rows.items():
                file = case_dir / f"scene_00146_r0_{method}.json"
                file.write_text(json.dumps(row), encoding="utf-8")
                cases[method] = (file, hashlib.sha256(file.read_bytes()).hexdigest())
            identity = "identity-146"
            last_name = "scene_00146_r0_none.json"
            prefix = {f"old-{index:04d}.json": "p" * 64 for index in range(1314)}
            prefix[last_name] = cases["none"][1]
            scene_files = [[f"scene-{index}", "other"] for index in range(440)]
            scene_files[146][1] = scene_sha
            originals = {
                "V1_job.json": {"run_id": run_id, "request_id": request_id,
                                "git_commit": "c" * 40,
                                "scientific_origin_commit": "o" * 40},
                "V1_status.json": {"run_id": run_id, "stage": "stopped",
                                   "env_step": 1315, "scene_index": 146,
                                   "repeat": 0, "method": "sca",
                                   "optimizer_progress": {"phase": "si", "evaluation": 571,
                                                          "elapsed_seconds": 1.4651551842689514}},
                "V1_refinement_checkpoint.json": {"identity": identity,
                                                  "completed_cases": 1315,
                                                  "last_case": str(cases["none"][0]),
                                                  "last_case_sha256": cases["none"][1]},
                "V1_refinement_manifest.json": {"identity": identity,
                                                "budget_cases": 3960,
                                                "scene_files": scene_files},
                "V1_prefix_sha256.json": prefix,
                "user_pause.json": {"reason": "User requested pause before restart",
                                    "stop_requests": [{"run_id": run_id,
                                                       "before_stage": "evaluating"}]},
            }
            for name, value in originals.items():
                (snapshot / name).write_text(json.dumps(value), encoding="utf-8")
            log = snapshot / "V1_power_iterations.jsonl"
            log.write_text("".join(json.dumps({"scene_id": "scene_0000146_r0",
                "repeat": 0, "method": "sca", "case_identity": "case-146",
                "phase": "si",
                "evaluation": 571 if index == 69 else index + 1,
                "elapsed_seconds": 1.4651551842689514 if index == 69 else 0.01 * index}) + "\n"
                for index in range(70)), encoding="utf-8")
            hashes = {name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
                      for name in report_results.INTERRUPTION_FILES}
            ledger = {"schema_version": report_results.INTERRUPTION_SCHEMA,
                      "experiment_id": "S2-P17", "event": "user_requested_pause",
                      "run_id": run_id, "request_id": request_id,
                      "role": "V1", "scene_index": 146, "repeat": 0,
                      "method": "sca", "scene_id": "scene_0000146_r0",
                      "source_scene_sha256": scene_sha,
                      "source_initial_case_sha256": initial_sha,
                      "initial_allocation_sha256": allocation_sha,
                      "origin_science_commit": "o" * 40,
                      "repair_commit": "c" * 40,
                      "snapshot_dir": str(snapshot), "files_sha256": hashes,
                      "resumed_case_sha256": {key: value[1] for key, value in cases.items()},
                      "pre_pause_last_none_sha256": cases["none"][1],
                      "interrupted_case_identity": "case-146",
                      "callback_lower_bounds": {"capacity_calls": 571,
                                                "elapsed_seconds": 1.4651551842689514,
                                                "accepted_callbacks": 70},
                      "missing_reason": "Original paused A exact total was not persisted"}
            ledger_path = root / "interruption_ledger.json"
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            stages = {"none": {"case_path": str(cases["none"][0]),
                               "case_sha256": cases["none"][1],
                               "failed_attempt_fallback": False},
                      "sca": {"case_sha256": cases["sca"][1],
                              "beams_sha256": "b" * 64,
                              "solver_evaluations": 900,
                              "failed_attempt_fallback": False},
                      "sca_si_lp": {"case_sha256": cases["sca_si_lp"][1],
                                    "reference_case_sha256": cases["sca"][1],
                                    "reference_allocation_sha256": "b" * 64,
                                    "solver_evaluations": 800,
                                    "complete_route_solver_evaluations": 1700,
                                    "failed_attempt_fallback": False}}
            refs = [{"sha256": "other"} for _ in range(440)]
            refs[146] = {"sha256": scene_sha}
            manifest = {"scientific_commit": "o" * 40,
                        "boundary_repair": {"repair_commit": "c" * 40}}
            jobs = {"V1_sca": {"directory": root, "job": {"run_id": run_id,
                                        "request_id": request_id,
                                        "git_commit": "c" * 40}}}
            inputs = {}
            result = load_interruption_ledger(ledger_path, controller, manifest,
                                              jobs, refs, {("V1", scene_sha, 0): source},
                                              {("V1", scene_sha, 0): stages}, inputs)
            self.assertEqual(result["original_A_capacity_calls_lower_bound"], 571)
            self.assertEqual(result["cumulative_A_capacity_calls_lower_bound"], 1471)
            self.assertIsNone(result["original_A_optimizer_seconds_exact"])
            self.assertIn(str(log.resolve()), inputs)
            ledger["files_sha256"]["V1_power_iterations.jsonl"] = "0" * 64
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            with self.assertRaisesRegex(AuditError, "原始证据SHA"):
                load_interruption_ledger(ledger_path, controller, manifest, jobs,
                                         refs, {("V1", scene_sha, 0): source},
                                         {("V1", scene_sha, 0): stages}, {})

    def test_refinement_manifest_does_not_replace_controller_science_commit(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"scientific_commit": "a" * 40}
            jobs = {"V0_sca": {"directory": root, "job": {}}}
            first_ref = {"sha256": "s" * 64}
            refinement_manifest = {"budget_cases": 3960,
                                   "scene_files": [["frozen-scene", first_ref["sha256"]]]}
            class FirstCaseReached(Exception):
                pass
            with patch.object(report_results, "json_file", return_value=refinement_manifest), \
                 patch.object(report_results, "sha256_file", return_value="d" * 64), \
                 patch.object(report_results, "_read_refinement_case",
                              side_effect=FirstCaseReached) as reader:
                with self.assertRaises(FirstCaseReached):
                    load_refinements(jobs, [first_ref],
                                     {("V0", first_ref["sha256"], 0): {}}, {}, manifest)
            self.assertEqual(reader.call_args.kwargs["expected_commit"], "a" * 40)

    def test_repair_profile_cannot_change_scientific_origin(self):
        with TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "run_manifest.json"
            names = ("V0_train", "V1_train", "V0_eval", "V1_eval", "H0_eval",
                     "V0_sca", "V1_sca", "H0_sca", "fixed_sca", "greedy_sca")
            manifest.write_text(json.dumps({
                "experiment_id": "S2-P17", "scientific_commit": "a" * 40,
                "target": {"ssh_port": 20424},
                "datasets": {name: {} for name in ("train", "validation", "test")},
                "jobs": {name: {} for name in names},
                "boundary_repair": {"repair_id": "sca_basic_threshold_exact_v1",
                                    "origin_science_commit": "b" * 40,
                                    "repair_commit": "c" * 40,
                                    "failed_run_id": "refine_sac_20261004_215930_e814197c"},
            }), encoding="utf-8")
            with self.assertRaisesRegex(AuditError, "来源身份"):
                load_manifest(manifest, {})

    def test_only_fallback_may_omit_exact_timing(self):
        self.assertIsNone(_check_timing({}, "sca_si_lp",
                                         {"initial_allocation_seconds": 1.0}, fallback=True))
        with self.assertRaisesRegex(AuditError, "不能补造"):
            _check_timing({"timing": {"total_pipeline_seconds": 0.0}},
                          "sca_si_lp", {"initial_allocation_seconds": 1.0}, fallback=True)
        with self.assertRaisesRegex(AuditError, "缺少实测"):
            _check_timing({}, "sca_si_lp", {"initial_allocation_seconds": 1.0})

    def test_missing_one_action_propagates_to_three_action_scene_time(self):
        refs, direct, refined = [], {}, {}
        for n in N_VALUES:
            for mother in range(20):
                for load in LOADS:
                    ref = {"sha256": f"scene-{n}-{mother}-{load}",
                           "beam_count": n, "mother_scene_id": f"mother-{mother}",
                           "load_multiplier": load}
                    refs.append(ref)
                    for role in ROLES:
                        for repeat in range(REPEATS[role]):
                            key = role, ref["sha256"], repeat
                            direct[key] = {"initial_allocation_seconds": 1.0}
                            full = {"initialization_seconds": 1.0,
                                    "physics_setup_seconds": 0.1,
                                    "reference_optimizer_wall_seconds": 0.2,
                                    "reference_verification_seconds": 0.1,
                                    "optimizer_wall_seconds": 0.4,
                                    "verification_seconds": 0.2,
                                    "total_pipeline_seconds": 2.0}
                            fallback = (role == "V0" and n == N_VALUES[0] and
                                        mother == 0 and load == LOADS[0] and repeat == 0)
                            if (role == "V0" and n == N_VALUES[0] and mother == 0 and
                                    load == LOADS[0] and repeat > 0):
                                # These two measured actions must not stand in for the missing third.
                                full["total_pipeline_seconds"] = 200.0
                            refined[key] = {
                                "sca": {"timing": {"optimizer_wall_seconds": 0.2,
                                                   "verification_seconds": 0.1,
                                                   "physics_setup_seconds": 0.1,
                                                   "total_pipeline_seconds": 1.4}},
                                "sca_si_lp": {"timing": None if fallback else full,
                                              "failed_attempt_fallback": fallback,
                                              "repair_verification_seconds": 0.01 if fallback else None,
                                              "source_commit": "repair" if fallback else "original"}}
        paused_scene = next(ref for ref in refs if ref["beam_count"] == N_VALUES[1])
        interruption = {"scene_sha256": paused_scene["sha256"], "repeat_index": 0,
                        "original_A_optimizer_seconds_lower_bound": 1.4651551842689514,
                        "original_A_capacity_calls_lower_bound": 571,
                        "cumulative_A_capacity_calls_lower_bound": 900 + 571,
                        "cumulative_complete_route_calls_lower_bound": 1700 + 571,
                        "missing_reason": "user pause before A persisted",
                        "ledger_sha256": "l" * 64}
        raw, summary = summarize_timing(refs, direct, refined, [interruption])
        affected = next(row for row in summary if row["role"] == "V0" and
                        row["beam_count"] == N_VALUES[0])
        self.assertEqual((affected["total_pipeline_seconds_complete_count"],
                          affected["total_pipeline_seconds_missing_count"]), (119, 1))
        self.assertEqual((affected["total_pipeline_seconds_scene_sum_complete_count"],
                          affected["total_pipeline_seconds_scene_sum_missing_count"]), (39, 1))
        self.assertGreater(affected["total_pipeline_seconds_mean"], 2.0)
        self.assertEqual(affected["total_pipeline_seconds_scene_sum_mean"], 6.0)
        self.assertEqual(affected["total_pipeline_seconds_route_scene_mean_mean"], 2.0)
        self.assertEqual(_timing_curve_values(summary, "V0")[0], 2.0)
        self.assertEqual(affected["total_pipeline_lower_bound_count"], 1)
        failed = next(row for row in raw if row["role"] == "V0" and
                      row["scene_sha256"] == refs[0]["sha256"] and row["repeat_index"] == 0)
        self.assertIsNone(failed["total_pipeline_seconds"])
        self.assertIsNone(failed["optimizer_wall_seconds"])
        self.assertGreater(failed["total_pipeline_seconds_lower_bound"], 2.4)
        self.assertEqual(failed["repair_verification_seconds"], 0.01)
        v1 = next(row for row in summary if row["role"] == "V1" and
                  row["beam_count"] == N_VALUES[1])
        self.assertEqual((v1["total_pipeline_seconds_complete_count"],
                          v1["total_pipeline_seconds_scene_sum_complete_count"]), (119, 39))
        paused = next(row for row in raw if row["role"] == "V1" and
                      row["scene_sha256"] == paused_scene["sha256"] and
                      row["repeat_index"] == 0)
        self.assertIsNone(paused["reference_optimizer_wall_seconds"])
        self.assertIsNone(paused["total_pipeline_seconds"])
        self.assertEqual(paused["successful_attempt_A_optimizer_seconds"], 0.2)
        self.assertEqual(paused["successful_attempt_total_pipeline_seconds"], 2.0)
        self.assertEqual(paused["cumulative_A_calls_lower_bound"], 1471)
        self.assertEqual(paused["cumulative_complete_route_calls_lower_bound"], 2271)
        self.assertGreater(paused["total_pipeline_seconds_lower_bound"], 3.46)


if __name__ == "__main__":
    unittest.main()
