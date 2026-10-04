"""Narrow S2-P17 repair provenance and missing-time reporting checks."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.variable_beams.report_results import (
    AuditError, LOADS, N_VALUES, REPEATS, ROLES, _check_timing, _timing_curve_values,
    load_manifest, summarize_timing,
)


class RepairReportTests(unittest.TestCase):
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
        raw, summary = summarize_timing(refs, direct, refined)
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


if __name__ == "__main__":
    unittest.main()
