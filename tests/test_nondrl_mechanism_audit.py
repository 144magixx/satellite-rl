"""Audits distinguish missing evidence from measured failure on fixed toys."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from satellite_rl.coverage.nondrl.mechanism_audit import (audit_direct_construction,
        audit_region_oracle, audit_scene_traces, audit_teacher_reachability)
from satellite_rl.coverage.nondrl.teachers import EvidenceWriter, digest, generate_teacher
from test_nondrl_learning import fixture


def source(folder):
    path = Path(folder) / "source.json"
    path.write_text(json.dumps({"format": "unit_arithmetic_source"}), encoding="utf-8")
    return [{"manifest_path": str(path), "manifest_sha256": digest(path), "source_commit": "unit_check"}]


def result(cv=.1, before=.2):
    metrics = {"load_cv_all": cv, "geometry_feasible": True,
               "unique_assignment": True, "demand_conserved": True, "beam_count": 2}
    initial = copy.deepcopy(metrics)
    initial["load_cv_all"] = before
    return {"metrics_all": metrics, "initial_metrics_all": initial, "initial_sha256": "same-I0",
            "physical": {"aircraft_service": {"demand_weighted_carried_fraction": 1.}},
            "initial_physical": {"aircraft_service": {"demand_weighted_carried_fraction": 1.}},
            "initial_layout": {"centers": [[45., 10.], [45.4, 10.5]], "assignment": [0, 0, 0, 1]},
            "trace": []}


def rows(route):
    values = []
    for index in range(20):
        for multiplier in (.75, 1., 1.25):
            item = result()
            if route == "06":
                item["trace"] = [{"source": "06_alternating_centroid_assignment", "accepted": True,
                                  "metrics_all": copy.deepcopy(item["metrics_all"])}]
            elif route == "07":
                item["trace"] = [{"source": "07_regroup_then_exact_footprint_projection",
                                  "metrics_all": copy.deepcopy(item["metrics_all"]),
                                  "layout": {"centers": item["initial_layout"]["centers"], "assignment": [0, 0, 1, 1]}}]
            else:
                item["trace"] = [{"source": "08_transport", "model": "row_conserving_entropy_soft_quota_v1",
                                  "row_residual": 1e-12},
                                 {"source": "08_OT_integerized_geometric_candidate", "accepted": True,
                                  "metrics_all": copy.deepcopy(item["metrics_all"])}]
            values.append({"dataset_index": index, "mother_scene_id": f"mother{index}",
                           "demand_multiplier": multiplier, "result": item})
    return values


def region_sample(name, multiplier, unknown=False):
    before = {"centers": [[45., 10.], [45.1, 10.1], [45.2, 10.2]],
              "assignment": [0, 0, 0, 1, 1, 2]}
    after = copy.deepcopy(before)
    after["assignment"] = [0, 0, 2, 1, 1, 2]
    cv = float(np.std([3., 2., 1.]) / 2.)
    objects = [{"id": "r0-2", "beams": [0, 2], "features": [0.] * 8},
               {"id": "r1-2", "beams": [1, 2], "features": [0.] * 8}]
    accepted = {"accepted": True, "layout": after, "metrics_all": {"load_cv_all": 0.},
                "physical": {"aircraft_service": {"demand_weighted_carried_fraction": 1.}}}
    rejected = {"accepted": False, "reason": "no_exact_CV_improvement", "layout": before,
                "metrics_all": {"load_cv_all": cv}}
    features = [[0., 0., 0., 0., 0., 1 / 6., 0., 1.] for _ in range(6)]
    return {"mother_id": name, "demand_multiplier": multiplier, "source_hash": "unit_source",
            "aircraft": features, "state": before, "scene_demands_bps": [multiplier] * 6,
            "scene_positions": [[45., 10.]] * 6, "initial_carry": 1.,
            "initial_metrics": {"load_cv_all": cv}, "initial_layout_sha256": "unit-I0",
            "objects": objects, "actions": [[0.] * 16] * 2,
            "known": [True, not unknown], "utility": [1., 0.],
            "trials": [{"object": objects[0], "label": {"known": True}, "result": accepted},
                       {"object": objects[1], "label": {"known": not unknown},
                        "result": rejected if not unknown else {"reason": "rebuild_no_incumbent_unknown"}}]}


class MechanismAuditTests(unittest.TestCase):
    def test_06_07_trace_gates_have_real_denominators(self):
        with tempfile.TemporaryDirectory() as temporary:
            sources = source(temporary)
            for route in ("06", "07"):
                report = audit_scene_traces(rows(route), route, sources)
                self.assertEqual(report["status"], "passed")
                self.assertEqual(report["statistics"]["actual_rows"], 60)
                self.assertEqual(len(report["statistics"]["improved_mothers"]), 20)
            incomplete = rows("06")[:-1]
            report = audit_scene_traces(incomplete, "06", sources)
            self.assertTrue(report["attention_required"])
            self.assertFalse(report["scientific_stop"])

    def test_measured_trace_failure_is_scientific_and_unknown_source_is_attention(self):
        with tempfile.TemporaryDirectory() as temporary:
            values = rows("06")
            for row in values:
                row["result"]["metrics_all"]["load_cv_all"] = .2
                row["result"]["trace"] = []
            report = audit_scene_traces(values, "06", source(temporary))
            self.assertEqual(report["status"], "science_stopped")
            self.assertFalse(report["attention_required"])
            broken = audit_scene_traces(values, "06", [{"path": str(Path(temporary) / "missing"), "sha256": "wrong"}])
            self.assertTrue(broken["attention_required"])

    def test_08_requires_both_controls_and_net_three_mothers(self):
        with tempfile.TemporaryDirectory() as temporary:
            values, controls = rows("08"), {"voronoi": rows("08"), "integer": rows("08")}
            for group in controls.values():
                for row in group:
                    row["result"]["metrics_all"]["load_cv_all"] = .2
            report = audit_scene_traces(values, "08", source(temporary), controls=controls)
            self.assertTrue(report["passed"])
            self.assertEqual(report["statistics"]["controls"]["integer"]["net_improved_mother_advantage"], 20)
            report = audit_scene_traces(values, "08", source(temporary))
            self.assertTrue(report["attention_required"])
            controls["integer"] = rows("08")
            report = audit_scene_traces(values, "08", source(temporary), controls=controls)
            self.assertTrue(report["scientific_stop"])

    def test_region_oracle_equal_mother_weight_and_unknown_not_negative(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = EvidenceWriter(temporary, "05")
            for index in range(20):
                for multiplier in (.75, 1., 1.25):
                    writer.append(region_sample(f"mother{index}", multiplier))
            writer.manifest.update(status="completed", trials=120)
            writer.flush()
            path = Path(temporary) / "manifest.json"
            report = audit_region_oracle(path, digest(path), mother_indices={f"mother{i}": i for i in range(20)})
            self.assertTrue(report["passed"], report["reason"])
            self.assertEqual(report["statistics"]["actual_mother_tiers"], 60)
            self.assertAlmostEqual(report["statistics"]["oracle_uniform_advantage"], .5)
        with tempfile.TemporaryDirectory() as temporary:
            writer = EvidenceWriter(temporary, "05")
            writer.append(region_sample("mother0", 1., unknown=True))
            writer.manifest.update(status="completed", trials=2)
            writer.flush()
            report = audit_region_oracle(Path(temporary) / "manifest.json", expected_mother_count=1,
                                          multipliers=(1.,), mother_indices={"mother0": 0})
            self.assertTrue(report["attention_required"])
            self.assertFalse(report["scientific_stop"])

    def test_raw_constructor_failure_not_hidden_by_feasible_fallback(self):
        problem = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            item = result(cv=0., before=.5)
            item["initial_layout"] = problem.initial.to_record()
            item["initial_sha256"] = problem.initial_sha256
            item["raw_layout"] = problem.initial.to_record()
            item["raw_layout"]["assignment"] = [0, 0, 0, 0]
            row = {"dataset_index": 0, "mother_scene_id": problem.scene.scene_id,
                   "demand_multiplier": 1., "result": item, "HR_result": result(cv=.5, before=.5)}
            row["HR_result"]["initial_sha256"] = problem.initial_sha256
            report = audit_direct_construction([row], [problem.scene], sources=source(temporary),
                                                expected_mother_count=1, multipliers=(1.,))
            self.assertTrue(report["scientific_stop"])
            self.assertEqual(report["statistics"]["raw_feasible_fraction"], 0.)

    def test_true_teacher_receipt_recomputes_geometry_cv_and_carry(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for route in ("02", "03"):
                with self.subTest(route=route), tempfile.TemporaryDirectory() as temporary:
                    result = generate_teacher([fixture()], route, temporary,
                                              limits={"max_trials": 128, "max_steps": 4, "max_seconds": 4.})
                    self.assertEqual(len(result["problem_receipts"]), 1)
                    report = audit_teacher_reachability(result["path"], result["sha256"], route,
                                mother_indices={"mother": 0}, expected_mother_count=1, multipliers=(1.,))
                    # Actual one-mother toy can verify reference closure, but
                    # cannot satisfy the six-independent-mother science gate.
                    self.assertEqual(report["status"], "science_stopped", report["reason"])
                    self.assertEqual(report["statistics"]["legal_reference_mothers"], [0])
        finally:
            torch.set_num_threads(previous)


if __name__ == "__main__":
    unittest.main()
