"""Tamper detection and deterministic selection checks for archived S1-02 data."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def module(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


audit = module("audit_stage1_balance")
selection = module("select_stage1_menus")


def equal_scene():
    # 100 uniquely assigned aircraft, each 10 Mbps, within all 100 beam footprints.
    # This is a deterministic ledger/geometry fixture, never an experiment result.
    position = [45.0, 13.0]
    metrics = {"coverage_demand": 1.0, "coverage_users": 1.0, "uncovered_demand_bps": 0.0,
               "uncovered_users": 0, "load_cv": 0.0, "load_std_bps": 0.0, "max_load_bps": 1e7,
               "empty_beams": 0, "active_beams": 100, "subject_cv": 0.0,
               "subject_std_bps": 0.0, "subject_mean_bps": 1e7, "subject_beams": 100,
               "subject_in_band_beams": 100, "subject_in_band_fraction_all_100": 1.0,
               "subject_max_mean_ratio": 1.0, "edge_beam_count": 0, "edge_beams": [],
               "certified_edge_region_count": 0, "edge_limit_met": True, "edge_demand_fraction": 0.0,
               "coverage_100": True, "coverage_99_9": True, "feasible": True,
               "balanced_target_met": True, "movement_mean_deg": 0.0, "movement_max_deg": 0.0}
    return {"scene_id": "deterministic_equal_demand_fixture", "status": "completed",
            "input": {"aircraft_ids": [f"u{i}" for i in range(100)],
                      "positions": [position[:] for _ in range(100)], "demands_bps": [1e7]*100,
                      "initial_centers": [position[:] for _ in range(100)]},
            "final_positions": [position[:] for _ in range(100)], "final_associations": list(range(100)),
            "beam_loads_bps": [1e7]*100, "metrics": metrics, "total_pipeline_seconds": 3.0,
            "balance_diagnostics": {"certified_regions": [], "qualified_edge_beam_ids": [],
                                    "stage_seconds": {"c1": 1.0, "verification": 2.0},
                                    "total_pipeline_seconds": 3.0}}


class AuditChecks(unittest.TestCase):
    def test_complete_ledger_passes_and_demand_tamper_fails(self):
        row = equal_scene()
        result = audit.audit_row(row)
        self.assertTrue(result["passed"], result["issues"])
        row["beam_loads_bps"][0] += 100
        self.assertFalse(audit.audit_row(row)["passed"])

    def test_rejects_dropping_geometrically_covered_user(self):
        row = equal_scene()
        row["final_associations"][0] = -1
        row["beam_loads_bps"][0] = 0
        result = audit.audit_row(row)
        self.assertFalse(result["passed"])
        self.assertIn("Geometric coverage differs from complete unique association", result["issues"])

    def test_rejects_invented_edge_region_and_false_low_cv(self):
        row = equal_scene()
        row["balance_diagnostics"]["certified_regions"] = [{
            "aircraft_indices": [0], "aircraft_ids": ["u0"], "witness_center": [45, 13], "demand_bps": 1e7}]
        result = audit.audit_row(row)
        self.assertFalse(result["passed"])
        self.assertTrue(any("invent" in issue or "isolated" in issue for issue in result["issues"]))

    def test_does_not_accept_adjustment_only_as_full_pipeline_time(self):
        row = equal_scene()
        row["total_pipeline_seconds"] = 2.0
        self.assertIn("Pipeline time differs from measured stage sum", audit.audit_row(row)["issues"])

    def test_rejects_reset_cumulative_movement_reference(self):
        row = equal_scene()
        row["input"]["movement_reference_centers"] = [[45.1, 13.0]]*100
        self.assertIn("Movement reference was reset after common preparation", audit.audit_row(row)["issues"])

    def test_finds_certifiable_light_isolated_component(self):
        users = audit.rays(np.array([[45., 13.], [45., 13.], [25., -10.]]))
        components = audit.expected_edge_components(users, np.array([1e9, 1e9, 1.]))
        self.assertEqual(components, [[2]])

    def test_explicit_initialization_failure_is_preserved_without_fake_layout(self):
        row = equal_scene()
        row.update(status="initialization_failed", metrics=None, final_positions=None,
                   final_associations=None, beam_loads_bps=None)
        row["input"]["initial_centers"] = None
        row["balance_diagnostics"].update(
            failure_reason="fewer than 100 distinct positive locations",
            common_start_sha256="a" * 64)
        result = audit.audit_row(row)
        self.assertTrue(result["passed"], result["issues"])
        summary_result = selection.summarize([equal_scene(), row])
        self.assertEqual(summary_result["scene_count"], 2)
        self.assertEqual(summary_result["valid_layout_scenes"], 1)
        self.assertEqual(summary_result["full_coverage_legal_scenes"], 1)


def summary(cv=.5, time=2., full=20, near=20, target=0, in_band=70):
    return {"full_coverage_legal_scenes": full, "near_coverage_legal_scenes": near,
            "balanced_target_scenes": target, "subject_cv_mean": cv,
            "subject_in_band_beams_mean": in_band, "subject_max_mean_ratio_mean": 2.,
            "full_pipeline_seconds_mean": time}


class SelectionChecks(unittest.TestCase):
    def test_coverage_precedes_low_cv(self):
        summaries = {label: summary() for label in selection.MENUS}
        summaries["X1"] = summary(.01, full=19, near=19)
        summaries["X2"] = summary(.4)
        result = selection.choose(summaries)
        self.assertEqual(result["quality_menu"], "X2")
        self.assertEqual(result["selected_menus"][0], "X0")
        self.assertEqual(len(set(result["selected_menus"])), 3)

    def test_efficiency_selection_and_deterministic_tie(self):
        summaries = {label: summary() for label in selection.MENUS}
        summaries["X1"] = summary(.3, time=3.)
        summaries["X2"] = summary(.4, time=1.)
        summaries["X3"] = summary(.4, time=1.)
        result = selection.choose(summaries)
        self.assertEqual(result["selected_menus"], ["X0", "X1", "X2"])

    def test_no_noninferior_candidate_is_recorded_without_rerun(self):
        summaries = {label: summary(.8, full=18, near=19) for label in selection.MENUS}
        summaries["X0"] = summary(.2)
        result = selection.choose(summaries)
        self.assertEqual(result["efficiency_eligible"], [])
        self.assertIn("no_efficiency", result["second_menu_reason"])


if __name__ == "__main__":
    unittest.main()
