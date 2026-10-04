"""Independent audit must reject tampered geometry, loads and denominators."""
import copy
from dataclasses import asdict
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from satellite_rl.coverage.reconstruction import ReconstructionConfig, ReconstructionState
from satellite_rl.coverage.reconstruction_runner import _evaluate_scene, _start_record
from tests.test_coverage_reconstruction import scene_fixture


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/stage1_reconstruction/audit.py"
SPEC = importlib.util.spec_from_file_location("independent_reconstruction_audit", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ReconstructionAuditTests(unittest.TestCase):
    def setUp(self):
        self.scene = scene_fixture(isolated=True)
        self.config = ReconstructionConfig(max_macro_steps=2)
        self.state = ReconstructionState.from_scene(self.scene, config=self.config)
        self.start = _start_record(self.scene, self.state, .25)
        self.row = _evaluate_scene(self.scene, self.state, self.start, "R0", self.config,
                                   policy=None, rng=np.random.default_rng(51001), device="cpu")
        self.coverage = {"reconstruction": asdict(self.config)}

    def test_independent_geometry_metrics_and_fixed_certificate(self):
        result = AUDIT.audit_row(self.scene, self.start, self.row, self.coverage)
        self.assertTrue(result["physical_metrics_defined"])
        self.assertEqual(result["quality_gain"], 0)
        self.assertEqual(result["metrics"]["coverage_users"], 1)

    def test_rejects_covered_but_unassigned_user(self):
        row = copy.deepcopy(self.row)
        row["users"][0]["assigned_beam"] = None
        with self.assertRaisesRegex(ValueError, "no association"):
            AUDIT.audit_row(self.scene, self.start, row, self.coverage)

    def test_rejects_tampered_load_metric_or_budget(self):
        for field in ("loads", "metric", "budget"):
            row = copy.deepcopy(self.row)
            if field == "loads":
                row["beam_load_demand_bps"][0] += 1
            elif field == "metric":
                row["metrics"]["subject_cv"] += .01
            else:
                row["geometry_calls"] = self.config.max_geometry_calls_total + 1
            with self.assertRaises(ValueError):
                AUDIT.audit_row(self.scene, self.start, row, self.coverage)

    def test_rejects_tampered_fixed_edge_certificate(self):
        start = copy.deepcopy(self.start)
        start["state"]["edge_beam_ids"] = []
        with self.assertRaisesRegex(ValueError, "certificate"):
            AUDIT.audit_row(self.scene, start, self.row, self.coverage)

    def test_rejects_wrong_method_and_noncompleted_task(self):
        config = {"coverage": {"reconstruction": {"route": "R0"}}}
        job = {"run_id": "job0", "kind": "evaluate", "reconstruction_route": "R0", "git_commit": "science"}
        result = {"run_id": "job0", "route": "R0", "source_commit": "science",
                  "config_hash": AUDIT.config_digest(config)}
        AUDIT.validate_method_contract(config, job, {"stage": "completed"}, result, "R0")
        for route, status in (("R1", "completed"), ("R0", "evaluating")):
            with self.assertRaises(ValueError):
                AUDIT.validate_method_contract(config, job, {"stage": status}, result, route)

    def test_rejects_wrong_validation_node_step(self):
        config = {"coverage": {"reconstruction": {"route": "P1"}}}
        job = {"run_id": "train", "kind": "train", "reconstruction_route": "P1"}
        status = {"stage": "stopped", "stop_reason": "mechanism_gate_pending", "env_step": 65536}
        node = {"scheduled_step": 65536, "actual_env_step": 65536,
                "config_hash": AUDIT.config_digest(config)}
        AUDIT.validate_method_contract(config, job, status, node, "P1", mechanism_step=65536)
        node["actual_env_step"] = 49152
        with self.assertRaisesRegex(ValueError, "validation node"):
            AUDIT.validate_method_contract(config, job, status, node, "P1", mechanism_step=65536)


if __name__ == "__main__":
    unittest.main()
