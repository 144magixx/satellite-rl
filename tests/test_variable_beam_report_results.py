"""Small fixtures for read-only S2-P17 reporting gates and independent arithmetic."""

import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.variable_beams.report_results import (
    AuditError, _sgm, audit_beams, generate, load_manifest,
)


class ReportAuditTests(unittest.TestCase):
    def test_recompute_two_beam_terminal_and_reject_modified_ratio(self):
        demands = [100.0, 200.0]
        capacities = [50.0, 300.0]
        beta = 150.0
        beams = []
        for index, (demand, capacity) in enumerate(zip(demands, capacities)):
            beams.append({"beam_id": index, "original_index": index,
                          "required_bps": demand, "capacity_bps": capacity,
                          "rate_bps": capacity, "effective_served_bps": min(capacity, demand),
                          "capacity_demand_ratio": capacity / demand,
                          "si": min(capacity / demand, 1.0),
                          "sgm": _sgm(capacity, demand, beta),
                          "basic_met": True, "slots": 1,
                          "power_w": [5.0, 10.0][index]})
        metrics = {"final_si_mean": 0.75,
                   "final_sgm_mean": sum(beam["sgm"] for beam in beams) / 2,
                   "final_si_total": 1.5, "basic_met_count": 2,
                   "zero_capacity_count": 0, "power_w": 15.0,
                   "complete": True}
        self.assertAlmostEqual(audit_beams(beams, metrics, 2)["effective_carried_bps"], 250.0)
        beams[0]["capacity_demand_ratio"] = 1.0
        with self.assertRaisesRegex(AuditError, "C/D"):
            audit_beams(beams, metrics, 2)

    def test_missing_job_matrix_refuses_report_without_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "controller_manifest.json"
            manifest.write_text(json.dumps({"experiment_id": "S2-P17",
                "scientific_commit": "a" * 40,
                "target": {"ssh_port": 20424},
                "datasets": {}, "jobs": {}}), encoding="utf-8")
            output = root / "report"
            with self.assertRaisesRegex(AuditError, "任务"):
                generate(manifest, output, skip_physics=True)
            self.assertFalse(output.exists())

    def test_wrong_execution_port_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "controller_manifest.json"
            names = ("V0_train", "V1_train", "V0_eval", "V1_eval", "H0_eval",
                     "V0_sca", "V1_sca", "H0_sca", "fixed_sca", "greedy_sca")
            manifest.write_text(json.dumps({"experiment_id": "S2-P17",
                "scientific_commit": "a" * 40,
                "target": {"ssh_port": 20418},
                "datasets": {name: {} for name in ("train", "validation", "test")},
                "jobs": {name: {} for name in names}}), encoding="utf-8")
            with self.assertRaisesRegex(AuditError, "20424"):
                load_manifest(manifest, {})

    def test_user_confirmed_520_scope_is_accepted_and_unconfirmed_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train_record.json"
            train.write_text("{}", encoding="utf-8")
            contract = {"experiment_id": "S2-P17", "status": "user_confirmed",
                        "decision": "eligible520", "mother_count": 520,
                        "derived_scene_count": 7280,
                        "user_message_reference": "user-turn-approved-520",
                        "train_record_path": str(train),
                        "train_record_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
                        "h0_compatibility_status": "passed",
                        "h0_compatibility_receipt": "h0-receipt.json"}
            scope = root / "approved_scope.json"
            scope.write_text(json.dumps(contract), encoding="utf-8")
            names = ("V0_train", "V1_train", "V0_eval", "V1_eval", "H0_eval",
                     "V0_sca", "V1_sca", "H0_sca", "fixed_sca", "greedy_sca")
            manifest = root / "controller_manifest.json"
            def write_manifest():
                manifest.write_text(json.dumps({"experiment_id": "S2-P17",
                    "scientific_commit": "a" * 40, "target": {"ssh_port": 20424},
                    "datasets": {name: {} for name in ("train", "validation", "test")},
                    "jobs": {name: {} for name in names},
                    "scope_sha256": hashlib.sha256(scope.read_bytes()).hexdigest(),
                    "scope": {key: contract[key] for key in ("decision", "mother_count",
                                                       "derived_scene_count", "user_message_reference")}}),
                    encoding="utf-8")
            write_manifest()
            self.assertEqual(load_manifest(manifest, {})["scope"]["derived_scene_count"], 7280)
            contract["status"] = "suggested_only"
            scope.write_text(json.dumps(contract), encoding="utf-8")
            write_manifest()
            with self.assertRaisesRegex(AuditError, "用户确认"):
                load_manifest(manifest, {})


if __name__ == "__main__":
    unittest.main()
