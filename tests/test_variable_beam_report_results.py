"""Small fixtures for read-only S2-P17 reporting gates and independent arithmetic."""

import json
import hashlib
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.variable_beams.report_results import (
    AuditError, _job_files_from_manifest, _sgm, audit_beams, generate, load_manifest,
    sha256_file, verify_reimported_files,
)


class ReportAuditTests(unittest.TestCase):
    def test_controller_job_path_is_a_file_and_siblings_share_one_run_directory(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "jobs" / "train_sac_example"
            entry = {"run_id": directory.name,
                     "job_path": str(directory / "job.json"),
                     "status_path": str(directory / "status.json"),
                     "results_path": str(directory / "results.json")}
            paths = _job_files_from_manifest(entry)
            self.assertEqual(paths[0], directory.resolve())
            self.assertEqual(paths[1], directory.resolve() / "job.json")
            self.assertEqual(paths[3], directory.resolve() / "config.json")
            with self.assertRaisesRegex(AuditError, "路径"):
                _job_files_from_manifest({**entry, "job_path": str(directory)})
            with self.assertRaisesRegex(AuditError, "路径"):
                _job_files_from_manifest({**entry,
                    "status_path": str(directory.parent / "other" / "status.json")})

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
                          "basic_met": True, "group": index, "freq": 10,
                          "slots": 1,
                          "power_w": [5.0, 10.0][index]})
        metrics = {"final_si_mean": 0.75,
                   "final_sgm_mean": sum(beam["sgm"] for beam in beams) / 2,
                   "final_si_total": 1.5, "basic_met_count": 2,
                   "zero_capacity_count": 0, "power_w": 15.0,
                   "allocated_slots": 2, "zero_slot_count": 0,
                   "complete": True}
        audited = audit_beams(beams, metrics, 2)
        self.assertAlmostEqual(audited["effective_carried_bps"], 250.0)
        self.assertEqual(audited["slots_total"], 2)
        self.assertEqual(audited["zero_slot_fraction"], 0)
        with self.assertRaisesRegex(AuditError, "槽数"):
            audit_beams(beams, {**metrics, "allocated_slots": 3}, 2)
        beams[0]["freq"] = 100
        with self.assertRaisesRegex(AuditError, "频率槽起点"):
            audit_beams(beams, metrics, 2)
        beams[0]["freq"] = 10
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

    def test_repacked_import_requires_source_sha_and_equal_arrays(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new = root / "prepared.npz", root / "imported.npz"
            np.savez_compressed(old, rate=np.array([[1.0]], dtype=np.float64))
            np.savez_compressed(new, rate=np.array([[1.0]], dtype=np.float64))
            with zipfile.ZipFile(new, "a") as archive:
                archive.comment = b"workbench repack"
            self.assertNotEqual(sha256_file(old), sha256_file(new))
            metadata = {"beam_count": 1, "mother_scene_id": "m0",
                        "load_multiplier": 0.75, "partition_role": "train",
                        "selected_time_utc": "2024-01-08T00:00:00Z",
                        "source_sha256": "s" * 64}
            prepared = {"files": [{**metadata, "path": str(old), "sha256": sha256_file(old)}]}
            imported = {"files": [{**metadata, "path": str(new), "sha256": sha256_file(new)}],
                        "source": {"reimport": {"files": [{"sha256": sha256_file(old)}]}}}
            verify_reimported_files(prepared, imported, {}, "train")
            imported["source"]["reimport"]["files"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(AuditError, "来源链"):
                verify_reimported_files(prepared, imported, {}, "train")
            imported["source"]["reimport"]["files"][0]["sha256"] = sha256_file(old)
            np.savez_compressed(new, rate=np.array([[2.0]], dtype=np.float64))
            imported["files"][0]["sha256"] = sha256_file(new)
            with self.assertRaisesRegex(AuditError, "数组不等价"):
                verify_reimported_files(prepared, imported, {}, "train")


if __name__ == "__main__":
    unittest.main()
