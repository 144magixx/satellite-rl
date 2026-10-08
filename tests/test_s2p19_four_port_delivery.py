"""Local ownership checks for the four-instance S2-P19 delivery tools."""

import json
import base64
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts.s2p19 import archive, ops, report


class FourPortDeliveryTests(unittest.TestCase):
    def test_instance_allowlist_and_api_mapping(self):
        for port, api_port in ((20418, 8900), (20424, 8901),
                               (20473, 8904), (20474, 8905)):
            item = ops.instance(port)
            self.assertEqual(item["api_port"], api_port)
            self.assertEqual(item["archive"],
                             f"{ops.REMOTE_BASE}/{port}/archive")
        with self.assertRaises(ValueError):
            ops.instance(20475)
        with self.assertRaises(ValueError):
            ops.api(20475, "/api/runs/x")
        self.assertEqual(ops.NEW_HOST_KEY, ops.transport.HOST_KEYS[20418])

    def test_new_port_host_key_pin_rejects_mismatch(self):
        observed = {}

        class SSHException(Exception):
            pass

        class FakeClient:
            def set_missing_host_key_policy(self, policy):
                observed["policy"] = policy

            def connect(self, host, **kwargs):
                observed["host"] = host
                observed["port"] = kwargs["port"]

        fake = types.SimpleNamespace(MissingHostKeyPolicy=object,
                                     SSHException=SSHException, SSHClient=FakeClient)
        with patch.dict(sys.modules, {"paramiko": fake}), \
                patch.object(ops.transport, "_password", return_value="fixture-secret"):
            ops.connect(20474)
            self.assertEqual(observed["port"], 20474)
            key = types.SimpleNamespace(asbytes=lambda: b"wrong-key")
            with self.assertRaises(SSHException):
                observed["policy"].missing_host_key(None, "host", key)
            raw = b"matching-key"
            pin = base64.b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
            with patch.object(ops, "NEW_HOST_KEY", pin):
                observed["policy"].missing_host_key(
                    None, "host", types.SimpleNamespace(asbytes=lambda: raw))

    def test_archive_rejects_cross_port_and_archive_root_before_copy(self):
        port = 20473
        run_id = "evaluate_sac_fixture"
        job = {"run_id": run_id, "experiment_id": "S2-P19",
               "archive_root": ops.instance(20418)["archive"]}
        detail = {"job": job, "status": {"run_id": run_id,
                                          "stage": "completed"}}
        with patch.object(ops, "api", return_value=detail) as api, \
                patch.object(ops, "remote_python") as remote:
            with self.assertRaises(ValueError):
                archive.archive_completed(port, run_id, "C0_final_direct_20418")
            api.assert_not_called()
            with self.assertRaises(ValueError):
                archive.archive_completed(port, run_id, "C0_final_direct_20473")
            remote.assert_not_called()

    def test_archive_accepts_legacy_logical_keys_but_keeps_completion_gate(self):
        port = 20418
        run_id = "evaluate_sac_fixture"
        detail = {"job": {"run_id": run_id, "experiment_id": "S2-P19",
                          "archive_root": ops.instance(port)["archive"]},
                  "status": {"run_id": run_id, "stage": "completed"}}
        for task_id in ("validation_greedy_direct", "teacher_validation", "C0"):
            with self.subTest(task_id=task_id), \
                    patch.object(ops, "api", return_value=detail), \
                    patch.object(ops, "remote_python", return_value={
                        "exit_code": 1, "stderr": "fixture stopped before remote copy"}) as remote:
                with self.assertRaisesRegex(RuntimeError, "fixture stopped"):
                    archive.archive_completed(port, run_id, task_id)
                remote.assert_called_once()
        incomplete = {**detail, "status": {"run_id": run_id, "stage": "evaluating"}}
        with patch.object(ops, "api", return_value=incomplete), \
                patch.object(ops, "remote_python") as remote:
            with self.assertRaises(ValueError):
                archive.archive_completed(port, run_id, "validation_greedy_direct")
            remote.assert_not_called()

    def test_report_merges_five_legacy_and_seven_four_lane_tasks(self):
        legacy = {
            "V2_final_direct_20418": "run_v2_direct",
            "V2_final_B_20418": "run_v2_B",
            "baseline_test_20418": "run_baseline",
            "baseline_greedy_B_20418": "run_greedy_B",
            "baseline_rule_B_20418": "run_rule_B",
        }
        new = [
            ("baseline_random_B_20418", 20418),
            ("C0_final_direct_20473", 20473), ("C0_final_B_20473", 20473),
            ("C1_final_direct_20424", 20424), ("C1_final_B_20424", 20424),
            ("I0_final_direct_20474", 20474), ("I0_final_B_20474", 20474),
        ]
        plan = {"experiment_id": "S2-P19", "completed_legacy": legacy,
                "tasks": [{"id": name, "port": port} for name, port in new] +
                         [{"id": "approve_C0_pilot", "port": 20418}]}
        self.assertEqual(len(report.final_tasks(plan)), 12)
        self.assertEqual(report.plan_ports(plan)["I0_final_direct_20474"], 20474)
        with tempfile.TemporaryDirectory() as temp:
            manifest = {"experiment_id": "S2-P19", "authorization": {"scope": ["C0"]}}
            summary = report.build(manifest, {"tasks": {}}, plan, {},
                                   Path(temp), render_plots=False)
            self.assertFalse(summary["formal_ready"])
            self.assertEqual(len(summary["planned_final_tasks"]), 12)
            self.assertEqual(summary["jobs"], {})
            self.assertEqual(summary["comparisons"], {})
        plan["tasks"][1]["port"] = 20418
        with self.assertRaises(report.AuditError):
            report.plan_ports(plan)

    def test_report_rejects_source_from_other_port(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = root / "results.json"
            results.write_text(json.dumps({"scenes": []}), encoding="utf-8")
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({"run_id": "run_c0", "port": 20418,
                                           "job_stage": "completed",
                                           "inventory_sha256": "a" * 64}),
                               encoding="utf-8")
            spec = {"run_id": "run_c0", "port": 20473,
                    "results_path": str(results),
                    "results_sha256": archive.digest(results),
                    "archive_receipt_path": str(receipt),
                    "archive_inventory_sha256": "a" * 64}
            with self.assertRaises(report.AuditError):
                report.source_for_task("C0_final_direct_20473", "run_c0",
                                       {"jobs": {"C0_final_direct_20473": spec}},
                                       {}, {}, 20473)

    def test_report_rejects_run_that_differs_from_adopted_plan(self):
        task_id = "C0_final_direct_20473"
        plan = {"experiment_id": "S2-P19", "tasks": [
            {"id": task_id, "port": 20473, "adopted_run_id": "expected_run"}]}
        manifest = {"experiment_id": "S2-P19", "authorization": {"scope": ["C0"]}}
        controller = {"tasks": {task_id: {"state": "completed", "run_id": "other_run"}}}
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(report.AuditError):
                report.build(manifest, controller, plan, {}, Path(temp),
                             render_plots=False)

    def test_archived_source_roundtrip_without_explicit_report_port(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mirror = root / "mirror"
            files = {
                "job/job.json": json.dumps({"run_id": "run_c0"}).encode(),
                "job/config.json": b"{}",
                "job/status.json": json.dumps({"run_id": "run_c0",
                                               "stage": "completed"}).encode(),
                "job/results.json": b'{"scenes":[]}',
                "source/source_manifest.json": b"{}",
            }
            for name, payload in files.items():
                path = mirror / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            receipt = {"port": 20473, "run_id": "run_c0", "job_stage": "completed",
                       "inventory_sha256": "a" * 64, "scene_roles": {},
                       "members": [{"member": name, "bytes": len(payload),
                                    "sha256": hashlib.sha256(payload).hexdigest()}
                                   for name, payload in files.items()]}
            (root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            spec, _ = archive.source_spec("C0_final_direct_20473", "run_c0",
                                          mirror, receipt)
            source, reason = report.source_for_task(
                "C0_final_direct_20473", "run_c0",
                {"jobs": {"C0_final_direct_20473": spec}}, {}, {})
            self.assertIsNone(reason)
            self.assertEqual(source["spec"]["port"], 20473)
            source, reason = report.source_for_task(
                "C0_final_direct_20473", "run_c0",
                {"jobs": {"C0_final_direct_20473": spec}}, {}, {}, 20473,
                require_port_evidence=True)
            self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
