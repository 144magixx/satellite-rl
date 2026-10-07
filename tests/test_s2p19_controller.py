"""S2-P19 controller contract checks without remote dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.s2p19 import controller, disk_guard


class ControllerContractTests(unittest.TestCase):
    def test_joint_decision_survives_first_resume_before_second_approval(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks, receipts = [], {}
            for arm, port in (("C0", 20418), ("C1", 20424)):
                pilot = f"{arm}_pilot_{port}"
                reference = root / f"{arm}.json"
                reference.write_text(json.dumps({"audit_path": f"/remote/{arm}.json",
                                                 "audit_sha256": arm.lower()[1] * 64}),
                                     encoding="utf-8")
                tasks.extend([
                    {"id": pilot, "phase": "pilot", "port": port,
                     "kind": "milestone", "requires": []},
                    {"id": f"approve_{arm}_pilot", "phase": "pilot_gate", "port": port,
                     "kind": "approve_pilot", "pilot_task_id": pilot, "requires": [pilot],
                     "gate_receipt_file": str(reference)}])
                receipts[pilot] = {"state": "pilot_gate_pending", "run_id": f"train_{arm.lower()}"}
            plan = {"schema": 1, "experiment_id": "S2-P19", "revision": 1,
                    "terminal": False, "scientific_commit": "a" * 40, "tasks": tasks}
            plan_path, state_path = root / "plan.json", root / "state.json"
            controller.write_json(plan_path, plan)
            controller.write_json(state_path, {"schema": 1, "experiment_id": "S2-P19",
                "scientific_commit": plan["scientific_commit"],
                "plan_sha256": controller.digest(plan_path), "plan_revision": 1,
                "task_hashes": controller._task_hashes(tasks), "tasks": receipts, "events": []})
            disk = {"observed_utc": "2026-10-07T00:00:00Z",
                    "archive_free_bytes": 100 * disk_guard.GIB,
                    "mount_identity": "nfs|/home/user|nfs", "root": "/group",
                    "s2p19_root_bytes": 0}
            reviewed = []

            def audit(port, run, reference, approved=False):
                reviewed.append((run, approved))
                return {"passed": True, "checkpoint_sha256": run.ljust(64, "x")}

            with (patch.object(controller, "PLAN", plan_path),
                  patch.object(controller, "STATE", state_path),
                  patch.object(controller, "JOINT_PILOT_DECISION", root / "joint.json"),
                  patch.object(controller, "_service_identity", return_value={}),
                  patch.object(controller, "_run_rows", return_value=[]),
                  patch.object(controller, "_scene_rows", return_value=[]),
                  patch.object(controller, "_model_rows", return_value=[]),
                  patch.object(controller.disk_guard, "sample", return_value=disk),
                  patch.object(controller, "_pilot_audit_from_receipt", side_effect=audit),
                  patch.object(controller.ops, "api", return_value={"stage": "queued"}) as api_mock):
                first = controller.step(dispatch=True)
                self.assertEqual(api_mock.call_count, 1)
                self.assertTrue((root / "joint.json").is_file())
                self.assertEqual(first["tasks"]["approve_C0_pilot"]["state"], "completed")
                saved = json.loads(state_path.read_text(encoding="utf-8"))
                saved["tasks"]["C0_pilot_20418"]["state"] = "running"
                controller.write_json(state_path, saved)
                second = controller.step(dispatch=True)
                self.assertEqual(api_mock.call_count, 2)
            self.assertEqual(second["tasks"]["approve_C1_pilot"]["state"], "completed")
            self.assertIn(("train_c0", True), reviewed)

    def test_pilot_audit_accepts_sha_bound_sources_from_both_isolated_archives(self):
        archives = {20418: "/shared/s2p19/20418/archive",
                    20424: "/shared/s2p19/20424/archive"}
        audit = {"experiment_id": "S2-P19", "run_id": "train_c0", "passed": True,
                 "criteria": {"same_cases": True}, "checkpoint_sha256": "c" * 64,
                 "source_files": [
                     {"path": archives[20418] + "/case.json", "sha256": "a" * 64},
                     {"path": archives[20424] + "/case.json", "sha256": "b" * 64}]}
        calls = []

        def remote(code, port, timeout=120):
            calls.append(code)
            value = ({"audit": audit, "checkpoint_sha256": "c" * 64}
                     if len(calls) == 1 else {"checked": 2})
            return {"exit_code": 0, "stdout": json.dumps(value) + "\n", "stderr": ""}

        with (patch.object(controller.ops, "instance",
                           side_effect=lambda port: {"archive": archives[port]}),
              patch.object(controller.ops, "remote_python", side_effect=remote)):
            observed = controller._pilot_audit_from_receipt(20418, "train_c0",
                {"audit_path": archives[20418] + "/audit.json", "audit_sha256": "d" * 64})
        self.assertTrue(observed["passed"])
        self.assertIn(archives[20424] + "/", calls[1])

    def test_failed_matched_pilot_audit_never_approves_either_arm(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = []
            receipts = {}
            for arm, port in (("C0", 20418), ("C1", 20424)):
                pilot = f"{arm}_pilot_{port}"
                tasks.append({"id": pilot, "phase": "pilot", "port": port,
                              "kind": "milestone", "requires": []})
                receipts[pilot] = {"state": "pilot_gate_pending", "run_id": f"train_{arm.lower()}",
                                   "workbench_stage": "stopped"}
            for arm, port in (("C0", 20418), ("C1", 20424)):
                reference = root / f"{arm}.json"
                reference.write_text(json.dumps({"audit_path": f"/remote/{arm}.json",
                                                  "audit_sha256": "a" * 64}), encoding="utf-8")
                tasks.append({"id": f"approve_{arm}_pilot", "phase": "pilot_gate", "port": port,
                              "kind": "approve_pilot", "pilot_task_id": f"{arm}_pilot_{port}",
                              "requires": [f"{arm}_pilot_{port}"],
                              "gate_receipt_file": str(reference)})
            plan = {"schema": 1, "experiment_id": "S2-P19", "revision": 1,
                    "terminal": False, "scientific_commit": "a" * 40, "tasks": tasks}
            plan_path, state_path = root / "plan.json", root / "state.json"
            controller.write_json(plan_path, plan)
            controller.write_json(state_path, {"schema": 1, "experiment_id": "S2-P19",
                "scientific_commit": plan["scientific_commit"],
                "plan_sha256": controller.digest(plan_path), "plan_revision": 1,
                "task_hashes": controller._task_hashes(tasks), "tasks": receipts, "events": []})
            disk = {"observed_utc": "2026-10-07T00:00:00Z",
                    "archive_free_bytes": 100 * disk_guard.GIB,
                    "mount_identity": "nfs|/home/user|nfs", "root": "/group",
                    "s2p19_root_bytes": 0}
            audits = {"train_c0": {"passed": True, "checkpoint_sha256": "c" * 64},
                      "train_c1": {"passed": False, "checkpoint_sha256": "d" * 64}}
            with (patch.object(controller, "PLAN", plan_path),
                  patch.object(controller, "STATE", state_path),
                  patch.object(controller, "JOINT_PILOT_DECISION", root / "joint.json"),
                  patch.object(controller, "_service_identity", return_value={}),
                  patch.object(controller, "_run_rows", return_value=[]),
                  patch.object(controller, "_scene_rows", return_value=[]),
                  patch.object(controller, "_model_rows", return_value=[]),
                  patch.object(controller.disk_guard, "sample", return_value=disk),
                  patch.object(controller, "_pilot_audit_from_receipt",
                               side_effect=lambda port, run, ref, approved=False: audits[run]),
                  patch.object(controller.ops, "api") as api_mock):
                result = controller.step(dispatch=True)
                repeated = controller.step(dispatch=True)
            api_mock.assert_not_called()
            self.assertEqual(result["tasks"]["approve_C0_pilot"]["state"], "gate_failed")
            self.assertEqual(result["tasks"]["approve_C1_pilot"]["state"], "gate_failed")
            self.assertEqual(result["tasks"]["C0_pilot_20418"]["state"], "not_selected")
            self.assertEqual(result["tasks"]["C1_pilot_20424"]["state"], "not_selected")
            self.assertEqual(repeated["tasks"]["approve_C0_pilot"]["state"], "gate_failed")
            self.assertEqual(repeated["tasks"]["C1_pilot_20424"]["state"], "not_selected")

    def test_typed_plan_and_reference(self):
        with TemporaryDirectory() as temporary:
            plan_path = Path(temporary) / "plan.json"
            plan = {"schema": 1, "experiment_id": "S2-P19", "scientific_commit": "a" * 40,
                    "revision": 1, "terminal": False,
                    "tasks": [
                        {"id": "dev", "phase": "preparation", "port": 20418,
                         "kind": "prepare", "request_id": "s2p19-dev", "disk_class": "dev_validation",
                         "body": {"preparation": {"preparation_type": "dev_validation"}}},
                        {"id": "import-dev", "phase": "preparation", "port": 20418,
                         "kind": "import_scene", "disk_class": "dev_validation",
                         "source_record_id": "s2p19_development_20261007", "expected_count": 72,
                         "requires": ["dev"], "body": {"path": {"$task": {"id": "dev",
                                      "path": ["result", "summary", "outputs", "development", "path"]}}}},
                    ]}
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            with patch.object(controller, "PLAN", plan_path):
                loaded, digest = controller.load_plan()
            self.assertEqual(len(loaded["tasks"]), 2)
            self.assertEqual(len(digest), 64)
            actual = controller._resolve(plan["tasks"][1]["body"], {"dev": {"state": "completed",
                "result": {"summary": {"outputs": {"development": {"path": "/x"}}}}}})
            self.assertEqual(actual["path"], "/x")
            plan["tasks"][1]["requires"] = ["future"]
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            with patch.object(controller, "PLAN", plan_path), self.assertRaises(ValueError):
                controller.load_plan()

    def test_shared_disk_reservation_blocks_second_large_job(self):
        first = disk_guard.PREDICTED_BYTES["final_B_one_stochastic_arm_2640"]
        observed = {"observed_utc": "2026-10-07T00:00:00+00:00",
                    "archive_free_bytes": first + disk_guard.RESERVE_BYTES + 1024,
                    "mount_identity": "nfs-server|/home/u2024110224|nfs",
                    "s2p19_root_bytes": 0}
        self.assertTrue(disk_guard.gate(observed, "final_B_one_stochastic_arm_2640")["admit"])
        self.assertFalse(disk_guard.gate(observed, "final_B_one_stochastic_arm_2640", first)["admit"])

    def test_archive_mount_identity_deduplicates_shared_roots(self):
        first = {"root": "/nfs/20418", "mount_identity": "nfs-server|/home/user|nfs",
                 "s2p19_root_bytes": 5 * disk_guard.GIB}
        second = {"root": "/nfs/20424", "mount_identity": first["mount_identity"],
                  "s2p19_root_bytes": 7 * disk_guard.GIB}
        self.assertEqual(controller._shared_root_bytes(20418, {20418: first, 20424: second}),
                         12 * disk_guard.GIB)
        second["mount_identity"] = "other-disk|/archive|ext4"
        self.assertEqual(controller._shared_root_bytes(20418, {20418: first, 20424: second}),
                         5 * disk_guard.GIB)

    def test_engineering_budget_warns_without_blocking_valid_nfs_space(self):
        observed = {"observed_utc": "2026-10-07T00:00:00+00:00",
                    "archive_free_bytes": 2_000 * disk_guard.GIB,
                    "mount_identity": "nfs-server|/home/user|nfs",
                    "s2p19_root_bytes": 250 * disk_guard.GIB}
        decision = disk_guard.gate(observed, "final_B_one_stochastic_arm_2640")
        self.assertTrue(decision["archive_schedule_warning"])
        self.assertTrue(decision["admit"])


if __name__ == "__main__":
    unittest.main()
