"""Read-only adoption of a workbench-approved S2-P19 pilot gate."""

import unittest
from unittest.mock import patch

from scripts.s2p19 import controller


RUN = "train_sac_20261007_152717_479365fb"
AUDIT = "/home/u2024110224/second_stage_satellite_rl/sac_routes_s2p19/20418/archive/ops/pilot_audit/C0_audit.json"


def proof():
    return {"pass": True, "run_id": RUN, "checkpoint_sha256": "c" * 64,
            "checkpoint_step": 100000, "audit_path": AUDIT,
            "audit_sha256": "a" * 64, "approved_at": "2026-10-07T10:00:00+00:00"}


def audit():
    return {"experiment_id": "S2-P19", "run_id": RUN, "checkpoint_sha256": "c" * 64,
            "checkpoint_step": 100000, "passed": True,
            "criteria": {"source_backed": True},
            "source_files": [{"path": "/frozen/source.json", "sha256": "f" * 64}]}


class PilotApprovalReconciliationTests(unittest.TestCase):
    task = {"id": "approve_C0_pilot", "kind": "approve_pilot", "port": 20418,
            "pilot_task_id": "C0_pilot_20418"}

    def status(self, saved_state="waiting", pilot_state="running"):
        receipts = {"approve_C0_pilot": {"state": saved_state},
                    "C0_pilot_20418": {"state": pilot_state, "run_id": RUN}}
        return controller._task_status(self.task, receipts, {}, {}, {})

    def test_waiting_adopts_real_workbench_proof_and_backed_audit_without_post(self):
        with (patch.object(controller, "_pilot_gate_proof", return_value=proof()) as proof_read,
              patch.object(controller, "_pilot_audit_from_receipt",
                           return_value=audit()) as audit_read,
              patch.object(controller.ops, "api") as api):
            observed = self.status()
        self.assertEqual(observed["state"], "completed")
        self.assertEqual(observed["approved_run_id"], RUN)
        self.assertEqual(observed["checkpoint_step"], 100000)
        self.assertTrue(observed["recovered_from_workbench_proof"])
        proof_read.assert_called_once_with(20418, RUN)
        audit_read.assert_called_once_with(20418, RUN, {
            "audit_path": AUDIT, "audit_sha256": "a" * 64}, approved=True)
        api.assert_not_called()

    def test_malformed_or_unbacked_proof_is_rejected(self):
        for field, value in (("run_id", "other"), ("pass", False),
                             ("checkpoint_step", 99999), ("audit_sha256", None)):
            bad = proof()
            bad[field] = value
            with self.subTest(field=field), patch.object(controller, "_pilot_gate_proof",
                                                   return_value=bad), patch.object(
                    controller, "_pilot_audit_from_receipt") as audit_read:
                with self.assertRaises(RuntimeError):
                    self.status()
                audit_read.assert_not_called()
        missing_sources = audit()
        missing_sources["source_files"] = []
        with (patch.object(controller, "_pilot_gate_proof", return_value=proof()),
              patch.object(controller, "_pilot_audit_from_receipt",
                           return_value=missing_sources)):
            with self.assertRaises(RuntimeError):
                self.status()

    def test_only_success_or_failed_approval_is_immutable(self):
        with patch.object(controller, "_pilot_gate_proof", return_value=proof()) as proof_read:
            for state in ("completed", "gate_failed"):
                self.assertEqual(self.status(saved_state=state)["state"], state)
            proof_read.assert_not_called()
        with (patch.object(controller, "_pilot_gate_proof", return_value=proof()),
              patch.object(controller, "_pilot_audit_from_receipt", return_value=audit())):
            self.assertEqual(self.status(saved_state="not_selected")["state"], "completed")


if __name__ == "__main__":
    unittest.main()
