"""Source-backed I0 development decision and no-train failure propagation."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.s2p19 import controller, disk_guard, i0_gate_audit


class I0GateOpsTests(unittest.TestCase):
    def test_i0_requires_source_backed_logging_runtime_on_matching_service(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_path = root / "logging_receipt.json"
            science = "47da239910485508f7c978461314dff92f8672ec"
            runtime = "13ea014f1c1709c4a82059fc89ba3092e75010e1"
            tasks = [
                {"id": "i0_scientific_gate", "phase": "engineering_gate", "port": 20418,
                 "kind": "milestone", "requires": []},
                {"id": "I0_pilot_20418", "phase": "pilot", "port": 20418,
                 "kind": "improvement", "requires": ["i0_scientific_gate"],
                 "request_id": "s2p19-i0-train-51001", "disk_class": "I0_pilot",
                 "engineering_receipt_file": str(receipt_path), "body": {"mode": "train"}},
            ]
            plan_path, state_path = root / "plan.json", root / "state.json"
            plan = {"schema": 1, "experiment_id": "S2-P19", "revision": 1,
                    "terminal": False, "scientific_commit": science,
                    "science_origin_commit": science, "service_commit": science,
                    "tasks": tasks}
            controller.write_json(plan_path, plan)
            controller.write_json(state_path, {"schema": 1, "experiment_id": "S2-P19",
                "scientific_commit": science, "science_origin_commit": science,
                "service_commit": science, "plan_sha256": controller.digest(plan_path),
                "plan_revision": 1, "task_hashes": controller._task_hashes(tasks),
                "tasks": {"i0_scientific_gate": {"state": "completed"}}, "events": []})
            disk = {"observed_utc": "2026-10-07T00:00:00Z",
                    "archive_free_bytes": 1_000 * disk_guard.GIB,
                    "mount_identity": "nfs|/home/user|nfs", "root": "/group",
                    "s2p19_root_bytes": 0}
            with (patch.object(controller, "PLAN", plan_path),
                  patch.object(controller, "STATE", state_path),
                  patch.object(controller, "_service_identity", return_value={}) as identity,
                  patch.object(controller, "_run_rows", return_value=[]),
                  patch.object(controller, "_scene_rows", return_value=[]),
                  patch.object(controller, "_model_rows", return_value=[]),
                  patch.object(controller.disk_guard, "sample", return_value=disk),
                  patch.object(controller.ops, "api") as api_mock):
                missing = controller.step(dispatch=False)
                self.assertIn("waiting_i0_logging_equivalence_receipt",
                              [row["state"] for row in missing["waiting"]])
                source = root / "paired_source.json"
                source.write_text('{"same":true}', encoding="utf-8")
                criteria = {key: True for key in ("numeric_equivalence",
                    "full_step_and_final_beam_logging", "resume_equivalence", "workbench_wiring")}
                proof = root / "proof.json"
                source_sha = hashlib.sha256(b"frozen source").hexdigest()
                proof.write_text(json.dumps({"science_origin_commit": science,
                    "runtime_commit": runtime, "passed": True, "criteria": criteria,
                    "source_files": {"src/satellite_rl/sac_improvement/agent.py": {
                        commit: source_sha for commit in (science,
                            "62202aff0e1190492338f98359cc2f274884341c", runtime)}},
                    "evidence": [{"path": str(source), "sha256": controller.digest(source)}]}),
                    encoding="utf-8")
                receipt_path.write_text(json.dumps({"experiment_id": "S2-P19", "arm": "I0",
                    "science_origin_commit": science, "runtime_commit": runtime,
                    "request_id": "s2p19-i0-train-51001", "passed": True,
                    "criteria": criteria, "proof": {"path": str(proof),
                                                 "sha256": controller.digest(proof)}}),
                    encoding="utf-8")
                with patch.object(controller.subprocess, "run",
                                  return_value=SimpleNamespace(stdout=b"frozen source")):
                    held = controller.step(dispatch=False)
                self.assertIn("waiting_exact_i0_runtime_service",
                              [row["state"] for row in held["waiting"]])
                self.assertTrue(all(call.args[1] == science for call in identity.call_args_list))
                api_mock.assert_not_called()

    def test_threshold_is_per_candidate_not_mixed_between_methods(self):
        rule = [(125, .006)] * 48 + [(50, .004)] * 24  # overall .00533, large .006
        random = [(125, .011)] * 48 + [(50, -.01)] * 24  # large passes, overall .004
        result = i0_gate_audit.candidate_thresholds({"rule": rule, "random": random})
        self.assertFalse(result["rule"]["passes_performance"])
        self.assertFalse(result["random"]["passes_performance"])
        random = [(125, .012)] * 48 + [(50, -.006)] * 24  # overall .006, large .012
        result = i0_gate_audit.candidate_thresholds({"rule": rule, "random": random})
        self.assertTrue(result["random"]["passes_performance"])

    def test_empty_slot_beam_with_power_is_rejected(self):
        beams = [{"beam_id": index, "group": index % 8, "slots": 0, "freq": 0,
                  "required_bps": 1.0, "rate_bps": 0.0, "power_w": 0.0}
                 for index in range(50)]
        self.assertEqual(i0_gate_audit.physical_beams(beams, 50)["si"], 0.0)
        beams[0]["power_w"] = 1.0
        with self.assertRaises(ValueError):
            i0_gate_audit.physical_beams(beams, 50)

    def test_failed_gate_marks_only_i0_not_selected_and_never_posts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = root / "gate.json"
            audit.write_text('{"passed":false}', encoding="utf-8")
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({"path": str(audit),
                "sha256": controller.digest(audit)}), encoding="utf-8")
            c0 = {"id": "C0_pilot_20418", "phase": "pilot", "port": 20418,
                  "kind": "constructive_train", "requires": [], "adopted_only": True,
                  "adopted_run_id": "train_sac_c0_saved", "request_id": "s2p19-c0-train-51001",
                  "expected_config_sha256": "a" * 64,
                  "expected_commit": "47da239910485508f7c978461314dff92f8672ec",
                  "workbench_kind": "train"}
            prior = [{"id": f"i0_dev_B_{group}_20418", "phase": "engineering_gate",
                      "port": 20418, "kind": "milestone", "requires": []}
                     for group in ("greedy", "rule", "random")]
            gate = {"id": "i0_scientific_gate", "phase": "engineering_gate", "port": 20418,
                    "kind": "milestone", "requires": [row["id"] for row in prior],
                    "receipt_file": str(receipt)}
            i0 = {"id": "I0_pilot_20418", "phase": "pilot", "port": 20418,
                  "kind": "improvement", "requires": ["i0_scientific_gate"],
                  "request_id": "s2p19-i0-train-51001", "disk_class": "I0_pilot",
                  "body": {"mode": "train"}}
            approve = {"id": "approve_I0_pilot", "phase": "pilot_gate", "port": 20418,
                       "kind": "approve_pilot", "pilot_task_id": "I0_pilot_20418",
                       "requires": ["I0_pilot_20418"]}
            plan = {"schema": 1, "experiment_id": "S2-P19", "revision": 1,
                    "terminal": False, "scientific_commit": c0["expected_commit"],
                    "tasks": [c0, *prior, gate, i0, approve]}
            plan_path = root / "plan.json"
            state_path = root / "state.json"
            controller.write_json(plan_path, plan)
            state = {"schema": 1, "experiment_id": "S2-P19",
                     "scientific_commit": plan["scientific_commit"],
                     "plan_sha256": controller.digest(plan_path), "plan_revision": 1,
                     "task_hashes": controller._task_hashes(plan["tasks"]),
                     "tasks": {"C0_pilot_20418": {"state": "running", "run_id": "train_sac_c0_saved"},
                               **{row["id"]: {"state": "completed"} for row in prior}},
                     "events": []}
            controller.write_json(state_path, state)
            disk = {"observed_utc": "2026-10-07T00:00:00Z",
                    "archive_free_bytes": 100 * disk_guard.GIB,
                    "mount_identity": "nfs|/home/user|nfs", "root": "/group/20418",
                    "s2p19_root_bytes": 0}
            run = {"run_id": "train_sac_c0_saved", "request_id": "s2p19-c0-train-51001",
                   "git_commit": plan["scientific_commit"], "kind": "train", "stage": "training"}
            with (patch.object(controller, "PLAN", plan_path),
                  patch.object(controller, "STATE", state_path),
                  patch.object(controller, "_service_identity", return_value={}),
                  patch.object(controller, "_run_rows", side_effect=lambda port: [run] if port == 20418 else []),
                  patch.object(controller, "_scene_rows", return_value=[]),
                  patch.object(controller, "_model_rows", return_value=[]),
                  patch.object(controller.disk_guard, "sample", return_value=disk),
                  patch.object(controller, "_adopted_config_sha", return_value="a" * 64),
                  patch.object(controller, "_verify_i0_gate_audit", return_value=False),
                  patch.object(controller.ops, "api") as api_mock):
                result = controller.step(dispatch=True)
            api_mock.assert_not_called()
            self.assertEqual(result["tasks"]["C0_pilot_20418"]["state"], "running")
            self.assertEqual(result["tasks"]["i0_scientific_gate"]["state"], "gate_failed")
            self.assertEqual(result["tasks"]["I0_pilot_20418"]["state"], "not_selected")
            self.assertEqual(result["tasks"]["approve_I0_pilot"]["state"], "not_selected")


if __name__ == "__main__":
    unittest.main()
