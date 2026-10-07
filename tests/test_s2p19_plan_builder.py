"""The initial phase plan binds API tasks through prior outputs, not run IDs."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.s2p19 import plan_builder


class PlanBuilderTests(unittest.TestCase):
    def test_bootstrap_has_typed_prep_teacher_and_three_pilots(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            commit = "a" * 40
            launch = root / "preflight" / f"workbench_{commit[:12]}.json"
            launch.parent.mkdir(parents=True)
            launch.write_text(json.dumps({"scientific_commit": commit,
                "instances": {"20418": {}, "20424": {}}}), encoding="utf-8")
            (root / "preflight/p18_v2_model.json").write_text(json.dumps({
                "source_dir": "/home/x/train_sac_20261006_084454_365eff71/final",
                "provenance": {"run_id": "train_sac_20261006_084454_365eff71", "env_step": 400000},
                "files": {"policy.pt": {"sha256": "78ed38b164141af3968c4498dc4a462b8baf060764f133076ea7c02c7d868cea"}}}),
                encoding="utf-8")
            c0 = {"training": {"total_steps": 400000, "pilot_stop_steps": 100000,
                               "demonstrations": {"mode": "none"}}}
            c1 = {"training": {"total_steps": 400000, "pilot_stop_steps": 100000,
                               "demonstrations": {"mode": "half"}}}
            catalog = {"experiment_id": "S2-P19", "defaults": {"C0": c0, "C1": c1,
                         "I0": {"total_steps": 400000, "pilot_stop_steps": 100000}}}
            plan_path = root / "controller" / "plan.json"
            with (patch.object(plan_builder, "_source_commit", return_value=commit),
                  patch.object(plan_builder.ops, "LOCAL", root),
                  patch.object(plan_builder.ops, "api", return_value=catalog),
                  patch.object(plan_builder.controller, "PLAN", plan_path),
                  patch.object(plan_builder, "CONTRACTS", Path("D:/paper/outputs/experiments/sac_routes_s2p19_20261007/contracts"))):
                report = plan_builder.bootstrap()
            self.assertEqual(report["task_count"], 21)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            tasks = {task["id"]: task for task in plan["tasks"]}
            self.assertEqual(plan["revision"], 1)
            self.assertFalse(plan["terminal"])
            self.assertEqual(tasks["prepare_final_20424"]["body"]["preparation"]["source_record"]["path"],
                             plan_builder.ref("prepare_mothers_20424", "result", "summary", "path"))
            self.assertEqual(tasks["C1_pilot_20424"]["body"]["config"]["training"]["demonstrations"]["manifest_path"],
                             plan_builder.ref("teacher_20424", "result", "summary", "manifest_path"))
            self.assertTrue({"C0_pilot_20418", "C1_pilot_20424", "I0_pilot_20418"} <= set(tasks))

    def test_phase_advance_adds_one_shared_baseline_job(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "controller/plan.json"
            state_path = root / "controller/state.json"
            gate_path = root / "controller/gates/retained_arms.json"
            gate_path.parent.mkdir(parents=True)
            audit_path = root / "retention_audit.json"
            audit_path.write_text('{"passed":true}', encoding="utf-8")
            gate_path.write_text(json.dumps({"audit_path": str(audit_path),
                "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                "retained_arms": ["C0", "C1", "I0"]}), encoding="utf-8")
            old = {"schema": 1, "experiment_id": "S2-P19", "scientific_commit": "a" * 40,
                   "revision": 1, "terminal": False, "tasks": [
                       {"id": "import_final_20418", "phase": "preparation", "port": 20418,
                        "kind": "milestone"},
                       {"id": "import_final_20424", "phase": "preparation", "port": 20424,
                        "kind": "milestone"},
                       {"id": "import_V2_model_20418", "phase": "preparation", "port": 20418,
                        "kind": "milestone"},
                       *({"id": f"{arm}_pilot_{port}", "phase": "pilot", "port": port,
                          "kind": "milestone"} for arm, port in
                         (("C0", 20418), ("C1", 20424), ("I0", 20418))),
                       *({"id": f"approve_{arm}_pilot", "phase": "pilot_gate", "port": port,
                          "kind": "milestone"} for arm, port in
                         (("C0", 20418), ("C1", 20424), ("I0", 20418))),
                   ]}
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(json.dumps(old), encoding="utf-8")
            receipts = {f"import_final_{port}": {"state": "completed", "scene_id": str(port)}
                        for port in (20418, 20424)}
            receipts["import_V2_model_20418"] = {"state": "completed", "model_id": "model_V2"}
            for arm, port in (("C0", 20418), ("C1", 20424), ("I0", 20418)):
                receipts[f"{arm}_pilot_{port}"] = {"state": "completed", "run_id": f"train_sac_{arm.lower()}"}
                receipts[f"approve_{arm}_pilot"] = {"state": "completed"}
            state_path.write_text(json.dumps({"tasks": receipts}), encoding="utf-8")

            def api(port, path):
                if path == "/api/s2p19/catalog":
                    return {"defaults": {"I0": {"total_steps": 400000}}}
                if path == "/api/models":
                    arm = "C0" if port == 20418 else "C1"
                    return {"models": [{"id": f"model_{arm}",
                        "path": f"/a/train_sac_{arm.lower()}/final",
                        "provenance": {"run_id": f"train_sac_{arm.lower()}", "env_step": 400000}}]}
                raise AssertionError(path)

            with (patch.object(plan_builder.ops, "LOCAL", root),
                  patch.object(plan_builder.ops, "api", side_effect=api),
                  patch.object(plan_builder.controller, "PLAN", plan_path),
                  patch.object(plan_builder.controller, "STATE", state_path)):
                result = plan_builder.advance_common_evaluation()
            updated = json.loads(plan_path.read_text(encoding="utf-8"))
            tasks = {task["id"]: task for task in updated["tasks"]}
            self.assertEqual(updated["revision"], 2)
            self.assertEqual(result["retained_arms"], ["C0", "C1", "I0"])
            self.assertEqual(tasks["baseline_test_20418"]["body"]["methods"],
                             ["greedy", "rule", "random"])
            self.assertEqual(tasks["C0_final_direct_20418"]["body"]["baselines"], [])
            self.assertEqual(len([task for task in tasks if task.startswith("baseline_greedy_B")]), 1)
            self.assertEqual(tasks["V2_final_direct_20418"]["body"]["model_id"],
                             plan_builder.ref("import_V2_model_20418", "model_id"))


if __name__ == "__main__":
    unittest.main()
