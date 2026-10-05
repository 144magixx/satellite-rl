"""R1 source reuse gate, import references and parallel baseline scheduling."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.stage1_reconstruction import controller as original
from scripts.stage1_reconstruction import rerun_controller as rerun


def plan():
    roles = {role: {"source_dataset_id": "dataset_" + role,
                    "dataset_sha256": letter * 64,
                    "source_run_id": "prepare_" + role,
                    "starts_sha256": letter.upper().lower() * 64,
                    "scene_count": count,
                    "qualified_count": 572 if role == "train" else count}
             for role, letter, count in (("train", "a", 576),
                                         ("validation", "b", 20),
                                         ("test", "c", 100))}
    return {"experiment_id": "S1-03-R1", "hostroot": rerun.NEW_BASE,
            "ports": [20418, 20424], "science_commit": "d" * 40,
            "config_template": {"schema_version": 4, "research_stage": "coverage",
                "training": {"seed": 51001, "total_steps": 262144,
                             "mechanism_gate_step": 65536}},
            "gates": original.GATES,
            "audit_commands": {mode: {"argv": [rerun.remote.DQN,
                                      f"{rerun.NEW_BASE}/{20424 if mode == 'formal' else 20418}"
                                      "/repo/scripts/stage1_reconstruction/audit.py"],
                                      "script_sha256": "f" * 64}
                               for mode in ("development", "mechanism", "formal")},
            "s2_release": {"controller_manifest": "/frozen/S2/manifest.json"},
            "source_reuse": {"source_group": rerun.OLD_GROUP,
                "source_target": "20418", "old_science_commit": rerun.OLD_COMMIT,
                "old_plan_sha256": "1" * 64,
                "old_source_manifest_sha256": "2" * 64,
                "new_source_manifest_sha256": "3" * 64,
                "roles": roles,
                "audit": {"path": rerun.NEW_BASE + "/20418/ops/reuse/reuse_audit.json",
                          "sha256": "4" * 64}}}


class RerunControllerTests(unittest.TestCase):
    def test_plan_rejects_budget_or_incomplete_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            valid = plan()
            path.write_text(json.dumps(valid), encoding="utf-8")
            loaded, digest = rerun.load_r1_plan(path)
            self.assertEqual(loaded["science_commit"], "d" * 40)
            self.assertEqual(len(digest), 64)
            valid["config_template"]["training"]["total_steps"] += 1
            path.write_text(json.dumps(valid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "full-budget"):
                rerun.load_r1_plan(path)
            valid = plan()
            valid["audit_commands"]["formal"]["argv"][0] = "python3"
            path.write_text(json.dumps(valid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must use DQN"):
                rerun.load_r1_plan(path)
            valid = plan()
            valid["audit_commands"]["development"]["argv"][1] = (
                rerun.OLD_BASE + "/20418/repo/scripts/stage1_reconstruction/audit.py")
            path.write_text(json.dumps(valid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "this group's frozen repo"):
                rerun.load_r1_plan(path)

    def test_old_start_hook_unchanged_and_r1_uses_import_record(self):
        old = object.__new__(original.Controller)
        old.state = {"jobs": {"prepare_train": "old_prepare_train"},
                     "imports": {"test_start_20424": "old_test_import"}}
        self.assertEqual(old._start_reference("train", 20418), "old_prepare_train")
        self.assertEqual(old._start_reference("test", 20424), "old_test_import")
        new = object.__new__(rerun.RerunController)
        new.state = {"jobs": {}, "imports": {"20418:train:start": "new_import"}}
        self.assertEqual(new._start_reference("train", 20418), "new_import")
        self.assertEqual(new.state["jobs"], {})

    def test_missing_reuse_audit_prevents_import_and_jobs(self):
        controller = object.__new__(rerun.RerunController)
        controller.plan = plan()
        controller.state = {"phase": "preparing", "source_reuse": None,
                            "jobs": {}, "imports": {}}
        controller._read_remote_json = lambda port, path: {"exists": False}
        self.assertFalse(controller._verify_reuse())
        self.assertEqual(controller.state["phase"], "awaiting_reuse_audit")
        self.assertEqual(controller.state["jobs"], {})

    def test_import_ids_are_not_fictitious_prepare_jobs(self):
        controller = object.__new__(rerun.RerunController)
        controller.plan = plan()
        controller.state = {"source_reuse": {"audit_sha256": "4" * 64},
                            "datasets": {}, "imports": {}, "jobs": {}, "observed": {}}
        role = "validation"
        expected = controller.plan["source_reuse"]["roles"][role]
        dataset = {"id": "new_dataset", "source_group": rerun.OLD_GROUP,
                   "source_target": rerun.SOURCE_TARGET,
                   "source_dataset_id": expected["source_dataset_id"],
                   "partition_role": role, "sha256": expected["dataset_sha256"],
                   "count": 20}
        start = {"id": "new_start", "source_group": rerun.OLD_GROUP,
                 "source_target": rerun.SOURCE_TARGET,
                 "source_run_id": expected["source_run_id"],
                 "partition_role": role, "sha256": expected["starts_sha256"],
                 "dataset_sha256": expected["dataset_sha256"]}
        def api(port, endpoint, body=None, **kwargs):
            return {"coverage/datasets": {"datasets": [dataset]},
                    "coverage/reconstruction/imports": {"starts": [start]}}[endpoint]
        controller.api = api
        self.assertTrue(controller._import_role(role, 20418))
        self.assertEqual(controller.state["datasets"]["20418:validation"], "new_dataset")
        self.assertEqual(controller.state["imports"]["20418:validation:start"], "new_start")
        self.assertEqual(controller.state["jobs"], {})

    def test_four_formal_baselines_are_submitted_even_when_first_is_active(self):
        controller = object.__new__(rerun.RerunController)
        controller.state = {"phase": "preparing", "jobs": {}, "imports": {},
                            "audits": {"development": {"passed": True}}, "observed": {}}
        controller._s2_released = lambda: True
        controller._verify_reuse = lambda: True
        controller._import_role = lambda role, port: True
        calls = []
        controller._job = lambda label, port, route, kind, role=None: (
            calls.append((label, port, route, kind, role)) or False)
        controller._lane_20424()
        self.assertEqual([row[0] for row in calls],
                         ["formal_R0", "formal_R1", "formal_R2", "formal_R3"])

    def test_failed_or_missing_development_gate_never_imports_jan31_or_submits_formal(self):
        controller = object.__new__(rerun.RerunController)
        controller.state = {"phase": "preparing", "jobs": {}, "imports": {},
                            "audits": {"development": {"passed": False}}, "observed": {}}
        controller._s2_released = lambda: True
        calls = []
        controller._verify_reuse = lambda: calls.append("reuse") or True
        controller._import_role = lambda role, port: calls.append("import") or True
        controller._job = lambda *args, **kwargs: calls.append("job") or False
        controller._lane_20424()
        self.assertEqual(calls, [])
        self.assertEqual(controller.state["phase"],
                         "awaiting_development_audit_for_independent_test")


if __name__ == "__main__":
    unittest.main()
