"""Adopt live S2-P19 jobs without ever recreating their scientific roles."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.s2p19 import controller, disk_guard, plan_builder


SCIENCE = plan_builder.SCIENCE_COMMIT
R1 = plan_builder.R1_MOTHER_COMMIT


def _fixture(root: Path):
    config_shas = {}
    jobs = {20418: [], 20424: []}

    def job(role, port, kind, commit, stage, token):
        run_id = f"{kind}_{role.lower()}_saved"
        request_id = f"s2p19-saved-{role.lower()}"
        config_sha = token * 64
        row = {"run_id": run_id, "request_id": request_id, "port": port,
               "kind": kind, "stage": stage, "git_commit": commit,
               "config_sha256": config_sha}
        jobs[port].append(row)
        config_shas[run_id] = config_sha
        return row

    dev = job("dev", 20418, "prepare", SCIENCE, "completed", "a")
    r1 = job("r1", 20424, "prepare", R1, "completed", "b")
    teacher = job("teacher", 20424, "prepare", SCIENCE, "completed", "c")
    direct = job("direct", 20418, "evaluate", SCIENCE, "completed", "d")
    groups = {group: job(group, 20418, "refine", SCIENCE,
                         "completed" if group == "greedy" else "evaluating" if group == "rule" else "queued",
                         token)
              for group, token in (("greedy", "e"), ("rule", "f"), ("random", "1"))}
    c0 = job("c0", 20418, "train", SCIENCE, "training", "2")
    c1 = job("c1", 20424, "train", SCIENCE, "training", "3")
    jobs[20424].append({"run_id": "prepare_old_failed", "request_id": "old-six-e-eight",
                        "kind": "prepare", "stage": "failed", "git_commit": "6e8"})
    scenes = {}
    imported = {}
    for port in (20418, 20424):
        scenes[port] = []
        for name, source_id, count in (
            ("train", "variable_beams_train_20261004", 8064),
            ("development", "s2p19_development_20261007", 72),
            ("validation", "s2p19_validation_20261007", 72),
        ):
            scene_id = f"scenes_{name}_{port}"
            scenes[port].append({"id": scene_id, "count": count,
                                 "source": {"reimport": {"record_id": source_id}}})
            imported[f"{name}_{port}"] = {"scene_id": scene_id, "count": count}
    manifest = {"experiment_id": "S2-P19", "authorization": {"scope": ["C0", "C1", "I0"]},
                "tasks": {"C0": c0, "C1": c1, "I0_development_gate": direct,
                          "mother_sources_R1": r1,
                          **{f"I0_dev_B_{group}": row for group, row in groups.items()}},
                "operations": {"preparation": {"dev_validation": dev, "teacher": teacher},
                               "scene_imports": imported}}
    (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pin = root / "preflight/p18_v2_model.json"
    pin.parent.mkdir(parents=True)
    pin.write_text(json.dumps({"source_dir": "/old/V2/final",
        "provenance": {"run_id": "train_sac_20261006_084454_365eff71", "env_step": 400000},
        "files": {"policy.pt": {"sha256": "4" * 64}}}), encoding="utf-8")

    def api(port, path):
        if path == "/api/s2p19/runs":
            return {"runs": jobs[port]}
        if path == "/api/scenes":
            return {"scenes": scenes[port]}
        if path == "/api/s2p19/catalog":
            return {"defaults": {"I0": {"seed": 51001, "total_steps": 400000}}}
        if path.startswith("/api/runs/") and path.endswith("/results"):
            return {"summary": {"path": "/saved/final/record.json", "sha256": "5" * 64}}
        raise AssertionError((port, path))

    return api, config_shas, jobs


class AdoptTests(unittest.TestCase):
    def test_harden_split_preserves_science_and_adopts_manual_final_prep(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "controller/plan.json"
            state_path = root / "controller/state.json"
            plan = {"schema": 1, "experiment_id": "S2-P19", "revision": 1,
                    "terminal": False, "scientific_commit": SCIENCE,
                    "tasks": [
                        {"id": "prepare_mothers_R1_20424", "phase": "preparation", "port": 20424,
                         "kind": "milestone", "requires": []},
                        {"id": "prepare_final_20424", "phase": "preparation", "port": 20424,
                         "kind": "prepare", "requires": ["prepare_mothers_R1_20424"],
                         "request_id": "s2p19-prepare-final-20424", "disk_class": "final",
                         "body": {"preparation": {"preparation_type": "final"}}},
                        {"id": "i0_scientific_gate", "phase": "engineering_gate", "port": 20418,
                         "kind": "milestone", "requires": []},
                        {"id": "I0_pilot_20418", "phase": "pilot", "port": 20418,
                         "kind": "improvement", "requires": ["i0_scientific_gate"],
                         "request_id": "s2p19-i0-train-51001", "disk_class": "I0_pilot",
                         "body": {"mode": "train"}},
                    ]}
            plan_path.parent.mkdir(parents=True)
            controller.write_json(plan_path, plan)
            old_sha = controller.digest(plan_path)
            controller.write_json(state_path, {"schema": 1, "experiment_id": "S2-P19",
                "scientific_commit": SCIENCE, "plan_sha256": old_sha,
                "plan_revision": 1, "task_hashes": controller._task_hashes(plan["tasks"]),
                "tasks": {"prepare_mothers_R1_20424": {"state": "completed"}},
                "events": [{"event": "adopted_existing_only"}]})
            final = {"run_id": "prepare_final_existing", "request_id": "s2p19-prepare-final-20424",
                     "git_commit": SCIENCE, "config_sha256": "a" * 64}
            (root / "run_manifest.json").write_text(json.dumps({"tasks": {"final880_preparation": final}}),
                                                    encoding="utf-8")

            def api(port, path):
                if path.endswith("/results"):
                    return {"summary": {"outputs": {"final": {
                        "count": 880, "path": "/archive/final/record.json", "sha256": "b" * 64}}}}
                return {"job": {"run_id": final["run_id"], "request_id": final["request_id"],
                                "git_commit": SCIENCE, "kind": "prepare"},
                        "status": {"stage": "completed", "env_step": 880,
                                   "heartbeat_at": "2026-10-07T00:00:00Z"}}

            runtime = "13ea014f1c1709c4a82059fc89ba3092e75010e1"
            with (patch.object(plan_builder.ops, "LOCAL", root),
                  patch.object(plan_builder.ops, "api", side_effect=api) as api_mock,
                  patch.object(plan_builder.controller, "PLAN", plan_path),
                  patch.object(plan_builder.controller, "STATE", state_path),
                  patch.object(plan_builder.controller, "_service_identity", return_value={}),
                  patch.object(plan_builder.controller, "_adopted_config_sha", return_value="a" * 64)):
                result = plan_builder.harden_live_plan(runtime)
            self.assertEqual(result["service_commit"], runtime)
            new_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            new_state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(new_plan["science_origin_commit"], SCIENCE)
            self.assertEqual(new_plan["service_commit"], runtime)
            self.assertEqual(new_plan["parent_plan_sha256"], old_sha)
            self.assertEqual(new_plan["revision"], 2)
            final_task = next(row for row in new_plan["tasks"] if row["id"] == "prepare_final_20424")
            self.assertTrue(final_task["adopted_only"])
            self.assertNotIn("body", final_task)
            self.assertEqual(final_task["adopted_run_id"], final["run_id"])
            self.assertEqual(new_state["tasks"]["prepare_final_20424"]["state"], "completed")
            self.assertEqual(new_state["service_commit"], runtime)
            self.assertFalse(any(len(call.args) > 2 and call.args[2] == "POST"
                                 for call in api_mock.call_args_list))

    def test_preview_checks_live_identity_without_publishing_plan(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            api, hashes, _ = _fixture(root)
            with (patch.object(plan_builder.ops, "LOCAL", root),
                  patch.object(plan_builder.ops, "api", side_effect=api),
                  patch.object(plan_builder.controller, "PLAN", root / "controller/plan.json"),
                  patch.object(plan_builder.controller, "STATE", root / "controller/state.json"),
                  patch.object(plan_builder.controller, "_service_identity", return_value={}),
                  patch.object(plan_builder.controller, "_adopted_config_sha",
                               side_effect=lambda port, run_id: hashes[run_id])):
                result = plan_builder.adopt(publish=False)
            self.assertEqual(result["status"], "adopt_preview_no_dispatch_no_plan_written")
            self.assertEqual(result["scientific_commit"], SCIENCE)
            self.assertFalse((root / "controller/plan.json").exists())
            self.assertFalse((root / "controller/state.json").exists())

    def test_adopts_exact_jobs_and_excludes_old_failed_mother(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            api, hashes, _ = _fixture(root)
            with (patch.object(plan_builder.ops, "LOCAL", root),
                  patch.object(plan_builder.ops, "api", side_effect=api) as api_mock,
                  patch.object(plan_builder.controller, "PLAN", root / "controller/plan.json"),
                  patch.object(plan_builder.controller, "STATE", root / "controller/state.json"),
                  patch.object(plan_builder.controller, "_service_identity", return_value={}),
                  patch.object(plan_builder.controller, "_adopted_config_sha",
                               side_effect=lambda port, run_id: hashes[run_id])):
                result = plan_builder.adopt()
            self.assertEqual(result["status"], "adopted_no_dispatch")
            self.assertFalse(any(call.args[2] == "POST" for call in api_mock.call_args_list
                                 if len(call.args) > 2))
            plan = json.loads((root / "controller/plan.json").read_text(encoding="utf-8"))
            state = json.loads((root / "controller/state.json").read_text(encoding="utf-8"))
            tasks = {row["id"]: row for row in plan["tasks"]}
            for name in ("C0_pilot_20418", "C1_pilot_20424", "i0_development_gate_20418",
                         "i0_dev_B_greedy_20418", "i0_dev_B_rule_20418", "i0_dev_B_random_20418"):
                self.assertTrue(tasks[name]["adopted_only"])
                self.assertNotIn("body", tasks[name])
                self.assertEqual(tasks[name]["adopted_run_id"], state["tasks"][name]["run_id"])
            self.assertEqual(tasks["prepare_mothers_R1_20424"]["expected_commit"], R1)
            self.assertEqual(plan["scientific_commit"], SCIENCE)
            self.assertFalse(any("old_failed" in row["id"] for row in plan["tasks"]))
            self.assertEqual(tasks["I0_pilot_20418"]["requires"][0], "i0_scientific_gate")

    def test_adopt_rejects_request_id_rebound_to_another_run(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            api, hashes, jobs = _fixture(root)
            jobs[20418][0]["request_id"] = "s2p19-saved-c0"  # Two live jobs claim C0's request.
            with (patch.object(plan_builder.ops, "LOCAL", root),
                  patch.object(plan_builder.ops, "api", side_effect=api),
                  patch.object(plan_builder.controller, "PLAN", root / "controller/plan.json"),
                  patch.object(plan_builder.controller, "STATE", root / "controller/state.json"),
                  patch.object(plan_builder.controller, "_service_identity", return_value={}),
                  patch.object(plan_builder.controller, "_adopted_config_sha",
                               side_effect=lambda port, run_id: hashes[run_id])):
                with self.assertRaises(RuntimeError):
                    plan_builder.adopt()
            self.assertFalse((root / "controller/plan.json").exists())

    def test_missing_adopted_run_never_posts_new_training(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = {"schema": 1, "experiment_id": "S2-P19", "revision": 1,
                    "terminal": False, "scientific_commit": SCIENCE,
                    "tasks": [{"id": "C0_pilot_20418", "phase": "pilot", "port": 20418,
                               "kind": "constructive_train", "requires": [], "adopted_only": True,
                               "adopted_run_id": "train_sac_missing", "request_id": "s2p19-c0-train-51001",
                               "expected_config_sha256": "a" * 64, "expected_commit": SCIENCE,
                               "workbench_kind": "train"}]}
            path = root / "controller/plan.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(plan), encoding="utf-8")
            disk = {"observed_utc": "2026-10-07T00:00:00Z", "archive_free_bytes": 100 * disk_guard.GIB,
                    "mount_identity": "nfs|/home/user|nfs", "root": "/group/20418",
                    "s2p19_root_bytes": 0}
            with (patch.object(controller, "PLAN", path),
                  patch.object(controller, "STATE", root / "controller/state.json"),
                  patch.object(controller, "_service_identity", return_value={}),
                  patch.object(controller, "_run_rows", return_value=[]),
                  patch.object(controller, "_scene_rows", return_value=[]),
                  patch.object(controller, "_model_rows", return_value=[]),
                  patch.object(controller.disk_guard, "sample", return_value=disk),
                  patch.object(controller.ops, "api") as api_mock):
                result = controller.step(dispatch=True)
            self.assertEqual(result["tasks"]["C0_pilot_20418"]["state"], "attention_required")
            api_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
