"""Bounded local DAG checks; all remote/workbench responses are explicit mocks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.stage1_nondrl12.controller import Controller, PORTS, stable_sha
from scripts.stage1_nondrl12.controller_pipeline import PipelineController
from scripts.stage1_nondrl12.launcher_pipeline import load_resume_inputs, verify_registered_roles, sha_bytes
from satellite_rl.workbench.coverage_nondrl import METHODS
from tests.test_nondrl_controller import inputs, FakeTransport


def mark(transport, record, stage, scientific_stop=False):
    record.update(stage=stage, scientific_stop=scientific_stop)
    for row in transport.jobs.values():
        if row["run_id"] == record["run_id"]:
            row.update(stage=stage, scientific_stop=scientific_stop)


def bootstrap(controller, transport, slow_route=None):
    for role in ("train", "development", "validation", "test"):
        count = controller.record(PORTS[0], role)["count"]
        for lane, port in enumerate(PORTS):
            job = controller.ensure(f"initial/{role}/{port}", port,
                controller.body("06", "I0", "prepare", role, port, preparation_mode="initial",
                                scene_indices=list(range(lane, count, 4))))
            mark(transport, job, "completed")
    for route in METHODS:
        count = controller.record(PORTS[0], "development")["count"]
        for method in controller.plan["gate_thresholds"][route]["development_methods"]:
            for lane, port in enumerate(PORTS):
                job = controller.ensure(f"development/{route}/{method}/{port}", port,
                    controller.body(route, method, "evaluate", "development", port,
                                    scene_indices=list(range(lane, count, 4))))
                mark(transport, job, "evaluating" if route == slow_route else "completed")
    controller.state["phase"] = "development"
    controller.save()


def gate(route, passed=True):
    return {"passed": passed, "status": "passed" if passed else "science_stopped",
            "gate_source": {"path": "/mock/gate_" + route, "sha256": "e" * 64}, "mock_gate": True}


class PipelineTests(unittest.TestCase):
    def test_slow10_does_not_block_ready01_or_strong_probe02(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, registry = inputs(); transport = FakeTransport()
            controller = PipelineController(tmp, plan, registry, transport)
            bootstrap(controller, transport, slow_route="10")
            called = []
            def measured(route):
                called.append(route)
                if route == "02":
                    probe = controller.ensure("teacher_probe/02", 20424,
                        controller.body("02", "main", "prepare", "development", 20424, preparation_mode="teacher_probe"))
                    return gate(route) if probe["stage"] == "completed" else None
                return gate(route, passed=route == "01")
            controller.make_gate = measured
            controller.workers_quiet = lambda: True
            controller.tick()
            self.assertNotIn("10", called)
            self.assertIn("teacher/01", controller.state["jobs"])
            self.assertIn("teacher_probe/02", controller.state["jobs"])
            mark(transport, controller.state["jobs"]["teacher/01"], "completed")
            mark(transport, controller.state["jobs"]["teacher_probe/02"], "completed")
            controller.tick()
            self.assertIn("train/01", controller.state["jobs"])
            self.assertIn("teacher/02", controller.state["jobs"])
            mark(transport, controller.state["jobs"]["teacher/02"], "completed")
            controller.tick()
            self.assertIn("train/02", controller.state["jobs"])
            self.assertNotIn("10", controller.state["routes"])
            self.assertFalse(any(k.startswith("test/") for k in controller.state["jobs"]))
            self.assertEqual(controller.state["pipeline_routes"]["10"]["phase"], "development_pending")

    def test_route_evidence_gap_is_attention_not_failure_and_other_train_advances(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, registry = inputs(); transport = FakeTransport()
            controller = PipelineController(tmp, plan, registry, transport)
            bootstrap(controller, transport, slow_route="10")
            def measured(route):
                if route == "02":
                    controller.attention("mock missing closed teacher evidence", route=route)
                    return None
                return gate(route, route == "01")
            controller.make_gate = measured
            controller.tick()
            mark(transport, controller.state["jobs"]["teacher/01"], "completed")
            controller.tick()
            self.assertIn("train/01", controller.state["jobs"])
            self.assertIn("02", controller.state["route_attention"])
            self.assertNotIn("02", controller.state["routes"])
            self.assertNotEqual(controller.state["phase"], "attention_required")
            self.assertFalse(controller.ready_for_test())

    def test_existing_bodies_and_requests_are_identical_after_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, registry = inputs(); transport = FakeTransport()
            old = Controller(tmp, plan, registry, transport)
            bootstrap(old, transport, slow_route="10")
            keys = deepcopy(old.state["jobs"])
            writes = len(transport.jobs)
            new = PipelineController(tmp, plan, registry, transport)
            new.make_gate = lambda _: None
            new.tick()
            self.assertEqual(len(transport.jobs), writes)
            for key, expected in keys.items():
                actual = new.state["jobs"][key]
                self.assertEqual((actual["run_id"], actual["body_sha256"]), (expected["run_id"], expected["body_sha256"]))
            self.assertEqual(new.plan_sha, old.plan_sha)
            self.assertEqual(new.registry_sha, old.registry_sha)

    def test_final_test_waits_for_training_and_process_quiet(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, registry = inputs(); plan["methods"]["06"] = ["main", "BH"]
            transport = FakeTransport(); controller = PipelineController(tmp, plan, registry, transport)
            bootstrap(controller, transport)
            controller.state["routes"] = {r: gate(r, r == "01") for r in METHODS}
            teacher = controller.ensure("teacher/01", 20418,
                controller.body("01", "main", "prepare", "train", 20418, preparation_mode="teacher"))
            mark(transport, teacher, "completed")
            controller.advance_learning()
            controller.workers_quiet = lambda: True
            controller.testing()
            self.assertFalse(any(k.startswith("test/") for k in controller.state["jobs"]))
            mark(transport, controller.state["jobs"]["train/01"], "completed")
            controller.workers_quiet = lambda: False
            controller.testing()
            self.assertFalse(any(k.startswith("test/") for k in controller.state["jobs"]))
            controller.workers_quiet = lambda: True
            controller.testing()
            self.assertEqual(sum(k.startswith("test/") for k in controller.state["jobs"]), 4)
            controller.testing()
            self.assertEqual(sum(k.startswith("test/") for k in controller.state["jobs"]), 4)

    def test_scientific_stop_does_not_retrain_or_skip_final_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, registry = inputs(); transport = FakeTransport()
            controller = PipelineController(tmp, plan, registry, transport)
            bootstrap(controller, transport, slow_route="10")
            controller.make_gate = lambda r: gate(r, r == "01")
            controller.tick()
            teacher = controller.state["jobs"]["teacher/01"]
            mark(transport, teacher, "stopped", scientific_stop=True)
            controller.tick()
            self.assertFalse(controller.state["routes"]["01"]["passed"])
            self.assertEqual(controller.state["routes"]["01"]["stopped_phase"], "teacher")
            self.assertNotIn("train/01", controller.state["jobs"])
            self.assertNotIn("10", controller.state["routes"])
            self.assertFalse(controller.ready_for_test())


class RegistryResumeTests(unittest.TestCase):
    def test_original_registry_bytes_reused_and_changed_timestamp_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); plan, registry = inputs()
            registry["observed_utc"] = "original_registration_time"
            plan_raw = json.dumps(plan, indent=2).encode()
            registry_raw = json.dumps(registry, indent=2).encode()
            (root / "plan.json").write_bytes(plan_raw)
            (root / "dataset_registry.json").write_bytes(registry_raw)
            (root / "state.json").write_text(json.dumps({"experiment_id": "S1-ND12",
                "plan_sha256": stable_sha(plan), "registry_sha256": stable_sha(registry)}))
            (root / "launcher.json").write_text(json.dumps({"scientific_commit": plan["scientific_commit"],
                "phase": "running_workbench_dag", "plan_sha256": sha_bytes(plan_raw)}))
            values, raw, hashes = load_resume_inputs(root / "plan.json", root, plan["scientific_commit"])
            self.assertEqual(raw["registry"], registry_raw)
            self.assertEqual((root / "dataset_registry.json").read_bytes(), registry_raw)
            registry["observed_utc"] = "different_resume_time"
            (root / "dataset_registry.json").write_text(json.dumps(registry))
            with self.assertRaises(ValueError):
                load_resume_inputs(root / "plan.json", root, plan["scientific_commit"])

    def test_registered_roles_verified_with_readonly_api_only(self):
        plan, registry = inputs()
        class Transport:
            calls = []
            def api(self, port, path, body=None):
                self.calls.append((port, path, body))
                return {"datasets": [{**r, "schema_version": 7, "partition_role": role,
                    "partition_manifest_sha256": registry["partition_manifest_sha256"]}
                    for role, r in registry["lanes"][str(port)].items()]}
        transport = Transport()
        verify_registered_roles(registry, transport)
        self.assertEqual(len(transport.calls), 4)
        self.assertTrue(all(body is None and path.endswith("/datasets") for _, path, body in transport.calls))


if __name__ == "__main__":
    unittest.main()
