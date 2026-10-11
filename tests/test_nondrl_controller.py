"""Controller publication and recovery tests use a deterministic fake transport."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.stage1_nondrl12.controller import Controller, validate_plan
from satellite_rl.workbench.coverage_nondrl import METHODS


def inputs():
    development = {"01": "O", "02": "T", "03": "HR", "04": "F", "05": "rule",
                   **{f"{i:02d}": "main" for i in range(6, 13)}}
    plan = {"experiment_id": "S1-ND12", "scientific_commit": "1" * 40,
            "methods": {r: ["main"] for r in METHODS},
            "gate_thresholds": {r: {"development_methods": [m], "primary_method": m,
                "minimum_improved_mothers": 6, "relative_cv_improvement": .01} for r, m in development.items()}}
    roles = {r: {"id": r + "_fixture", "sha256": str(i) * 64, "path": "/fixture/" + r,
                 "count": 20} for i, r in enumerate(("train", "development", "validation", "test"), 1)}
    registry = {"experiment_id": "S1-ND12", "scientific_commit": "1" * 40,
                "partition_manifest_sha256": "a" * 64,
                "lanes": {str(p): deepcopy(roles) for p in (20418, 20424, 20473, 20474)}}
    return plan, registry


class FakeTransport:
    def __init__(self):
        self.requests, self.jobs, self.uploads = [], {}, []
        self.resume_error = False

    def api(self, port, path, body=None):
        self.requests.append((port, path, body))
        if path.endswith("/resume"):
            if self.resume_error:
                raise ValueError("no recoverable checkpoint")
            return {"stage": "queued"}
        if body is None:
            return {"runs": [r for (p, _), r in self.jobs.items() if p == port]}
        request = body["request_id"]
        key = (port, request)
        if key not in self.jobs:
            self.jobs[key] = {"run_id": "run_" + str(len(self.jobs)), "git_commit": "1" * 40,
                              "stage": "queued", "scientific_stop": False}
        return self.jobs[key]

    def upload(self, path, remote, port):
        self.uploads.append((path, remote, port))


class NondrlControllerTests(unittest.TestCase):
    def test_fails_closed_on_changed_registry_or_learning_before_gate(self):
        plan, registry = inputs()
        validate_plan(plan, registry)
        registry["lanes"]["20424"]["test"]["sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            validate_plan(plan, registry)
        plan, registry = inputs()
        plan["gate_thresholds"]["01"]["development_methods"] = ["MO"]
        plan["gate_thresholds"]["01"]["primary_method"] = "MO"
        with self.assertRaises(ValueError):
            validate_plan(plan, registry)

    def test_task_retry_is_idempotent_and_body_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, registry = inputs()
            transport = FakeTransport()
            controller = Controller(tmp, plan, registry, transport)
            body = controller.body("06", "I0", "prepare", "development", 20418,
                                   preparation_mode="initial", scene_indices=[0, 4, 8, 12, 16])
            first = controller.ensure("initial/development/20418", 20418, body)
            second = controller.ensure("initial/development/20418", 20418, body)
            self.assertEqual(first["run_id"], second["run_id"])
            self.assertEqual(len(transport.jobs), 1)
            bad = deepcopy(body)
            bad["scene_indices"] = [1]
            with self.assertRaises(ValueError):
                controller.ensure("initial/development/20418", 20418, bad)
            restored = Controller(tmp, plan, registry, transport)
            self.assertEqual(restored.state["jobs"], controller.state["jobs"])

    def test_failed_resume_needs_attention_science_stop_never_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, registry = inputs()
            transport = FakeTransport()
            controller = Controller(tmp, plan, registry, transport)
            body = controller.body("06", "I0", "prepare", "development", 20418, preparation_mode="initial")
            controller.ensure("initial/development/20418", 20418, body)
            list(transport.jobs.values())[0].update(stage="stopped", scientific_stop=True, stop_reason="measured_gate_failure")
            self.assertTrue(controller.refresh())
            self.assertEqual(sum(p.endswith("/resume") for _, p, _ in transport.requests), 0)
            list(transport.jobs.values())[0].update(stage="failed", scientific_stop=False)
            transport.resume_error = True
            self.assertFalse(controller.refresh())
            self.assertEqual(controller.state["phase"], "attention_required")

    def test_test_dispatch_serializes_timing_per_lane(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, registry = inputs()
            plan["methods"]["06"] = ["main", "assign_only"]
            transport = FakeTransport()
            controller = Controller(tmp, plan, registry, transport)
            controller.state["phase"] = "test"
            controller.state["routes"] = {r: {"passed": r == "06"} for r in METHODS}
            controller.testing()
            self.assertEqual(len(transport.jobs), 4)
            controller.testing()
            self.assertEqual(len(transport.jobs), 4)
            for row in transport.jobs.values():
                row["stage"] = "completed"
            controller.refresh()
            controller.testing()
            self.assertEqual(len(transport.jobs), 8)

    def test_gate_uses_twenty_mothers_and_missing_mechanism_is_attention(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, registry = inputs()
            plan["gate_thresholds"]["05"]["required_mechanism"] = "oracle advantage must be measured"
            transport = FakeTransport()
            controller = Controller(tmp, plan, registry, transport)
            rows = [{"index": i, "multiplier": m, "mother_id": str(i), "cv": .9, "initial_cv": 1.,
                     "carry": .8, "initial_carry": .8, "feasible": True}
                    for i in range(20) for m in (.75, 1., 1.25)]
            controller.summaries = lambda _: [{"rows": rows, "source_commit": "1" * 40,
                                               "manifest_sha256": "e" * 64}]
            gate = controller.make_gate("06")
            self.assertTrue(gate["passed"])
            self.assertEqual(gate["mother_count"], 20)
            self.assertEqual(gate["actual_row_count"], 60)
            self.assertEqual(len(gate["improved_mothers"]), 20)
            missing = controller.make_gate("05")
            self.assertIsNone(missing)
            self.assertEqual(controller.state["phase"], "attention_required")
            self.assertNotEqual(controller.state.get("stop_reason"), "science_stopped")


if __name__ == "__main__":
    unittest.main()
