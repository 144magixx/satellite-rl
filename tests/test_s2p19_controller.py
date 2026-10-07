"""S2-P19 controller contract checks without remote dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.s2p19 import controller, disk_guard


class ControllerContractTests(unittest.TestCase):
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
                    "home_free_bytes": first + disk_guard.RESERVE_BYTES + 1024}
        self.assertTrue(disk_guard.gate(observed, "final_B_one_stochastic_arm_2640")["admit"])
        self.assertFalse(disk_guard.gate(observed, "final_B_one_stochastic_arm_2640", first)["admit"])


if __name__ == "__main__":
    unittest.main()
