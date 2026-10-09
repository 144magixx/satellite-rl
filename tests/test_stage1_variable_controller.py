"""S1-V10 four-lane graph and handoff manifest checks."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.stage1_variable import controller


class Stage1VariableControllerTests(unittest.TestCase):
    def test_graph_has_exact_authorized_routes_budgets_and_sequence_pair(self):
        tasks = controller.TASKS
        self.assertEqual(len(tasks), 28)
        self.assertEqual({tasks[f"dev_R{i:02d}"]["route"] for i in
                          (1, 2, 3, 4, 5, 6, 7, 10, 11, 12)},
                         {f"R{i:02d}" for i in (1, 2, 3, 4, 5, 6, 7, 10, 11, 12)})
        self.assertEqual({tasks[key]["budget"] for key in ("verify_R08", "verify_R09")}, {2048})
        self.assertEqual({tasks[key]["budget"] for key in ("train_R08", "train_R09")}, {262144})
        self.assertEqual(tasks["sequence_R04"]["port"], tasks["sequence_R12"]["port"])
        self.assertEqual(tasks["sequence_R12"]["expected_count"], 360)
        self.assertEqual(tasks["test_R12"]["expected_count"], 300)
        self.assertTrue(tasks["test_R12"]["cold_init"])
        self.assertEqual(set(tasks), set(controller.DEVELOPMENT + controller.MAIN +
                                         list(controller.LEARNED_FINAL)))

    def test_manifest_is_compact_single_handoff_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "run_manifest.json"
            state = {"scientific_commit": "a" * 40, "phase": "development",
                     "need_attention": {}, "observed_utc": "2026-10-09T01:00:00+00:00",
                     "dataset_sha256": {"test": "b" * 64},
                     "tasks": {"dev_R01": {"run_id": "evaluate_1", "request_id": "s1v10_dev_R01",
                                            "port": 20473, "role": "development", "route": "R01",
                                            "kind": "evaluate", "purpose": "verification",
                                            "last_observed_utc": "2026-10-09T01:00:00+00:00"}},
                     "audits": {}, "last_error": None}
            with patch.object(controller, "RUN_MANIFEST", target):
                controller.write_run_manifest(state)
            value = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(value["experiment_id"], "S1-V10")
            self.assertEqual(value["scientific_commit"], "a" * 40)
            self.assertEqual(value["data_sha256"]["test"], "b" * 64)
            self.assertEqual(value["tasks"]["dev_R01"]["port"], 20473)
            self.assertEqual(value["budget"]["formal_total_decisions"], 524288)
            self.assertEqual(value["followup_id"], "s1-v10")


if __name__ == "__main__":
    unittest.main()
