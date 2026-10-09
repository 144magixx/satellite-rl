"""Final S1-V10 archive lanes cannot overwrite one another."""
import unittest
from pathlib import Path
import json
import tempfile
from unittest.mock import patch

from scripts.stage1_variable.archive_inventory import bundle_local_path, fetch_completed_analysis
from scripts.stage1_variable import archive_inventory, controller


class VariableArchiveTests(unittest.TestCase):
    def test_two_lanes_have_distinct_local_zip_targets(self):
        name = "/home/u2024110224/s1_v10/ops/s1v10_final_" + "a" * 40 + ".zip"
        first = bundle_local_path(20418, name)
        second = bundle_local_path(20424, name)
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent.name, "20418")
        self.assertEqual(second.parent.name, "20424")

    def test_unexpected_archive_target_is_rejected(self):
        with self.assertRaises(ValueError):
            bundle_local_path(20473, "/tmp/other.zip")
        with self.assertRaises(ValueError):
            bundle_local_path(99999, "/tmp/s1v10_final_a.zip")

    def test_completed_only_sync_does_not_require_final_archive_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "controller" / "state.json"
            state.parent.mkdir()
            state.write_text(json.dumps({"scientific_commit": "a" * 40,
                                         "phase": "execution", "tasks": {},
                                         "audits": {}}))
            with patch.object(controller, "STATE", state), \
                 patch.object(archive_inventory.ops, "LOCAL", root), \
                 patch.object(archive_inventory, "LOCAL", root / "delivery"):
                result = fetch_completed_analysis("a" * 40)
            self.assertEqual(result["tasks"], 0)
            self.assertFalse(result["formal_final_archive"])
            manifest = json.loads((root / "synced_jobs" / "manifest.json").read_text())
            self.assertEqual(manifest["sync_kind"], "audited_completed_tasks_only")


if __name__ == "__main__":
    unittest.main()
