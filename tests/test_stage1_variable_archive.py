"""Final S1-V10 archive lanes cannot overwrite one another."""
import unittest
from pathlib import Path
import json
import tempfile
from unittest.mock import patch

from scripts.stage1_variable.archive_inventory import bundle_local_path, fetch_completed_analysis
from scripts.stage1_variable import archive_inventory, archive_jobs, controller


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

    def test_completed_only_archive_delegates_without_final_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "controller" / "state.json"
            state.parent.mkdir()
            state.write_text(json.dumps({"scientific_commit": "a" * 40,
                                         "phase": "execution", "tasks": {},
                                         "audits": {}}))
            with patch.object(controller, "STATE", state), \
                 patch.object(archive_inventory, "STATE", state), \
                 patch.object(archive_jobs, "archive_all_completed",
                              return_value={"archived": 0, "tasks": {}}) as archive:
                result = fetch_completed_analysis("a" * 40)
            self.assertEqual(result["archived"], 0)
            archive.assert_called_once_with()

    def test_lane_wide_final_bundle_is_refused(self):
        with patch.object(archive_inventory, "_state", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "job ZIP64 packages"):
                archive_inventory.bundle_download("a" * 40)

    def test_shared_raw_source_chooses_disk_from_actual_zip_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            droot, croot = base / "D", base / "C"
            gib = 1 << 30
            def usage(path):
                amount = 12 * gib if Path(path).is_relative_to(droot) else 14 * gib
                return type("Usage", (), {"free": amount})()
            remote = {"bytes": 5 * gib, "sha256": "a" * 64,
                      "path": "/remote/one_raw_source.zip"}
            with patch.object(archive_inventory, "LOCAL", droot), \
                 patch.object(archive_jobs, "BACKUP_LOCAL", croot), \
                 patch.object(archive_inventory.shutil, "disk_usage", side_effect=usage):
                target, gate = archive_inventory.source_bundle_target("b" * 40, remote)
            self.assertTrue(target.is_relative_to(croot))
            self.assertEqual(gate["missing_compressed_bytes"], 5 * gib)
            self.assertEqual(gate["reserved_bytes_each_drive"], 8 * gib)


if __name__ == "__main__":
    unittest.main()
