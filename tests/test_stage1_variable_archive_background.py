"""Bounded S1-V10 archive handoff records one owner and never marks final."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.stage1_variable import archive_background


class ArchiveBackgroundTests(unittest.TestCase):
    def test_bounded_batch_saves_pid_birth_and_nonfinal_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = {"archived_new": 1, "already_archived": 4,
                      "remaining_audited": 0, "tasks": {},
                      "all_controller_tasks_audited": False}
            with patch.object(archive_background, "ROOT", root), \
                 patch.object(archive_background, "LOCK", root / "one.lock"), \
                 patch.object(archive_background, "STATUS", root / "status.json"), \
                 patch.object(archive_background, "RECEIPT", root / "receipt.json"), \
                 patch.object(archive_background, "_source_gate", return_value="a" * 40), \
                 patch.object(archive_background.archive_jobs, "archive_all_completed",
                              return_value=result) as archive:
                receipt = archive_background.run("a" * 40, 32)
            archive.assert_called_once()
            self.assertEqual(archive.call_args.kwargs["limit"], 32)
            self.assertEqual(receipt["stage"], "completed")
            self.assertFalse(receipt["formal_final_archive_closed"])
            self.assertGreater(receipt["birth"], 0)
            saved = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["pid"], receipt["pid"])
            self.assertEqual(json.loads((root / "status.json").read_text())["stage"],
                             "completed")

    def test_failure_keeps_status_and_does_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(archive_background, "ROOT", root), \
                 patch.object(archive_background, "LOCK", root / "one.lock"), \
                 patch.object(archive_background, "STATUS", root / "status.json"), \
                 patch.object(archive_background, "RECEIPT", root / "receipt.json"), \
                 patch.object(archive_background, "_source_gate", return_value="a" * 40), \
                 patch.object(archive_background.archive_jobs, "archive_all_completed",
                              side_effect=RuntimeError("immutable member mismatch")):
                with self.assertRaisesRegex(RuntimeError, "immutable member mismatch"):
                    archive_background.run("a" * 40, 32)
            saved = json.loads((root / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["stage"], "failed")
            self.assertFalse((root / "receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
