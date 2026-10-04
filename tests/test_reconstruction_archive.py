"""Bounded S1-03 archive identity and before/after SHA checks."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.stage1_reconstruction import archive_completed as archive


class FakeSftp:
    def __init__(self, files):
        self.files = files

    @contextmanager
    def open(self, path, mode):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield io.BytesIO(self.files[path])

    def get(self, path, target):
        Path(target).write_bytes(self.files[path])


class ArchiveTests(unittest.TestCase):
    def test_active_job_is_read_only_and_not_eligible_for_inventory(self):
        run_id = "evaluate_ppo_20261005_042834_c4be0794"
        directory, _, source_parent = archive.job_paths(20418, run_id)
        source = source_parent + "/" + "e" * 40
        files = {
            directory + "/job.json": json.dumps({"run_id": run_id,
                "git_commit": "e" * 40, "research_stage": "coverage",
                "execution_target": "u2024110224@10.160.17.102:20418",
                "source_root": source, "kind": "evaluate"}).encode(),
            directory + "/status.json": b'{"stage":"evaluating"}',
            directory + "/config.json": b'{"schema_version":4,"research_stage":"coverage",'
                b'"coverage":{"reconstruction":{"route":"R1"}}}',
        }
        self.assertIsNone(archive.completed_job(FakeSftp(files),
            {"science_commit": "e" * 40}, "dev_R1", 20418, run_id))

    def test_transfer_publishes_only_after_matching_remote_second_sha(self):
        run_id = "evaluate_ppo_20261005_042725_c227e926"
        remote_path = archive.job_paths(20418, run_id)[0] + "/job.json"
        data = b'{"run_id":"original"}'
        item = {"port": 20418, "group": "job:" + run_id,
                "root": archive.job_paths(20418, run_id)[0],
                "relative": "job.json", "remote_path": remote_path,
                "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        with tempfile.TemporaryDirectory() as temp:
            local = Path(temp) / "S1-03"
            with (patch.object(archive, "LOCAL", local),
                  patch.object(archive, "LEDGER", local / "archive_ledger.jsonl"),
                  patch.object(archive, "after_hash", return_value={remote_path: {
                      "sha256": item["sha256"], "bytes": item["bytes"]}})):
                events = archive.transfer_group(None, FakeSftp({remote_path: data}),
                    [item], {run_id: {"port": 20418, "label": "dev_R0"}}, {})
                self.assertEqual(len(events), 1)
                self.assertEqual((local / "jobs" / run_id / "job.json").read_bytes(), data)
                self.assertEqual(events[0]["sha256"], item["sha256"])
                self.assertEqual(len((local / "archive_ledger.jsonl").read_text(
                    encoding="utf-8").splitlines()), 1)

    def test_changed_remote_second_sha_never_publishes(self):
        run_id = "evaluate_ppo_20261005_042725_c227e926"
        remote_path = archive.job_paths(20418, run_id)[0] + "/results.json"
        data = b'{"scene_count":20}'
        item = {"port": 20418, "group": "job:" + run_id,
                "root": archive.job_paths(20418, run_id)[0],
                "relative": "results.json", "remote_path": remote_path,
                "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        with tempfile.TemporaryDirectory() as temp:
            local = Path(temp) / "S1-03"
            with (patch.object(archive, "LOCAL", local),
                  patch.object(archive, "LEDGER", local / "archive_ledger.jsonl"),
                  patch.object(archive, "after_hash", return_value={remote_path: {
                      "sha256": "0" * 64, "bytes": item["bytes"]}})):
                with self.assertRaisesRegex(RuntimeError, "after-SHA"):
                    archive.transfer_group(None, FakeSftp({remote_path: data}),
                        [item], {run_id: {"port": 20418, "label": "dev_R0"}}, {})
                self.assertFalse((local / "jobs" / run_id / "results.json").exists())
                self.assertFalse((local / "archive_ledger.jsonl").exists())
                self.assertTrue(list((local / "archive_conflicts").rglob("*.bin")))

    def test_source_manifest_requires_full_matching_sha_inventory(self):
        source = archive.archive_root(20418) + "/outputs/workbench/source_snapshots/" + "e" * 40
        data = b"frozen source"
        digest = hashlib.sha256(data).hexdigest()
        manifest = {"git_commit": "e" * 40, "format_version": 1,
                    "files": {"src/satellite_rl/example.py": digest}}
        encoded = json.dumps(manifest).encode()
        entries = [
            {"root": source, "relative": "source_manifest.json",
             "sha256": hashlib.sha256(encoded).hexdigest()},
            {"root": source, "relative": "src/satellite_rl/example.py",
             "sha256": digest},
        ]
        sftp = FakeSftp({source + "/source_manifest.json": encoded})
        self.assertEqual(archive.verify_source_manifest(sftp, source, entries,
                                                        "e" * 40), manifest)
        entries.pop()
        with self.assertRaisesRegex(ValueError, "complete file inventory"):
            archive.verify_source_manifest(sftp, source, entries, "e" * 40)


if __name__ == "__main__":
    unittest.main()
