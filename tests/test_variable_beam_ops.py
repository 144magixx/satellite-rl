"""Training mother identity cannot hide a reused source timestamp."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

OPERATIONS = Path(__file__).resolve().parents[1] / "scripts" / "variable_beams"
sys.path.insert(0, str(OPERATIONS))
from controller import validate_train_mother_times  # noqa: E402
import remote_server  # noqa: E402


class TrainingMotherTimeTests(unittest.TestCase):
    def test_one_utc_instant_per_distinct_mother(self):
        records = [
            {"mother_scene_id": "old", "selected_time_utc": "2024-01-08T00:00:00Z"},
            {"mother_scene_id": "old", "selected_time_utc": "2024-01-08T00:00:00.000+00:00"},
            {"mother_scene_id": "new", "selected_time_utc": "2024-01-09T00:00:00Z"},
        ]
        validate_train_mother_times(records, 2)
        with self.assertRaisesRegex(RuntimeError, "复用"):
            validate_train_mother_times(records + [
                {"mother_scene_id": "copied", "selected_time_utc": "2024-01-08T00:00:00+00:00"}], 3)
        with self.assertRaisesRegex(RuntimeError, "多个"):
            validate_train_mother_times(records + [
                {"mother_scene_id": "old", "selected_time_utc": "2024-01-08T00:01:00Z"}], 2)

    def test_rejects_wrong_day_naive_time_and_missing_mother(self):
        for record in (
                {"mother_scene_id": "new", "selected_time_utc": "2024-01-10T00:00:00Z"},
                {"mother_scene_id": "new", "selected_time_utc": "2024-01-08T00:00:00"},
                {"mother_scene_id": "", "selected_time_utc": "2024-01-08T00:00:00Z"}):
            with self.subTest(record=record), self.assertRaises(RuntimeError):
                validate_train_mother_times([record], 1)


class WorkbenchModeSwitchTests(unittest.TestCase):
    def test_training_and_warmup_block_switch_to_one_worker(self):
        commit = "a" * 40
        for stage in ("warmup", "training"):
            with self.subTest(stage=stage), TemporaryDirectory() as temp:
                launch = Path(temp) / "launch.json"
                # ``ensure_server`` compares the live ``/system`` payload against the
                # launch receipt, resolving both sides with ``Path.resolve()``. On
                # Windows a POSIX-style constant such as ``/home/...`` resolves to a
                # drive-relative path, so the fixture must publish the *resolved*
                # values; otherwise the identity check fails before the test reaches
                # the "active or queued" branch it is meant to cover.
                repo = Path(remote_server.REPO).resolve()
                archive = Path(remote_server.ARCHIVE).resolve()
                package_file = repo / "src/satellite_rl/__init__.py"
                launch.write_text(
                    json.dumps({"commit": commit, "pid": 123, "start_ticks": 456,
                                "max_workers": 2, "repo": str(repo),
                                "package_file": str(package_file)}),
                    encoding="utf-8")
                system = {"source_root": str(repo), "archive_root": str(archive),
                          "package_file": str(package_file), "cuda": True}

                def api(path):
                    return system if path == "/system" else {"runs": [{"stage": stage}]}

                with (patch.object(remote_server, "LAUNCH", launch),
                      patch.object(remote_server, "REPO", repo),
                      patch.object(remote_server, "ARCHIVE", archive),
                      patch.object(remote_server, "preflight", return_value={}),
                      patch.object(remote_server, "port_open", return_value=True),
                      patch.object(remote_server, "proc_birth", return_value=456),
                      patch.object(remote_server, "api", side_effect=api),
                      patch.object(remote_server.os, "kill") as kill):
                    with self.assertRaisesRegex(RuntimeError, "active or queued"):
                        remote_server.ensure_server(commit, "evaluation")
                    kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
