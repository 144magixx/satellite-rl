"""Exercise the remote source-builder program locally with frozen small assets."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.stage1_drl_r04 import archive


class SourceArchiveExecutionTests(unittest.TestCase):
    def test_model_endpoints_keep_distinct_relative_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder).resolve()
            store = base / "20418/archive"
            run = "train_one"
            jobdir = store / "outputs/workbench/jobs" / run
            jobdir.mkdir(parents=True)
            source = store / "source"
            source.mkdir()
            (source / "source_manifest.json").write_text(json.dumps({"git_commit": "a" * 40, "files": {}}))
            models = store / "models" / run
            for endpoint in ("final", "pilot_endpoint"):
                p = models / endpoint
                p.mkdir(parents=True)
                (p / "policy.pt").write_bytes(endpoint.encode())
                (p / "model.json").write_text(json.dumps({"selection": endpoint}))
            job = {"run_id": run, "request_id": "one", "experiment_id": "S1-D12A",
                   "git_commit": "a" * 40, "archive_root": str(store), "variant": "P0",
                   "source_root": str(source), "model_dir": str(models)}
            for name, value in {"job.json": job, "config.json": {"datasets": {}},
                                "status.json": {"run_id": run, "stage": "completed"},
                                "launch.json": {}}.items():
                (jobdir / name).write_text(json.dumps(value))
            request = {"port": 20418, "run_id": run, "request_id": "one", "variant": "P0",
                       "base": str(base), "archive": str(store), "source_commit": "a" * 40,
                       "destination": str(base / "export"), "preparation_day": None,
                       "max_files": 100, "max_bytes": 1 << 20, "reserve": archive.RESERVE,
                       "part_bytes": 1 << 20}
            output = io.StringIO()
            with patch("shutil.disk_usage", return_value=type("Space", (), {"free": 1 << 40})()), contextlib.redirect_stdout(output):
                exec(compile(archive.REMOTE_BUILD, "job_builder", "exec"), {"REQUEST_JSON": json.dumps(request)})
            receipt = json.loads(output.getvalue())
            model_volume = next(v for v in receipt["volumes"] if v["file"] == "models.zip")
            with zipfile.ZipFile(model_volume["remote_path"]) as packed:
                for endpoint in ("final", "pilot_endpoint"):
                    self.assertEqual(packed.read(f"models/20418/{run}/model_dir/{endpoint}/policy.pt"), endpoint.encode())

    def test_three_source_volumes_are_built_after_capacity_check(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            entries = []
            for port, (role, day) in archive.ASSIGNMENT.items():
                run = "prepare_" + str(port)
                source = base / "source" / role / day
                source.mkdir(parents=True)
                payload = (day * 400).encode()
                name = day + ".tar"
                (source / name).write_bytes(payload)
                pin = source / "source_preflight.json"
                pin.write_text(json.dumps({"archive_name": name, "expected_bytes": len(payload)}))
                root = base / str(port) / "archive/outputs/workbench/jobs" / run
                root.mkdir(parents=True)
                values = {"job.json": {"experiment_id": "S1-D12A", "run_id": run,
                          "git_commit": "a" * 40, "source_date": day,
                          "dataset_preparation": {"archive_name": name, "files": {
                              "source_preflight.json": {"sha256": archive.digest(pin)}}}},
                          "status.json": {"stage": "completed", "pid": None},
                          "launch.json": {"pid": None},
                          "artifacts.json": {"raw_archive_sha256": hashlib.sha256(payload).hexdigest()}}
                for filename, value in values.items():
                    (root / filename).write_text(json.dumps(value))
                entries.append({"port": port, "role": role, "day": day, "run_id": run})
            dest = base / "bundle"
            request = {"base": str(base), "destination": str(dest),
                       "source_commit": "a" * 40, "preparations": entries,
                       "reserve": archive.RESERVE, "builder_ssh_port": 20473}
            output = io.StringIO()
            free = type("Space", (), {"free": 1 << 40})()
            with patch("shutil.disk_usage", return_value=free), contextlib.redirect_stdout(output):
                exec(compile(archive.REMOTE_SOURCE_BUILD, "source_builder", "exec"),
                     {"REQUEST_JSON": json.dumps(request)})
            receipt = json.loads(output.getvalue())
            self.assertEqual(len(receipt["volumes"]), 3)
            self.assertEqual(receipt["builder_ssh_port"], 20473)
            self.assertFalse((dest / ".lock").exists())
            for volume in receipt["volumes"]:
                self.assertEqual(archive.verify_volume(volume["remote_path"], volume), 2)
                with zipfile.ZipFile(volume["remote_path"]) as packed:
                    self.assertTrue(all(x.compress_type == zipfile.ZIP_DEFLATED
                                        for x in packed.infolist()))

if __name__ == "__main__":
    unittest.main()
