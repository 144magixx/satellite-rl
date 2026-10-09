"""D12A archive identity and streaming ZIP evidence checks; no SSH calls."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.stage1_drl_r04 import archive
from scripts.stage1_drl_r04.report import Evidence


def volume(directory, filename, contents):
    path = Path(directory) / filename
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as out:
        for name, data in contents.items():
            out.writestr(name, data)
    return {"file": filename, "bytes": path.stat().st_size,
            "sha256": archive.digest(path),
            "members": {name: {"bytes": len(data),
                                "sha256": hashlib.sha256(data).hexdigest()}
                        for name, data in contents.items()}}


class ArchiveTests(unittest.TestCase):
    def test_plan_combines_controller_and_three_preparation_tasks(self):
        commit = "a" * 40
        state = {"experiment_id": "S1-D12A", "scientific_commit": commit,
                 "tasks": {"P0": {"port": 20418, "run_id": "train-a",
                                  "request_id": "req-a", "kind": "train", "variant": "P0"}}}
        prep = {"experiment_id": "S1-D12A", "scientific_commit": commit,
                "tasks": {day: {"port": port, "role": role, "run_id": f"prep-{port}",
                                 "request_id": f"req-{port}", "git_commit": commit}
                          for port, (role, day) in archive.ASSIGNMENT.items()}}
        result = archive.plan(state, prep)
        self.assertEqual(len(result["tasks"]), 4)
        self.assertEqual(result["tasks"]["prepare_2024-02-07"]["port"], 20473)
        broken = json.loads(json.dumps(prep))
        broken["tasks"].pop("2024-02-09")
        with self.assertRaisesRegex(ValueError, "three official"):
            archive.plan(state, broken)

    def test_member_stream_hash_rejects_tamper_and_unsafe_names(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = volume(directory, "part.zip", {"20418/run-a/job.json": b"{}"})
            self.assertEqual(archive.verify_volume(Path(directory) / "part.zip", expected), 1)
            changed = json.loads(json.dumps(expected))
            changed["members"]["20418/run-a/job.json"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "member SHA"):
                archive.verify_volume(Path(directory) / "part.zip", changed)
            unsafe = volume(directory, "unsafe.zip", {"../escape": b"bad"})
            with self.assertRaisesRegex(ValueError, "unsafe"):
                archive.verify_volume(Path(directory) / "unsafe.zip", unsafe)

    def test_preparation_source_ref_requires_one_verified_shared_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobdir, sourcedir = root / "job", root / "sources"
            jobdir.mkdir(); sourcedir.mkdir()
            prefix = "20473/prepare-1/"
            part = volume(jobdir, "job.zip", {prefix + name: b"{}" for name in
                                               ("job.json", "config.json", "status.json")})
            receipt = {"experiment_id": "S1-D12A", "port": 20473,
                       "run_id": "prepare-1", "volumes": [part],
                       "source_refs": [{"source_bundle_member":
                                        "shared_sources/2024-02-07/day.tar",
                                        "sha256": hashlib.sha256(b"raw").hexdigest(),
                                        "bytes": 3,
                                        "source_preflight_sha256": hashlib.sha256(b"pin").hexdigest()}]}
            with self.assertRaisesRegex(ValueError, "receipt required"):
                archive.verify_receipt(receipt, jobdir)
            files = {}
            for day in ("2024-02-07", "2024-02-08", "2024-02-09"):
                files[f"shared_sources/{day}/day.tar"] = b"raw"
                files[f"shared_sources/{day}/source_preflight.json"] = b"pin"
            source_part = volume(sourcedir, "shared_sources.zip", files)
            source_receipt = {"experiment_id": "S1-D12A", "run_id": "shared_sources",
                              "volumes": [source_part]}
            self.assertTrue(archive.verify_receipt(receipt, jobdir,
                                                   source_receipt, sourcedir)["closed"])
            broken = json.loads(json.dumps(source_receipt))
            broken["volumes"][0]["members"]["shared_sources/2024-02-07/day.tar"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "member SHA"):
                archive.verify_receipt(receipt, jobdir, broken, sourcedir)

    def test_remote_programs_compile_without_contacting_server(self):
        compile(archive.REMOTE_BUILD, "remote_job_archive", "exec")
        compile(archive.REMOTE_SOURCE_BUILD, "remote_shared_source", "exec")

    def test_archive_zip_directory_is_direct_report_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / "20418" / "run-a"
            job.mkdir(parents=True)
            volume(job, "job_000.zip", {"20418/run-a/job.json": b'{"run_id":"run-a"}',
                                        "20418/run-a/results.jsonl": b'{"row":1}\n'})
            evidence = Evidence(root)
            try:
                result, error = evidence.job_json(20418, "run-a", "job.json")
                self.assertIsNone(error)
                self.assertEqual(result["run_id"], "run-a")
                self.assertEqual(list(evidence.job_iter_jsonl(20418, "run-a", "results.jsonl")),
                                 [{"row": 1}])
            finally:
                evidence.close()


if __name__ == "__main__":
    unittest.main()
