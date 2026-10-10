"""D12A archive identity and streaming ZIP evidence checks; no SSH calls."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
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
    def test_canonical_download_creates_index_for_backup_only_volumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backup = root / "C"
            backup.mkdir()
            packed = volume(backup, "part.zip", {"data.json": b"{}"})
            receipt = {"run_id": "one", "volumes": [packed]}
            client = MagicMock()
            with (patch.object(archive, "LOCAL", root / "D"),
                  patch.object(archive, "plan_canonical", return_value={"part.zip": str(backup / "part.zip")}),
                  patch.object(archive.ops, "connect", return_value=client)):
                directory = archive._download(20473, receipt)
            self.assertTrue(directory.is_dir())
            (directory / "receipt.json").write_text(json.dumps(receipt))
            self.assertEqual(packed["canonical_path"], str(backup / "part.zip"))
            self.assertFalse((directory / "part.zip").exists())

    def test_canonical_space_plan_rejects_below_reserve(self):
        receipt = {"run_id": "shared_sources", "volumes": [
            {"file": "day.zip", "bytes": 100, "sha256": "0" * 64}]}
        with tempfile.TemporaryDirectory() as directory:
            with (patch.object(archive, "LOCAL", Path(directory) / "D"),
                  patch.object(archive, "BACKUP", Path(directory) / "C")):
                with self.assertRaisesRegex(RuntimeError, "8GiB reserve"):
                    archive.plan_canonical(20418, receipt, {
                        "local": archive.RESERVE + 99,
                        "backup": archive.RESERVE + 40})

    def test_canonical_plan_keeps_room_for_second_large_volume(self):
        receipt = {"run_id": "task", "volumes": [
            {"file": name, "bytes": size, "sha256": "0" * 64}
            for name, size in (("a.zip", 7), ("b.zip", 4), ("c.zip", 4))]}
        with tempfile.TemporaryDirectory() as directory:
            with (patch.object(archive, "LOCAL", Path(directory) / "D"),
                  patch.object(archive, "BACKUP", Path(directory) / "C")):
                choices = archive.plan_canonical(20418, receipt, {
                    "local": archive.RESERVE + 8,
                    "backup": archive.RESERVE + 7})
        self.assertIn("C", choices["a.zip"])
        self.assertIn("D", choices["b.zip"])
        self.assertIn("D", choices["c.zip"])

    def test_canonical_plan_prefers_extra_and_preserves_existing_d_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            local, backup, extra = (base / name for name in ("D", "C", "E"))
            existing = local / "20418" / "task"
            existing.mkdir(parents=True)
            saved = volume(existing, "old.zip", {"20418/task/old.json": b"old"})
            receipt = {"run_id": "task", "volumes": [
                saved, {"file": "new.zip", "bytes": 100, "sha256": "0" * 64}]}
            with (patch.object(archive, "LOCAL", local),
                  patch.object(archive, "BACKUP", backup),
                  patch.object(archive, "EXTRA", extra)):
                choices = archive.plan_canonical(20418, receipt, {
                    "local": archive.RESERVE + 200,
                    "backup": archive.RESERVE + 200,
                    "extra": archive.RESERVE + 100})
            self.assertEqual(choices["old.zip"], str((existing / "old.zip").resolve()))
            self.assertEqual(choices["new.zip"], str((extra / "20418" / "task" / "new.zip").resolve()))

    def test_report_reads_three_drive_canonical_zips_and_checks_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            index, backup, extra = (base / "D" / "20418" / "task",
                                    base / "C", base / "E")
            index.mkdir(parents=True)
            backup.mkdir()
            extra.mkdir()
            member = "20418/task/job.json"
            saved_d = volume(index, "status.zip", {"20418/task/status.json": b'{"stage": "completed"}'})
            saved_c = volume(backup, "config.zip", {"20418/task/config.json": b'{"seed": 51001}'})
            saved_e = volume(extra, "job.zip", {member: b'{"ok": true}'})
            saved_c["canonical_path"] = str((backup / "config.zip").resolve())
            saved_e["canonical_path"] = str((extra / "job.zip").resolve())
            (index / "receipt.json").write_text(json.dumps({"volumes": [saved_d, saved_c, saved_e]}))
            evidence = Evidence(base / "D")
            try:
                data, error = evidence.job_json(20418, "task", "job.json")
                self.assertIsNone(error)
                self.assertEqual(data, {"ok": True})
                self.assertEqual(evidence.sha(20418, "task", "job.json")[0],
                                 saved_e["members"][member]["sha256"])
                self.assertEqual(evidence.job_json(20418, "task", "status.json")[0],
                                 {"stage": "completed"})
                self.assertEqual(evidence.job_json(20418, "task", "config.json")[0],
                                 {"seed": 51001})
            finally:
                evidence.close()
            (extra / "job.zip").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "wrong size|SHA differs"):
                Evidence(base / "D")

    def test_canonical_plan_reuses_verified_existing_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            local, backup = Path(directory) / "D", Path(directory) / "C"
            target = backup / "20418" / "shared_sources"
            target.mkdir(parents=True)
            saved = volume(target, "day.zip", {"shared_sources/2024-02-07/day.tar": b"raw"})
            receipt = {"run_id": "shared_sources", "volumes": [saved]}
            with (patch.object(archive, "LOCAL", local),
                  patch.object(archive, "BACKUP", backup)):
                choices = archive.plan_canonical(20418, receipt, {
                    "local": archive.RESERVE,
                    "backup": archive.RESERVE})
            self.assertEqual(choices["day.zip"], str((target / "day.zip").resolve()))
            (target / "day.zip").write_bytes(b"tampered")
            with (patch.object(archive, "LOCAL", local),
                  patch.object(archive, "BACKUP", backup)):
                with self.assertRaisesRegex(ValueError, "ZIP byte SHA"):
                    archive.plan_canonical(20418, receipt, {
                        "local": archive.RESERVE,
                        "backup": archive.RESERVE})

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

    def test_cross_drive_receipts_close_source_refs_and_report_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobdir = root / "D" / "20473" / "prepare-1"
            sourcedir = root / "D" / "20418" / "shared_sources"
            cdir = root / "C" / "20418" / "shared_sources"
            jobdir.mkdir(parents=True)
            sourcedir.mkdir(parents=True)
            cdir.mkdir(parents=True)
            prefix = "20473/prepare-1/"
            part = volume(jobdir, "job.zip", {prefix + name: b"{}" for name in
                                               ("job.json", "config.json", "status.json")})
            part["canonical_path"] = str((jobdir / "job.zip").resolve())
            receipt = {"experiment_id": "S1-D12A", "port": 20473,
                       "run_id": "prepare-1", "volumes": [part],
                       "source_refs": [{"source_bundle_member":
                                        "shared_sources/2024-02-07/day.tar",
                                        "sha256": hashlib.sha256(b"raw").hexdigest(),
                                        "bytes": 3,
                                        "source_preflight_sha256": hashlib.sha256(b"pin").hexdigest()}]}
            (jobdir / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            volumes = []
            for day in ("2024-02-07", "2024-02-08", "2024-02-09"):
                source_part = volume(cdir, f"{day}.zip", {
                    f"shared_sources/{day}/day.tar": b"raw",
                    f"shared_sources/{day}/source_preflight.json": b"pin"})
                source_part["canonical_path"] = str((cdir / source_part["file"]).resolve())
                volumes.append(source_part)
            source_receipt = {"experiment_id": "S1-D12A", "run_id": "shared_sources",
                              "volumes": volumes}
            (sourcedir / "receipt.json").write_text(json.dumps(source_receipt), encoding="utf-8")
            self.assertTrue(archive.verify_receipt(receipt, jobdir, source_receipt,
                                                   sourcedir)["closed"])
            evidence = Evidence(root / "D")
            try:
                job, error = evidence.job_json(20473, "prepare-1", "job.json")
                self.assertIsNone(error)
                self.assertEqual(job, {})
                with patch.object(evidence, "_read", side_effect=AssertionError("full read")):
                    self.assertEqual(evidence.sha(20473, "prepare-1", "job.json"),
                                     (hashlib.sha256(b"{}").hexdigest(), None))
                    self.assertTrue(evidence.has_digest(hashlib.sha256(b"raw").hexdigest(),
                                                        "day.tar"))
            finally:
                evidence.close()
            volumes[0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "ZIP byte SHA"):
                archive.verify_receipt(receipt, jobdir, source_receipt, sourcedir)


if __name__ == "__main__":
    unittest.main()
