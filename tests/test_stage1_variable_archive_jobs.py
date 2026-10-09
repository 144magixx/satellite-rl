"""Completed S1-V10 ZIP packages are the report's sole large-file source."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.s2p19.archive_sources import member_ref, open_text_source
from scripts.stage1_variable import archive_jobs


COMMIT = "b31187fcaf79e90b70d85db4cdfec5a4150020de"
RUN = "train_ppo_20261009_094809_f22e8020"


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class CompletedJobArchiveTests(unittest.TestCase):
    def fixture(self, base, *, stage="completed"):
        root = base / "20418"
        jobdir = root / "outputs/workbench/jobs" / RUN
        jobdir.mkdir(parents=True)
        source = root / "outputs/workbench/source_snapshots" / COMMIT
        file = source / "src/satellite_rl/__init__.py"
        file.parent.mkdir(parents=True)
        file.write_text("# frozen source\n", encoding="utf-8")
        source_sha = archive_jobs.digest(file)
        _json(source / "source_manifest.json", {
            "git_commit": COMMIT,
            "files": {"src/satellite_rl/__init__.py": source_sha}})
        model = root / "models/runs/workbench" / RUN / "final"
        model.mkdir(parents=True)
        (model / "policy.pt").write_bytes(b"weights")
        _json(jobdir / "job.json", {
            "run_id": RUN, "request_id": "s1v10_verify_R08_20418_s51001",
            "experiment_id": "S1-V10", "archive_root": str(root.resolve()),
            "git_commit": COMMIT, "source_root": str(source.resolve()),
            "model_dir": str(model.parent.resolve()), "kind": "train"})
        _json(jobdir / "status.json", {"run_id": RUN, "stage": stage,
                                      "env_step": 2048, "pid": 999999})
        _json(jobdir / "config.json", {"datasets": {}})
        _json(jobdir / "launch.json", {"pid": 999999,
                                       "process_start_time": 1_700_000_000.0})
        (jobdir / "episodes.jsonl").write_text(
            "".join(json.dumps({"episode": index, "payload": "x" * 1000}) + "\n"
                    for index in range(250)), encoding="utf-8")
        return root, jobdir

    def build(self, root):
        request = {"archive_root": str(root.resolve()), "port": 20418,
                   "task_key": "verify_R08", "run_id": RUN,
                   "request_id": "s1v10_verify_R08_20418_s51001",
                   "source_commit": COMMIT, "expected_step": 2048,
                   "raw_result_sha256": None,
                   "controller_audit_sha256": "f" * 64,
                   "part_bytes": 1024 * 1024, "compression_level": 5,
                   "max_files": 100, "max_volumes": 10,
                   "max_bytes": 10 * (1 << 20)}
        code = "REQUEST_JSON=" + repr(json.dumps(request)) + "\n" + archive_jobs.REMOTE_BUILD
        output = io.StringIO()
        with redirect_stdout(output):
            exec(compile(code, "<S1 archive closure>", "exec"), {})
        return json.loads(output.getvalue().splitlines()[-1])

    def test_completed_job_package_verifies_without_large_plain_mirror(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, jobdir = self.fixture(base)
            handoff = self.build(root)
            receipt_path = Path(handoff["receipt_path"])
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["schema"], "s1v10.job_archive.v1")
            self.assertEqual(receipt["source_commit"], COMMIT)
            self.assertIn("job/episodes.jsonl", {row["member"] for row in receipt["members"]})
            self.assertIn("source/source_manifest.json", {row["member"] for row in receipt["members"]})
            self.assertIn("weights/model_dir/final/policy.pt",
                          {row["member"] for row in receipt["members"]})
            members = {item["member"]: item for item in receipt["members"]}
            for volume in receipt["volumes"]:
                path = receipt_path.parent / volume["name"]
                with zipfile.ZipFile(path) as bundle:
                    expected = [members[name] for name in bundle.namelist()]
                archive_jobs._verify_volume(path, expected, volume["sha256"])
            ref = member_ref(receipt_path, "job/episodes.jsonl")
            with open_text_source(ref) as stream:
                self.assertEqual(sum(1 for _ in stream), 250)
            raw_size = (jobdir / "episodes.jsonl").stat().st_size
            self.assertLess(sum(v["bytes"] for v in receipt["volumes"]), raw_size)
            receipt["controller_scientific_commit"] = COMMIT
            refs = {item["member"]: member_ref(receipt_path, item["member"])
                    for item in receipt["members"]}
            _json(base / "manifest.json", {
                "scientific_commit": COMMIT, "tasks": {},
                "files": [{"port": 20474, "run_id": "older_completed_job",
                           "path": str(base / "older.json"), "sha256": "a" * 64,
                           "bytes": 2}]})
            with patch.object(archive_jobs, "ANALYSIS_MANIFEST", base / "manifest.json"), \
                 patch.object(archive_jobs, "LOCAL", base / "local"), \
                 patch.object(archive_jobs.ops, "LOCAL", base / "local"):
                result = archive_jobs._merge_analysis_manifest(
                    "verify_R08", 20418, RUN, COMMIT, receipt_path.parent, receipt, [], refs)
            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            row = next(item for item in manifest["files"] if item["name"] == "episodes.jsonl")
            self.assertEqual(row["source_ref"], ref)
            self.assertNotIn("path", row)

    def test_active_job_and_tampered_volume_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, jobdir = self.fixture(Path(temporary), stage="training")
            with self.assertRaisesRegex(RuntimeError, "completed"):
                self.build(root)
            status = json.loads((jobdir / "status.json").read_text(encoding="utf-8"))
            status["stage"] = "completed"
            _json(jobdir / "status.json", status)
            handoff = self.build(root)
            receipt = json.loads(Path(handoff["receipt_path"]).read_text(encoding="utf-8"))
            volume = receipt["volumes"][0]
            path = Path(volume["remote_path"])
            path.write_bytes(path.read_bytes()[:-1] + b"0")
            with self.assertRaisesRegex(ValueError, "ZIP SHA"):
                archive_jobs._verify_volume(path, receipt["members"], volume["sha256"])

    def test_download_gate_uses_actual_compressed_bytes_and_eight_gib(self):
        with patch.object(archive_jobs.shutil, "disk_usage",
                          return_value=type("Usage", (), {"free": 8 * (1 << 30) + 99})()):
            with self.assertRaisesRegex(RuntimeError, "8 GiB reserve"):
                archive_jobs._require_local_capacity(Path("."), 100)
            accepted = archive_jobs._require_local_capacity(Path("."), 99)
            with self.assertRaisesRegex(RuntimeError, "8 GiB reserve"):
                archive_jobs._require_local_capacity(Path("."), 80, 20)
        self.assertEqual(accepted["missing_compressed_bytes"], 99)

    def test_bounded_archive_skips_verified_completed_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            state = base / "state.json"
            _json(state, {"tasks": {"verify_R08": {}, "verify_R09": {}},
                          "audits": {"verify_R08": {}, "verify_R09": {}}})
            receipt = base / "receipt.json"
            _json(receipt, {"schema": "s1v10.job_archive.v1"})
            _json(base / "download_receipt.json", {
                "all_volume_and_member_shas_verified": True,
                "task_key": "verify_R08", "run_id": RUN, "port": 20418})
            manifest = base / "manifest.json"
            _json(manifest, {"tasks": {"verify_R08": {"archive_receipt": str(receipt),
                                                       "run_id": RUN, "port": 20418}}})
            with patch.object(archive_jobs.controller, "STATE", state), \
                 patch.object(archive_jobs, "ANALYSIS_MANIFEST", manifest), \
                 patch.object(archive_jobs, "archive_completed",
                              return_value={"archived": True}) as archive:
                result = archive_jobs.archive_all_completed(limit=1)
            archive.assert_called_once_with("verify_R09")
            self.assertEqual(result["already_archived"], 1)
            self.assertEqual(result["archived_new"], 1)

    def test_preparation_job_uses_its_original_frozen_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state.json"
            old = "f" * 40
            prepare = {"port": 20418, "run_id": "prepare_ppo_20261009_094017_5797b0d1",
                       "request_id": "s1v10_prepare_20240203_20418",
                       "role": "validation", "date": "2024-02-03", "git_commit": old}
            _json(state, {"scientific_commit": COMMIT,
                          "preparation_source_commit": old,
                          "prepares": {"20418": prepare}, "tasks": {}, "audits": {}})
            detail = {"job": {"run_id": prepare["run_id"], "request_id": prepare["request_id"],
                              "experiment_id": "S1-V10", "kind": "prepare",
                              "archive_root": archive_jobs.ops.instance(20418)["root"],
                              "git_commit": old},
                      "status": {"stage": "completed", "process_alive": False},
                      "artifacts": {}}
            with patch.object(archive_jobs.controller, "STATE", state), \
                 patch.object(archive_jobs.ops, "api", return_value={"status": 200, "body": detail}):
                _, task, _, _, _, source = archive_jobs._task_identity("prepare_20418")
            self.assertEqual(task["kind"], "prepare")
            self.assertEqual(source, old)

    def test_actual_zip_volumes_can_span_d_and_c_with_both_reserves(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            droot, croot = base / "D" / "job_archives", base / "C" / "job_archives"
            gib = 1 << 30
            def usage(path):
                free = 14 * gib if Path(path) == droot else 12 * gib
                return type("Usage", (), {"free": free})()
            receipt = {"members": [], "volumes": [
                {"name": "part_0001.zip", "bytes": 5 * gib, "sha256": "a" * 64},
                {"name": "part_0002.zip", "bytes": 3 * gib, "sha256": "b" * 64}]}
            with patch.object(archive_jobs, "LOCAL", droot), \
                 patch.object(archive_jobs, "BACKUP_LOCAL", croot), \
                 patch.object(archive_jobs.shutil, "disk_usage", side_effect=usage):
                metadata, paths, gate = archive_jobs._select_volume_locations(
                    receipt, 20418, RUN, "package", 2000)
            self.assertTrue(metadata.is_relative_to(droot))
            self.assertTrue(paths["part_0001.zip"].is_relative_to(droot))
            self.assertTrue(paths["part_0002.zip"].is_relative_to(croot))
            self.assertEqual(gate["reserved_bytes_each_drive"], 8 * gib)
            receipt["volumes"] = [{"name": "part_0001.zip", "bytes": 7 * gib,
                                   "sha256": "c" * 64}]
            with patch.object(archive_jobs, "LOCAL", droot), \
                 patch.object(archive_jobs, "BACKUP_LOCAL", croot), \
                 patch.object(archive_jobs.shutil, "disk_usage", side_effect=usage):
                with self.assertRaisesRegex(RuntimeError, "actual ZIP volume exceeds"):
                    archive_jobs._select_volume_locations(receipt, 20418, RUN,
                                                          "package", 2000)


if __name__ == "__main__":
    unittest.main()
