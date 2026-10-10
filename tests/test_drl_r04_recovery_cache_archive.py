"""Execute the source builder against a small, complete recovery-cache fixture."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.stage1_drl_r04 import archive


class RecoveryCacheArchiveTests(unittest.TestCase):
    def test_verified_cache_excluded_and_mutations_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder).resolve()
            commit = "a" * 40
            proof_path = base / "20473" / "ops" / "source_recovery_cache_audit.json"
            proof_path.parent.mkdir(parents=True)
            entries = []
            for port, role, day in ((20473, "validation", "2024-02-07"),
                                    (20474, "test", "2024-02-08"),
                                    (20418, "test", "2024-02-09")):
                name = f"day-{day}.tar"
                source = base / "source" / role / day
                source.mkdir(parents=True)
                raw = source / name
                raw.write_bytes(bytes(range(256)) + bytes(range(144)) if day == "2024-02-07" else b"other")
                pin = source / "source_preflight.json"
                pin.write_text(json.dumps({"archive_name": name, "expected_bytes": raw.stat().st_size,
                                           "asset_id": 150440724}), encoding="utf-8")
                run_id = f"prepare-{port}"
                jobdir = base / str(port) / "archive" / "outputs" / "workbench" / "jobs" / run_id
                jobdir.mkdir(parents=True)
                (jobdir / "job.json").write_text(json.dumps({"run_id": run_id,
                    "experiment_id": "S1-D12A", "git_commit": commit, "source_date": day,
                    "dataset_preparation": {"archive_name": name,
                        "files": {"source_preflight.json": {"sha256": archive.digest(pin)}}}}),
                    encoding="utf-8")
                (jobdir / "status.json").write_text('{"stage":"completed"}', encoding="utf-8")
                (jobdir / "launch.json").write_text('{"pid":0}', encoding="utf-8")
                (jobdir / "artifacts.json").write_text(json.dumps({
                    "raw_archive_sha256": archive.digest(raw)}), encoding="utf-8")
                entries.append({"port": port, "role": role, "day": day, "run_id": run_id})
                if day == "2024-02-07":
                    feb07_source, feb07_raw = source, raw
                    chunks = source / "chunks"
                    chunks.mkdir()
                    manifest = {"chunk_count": 400, "chunk_size": 1, "expected_bytes": 400,
                                "chunks": []}
                    cache = []
                    for index, byte in enumerate(raw.read_bytes()):
                        path = chunks / f"chunk_{index:04d}.bin"
                        path.write_bytes(bytes([byte]))
                        sha = archive.digest(path)
                        manifest["chunks"].append({"index": index, "sha256": sha})
                        cache.append({"path": str(path), "bytes": 1, "sha256": sha,
                                      "canonical_offset": index,
                                      "matches_canonical_bytes": True,
                                      "receipt_sha256_matches": True,
                                      "receipt_bytes_matches": True})
                    manifest_path = chunks / "manifest.json"
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    partial = source / (name + ".part")
                    partial.write_bytes(raw.read_bytes()[:7])
                    cache.append({"path": str(partial), "bytes": 7,
                                  "sha256": archive.digest(partial), "canonical_offset": 0,
                                  "matches_canonical_bytes": True,
                                  "receipt_sha256_matches": True,
                                  "receipt_bytes_matches": True})
                    proof = {"experiment_id": "S1-D12A", "role": role, "date_utc": day,
                             "all_reconstructable": True,
                             "source_preflight_sha256": archive.digest(pin),
                             "canonical_tar": {"path": str(raw), "bytes": 400,
                                               "sha256": archive.digest(raw),
                                               "expected_sha256": archive.digest(raw)},
                             "metadata_files": [{"path": str(manifest_path),
                                 "sha256": archive.digest(manifest_path), "record": manifest}],
                             "cache_files": cache, "chunk_count": 400,
                             "checked_bytes": 400, "unexpected": []}
            proof_path.write_text(json.dumps(proof), encoding="utf-8")
            request = {"base": str(base), "source_commit": commit, "preparations": entries,
                       "reserve": 0, "recovery_proof_path": str(proof_path),
                       "recovery_proof_sha256": archive.digest(proof_path)}
            def build(destination):
                payload = {**request, "destination": str(base / destination)}
                output = io.StringIO()
                with redirect_stdout(output):
                    exec(compile(archive.REMOTE_SOURCE_BUILD, "source_builder", "exec"),
                         {"REQUEST_JSON": json.dumps(payload), "__name__": "source_builder"})
                return json.loads(output.getvalue().splitlines()[-1])
            receipt = build("ok")
            self.assertEqual(len(receipt["excluded_recovery_cache"]["files"]), 401)
            self.assertEqual(len(archive.verify_source_receipt(receipt, base / "ok")), 8)
            self.assertFalse(any("chunk_0000.bin" in name for volume in receipt["volumes"]
                                 for name in volume["members"]))
            (feb07_source / "chunks" / "chunk_0000.bin").write_bytes(b"X")
            with self.assertRaisesRegex(RuntimeError, "chunk manifest SHA|cache receipt"):
                build("tampered")
            (feb07_source / "chunks" / "chunk_0000.bin").write_bytes(feb07_raw.read_bytes()[:1])
            (feb07_source / "unexpected.bin").write_bytes(b"extra")
            with self.assertRaisesRegex(RuntimeError, "unexpected shared source file"):
                build("unknown")


if __name__ == "__main__":
    unittest.main()
