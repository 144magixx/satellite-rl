"""Integrity checks for read-only S2-P19 archive volumes and report index."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.s2p19.archive import (
    digest, materialize_mirror, merge_index, mirror_member, safe_member,
    source_spec, verify_volume)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class S2P19ArchiveTests(unittest.TestCase):
    def test_mirror_keeps_only_small_identity_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            volume = root / "part_0001.zip"
            files = {"job/job.json": b"{}", "job/results.json": b"x" * 1000,
                     "job/cases/case_1.json": b"{}", "job/updates.jsonl": b"{}\n",
                     "scenes/test/scene_1.npz": b"npz"}
            with zipfile.ZipFile(volume, "w") as bundle:
                for name, payload in files.items():
                    bundle.writestr(name, payload)
            items = [{"member": name, "bytes": len(payload), "sha256": _sha(payload)}
                     for name, payload in files.items()]
            mirror = root / "mirror"
            self.assertEqual(materialize_mirror(volume, items, mirror), 1)
            self.assertTrue((mirror / "job/job.json").is_file())
            self.assertFalse((mirror / "job/results.json").exists())
            self.assertFalse((mirror / "job/cases/case_1.json").exists())
            self.assertFalse((mirror / "job/updates.jsonl").exists())
            self.assertFalse((mirror / "scenes/test/scene_1.npz").exists())

    def test_volume_member_sha_and_safe_path_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "part_0001.zip"
            payload = b'{"run_id":"fixture"}'
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
                output.writestr("job/job.json", payload)
            item = {"member": "job/job.json", "bytes": len(payload), "sha256": _sha(payload)}
            verify_volume(archive, [item], digest(archive))
            self.assertEqual(materialize_mirror(archive, [item], root / "mirror"), 1)
            self.assertEqual((root / "mirror/job/job.json").read_bytes(), payload)
            with self.assertRaises(ValueError):
                verify_volume(archive, [{**item, "sha256": "0" * 64}], digest(archive))
            with self.assertRaises(ValueError):
                verify_volume(archive, [item], "0" * 64)
            with self.assertRaises(ValueError):
                safe_member("../outside.json")

    def test_source_index_binds_job_config_scene_case_and_training_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mirror = root / "mirror"
            files = {
                "job/job.json": b"{}", "job/config.json": b"{}",
                "job/status.json": b"{}", "job/results.json": b"{}",
                "job/episodes.jsonl": b"{}\n", "job/updates.jsonl": b"{}\n",
                "source/source_manifest.json": b"{}",
                "job/cases/scene_00000.json": b"{}",
                "scenes/validation/scene_00000.npz": b"npz",
            }
            for name, content in files.items():
                path = mirror / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            receipt = {"inventory_sha256": "a" * 64,
                       "scene_roles": {"validation": {"count": 1}},
                       "members": [{"member": name, "sha256": _sha(content), "bytes": len(content)}
                                   for name, content in files.items()]}
            spec, training = source_spec("C0", "run_a", mirror, receipt)
            self.assertEqual(spec["scene_root"], str(mirror / "scenes/validation"))
            self.assertEqual(spec["case_root"], str(mirror / "job/cases"))
            self.assertEqual(spec["results_sha256"], _sha(b"{}"))
            self.assertEqual(training["episodes"]["sha256"], _sha(b"{}\n"))
            self.assertEqual(training["updates"]["sha256"], _sha(b"{}\n"))
            self.assertFalse(mirror_member("job/updates.jsonl"))
            self.assertFalse(mirror_member("job/results.json"))
            index = root / "source_index.json"
            merge_index(index, "C0", spec, training)
            saved = json.loads(index.read_text(encoding="utf-8"))
            self.assertEqual(saved["jobs"]["C0"]["run_id"], "run_a")
            self.assertEqual(saved["training"]["C0"]["run_id"], "run_a")
            with self.assertRaises(ValueError):
                merge_index(index, "C0", {**spec, "run_id": "run_b"}, training)


if __name__ == "__main__":
    unittest.main()
