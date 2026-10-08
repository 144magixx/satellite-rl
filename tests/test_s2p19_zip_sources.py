"""Large S2-P19 source members are streamed from verified ZIP volumes."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from scripts.s2p19 import archive, archive_sources, report


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


class ZipSourceTests(unittest.TestCase):
    def test_training_jsonl_members_stream_with_original_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mirror = root / "mirror"
            files = {"job/job.json": b'{"run_id":"train_i0"}',
                     "job/config.json": b"{}",
                     "job/status.json": b'{"run_id":"train_i0","stage":"completed"}',
                     "source/source_manifest.json": b"{}",
                     "job/episodes.jsonl": b'{"env_step":2000,"episode":1,"return":0.5}\n',
                     "job/evaluations.jsonl": b'{"env_step":2000,"mean_si":0.9}\n',
                     "job/updates.jsonl": b'{"env_step":2000,"updates":1,"actor_loss":0.2,"critic1_loss":0.3,"critic2_loss":0.4,"alpha":0.01}\n'}
            volume = root / "part_0001.zip"
            with zipfile.ZipFile(volume, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for name, payload in files.items():
                    bundle.writestr(name, payload)
            for name in archive.MIRROR_NAMES:
                path = mirror / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(files[name])
            receipt = {"scene_roles": {}, "inventory_sha256": "a" * 64,
                       "port": 20418, "members": [
                           {"member": name, "bytes": len(payload), "sha256": sha(payload)}
                           for name, payload in files.items()],
                       "volumes": [{"name": volume.name, "sha256": archive.digest(volume)}]}
            (root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            _, training = archive.source_spec("I0", "train_i0", mirror, receipt)
            self.assertEqual(training["updates"]["member"], "job/updates.jsonl")
            self.assertFalse((mirror / "job/updates.jsonl").exists())
            sources = {"training": {"I0": training}}
            manifest = {"tasks": {"I0": {"run_id": "train_i0"}}}
            curves = report.training_curves(sources, manifest)
            self.assertEqual({row["metric"] for row in curves},
                             {"episode_return", "episode_return_trailing_mean",
                              "validation_si"})
            diagnostics, missing, figures = report.learning_diagnostics(
                sources, manifest, root, arms=("I0",), render_plots=False)
            self.assertEqual(missing, [])
            self.assertEqual(figures, [])
            self.assertEqual(diagnostics["raw_row_count"], 4)

    def test_results_cross_chunk_case_and_scene_from_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mirror = root / "mirror"
            case = {"scene_sha256": "s" * 64,
                    "beams": [{"beam_id": 0}], "metrics": {"si": 0.5}}
            case_bytes = json.dumps(case).encode()
            scene_buf = io.BytesIO()
            np.savez(scene_buf, beam_location=np.array([[45.0, 10.0], [46.0, 11.0]]),
                     beam_G_Tx_max=np.full((2, 1), 50.0),
                     UE_G_Rx_max=np.full((2, 1), 40.0),
                     distances=np.array([38000.0, 38100.0]),
                     rate=np.array([[100.0], [200.0]]),
                     noise_temperature=np.full((2, 1), 290.0),
                     original_indices=np.array([0, 1]))
            scene_bytes = scene_buf.getvalue()
            results = json.dumps({"metadata": {"run_id": "run_zip"},
                "scenes": [{"id": 1, "long_note": "x" * 140000}, {"id": 2}]}).encode()
            files = {"job/job.json": b'{"run_id":"run_zip"}',
                     "job/config.json": b"{}",
                     "job/status.json": b'{"run_id":"run_zip","stage":"completed"}',
                     "source/source_manifest.json": b"{}",
                     "job/results.json": results,
                     "job/cases/case_00001.json": case_bytes,
                     "scenes/test/scene_0000000.npz": scene_bytes}
            volume = root / "part_0001.zip"
            with zipfile.ZipFile(volume, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for name, payload in files.items():
                    bundle.writestr(name, payload)
            for name in archive.MIRROR_NAMES:
                path = mirror / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(files[name])
            receipt = {"schema": "s2p19.archive.v1", "experiment_id": "S2-P19",
                       "run_id": "run_zip", "port": 20473, "job_stage": "completed",
                       "inventory_sha256": "a" * 64,
                       "scene_roles": {"test": {"count": 1}},
                       "members": [{"member": name, "bytes": len(payload),
                                    "sha256": sha(payload)} for name, payload in files.items()],
                       "volumes": [{"name": volume.name, "bytes": volume.stat().st_size,
                                    "sha256": archive.digest(volume)}]}
            (root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            spec, _ = archive.source_spec("C0_final_direct_20473", "run_zip", mirror,
                                          receipt)
            self.assertIsNone(spec["results_path"])
            self.assertEqual(spec["results_source"]["member"], "job/results.json")
            self.assertFalse((mirror / "job/results.json").exists())
            source, reason = report.source_for_task(
                "C0_final_direct_20473", "run_zip",
                {"jobs": {"C0_final_direct_20473": spec}}, {}, {}, 20473,
                require_port_evidence=True)
            self.assertIsNone(reason)
            info = {}
            self.assertEqual([row["id"] for row in report.iter_saved_scenes(
                source["source"], info)], [1, 2])
            self.assertEqual(info["metadata"]["run_id"], "run_zip")
            saved_case, label, case_sha = report._resolve_case(
                {"case_path": "/remote/case_00001.json", "case_sha256": sha(case_bytes)},
                spec, task_id="fixture")
            self.assertEqual(saved_case["metrics"]["si"], 0.5)
            self.assertIn("!job/cases/case_00001.json", label)
            self.assertEqual(case_sha, sha(case_bytes))
            scene_row = {"scene": {"source_path": "/remote/scene_0000000.npz"}}
            scene_label, content, static = report._resolve_scene_source(
                scene_row, scene_row, spec, sha(scene_bytes), "fixture")
            self.assertIn("!scenes/test/scene_0000000.npz", scene_label)
            self.assertEqual((content, static), report.scene_content_bytes(scene_bytes))
            damaged = {**source["source"], "sha256": "0" * 64}
            with self.assertRaises(ValueError):
                list(report.iter_saved_scenes(damaged, {}))
            with self.assertRaises(ValueError):
                archive_sources.read_bytes_source({**spec["results_source"],
                                                   "zip_sha256": "0" * 64})


if __name__ == "__main__":
    unittest.main()
