"""Durable row commits and same-input resumption rather than duplicate evaluation."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from satellite_rl.coverage.nondrl.runner import RowStore, canonical_sha, scaled_scene, run_nondrl_job
from satellite_rl.coverage.scenes import CoverageScene, CoverageDataset
from satellite_rl.coverage.variable_dataset import save_variable_dataset
from satellite_rl.workbench.coverage_nondrl import default_nondrl_config
from satellite_rl.workbench.storage import digest, read_json


class NondrlRunnerTests(unittest.TestCase):
    def test_complete_rows_survive_resume_and_orphan_not_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RowStore(tmp, {"dataset_sha256": "abc"})
            store.append("0:0.75", {"assignment": np.array([0, 1]), "cv": None})
            (Path(tmp) / "scene_000001.json.gz.partial").write_bytes(b"incomplete")
            resumed = RowStore(tmp, {"dataset_sha256": "abc"}, resume=True)
            self.assertEqual(resumed.keys, {"0:0.75"})
            self.assertEqual(resumed.load(resumed.manifest["rows"][0])["assignment"], [0, 1])
            with self.assertRaises(ValueError):
                resumed.append("0:0.75", {})
            with self.assertRaises(ValueError):
                RowStore(tmp, {"dataset_sha256": "different"}, resume=True)

    def test_corruption_never_silently_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RowStore(tmp, {})
            store.append("0:1", {"layout": [1, 2]})
            shard = Path(tmp) / store.manifest["rows"][0]["file"]
            shard.write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                RowStore(tmp, {}, resume=True)

    def test_multipliers_do_not_mutate_mother(self):
        scene = CoverageScene("mother", np.array([[45., 13.], [45.1, 13.1]]),
                              np.array([0., 10.]), np.empty((0, 2)), ["a", "b"], {}, "rawsha")
        scaled = scaled_scene(scene, 1.25)
        np.testing.assert_array_equal(scene.demands, [0., 10.])
        np.testing.assert_array_equal(scaled.demands, [0., 12.5])
        self.assertEqual(scaled.metadata["mother_scene_id"], "mother")
        scaled.positions[0] = 0
        self.assertEqual(scene.positions[0, 0], 45.)
        self.assertEqual(canonical_sha({"a": np.array([1])}), canonical_sha({"a": [1]}))

    def test_actual_initial_and_bh_worker_keep_same_i0(self):
        """Tiny real Europe geometry and downstream RF; no formal training."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = CoverageScene("mother", np.array([[45., 13.], [45.03, 13.04], [45.07, 13.08]]),
                                  np.array([1.e6, 2.e6, 0.]), np.empty((0, 2)), ["a", "b", "c"],
                                  {"source_dates": ["2024-01-01"], "selected_time_utc": "2024-01-01T00:00:00Z"}, "rawsha")
            npz = root / "data" / "inputs.npz"
            save_variable_dataset(CoverageDataset([scene], {"partition_role": "development"}), npz)
            record = {"path": str(npz), "sha256": digest(npz), "partition_role": "development"}
            config = default_nondrl_config("06")
            config["preparation"]["initial_workers"] = 2
            config["datasets"] = {"development": record}
            initial_dir = root / "initial"
            initial_dir.mkdir()
            job = {"run_id": "short_initial", "git_commit": "engineering", "kind": "prepare",
                   "preparation_mode": "initial", "dataset_role": "development", "execution_device": "cpu",
                   "dataset_sha256": {"development": record["sha256"]}}
            run_nondrl_job(initial_dir, job, config, preflight={"engineering": True})
            self.assertEqual(read_json(initial_dir / "status.json")["stage"], "completed")
            eval_dir = root / "evaluate"
            eval_dir.mkdir()
            config["coverage"]["nondrl"]["method"] = "BH"
            eval_job = {**job, "run_id": "short_evaluate", "kind": "evaluate", "preparation_mode": None,
                        "initial_source": {"path": str(initial_dir / "initial_manifest.json"),
                                           "sha256": digest(initial_dir / "initial_manifest.json")}}
            run_nondrl_job(eval_dir, eval_job, config, preflight={"engineering": True})
            self.assertEqual(read_json(eval_dir / "status.json")["stage"], "completed")
            first = RowStore(initial_dir / "raw_scenes", read_json(initial_dir / "raw_scenes/manifest.json")["provenance"], resume=True)
            second = RowStore(eval_dir / "raw_scenes", read_json(eval_dir / "raw_scenes/manifest.json")["provenance"], resume=True)
            self.assertEqual(len(second.keys), 3)
            for a, b in zip(first.manifest["rows"], second.manifest["rows"]):
                before, after = first.load(a), second.load(b)
                self.assertEqual(before["result"]["initial_sha256"], after["result"]["initial_sha256"])
                self.assertTrue(after["result"]["metrics_all"]["unique_assignment"])
                self.assertTrue(after["result"]["metrics_all"]["demand_conserved"])
                self.assertGreater(after["end_to_end_segment_sum_seconds"], 0)
            # Re-entering the same task does not perform extra evaluations.
            run_nondrl_job(eval_dir, eval_job, config, resume=True, preflight={"engineering": True})
            self.assertEqual(len(read_json(eval_dir / "raw_scenes/manifest.json")["rows"]), 3)


if __name__ == "__main__":
    unittest.main()
