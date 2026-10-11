"""Durable row commits and same-input resumption rather than duplicate evaluation."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from satellite_rl.coverage.nondrl.runner import RowStore, canonical_sha, scaled_scene
from satellite_rl.coverage.scenes import CoverageScene


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


if __name__ == "__main__":
    unittest.main()
