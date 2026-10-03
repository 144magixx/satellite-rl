"""Variable-count aviation generation and dataset input guards."""
from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from satellite_rl.workbench.aviation import build_aviation_scenes, CSV_UNITS
from satellite_rl.workbench.datasets import csv_scene


MODEL = {"persons_per_aircraft": 180, "satellite_access_fraction": 0.2,
         "active_fraction": 0.3, "rate_mbps_per_active_person": 2.0,
         "aircraft_base_mbps": 0, "lognormal_sigma": 0.35,
         "sampling": "binomial", "window_seconds": 60,
         "parameter_source": "G11 aviation fixture"}
TIME = "2024-01-08T12:00:00Z"


class VariableBeamDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "mother.csv"
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["aircraft_id", "timestamp", "lat", "lon", "altitude_m"])
            for i in range(220):
                writer.writerow([f"plane{i:03d}", TIME, 42 + i / 100, 12 + (i % 30) / 100, 11000])

    def build(self, count, load=1.0):
        return build_aviation_scenes([self.path], [TIME], source_format="csv_snapshot",
                                     units=CSV_UNITS, demand_model=MODEL, seed=51001,
                                     max_record_age_seconds=60, partition_role="train",
                                     beam_count=count, load_multiplier=load,
                                     min_positive_aircraft=200)

    def test_variable_counts_and_load_preserve_users_and_demand(self):
        for count in (1, 50, 100, 200):
            with self.subTest(count=count):
                arrays, source = self.build(count)
                scene, snapshot = arrays[0], source["snapshots"][0]
                self.assertEqual(scene["rate"].shape, (count, 1))
                self.assertEqual(len(set(scene["original_indices"])), count)
                self.assertTrue((scene["rate"] > 0).all())
                self.assertTrue(np.isclose(scene["rate"].sum(), snapshot["covered_demand_bps"], rtol=1e-12))
                self.assertTrue(np.isclose(snapshot["covered_demand_bps"] + snapshot["uncovered_demand_bps"],
                                           snapshot["total_demand_bps"], rtol=1e-12))
                self.assertEqual(snapshot["counts"]["positive_demand_aircraft"], 220)
                self.assertEqual(source["beam_count"], count)
        base, _ = self.build(50, 1.0)
        lighter, provenance = self.build(50, 0.75)
        np.testing.assert_allclose(base[0]["beam_location"], lighter[0]["beam_location"])
        np.testing.assert_allclose(base[0]["rate"] * 0.75, lighter[0]["rate"], rtol=1e-12)
        self.assertEqual(provenance["load_multiplier"], 0.75)

    def test_rejects_invalid_counts_and_zero_or_nonfinite_load(self):
        for count in (0, 201, True):
            with self.assertRaisesRegex(ValueError, "实际波束数"):
                self.build(count)
        for load in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                self.build(50, load)

    def test_csv_scene_accepts_dynamic_europe_count_and_preserves_identity(self):
        path = self.path.parent / "beams.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["beam_id", "lat", "lon", "demand_mbps", "group"])
            for i in range(200):
                writer.writerow([(i + 7) % 200, 40 + i / 100, 10, 1 + i, i % 8])
        scene = csv_scene(path, profile="europe_v1")
        self.assertEqual(scene["rate"].shape, (200, 1))
        self.assertEqual(scene["original_indices"].tolist()[:2], [7, 8])
        with self.assertRaisesRegex(ValueError, "中国旧版"):
            csv_scene(path, profile="china_legacy")

    def test_workbench_import_registers_and_freezes_mixed_counts(self):
        from satellite_rl.workbench.service import Application
        from satellite_rl.workbench.datasets import freeze_scene_set
        directory = self.path.parent / "mixed"
        directory.mkdir()
        original = []
        for n in (50, 200):
            arrays, _ = self.build(n)
            original.append(arrays[0])
            np.savez_compressed(directory / f"n{n:03d}.npz", **arrays[0])
        app = Application(self.path.parent / "archive", source_root=self.path.parent)
        record = app.api("POST", ["scenes", "import"], {}, {
            "path": str(directory), "name": "mixed-count import fixture", "profile": "europe_v1"})
        self.assertEqual(record["count"], 2)
        self.assertEqual([ref["beam_count"] for ref in record["files"]], [50, 200])
        frozen = freeze_scene_set(record, self.path.parent / "frozen", "europe_v1")
        for expected, ref in zip(original, frozen["files"]):
            with np.load(ref["path"], allow_pickle=False) as actual:
                np.testing.assert_array_equal(actual["rate"], expected["rate"])


if __name__ == "__main__":
    unittest.main()
