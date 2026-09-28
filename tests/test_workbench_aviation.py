"""Deterministic adapter checks; test positions are synthetic, not flight data."""

import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from satellite_rl.envs.regions import get_profile
from satellite_rl.workbench.aviation import (
    AviationSceneError, CSV_UNITS, _cover, _read_sources, build_aviation_scenes,
)


MODEL = {"persons_per_aircraft": 180, "satellite_access_fraction": 0.5,
         "active_fraction": 0.2, "rate_mbps_per_active_person": 2,
         "aircraft_base_mbps": 1, "parameter_source": "unit test engineering assumptions"}
TIME = "2026-09-27T12:00:00Z"


class AviationAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self, count=100):
        return [{"aircraft_id": f"ac{i:04}", "timestamp": TIME,
                 "lat": 45 + (i // 10) * 0.01, "lon": 5 + (i % 10) * 0.01,
                 "altitude_m": 10000} for i in range(count)]

    def write(self, rows, name="positions.csv"):
        path = self.root / name
        keys = ["aircraft_id", "timestamp", "lat", "lon", "altitude_m"]
        if any("persons" in row for row in rows):
            keys.append("persons")
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, keys)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def build(self, path, **kwargs):
        options = {"selected_times": [TIME], "units": CSV_UNITS, "demand_model": MODEL,
                   "seed": 51001, "partition_role": "train"}
        options.update(kwargs)
        return build_aviation_scenes([path], **options)

    def test_units_and_demand_conservation(self):
        arrays, source = self.build(self.write(self.rows()))
        scene = arrays[0]
        self.assertEqual(scene["rate"].shape, (100, 1))
        # 180 * .5 * .2 * 2 + 1 = 37 Mbit/s per aircraft.
        np.testing.assert_allclose(scene["rate"], 37e6)
        snap = source["snapshots"][0]
        self.assertAlmostEqual(snap["total_demand_bps"], 3.7e9)
        self.assertAlmostEqual(snap["covered_demand_bps"] + snap["uncovered_demand_bps"], 3.7e9)
        self.assertEqual(snap["counts"]["covered_aircraft"], 100)
        self.assertEqual(source["source_dates"], ["2026-09-27"])
        self.assertEqual(source["type"], "aviation_imported_positions_synthetic_demand")
        self.assertEqual(source["position_source_status"], "user_supplied_not_independently_verified")
        json.dumps(source, allow_nan=False)
        with self.assertRaises(ValueError):
            self.build(self.write(self.rows()), units={**CSV_UNITS, "altitude_m": "ft"})
        with self.assertRaises(ValueError):
            self.build(self.write(self.rows()), units=None)

    def test_future_not_used_and_latest_ground_not_resurrected(self):
        rows = self.rows(102)
        rows.extend([
            {**rows[0], "timestamp": "2026-09-27T12:00:01Z", "lat": 70},
            {**rows[100], "timestamp": "2026-09-27T11:59:59Z"},
            {**rows[100], "timestamp": TIME, "altitude_m": "ground"},
        ])
        # Drop the original equal-time airborne record, avoiding a real conflict.
        del rows[100]
        arrays, source = self.build(self.write(rows))
        records = source["snapshots"][0]["source_observations"]
        self.assertEqual(next(r["lat"] for r in records if r["aircraft_id"] == "ac0000"), 45)
        self.assertNotIn("ac0100", [r["aircraft_id"] for r in records])
        self.assertEqual(source["snapshots"][0]["counts"]["future_records"], 1)
        self.assertEqual(source["snapshots"][0]["counts"]["ground_aircraft"], 1)
        self.assertGreater(np.sum(arrays[0]["rate"]), 0)

    def test_expired_outside_elevation_invalid_and_insufficient(self):
        rows = self.rows(104)
        rows[100]["timestamp"] = "2026-09-27T11:00:00Z"
        rows[101]["lon"] = 90
        rows[102]["lat"] = 80
        rows[103]["lat"] = "invalid"
        _, source = self.build(self.write(rows))
        counts = source["snapshots"][0]["counts"]
        self.assertEqual(counts["selected_aircraft"], 100)
        self.assertEqual(counts["expired_aircraft"], 1)
        self.assertEqual(counts["outside_region_aircraft"], 1)
        self.assertEqual(counts["below_elevation_aircraft"], 1)
        self.assertEqual(counts["invalid_aircraft"], 1)
        with self.assertRaises(AviationSceneError) as caught:
            self.build(self.write(self.rows(99)))
        self.assertEqual(caught.exception.diagnostics["selected_aircraft"], 99)

    def test_invalid_latest_not_replaced_by_valid_old_position(self):
        rows = self.rows(101)
        rows[100]["timestamp"] = "2026-09-27T11:59:59Z"
        rows.append({**rows[100], "timestamp": TIME, "lat": "not-a-position"})
        _, source = self.build(self.write(rows))
        self.assertEqual(source["snapshots"][0]["counts"]["invalid_aircraft"], 1)
        self.assertEqual(source["snapshots"][0]["counts"]["selected_aircraft"], 100)

    def test_duplicate_aircraft_across_files_and_conflicts(self):
        path = self.write(self.rows())
        other = self.write(self.rows(), "other.csv")
        arrays, source = build_aviation_scenes([other, path], [TIME], units=CSV_UNITS,
                                              demand_model=MODEL, seed=51001, partition_role="train")
        self.assertEqual(source["snapshots"][0]["counts"]["selected_aircraft"], 100)
        self.assertEqual(source["snapshots"][0]["counts"]["duplicate_historical_records"], 100)
        self.assertAlmostEqual(arrays[0]["rate"].sum(), 3.7e9)
        rows = self.rows()
        rows[0]["lon"] = 6
        self.write(rows, "other.csv")
        with self.assertRaises(AviationSceneError):
            build_aviation_scenes([path, other], [TIME], units=CSV_UNITS,
                                  demand_model=MODEL, seed=51001, partition_role="test")

    def test_coverage_exclusive_and_uncovered_retained(self):
        lat, lon = np.meshgrid(np.linspace(30, 60, 11), np.linspace(-20, 40, 11))
        positions = np.column_stack((lat.ravel(), lon.ravel()))
        demand = np.arange(1, len(positions) + 1, dtype=float) * 1e6
        _, rates, association, angles = _cover(positions, demand, get_profile("europe_v1"))
        self.assertTrue((rates > 0).all())
        self.assertTrue((association < 0).any())
        self.assertTrue((angles[association >= 0] <= 0.25 + 1e-10).all())
        self.assertAlmostEqual(rates.sum() + demand[association < 0].sum(), demand.sum())
        for beam in range(100):
            self.assertAlmostEqual(rates[beam], demand[association == beam].sum())
        with self.assertRaises(AviationSceneError):
            _cover(np.tile([[45, 5]], (100, 1)), np.ones(100), get_profile("europe_v1"))

    def test_reproducible_independent_of_time_order_and_zero_rejection(self):
        path = self.write(self.rows())
        model = {**MODEL, "sampling": "binomial", "lognormal_sigma": 0.2}
        later = "2026-09-27T12:00:05Z"
        a, meta_a = self.build(path, demand_model=model, selected_times=[TIME, later])
        b, meta_b = self.build(path, demand_model=model, selected_times=[later, TIME])
        np.testing.assert_array_equal(a[0]["rate"], b[1]["rate"])
        np.testing.assert_array_equal(a[1]["rate"], b[0]["rate"])
        self.assertEqual(meta_a["snapshots"][0], meta_b["snapshots"][1])
        with self.assertRaises(AviationSceneError):
            self.build(path, demand_model={**MODEL, "active_fraction": 0, "aircraft_base_mbps": 0})

    def test_readsb_gzip_feet_offsets_ground_stale_and_future(self):
        paths = []
        base = 1790510400  # 2026-09-27 12:00 UTC
        for index, row in enumerate(self.rows(102)):
            path = self.root / f"{index:06x}.json.gz"
            altitude = "ground" if index == 100 else 30000
            flags = 1 if index == 101 else 0
            content = {"icao": f"{index:06x}", "timestamp": base,
                       "trace": [[0, row["lat"], row["lon"], altitude, 400, 30, flags],
                                 [5, 70, 10, 30000, 400, 30, 0]]}
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                json.dump(content, stream)
            paths.append(path)
        _, source = build_aviation_scenes(paths, [TIME], source_format="readsb_trace",
                                          demand_model=MODEL, seed=51001, partition_role="test")
        snap = source["snapshots"][0]
        self.assertEqual(snap["counts"]["ground_aircraft"], 1)
        self.assertEqual(snap["counts"]["stale_aircraft"], 1)
        self.assertEqual(snap["counts"]["future_records"], 102)
        self.assertAlmostEqual(snap["source_observations"][0]["altitude_m"], 9144)

    def test_readsb_gzip_magic_with_json_name_and_decompressed_limit(self):
        payload = json.dumps({"icao": "abc123", "timestamp": 1790510400,
                              "trace": [[0, 45, 5, 30000, None, None, 0]],
                              "padding": "x" * 2048}).encode()
        plain = self.root / "plain.json"
        hidden_gzip = self.root / "compressed.json"
        named_gzip = self.root / "compressed.json.gz"
        plain.write_bytes(payload)
        hidden_gzip.write_bytes(gzip.compress(payload))
        named_gzip.write_bytes(gzip.compress(payload))
        parsed = [_read_sources([path], "readsb_trace", None)[0][0]
                  for path in (plain, hidden_gzip, named_gzip)]
        for observation in parsed[1:]:
            self.assertEqual({k: v for k, v in observation.items() if k != "source_sha256"},
                             {k: v for k, v in parsed[0].items() if k != "source_sha256"})
        with patch("satellite_rl.workbench.aviation.MAX_BYTES", 256):
            with self.assertRaisesRegex(ValueError, "解压后超过"):
                _read_sources([hidden_gzip], "readsb_trace", None)
        named_gzip.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "缺少gzip文件头"):
            _read_sources([named_gzip], "readsb_trace", None)

    def test_persons_override_requires_unit_and_utc_required(self):
        rows = self.rows()
        rows[0]["persons"] = 100
        path = self.write(rows)
        with self.assertRaises(ValueError):
            self.build(path)
        _, source = self.build(path, units={**CSV_UNITS, "persons": "count"})
        first = source["snapshots"][0]["source_observations"][0]
        self.assertEqual(first["persons"], 100)
        self.assertAlmostEqual(first["demand_bps"], 21e6)
        with self.assertRaises(ValueError):
            self.build(path, selected_times=["2026-09-27T12:00:00"])

    def test_descending_order_keeps_cluster_identity_and_association(self):
        rows = self.rows()
        for index, row in enumerate(rows):
            row["persons"] = 100 + index
        arrays, source = self.build(self.write(rows), units={**CSV_UNITS, "persons": "count"})
        scene, snapshot = arrays[0], source["snapshots"][0]
        self.assertTrue((np.diff(scene["rate"].ravel()) <= 0).all())
        self.assertEqual(scene["original_indices"].tolist(), snapshot["beam_order_original_indices"])
        for aircraft in snapshot["source_observations"]:
            index = aircraft["processing_index"]
            self.assertEqual(scene["original_indices"][index], aircraft["beam_id"])
            self.assertAlmostEqual(scene["rate"][index, 0], aircraft["demand_bps"])
            np.testing.assert_allclose(scene["beam_location"][index], [aircraft["lat"], aircraft["lon"]])


if __name__ == "__main__":
    unittest.main()
