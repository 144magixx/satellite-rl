"""Preparation contract and partition checks without official download."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from satellite_rl.workbench import s2p19_data as data


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mother_fixture(root: Path, count: int, kind: str) -> dict:
    rows = []
    files = []
    snapshots = []
    for index in range(count):
        day = "2024-01-12" if kind == "dev_validation" else data.FINAL_DATES[index // 20]
        stamp = datetime.fromisoformat(day).replace(tzinfo=timezone.utc) + timedelta(minutes=5 * index)
        selected = stamp.isoformat().replace("+00:00", "Z")
        path = root / f"source_{index:03d}.csv"
        path.write_text("aircraft_id,timestamp,lat,lon,altitude_m\na,2024-01-12T00:00:00Z,40,10,10000\n",
                        encoding="utf-8")
        item = {"path": str(path), "sha256": sha(path), "selected_time_utc": selected}
        rows.append(item)
        files.append(item)
        snapshots.append({"selected_time_utc": selected,
                          "counts": {"selected_aircraft": 205, "zero_demand_aircraft": 0}})
    source = {"seed": data.SEED, "demand_model": {"window_seconds": 60}}
    if kind == "dev_validation":
        source.update(mother_scene_count=12, mother_scene_inputs=rows)
    else:
        source.update(files=files, snapshots=snapshots)
    record = {"count": 144 if kind == "dev_validation" else 40,
              "profile": "europe_v1", "source": source}
    record_path = root / "record.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    return {"preparation_type": kind,
            "source_record": {"path": str(record_path), "sha256": sha(record_path)},
            "seed": data.SEED, "environment_profile": "europe_v1"}


class S2P19DataTests(unittest.TestCase):
    def test_dev_freeze_and_partition_are_stable(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = mother_fixture(root, 12, "dev_validation")
            frozen = data.freeze_preparation_config(raw, root / "job" / "input_sources")
            self.assertEqual(len(data.input_references(frozen)), 13)
            partitions = data._partitions(frozen)
            self.assertEqual([(p, len(m), len(n)) for p, m, n in partitions],
                             [("development", 6, 6), ("validation", 6, 6)])
            self.assertNotEqual(set(item["selected_time_utc"] for item in partitions[0][1]),
                                set(item["selected_time_utc"] for item in partitions[1][1]))
            Path(raw["source_record"]["path"]).write_text("{}", encoding="utf-8")
            self.assertEqual(data.validate_preparation_config(frozen)["preparation_type"],
                             "dev_validation")
            with self.assertRaises(ValueError):
                data.validate_preparation_config(raw)

    def test_dev_generation_has_72_scenes_per_partition(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = data.freeze_preparation_config(
                mother_fixture(root, 12, "dev_validation"), root / "job/input_sources")

            def fake_build(*args, **kwargs):
                return ([{"rate": np.array([2.0]), "beam_count": np.array(kwargs["beam_count"])}],
                        {"snapshots": [{"source_observations": [], "total_demand_bps": 4.0,
                                        "uncovered_demand_bps": 2.0,
                                        "covered_aircraft_fraction": 1.0}]})

            with patch.object(data, "build_aviation_scenes", side_effect=fake_build):
                result = data.run_preparation(root / "job", config)
                resumed = data.run_preparation(root / "job", config)
            self.assertEqual(result["outputs"]["development"]["count"], 72)
            self.assertEqual(result["outputs"]["validation"]["count"], 72)
            self.assertTrue(all(item["reused"] for item in resumed["outputs"].values()))
            for partition in ("development", "validation"):
                record = json.loads((root / "job/prepared" / partition / "record.json").read_text(encoding="utf-8"))
                self.assertEqual(len(record["files"]), 72)
                self.assertEqual(record["source"]["partition_role"], partition)

    def test_final_metadata_and_same_day_eligibility_fallback(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = mother_fixture(root, 40, "final")
            config = data.validate_preparation_config(raw)
            self.assertEqual(len(config["mother_sources"]), 40)
            candidates = [{"eligible_200": True, "rejection_reason": None} for _ in range(288)]
            target = int(.5 * 288 / 20)
            candidates[target] = {"eligible_200": False, "rejection_reason": "too_few"}
            selected, substitutions = data._selected_indices(candidates)
            self.assertEqual(len(selected), 20)
            self.assertEqual(selected[0], target + 1)
            self.assertEqual(substitutions[0]["reason"], "too_few")


if __name__ == "__main__":
    unittest.main()
