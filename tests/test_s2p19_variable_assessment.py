"""P19 final assessment is 40 mothers × 11 N × two loads; older 440 stays fixed."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from satellite_rl.envs.regions import profile_metadata
from satellite_rl.envs.satellite import BeamInfo
from satellite_rl.workbench.datasets import freeze_scene_set, import_scenes, scene_set_files
from satellite_rl.workbench.runner import _validate_variable_assessment_scene_set
from satellite_rl.workbench.storage import Store


COUNTS = (50, 60, 75, 90, 100, 125, 140, 150, 175, 180, 200)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def p19_fixture():
    mothers, records, files = [], [], []
    for index in range(40):
        day = datetime(2024, 2, 1 if index < 20 else 2, tzinfo=timezone.utc)
        stamp = (day + timedelta(minutes=2 + 5 * (index % 20))).isoformat().replace("+00:00", "Z")
        mother_sha = sha(f"mother-{index}")
        mother_id = f"final_{index:05d}_{sha(stamp)[:12]}"
        mothers.append({"selected_time_utc": stamp, "path": f"/frozen/mother_{index:05d}.csv",
                        "sha256": mother_sha})
        for n in COUNTS:
            for load in (0.75, 1.25):
                scene_id = f"scene_{index:05d}_n{n:03d}_l{int(load * 100):03d}"
                scene_sha = sha(scene_id)
                records.append({"path": f"/frozen/{scene_id}.npz", "sha256": scene_sha,
                                "scene_id": scene_id, "mother_scene_id": mother_id,
                                "beam_count": n, "load_multiplier": load,
                                "partition_role": "test", "selected_time_utc": stamp,
                                "source_sha256": mother_sha})
                files.append((Path(f"/frozen/{scene_id}.npz"), scene_sha))
    source = {"type": "aviation_variable_beams_s2p19", "experiment_id": "S2-P19",
              "partition_role": "test", "mother_scene_count": 40,
              "source_dates": ["2024-02-01", "2024-02-02"],
              "snapshot_dates": ["2024-02-01", "2024-02-02"],
              "load_multipliers": [0.75, 1.25], "mother_scene_inputs": mothers}
    return {"source": source, "files": records}, files


class AssessmentGridTests(unittest.TestCase):
    def test_real_import_freeze_renames_cases_without_losing_p19_identity(self):
        record, _ = p19_fixture()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            prepared.mkdir()
            profile = profile_metadata("europe_v1")
            np.random.seed(51001)
            base = {}
            for n in COUNTS:
                beam = BeamInfo(profile="europe_v1", count=n)
                base[n] = {key: np.array(value, copy=True) for key, value in beam.beam_info.items()}
                base[n]["original_indices"] = np.array(beam.shuffle_indices, copy=True)
            for row in record["files"]:
                mother_index = int(row["mother_scene_id"].split("_")[1])
                arrays = {key: np.array(value, copy=True) for key, value in base[row["beam_count"]].items()}
                arrays["rate"] = np.asarray(arrays["rate"], dtype=np.float64) * (
                    row["load_multiplier"] * (1 + mother_index * 1e-4))
                path = prepared / (row["scene_id"] + ".npz")
                np.savez_compressed(path, **arrays)
                row["path"] = str(path)
                row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            record.update(id="s2p19_final_fixture", name="S2-P19 final fixture", count=880,
                          path=str(prepared), profile="europe_v1",
                          environment_version=profile["environment_version"],
                          environment_profile=profile)
            (prepared / "record.json").write_text(json.dumps(record), encoding="utf-8")
            store = Store(root, source_root=root)
            imported = import_scenes(store, prepared, name="P19 imported", profile="europe_v1")
            self.assertEqual(imported["count"], 880)
            self.assertNotEqual(imported["files"][0]["scene_id"], record["files"][0]["scene_id"])
            frozen = freeze_scene_set(imported, root / "frozen", "europe_v1")
            self.assertNotEqual(frozen["files"][0]["scene_id"], record["files"][0]["scene_id"])
            _validate_variable_assessment_scene_set("S2-P19", frozen, scene_set_files(frozen))

    def test_p19_complete_880_grid_is_accepted(self):
        record, files = p19_fixture()
        self.assertEqual(len(files), 880)
        _validate_variable_assessment_scene_set("S2-P19", record, files)

    def test_p19_rejects_440_missing_combination_and_duplicate_mother(self):
        record, files = p19_fixture()
        with self.assertRaisesRegex(ValueError, "880"):
            _validate_variable_assessment_scene_set("S2-P19",
                {"source": record["source"], "files": record["files"][:440]}, files[:440])
        missing = deepcopy(record)
        missing["files"][-1]["beam_count"] = 180
        missing["files"][-1]["scene_id"] = missing["files"][-3]["scene_id"]
        with self.assertRaises(ValueError):
            _validate_variable_assessment_scene_set("S2-P19", missing, files)
        duplicate = deepcopy(record)
        duplicate["source"]["mother_scene_inputs"][1]["selected_time_utc"] = (
            duplicate["source"]["mother_scene_inputs"][0]["selected_time_utc"])
        with self.assertRaises(ValueError):
            _validate_variable_assessment_scene_set("S2-P19", duplicate, files)

    def test_p19_rejects_wrong_source_partition_or_scene_sha(self):
        record, files = p19_fixture()
        for field, value in (("experiment_id", "S2-P17"), ("partition_role", "validation"),
                             ("source_dates", ["2024-01-29"])):
            bad = deepcopy(record)
            bad["source"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                _validate_variable_assessment_scene_set("S2-P19", bad, files)
        bad = deepcopy(record)
        bad["files"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            _validate_variable_assessment_scene_set("S2-P19", bad, files)
        bad = deepcopy(record)
        bad["files"][0]["source_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            _validate_variable_assessment_scene_set("S2-P19", bad, files)

    def test_p17_p18_keep_their_440_contract(self):
        record = {"files": [{"beam_count": 100, "partition_role": "test",
                             "sha256": sha(str(i))} for i in range(440)]}
        files = [(Path(f"/frozen/{i}.npz"), sha(str(i))) for i in range(440)]
        for experiment in ("S2-P17", "S2-P18"):
            _validate_variable_assessment_scene_set(experiment, record, files)
            with self.assertRaisesRegex(ValueError, "440"):
                _validate_variable_assessment_scene_set(experiment,
                    {"files": record["files"][:439]}, files[:439])


if __name__ == "__main__":
    unittest.main()
