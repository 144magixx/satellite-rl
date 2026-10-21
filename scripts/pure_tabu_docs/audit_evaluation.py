"""Audit pairing, common inputs, initializations, and finite publication values."""
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2] / "outputs/experiments/s2_pure_tabu_20261021/eval"
EXPECTED = {"conf_d432_n150": 88, "dev_d432_n150": 20, "dev_d216_n150": 20,
            "dev_d864_n150": 20, "dev_d432_n200": 20}


def main():
    totals = Counter()
    scenes, initials = {}, {}
    rows = []
    for path in ROOT.glob("*/results.csv"):
        for row in csv.DictReader(path.open(encoding="utf-8")):
            rows.append(row)
            key = (row["set"], row["method"], int(row["seed"]))
            totals[key] += 1
            for field, value in row.items():
                if field.startswith(("before_", "after_")) or field.endswith("_seconds"):
                    assert np.isfinite(float(value)), (key, row["scene_id"], field)
            assert int(row["worker"]) == int(row["scene_index"]) % 4
            assert int(row["core"]) == [106, 110, 114, 118][int(row["worker"])]
            if row["method"].startswith(("PTS", "GTS")):
                budget = int(row["method"][3:])
                assert int(row["probes"]) <= budget
                assert float(row["before_W"]) + 1e-10 >= float(row["initial_W"])
            with np.load(ROOT / row["beams"]) as arrays:
                scene = (row["set"], row["scene_id"])
                input_keys = ("beam_ids", "demand_bps", "aircraft_count")
                if scene in scenes:
                    for name in input_keys:
                        np.testing.assert_array_equal(arrays[name], scenes[scene][name])
                else:
                    scenes[scene] = {name: arrays[name].copy() for name in input_keys}
                family = "random" if row["method"] == "Random0" or row["method"].startswith("PTS") else "greedy"
                initial_key = (*scene, family, row["seed"])
                names = ("initial_groups", "initial_starts", "initial_counts", "initial_power")
                if initial_key in initials:
                    for name in names:
                        np.testing.assert_array_equal(arrays[name], initials[initial_key][name])
                else:
                    initials[initial_key] = {name: arrays[name].copy() for name in names}
                for name in ("capacity_before", "capacity_after", "demand_bps", "power_before", "power_after"):
                    assert np.isfinite(arrays[name]).all(), (key, name)
    for dataset, count in EXPECTED.items():
        for method in ("Random0", "PTS32", "PTS96", "PTS320", "G1", "GTS32", "GTS96", "GTS320"):
            assert totals[(dataset, method, 51001)] == count
    for seed in (51002, 51003):
        for method in ("Random0", "PTS32", "PTS96", "PTS320"):
            assert totals[("dev_d432_n150", method, seed)] == 20
    assert len(rows) == 1504 and len(totals) == 48 and len(scenes) == 168
    print("1504 rows / 168 paired scenes / 48 groups: finite values, common inputs, common initializations, budgets and CPU mapping verified.")


if __name__ == "__main__":
    main()
