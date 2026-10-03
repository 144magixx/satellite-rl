"""S2-P17 mother split and held-out count contracts (no scene generation)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/variable_beams/prepare_dataset.py"
SPEC = importlib.util.spec_from_file_location("variable_beam_prepare", SCRIPT)
PREP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREP)


def record(role, dates, positive):
    rows = [{"selected_time_utc": f"{date}T{index:02d}:00:00Z",
             "counts": {"selected_aircraft": count, "zero_demand_aircraft": 0}}
            for index, (date, count) in enumerate(zip(dates, positive))]
    return {"count": len(rows), "source": {"partition_role": role,
            "files": [{"path": f"t{index:03d}.csv", "sha256": "a" * 64}
                      for index in range(len(rows))], "snapshots": rows}}


class VariablePreparationContractTests(unittest.TestCase):
    def test_validation_selects_twelve_fixed_ranks_among_eighteen_eligible(self):
        source = record("validation", ["2024-01-12"] * 20,
                        [161, 122] + [200 + i for i in range(18)])
        chosen = PREP.chosen_mothers(source, "validation")
        expected = [2 + int((i + 0.5) * 18 / 12) for i in range(12)]
        self.assertEqual([int(item[0]["path"][1:4]) for item in chosen], expected)

    def test_training_blocks_ineligible_mothers_before_generation(self):
        source = record("train", ["2024-01-08"] * 3, [220, 199, 400])
        with self.assertRaisesRegex(ValueError, "不足200"):
            PREP.chosen_mothers(source, "train")

    def test_train_counts_are_distinct_and_never_use_final_holdouts(self):
        counts = [PREP.selected_counts("train", index) for index in range(576)]
        self.assertTrue(all(len(row) == 7 and len(set(row)) == 7 and row[0] == 100 for row in counts))
        self.assertTrue(all(not PREP.HELD_OUT.intersection(row) for row in counts))
        self.assertEqual(PREP.selected_counts("validation", 0), (50, 80, 100, 120, 160, 200))
        self.assertEqual(PREP.selected_counts("test", 0), (50, 60, 75, 90, 100, 125, 140, 150, 175, 180, 200))


if __name__ == "__main__":
    unittest.main()
