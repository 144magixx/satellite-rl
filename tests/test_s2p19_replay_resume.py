"""Exact variable-width replay restoration at a full S2-P19 ring boundary."""

import copy
import unittest

import numpy as np

from satellite_rl.agents.workbench_algorithms import Replay


WIDTH = 4046  # S2-P19 C0/C1 validated natural-context observation width.
BUCKETS = ("anchor_100", "lower", "middle", "upper")


def add(replay, marker, beam_count, bucket):
    state = np.full(WIDTH, marker, dtype=np.float32)
    replay.add(state, (0, 0, 0), float(marker), state + 1, False,
               bucket=bucket, entity_count=beam_count)


class VariableWidthReplayResumeTests(unittest.TestCase):
    def test_empty_ring_checkpoint_restores_without_inventing_beam_counts(self):
        replay = Replay(4, 51042, "mixed_four_buckets", padded_width=WIDTH)
        saved = replay.state_dict()
        self.assertEqual(saved["rows"], [])
        self.assertEqual(saved["slot_entity_counts"], [])
        restored = Replay(4, 0, "mixed_four_buckets", padded_width=WIDTH)
        restored.load_state_dict(saved)
        self.assertEqual(restored.slot_entity_counts, [])
        formal_capacity = Replay(20000, 51042, "mixed_four_buckets", padded_width=WIDTH)
        add(formal_capacity, 0, 100, "anchor_100")
        self.assertEqual(formal_capacity.state_dict()["slot_entity_counts"], [100])
        self.assertEqual(formal_capacity.rows[0][0].shape, (4046,))

    def test_full_ring_replacement_keeps_one_count_per_real_row(self):
        replay = Replay(4, 51042, "mixed_four_buckets", padded_width=WIDTH)
        self.assertEqual(replay.slot_entity_counts, [])
        for marker, count in enumerate((50, 80, 100, 200)):
            add(replay, marker, count, BUCKETS[marker])
        self.assertEqual(replay.slot_entity_counts, [50, 80, 100, 200])
        add(replay, 4, 120, BUCKETS[0])
        self.assertEqual(len(replay.rows), len(replay.slot_entity_counts))
        self.assertEqual(replay.slot_entity_counts, [120, 80, 100, 200])
        self.assertEqual(replay.next_position, 1)
        self.assertEqual(len(replay.state_dict()["slot_entity_counts"]), 4)

    def test_historical_unused_zero_tail_is_canonicalized_without_sampling_change(self):
        replay = Replay(4, 51042, "mixed_four_buckets", padded_width=WIDTH)
        for marker, count in enumerate((50, 80, 100, 200)):
            add(replay, marker, count, BUCKETS[marker])
        add(replay, 4, 120, BUCKETS[0])
        saved = replay.state_dict()
        historical = copy.deepcopy(saved)
        historical["slot_entity_counts"].append(0)
        restored = Replay(4, 51042, "mixed_four_buckets", padded_width=WIDTH)
        restored.load_state_dict(historical)
        self.assertEqual(restored.slot_entity_counts, saved["slot_entity_counts"])
        normalized = restored.state_dict()
        for key in saved.keys() - {"rows"}:
            self.assertEqual(normalized[key], saved[key])
        for actual, expected in zip(normalized["rows"], saved["rows"]):
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[3], expected[3])
            self.assertEqual(actual[1:3], expected[1:3])
            self.assertEqual(actual[4:], expected[4:])
        for _ in range(3):
            original = replay.sample(4, "cpu")
            resumed = restored.sample(4, "cpu")
            for left, right in zip(original, resumed):
                np.testing.assert_array_equal(left.numpy(), right.numpy())
        add(restored, 5, 160, BUCKETS[1])
        self.assertEqual(restored.slot_entity_counts, [120, 160, 100, 200])

    def test_only_exact_full_ring_zero_tail_is_accepted(self):
        replay = Replay(4, 1, "mixed_four_buckets", padded_width=WIDTH)
        add(replay, 0, 50, BUCKETS[0])
        partial = replay.state_dict()
        partial["slot_entity_counts"].append(0)
        with self.assertRaisesRegex(ValueError, "缺真实束数记录"):
            Replay(4, 1, "mixed_four_buckets", padded_width=WIDTH).load_state_dict(partial)
        for marker, count in enumerate((80, 100, 200), start=1):
            add(replay, marker, count, BUCKETS[marker])
        full = replay.state_dict()
        for bad_tail in (1, 50):
            damaged = copy.deepcopy(full)
            damaged["slot_entity_counts"].append(bad_tail)
            with self.assertRaisesRegex(ValueError, "缺真实束数记录"):
                Replay(4, 1, "mixed_four_buckets", padded_width=WIDTH).load_state_dict(damaged)
        missing = copy.deepcopy(full)
        missing["slot_entity_counts"] = missing["slot_entity_counts"][:-1]
        with self.assertRaisesRegex(ValueError, "缺真实束数记录"):
            Replay(4, 1, "mixed_four_buckets", padded_width=WIDTH).load_state_dict(missing)
        invalid = copy.deepcopy(full)
        invalid["slot_entity_counts"][1] = 0
        with self.assertRaisesRegex(ValueError, "真实束数非法"):
            Replay(4, 1, "mixed_four_buckets", padded_width=WIDTH).load_state_dict(invalid)
        with self.assertRaisesRegex(ValueError, "capacity changed"):
            Replay(5, 1, "mixed_four_buckets", padded_width=WIDTH).load_state_dict(full)

    def test_legacy_deque_full_capacity_keeps_counts_aligned(self):
        replay = Replay(2, 7, "none", padded_width=WIDTH)
        for marker, count in enumerate((50, 100, 200)):
            state = np.full(WIDTH, marker, dtype=np.float32)
            replay.add(state, (0, 0, 0), 0, state, False, entity_count=count)
        self.assertEqual(replay.slot_entity_counts, [100, 200])
        self.assertEqual([int(row[0][0]) for row in replay.rows], [1, 2])
        restored = Replay(2, 7, "none", padded_width=WIDTH)
        restored.load_state_dict(replay.state_dict())
        self.assertEqual(restored.slot_entity_counts, [100, 200])


if __name__ == "__main__":
    unittest.main()
