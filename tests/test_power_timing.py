"""Measured power-refinement stages and repeated-scene timing summaries."""

import unittest

import numpy as np

from satellite_rl.power_refinement.timing import (
    aggregate_scene_timing, build_timing,
)


def timing_row(scene, repeat, total):
    timing = build_timing(1.0, 0.2, total - 1.25, 0.1, 0.05)
    return {"source_scene_sha256": scene, "repeat": repeat, "timing": timing}


class TimingTests(unittest.TestCase):
    def test_measured_stages_and_solver_callback_partition_add_up(self):
        measured = build_timing(1.5, 0.25, 2.75, 0.4, 0.1)
        self.assertAlmostEqual(measured["optimizer_wall_seconds"],
                               measured["optimizer_callback_io_seconds"]
                               + measured["optimizer_excluding_callback_seconds"])
        self.assertAlmostEqual(measured["total_pipeline_seconds"],
                               measured["initialization_seconds"]
                               + measured["physics_setup_seconds"]
                               + measured["optimizer_wall_seconds"]
                               + measured["verification_seconds"])
        self.assertAlmostEqual(measured["total_pipeline_seconds"], 4.6)
        for values in ((-1, 0, 1, 0, 0), (0, np.nan, 1, 0, 0),
                       (0, 0, 1, 2, 0), (0, 0, 1, 1.0000005, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                build_timing(*values)

    def test_three_actions_are_summed_within_each_scene(self):
        rows = ([timing_row("scene_a", repeat, 2 + repeat) for repeat in range(3)]
                + [timing_row("scene_b", repeat, 5 + repeat) for repeat in range(3)])
        summary = aggregate_scene_timing(rows)
        self.assertTrue(summary["timing_available"])
        self.assertEqual(summary["timed_scenes"], 2)
        self.assertEqual(summary["timed_allocations"], 6)
        self.assertEqual(summary["percentile_method"], "numpy_linear")
        self.assertAlmostEqual(summary["single_allocation_total_pipeline_seconds_mean"], 4.5)
        self.assertAlmostEqual(summary["single_allocation_total_pipeline_seconds_median"], 4.5)
        self.assertAlmostEqual(summary["single_allocation_total_pipeline_seconds_p95"], 6.75)
        self.assertAlmostEqual(summary["scene_repeats_sum_total_pipeline_seconds_mean"], 13.5)
        self.assertAlmostEqual(summary["scene_repeats_sum_total_pipeline_seconds_median"], 13.5)
        self.assertAlmostEqual(summary["scene_repeats_sum_total_pipeline_seconds_p95"], 17.55)
        self.assertAlmostEqual(summary["scene_repeats_sum_initialization_seconds_mean"], 3)
        self.assertAlmostEqual(summary["scene_repeats_sum_physics_setup_seconds_mean"], 0.6)

    def test_historical_missing_timing_stays_unavailable(self):
        self.assertEqual(aggregate_scene_timing([]), {"timing_available": False})
        rows = [timing_row("scene_a", 0, 2),
                {"source_scene_sha256": "scene_a", "repeat": 1}]
        self.assertEqual(aggregate_scene_timing(rows), {"timing_available": False})

    def test_duplicate_scene_repeat_is_rejected(self):
        rows = [timing_row("scene_a", 0, 2), timing_row("scene_a", 0, 3)]
        with self.assertRaisesRegex(ValueError, "Duplicate scene/repeat"):
            aggregate_scene_timing(rows)


if __name__ == "__main__":
    unittest.main()
