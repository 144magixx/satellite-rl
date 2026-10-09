import unittest
from types import SimpleNamespace

import numpy as np

from satellite_rl.coverage.variable_config import VariableConfig
from satellite_rl.coverage.variable_design import Layout
from satellite_rl.coverage.variable_downstream import (
    WidthPattern, allocator_config, evaluate_layout, peak_gain, transmit_pattern,
    solve_feedback, width_physics)
from satellite_rl.envs.satellite import AntennaPattern
from satellite_rl.power_refinement.physics import FixedAllocationPhysics


def scene(positions, demands):
    return SimpleNamespace(positions=np.asarray(positions, dtype=float),
                           demands=np.asarray(demands, dtype=float),
                           aircraft_ids=[f"u{i}" for i in range(len(demands))])


class VariableDownstreamTest(unittest.TestCase):
    def test_fixed_pattern_and_width_scaling(self):
        original = AntennaPattern()
        adapted = WidthPattern()
        for angle in (0, 0.1, 0.25, 0.251, 0.5, 1.5):
            self.assertAlmostEqual(original.transmitting_antenna_pattern(50, angle),
                                   adapted.transmitting_antenna_pattern(50, angle), places=12)
        self.assertLess(peak_gain(0.7), peak_gain(0.5))
        self.assertGreater(peak_gain(0.35), peak_gain(0.5))
        for width in (0.35, 0.5, 0.7):
            self.assertAlmostEqual(float(transmit_pattern(peak_gain(width), width/2, width)),
                                   peak_gain(width) - 3, places=12)

    def test_fixed_terminal_matches_original_independent_phy(self):
        s = scene([[45 + i*.08, 13 + i*.06] for i in range(10)], [200e6]*10)
        layout = Layout(s.positions, np.full(10, .5), np.arange(10))
        result = evaluate_layout(s, layout, {"coverage": {"variable": {}}})
        arrays = {key: np.asarray(value) for key, value in result["physical_scene"].items()}
        rows = result["actual_allocations"]
        reference = FixedAllocationPhysics.from_scene(arrays, allocator_config(), rows)
        powers = [row["power_w"] for row in rows]
        expected = reference.evaluate(powers).capacity_bps
        actual = [row["capacity_bps"] for row in result["beams"]]
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-3)
        self.assertLessEqual(result["metrics"]["total_power_w"], 3000)
        self.assertEqual(result["calls"]["allocation_decisions"], 10)
        np.testing.assert_allclose(result["aircraft_service"]["capacity_bps"], expected,
                                   rtol=2e-8, atol=2e-3)

    def test_variable_width_terminal_and_airtime_no_duplication(self):
        s = scene([[45, 13], [45.5, 13.4], [46, 14], [46.1, 14.1]],
                  [100e6, 100e6, 150e6, 50e6])
        layout = Layout(s.positions[[0, 1, 2]], [.35, .5, .7], [0, 1, 2, 2])
        result = evaluate_layout(s, layout, {"coverage": {"variable": {}}}, route="R05")
        fractions = result["aircraft_service"]["airtime_fraction"]
        self.assertAlmostEqual(fractions[2] + fractions[3], 1)
        full = np.asarray(result["aircraft_service"]["full_time_rate_bps"])
        capacities = np.asarray(result["aircraft_service"]["capacity_bps"])
        np.testing.assert_allclose(capacities, full * fractions)
        self.assertLess(capacities[2], full[2])
        self.assertEqual(len(result["beams"]), 3)
        self.assertTrue(result["metrics"]["complete_allocation"])

    def test_uncovered_and_zero_demand_are_not_removed_from_denominator(self):
        s = scene([[45, 13], [55, 25], [44, 12]], [100e6, 300e6, 0])
        layout = Layout(s.positions[[0, 2]], [.5, .5], [0, -1, 1])
        result = evaluate_layout(s, layout, {"coverage": {"variable": {}}})
        self.assertEqual(result["metrics"]["total_demand_bps"], 400e6)
        self.assertLessEqual(result["metrics"]["demand_weighted_carried_fraction"], .25)
        self.assertEqual(result["beams"][1]["status"], "geometry_only_zero_demand")
        self.assertIsNone(result["beams"][1]["si"])
        self.assertEqual(result["aircraft_service"]["capacity_bps"][1], 0)
        self.assertEqual(result["calls"]["aircraft_link_checks"], 1)
        self.assertEqual(result["metrics"]["processed_count"], 1)

    def test_feedback_cost_includes_screening_and_final_is_separate(self):
        s = scene([[45, 13], [45.5, 13.4], [46, 14]], [400e6, 300e6, 100e6])
        cfg = VariableConfig(max_actions=2)
        layout, info = solve_feedback(s, cfg, config={"coverage": {"variable": {}}})
        self.assertGreater(info["actual_geometry_calls"], 0)
        self.assertGreater(info["actual_ray_pair_checks"], 0)
        self.assertLessEqual(info["physical_feedback_calls"], 16)
        self.assertEqual(info["final_independent_allocation_calls"], 1)
        self.assertEqual(info["total_allocation_call_limit_including_final"], 17)
        self.assertLessEqual(info["geometry_budget_charges"], cfg.max_geometry_calls)


if __name__ == "__main__":
    unittest.main()
