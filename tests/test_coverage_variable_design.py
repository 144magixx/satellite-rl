"""Deterministic geometry, conservation and route distinctions for variable N."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from satellite_rl.coverage.variable_config import CORE_ROUTES, VariableConfig
from satellite_rl.coverage.variable_design import (
    Layout, VariableAction, _construct, _objective, _robust_demand_patterns, _transport_lp,
    apply_action, candidates, geometry_audit, initial_layout, metrics, solve,
)


def scene(positions, demands):
    points = np.asarray(positions, dtype=float)
    return SimpleNamespace(positions=points, demands=np.asarray(demands, dtype=float),
                           aircraft_ids=[f"flight-{i}" for i in range(len(points))])


class VariableDesignTests(unittest.TestCase):
    def setUp(self):
        self.scene = scene([[45, 10], [45.01, 10.01], [46, 11],
                            [46.01, 11.01], [47, 12], [47.01, 12.01]],
                           np.array([80, 40, 90, 30, 70, 60]) * 1e6)
        self.cfg = VariableConfig(max_beams=6, max_actions=3, max_candidates=32)

    def test_each_core_route_preserves_all_aircraft_and_demand(self):
        methods = set()
        route_info = {}
        for route in CORE_ROUTES:
            with self.subTest(route=route):
                layout, info = solve(self.scene, self.cfg, route)
                m = metrics(self.scene, layout, self.cfg)
                self.assertEqual(info["status"], "geometry_feasible")
                self.assertTrue(m["coverage_100"])
                self.assertTrue(m["demand_conserved"])
                self.assertAlmostEqual(sum(m["beam_demand_bps"]), sum(self.scene.demands))
                self.assertLessEqual(len(layout.centers), self.cfg.max_beams)
                self.assertEqual(layout.aircraft_ids, tuple(self.scene.aircraft_ids))
                methods.add(info["initialization"]["initialization"])
                route_info[route] = info
        self.assertGreaterEqual(len(methods), 2)
        self.assertEqual(route_info["R02"]["initialization"]["mip"]["solver"], "scipy.optimize.milp")
        self.assertEqual(route_info["R06"]["initialization"]["transport"]["solver"],
                         "scipy.optimize.linprog/highs")

    def test_cap_failure_is_explicit_and_does_not_drop_aircraft(self):
        separated = scene([[40, -10], [50, 10], [60, 30]], [1, 2, 3])
        layout, info = solve(separated, VariableConfig(max_beams=2), "R02")
        self.assertEqual(info["status"], "failed_full_coverage")
        self.assertFalse(info["metrics"]["coverage_100"])
        self.assertEqual(info["metrics"]["total_demand_bps"], 6)
        self.assertGreater(info["metrics"]["uncovered_demand_bps"], 0)

    def test_construct_menu_stop_gate_and_atomic_rejection(self):
        layout, _ = initial_layout(self.scene, self.cfg, "R09")
        menu = candidates(self.scene, layout, self.cfg, mode="construct")
        self.assertEqual(menu[0][0].operation, "stop")
        self.assertEqual(menu[0][1].shape, (16,))
        self.assertTrue(any(a.operation == "add" for a, _ in menu))
        same, info = apply_action(self.scene, layout, self.cfg, menu[0][0])
        self.assertIs(same, layout)
        self.assertFalse(info["accepted"])
        self.assertEqual(info["reason"], "stop_before_full_coverage")
        action = next(a for a, _ in menu if a.operation == "add")
        grown, info = apply_action(self.scene, layout, self.cfg, action)
        self.assertTrue(info["accepted"])
        self.assertGreater(metrics(self.scene, grown, self.cfg)["covered_users"], 0)
        self.assertEqual(metrics(self.scene, layout, self.cfg)["covered_users"], 0)
        self.assertEqual(VariableAction.from_record(action.to_record()), action)
        self.assertEqual(Layout.from_record(grown.to_record()).to_record(), grown.to_record())

    def test_satellite_footprint_and_nonfinite_rejected(self):
        invalid = Layout([[45, 10]], [0.5], [0, 0, 0, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "outside exact satellite footprint"):
            metrics(self.scene, invalid, self.cfg)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            metrics(self.scene, Layout([[np.nan, 10]], [0.5], [-1] * 6), self.cfg)
        nearby = scene([[45, 10], [45.01, 10.01]], [1, 1])
        with self.assertRaisesRegex(ValueError, "empty artificial"):
            metrics(nearby, Layout([[45, 10], [45.01, 10.01]], [0.5, 0.5],
                                   [0, 0]), self.cfg)

    def test_uniform_demand_multiplier_preserves_cv_not_capacity_pressure(self):
        layout, _ = initial_layout(self.scene, self.cfg, "R02")
        low = metrics(self.scene, layout, self.cfg)
        high_scene = scene(self.scene.positions, self.scene.demands * 1.25)
        high = metrics(high_scene, layout, self.cfg)
        self.assertAlmostEqual(low["load_cv"], high["load_cv"], places=12)
        self.assertAlmostEqual(high["total_demand_bps"] / low["total_demand_bps"], 1.25)
        self.assertGreaterEqual(high["reference_unserved_bps"], low["reference_unserved_bps"])

    def test_budget_and_warm_start_identity(self):
        layout, _ = initial_layout(self.scene, self.cfg, "R02")
        action = VariableAction("split", (0,), int(np.flatnonzero(layout.assignment == 0)[0]))
        same, info = apply_action(self.scene, layout, self.cfg, action,
                                  {"geometry_calls": 2, "max_geometry_calls": 2})
        self.assertIs(same, layout)
        self.assertEqual(info["reason"], "geometry_budget_exhausted")
        warm, record = initial_layout(self.scene, self.cfg, "R12", previous=layout)
        self.assertEqual(record["initialization"], "previous_revalidated")
        self.assertTrue(metrics(self.scene, warm, self.cfg)["coverage_100"])
        changed = scene(self.scene.positions, self.scene.demands)
        changed.aircraft_ids[0] = "different-flight"
        changed_layout, changed_info = initial_layout(changed, self.cfg, "R12", previous=layout)
        self.assertEqual(changed_info["initialization"], "previous_revalidated")
        self.assertEqual(changed_layout.aircraft_ids, tuple(changed.aircraft_ids))
        extended = scene(np.vstack((self.scene.positions, [[48, 13]])),
                         np.append(self.scene.demands, 20e6))
        extended_layout, extended_info = initial_layout(extended, self.cfg, "R12", previous=layout)
        self.assertTrue(metrics(extended, extended_layout, self.cfg)["demand_conserved"])
        self.assertEqual(len(extended_layout.assignment), 7)

    def test_width_version_and_empty_scene(self):
        wide, _ = initial_layout(self.scene, self.cfg, "R05")
        self.assertTrue(np.allclose(wide.widths, 0.7))
        empty = scene(np.empty((0, 2)), np.empty(0))
        layout, info = solve(empty, self.cfg, "R01")
        self.assertEqual(len(layout.centers), 0)
        self.assertTrue(info["metrics"]["coverage_100"])
        with self.assertRaisesRegex(ValueError, "200"):
            VariableConfig(max_beams=201)

    def test_r01_overload_triggers_real_split_and_low_demand_uses_one_beam(self):
        s = scene([[45, 10], [45.01, 10.01]], [400e6, 400e6])
        cfg = VariableConfig(max_beams=2, max_actions=3)
        high, high_info = solve(s, cfg, "R01")
        self.assertEqual(len(high.centers), 2)
        self.assertTrue(any(row.get("selected_for_commit") for row in high_info["action_log"]))
        low = scene(s.positions, s.demands * 0.75)
        low_layout, _ = solve(low, cfg, "R01")
        self.assertEqual(len(low_layout.centers), 1)

    def test_r02_mip_certificate_is_limited_to_its_candidate_set(self):
        layout, info = solve(self.scene, self.cfg, "R02")
        mip = info["initialization"]["mip"]
        self.assertEqual(mip["status"], 0)
        self.assertEqual(mip["fallback"], None)
        self.assertEqual(mip["certificate_scope"], "screened_discrete_candidate_sites_only")
        self.assertEqual(len(layout.centers), 1)

    def test_r06_transport_relaxation_is_split_then_integer_repaired(self):
        s = scene([[45, 10], [45.1, 10.1], [45.2, 10.2]], [300e6, 800e6, 300e6])
        cfg = VariableConfig(max_beams=3)
        seed = Layout([[45, 10], [45.2, 10.2]], [0.5, 0.5], [0, 0, 1],
                      tuple(s.aircraft_ids))
        repaired, info = _transport_lp(s, seed, cfg)
        self.assertEqual(info["status"], 0)
        self.assertGreater(info["fractionally_split_users"], 0)
        self.assertGreater(info["fractional_to_integer_load_l1_bps"], 0)
        self.assertTrue(metrics(s, repaired, cfg)["coverage_100"])
        self.assertTrue(metrics(s, repaired, cfg)["demand_conserved"])

    def test_r08_uses_exact_r04_initial_layout(self):
        a, _ = initial_layout(self.scene, self.cfg, "R04")
        b, _ = initial_layout(self.scene, self.cfg, "R08")
        self.assertEqual(a.to_record(), b.to_record())

    def test_stable_beam_ids_are_not_reused_after_compaction(self):
        s = scene([[45, 10], [45.01, 10.01]], [400e6, 400e6])
        prior = Layout([[45, 10]], [0.5], [0, 0], tuple(s.aircraft_ids),
                       ("b0",), next_beam_serial=2)
        after, info = apply_action(s, prior, VariableConfig(max_beams=2),
                                   VariableAction("split", (0,), 0))
        self.assertTrue(info["accepted"])
        self.assertEqual(after.beam_ids, ("b0", "b2"))
        self.assertEqual(after.next_beam_serial, 3)
        self.assertEqual(Layout.from_record(after.to_record()).to_record(), after.to_record())

    def test_exact_200_business_beams_without_padding_or_truncation(self):
        positions = [(lat, lon) for lat in np.linspace(36, 64, 10)
                     for lon in np.linspace(-10, 30, 20)]
        s = scene(positions, np.full(200, 700e6))
        cfg = VariableConfig(max_beams=200, max_actions=200)
        layout, info = solve(s, cfg, "R01")
        self.assertEqual(info["status"], "geometry_feasible")
        self.assertEqual(len(layout.centers), 200)
        self.assertEqual(metrics(s, layout, cfg)["empty_beams"], 0)
        self.assertEqual(info["metrics"]["covered_users"], 200)
        self.assertEqual(info["metrics"]["total_demand_bps"], 140e9)

    def test_geometry_feasible_does_not_certify_reference_capacity(self):
        s = scene([[45, 10], [45.01, 10.01]], [400e6, 400e6])
        layout, info = solve(s, VariableConfig(max_beams=1), "R02")
        m = metrics(s, layout, VariableConfig(max_beams=1))
        self.assertEqual(info["status"], "geometry_feasible")
        self.assertTrue(m["geometry_feasible"])
        self.assertEqual(m["reference_overloaded_beams"], 1)
        self.assertFalse(m["reference_capacity_feasible"])
        self.assertIn("planning_proxy", m["reference_capacity_scope"])

    def test_r11_objective_and_pattern_sha_invariant_to_aircraft_row_order(self):
        s = scene([[45, 10], [45.01, 10.01], [46, 11]], [5e8, 5e8, 2e8])
        cfg = VariableConfig(max_beams=2, robust_perturbation=.25)
        layout = Layout([[45, 10], [46, 11]], [.5, .5], [0, 0, 1],
                        tuple(s.aircraft_ids))
        order = [2, 0, 1]
        reordered = scene(s.positions[order], s.demands[order])
        reordered.aircraft_ids = [s.aircraft_ids[u] for u in order]
        reordered_layout = Layout(layout.centers, layout.widths,
                                  layout.assignment[order], tuple(reordered.aircraft_ids),
                                  layout.beam_ids)
        self.assertAlmostEqual(_objective(s, layout, cfg, route="R11"),
                               _objective(reordered, reordered_layout, cfg, route="R11"), places=12)
        _, original_hashes = _robust_demand_patterns(s, cfg)
        _, reordered_hashes = _robust_demand_patterns(reordered, cfg)
        self.assertEqual(original_hashes, reordered_hashes)
        self.assertEqual(len(original_hashes), 4)

    def test_r12_fallback_retires_old_ids_and_uses_next_serial(self):
        cfg = VariableConfig(max_beams=3)
        old_scene = scene([[45, 10], [46, 11]], [1e8, 1e8])
        new_scene = scene([[50, 20], [51, 21]], [1e8, 1e8])
        new_scene.aircraft_ids = ["u1", "u3"]
        previous, _ = initial_layout(old_scene, cfg, "R12")
        rebuilt, info = initial_layout(new_scene, cfg, "R12", previous=previous)
        self.assertTrue(info["warm_start_fallback"])
        self.assertEqual(info["warm_start_fallback_reason"],
                         "previous_layout_does_not_cover_current_aircraft")
        self.assertTrue(set(previous.beam_ids).isdisjoint(rebuilt.beam_ids))
        self.assertEqual(info["retired_beam_ids"], sorted(previous.beam_ids))
        self.assertEqual(info["created_beam_ids"], sorted(rebuilt.beam_ids))
        self.assertGreaterEqual(rebuilt.next_beam_serial, previous.next_beam_serial)
        self.assertFalse(info["hysteresis_implemented"])
        self.assertEqual(info["switch_policy"], "switch_cost_only_no_hysteresis_v1")

    def test_public_geometry_audit_counts_nested_core_calls(self):
        with geometry_audit() as audit:
            layout, init = initial_layout(self.scene, self.cfg, "R04")
            menu = candidates(self.scene, layout, self.cfg)
            metrics(self.scene, layout, self.cfg)
        self.assertGreaterEqual(audit["matrix_calls"], init["actual_geometry_calls"] + 2)
        self.assertGreater(audit["ray_pair_checks"], 0)
        self.assertTrue(menu)

    def test_equal_score_candidate_order_survives_successive_coverage(self):
        # Three disconnected, equal-demand users force three scans.  The
        # frozen constructor chooses the lexicographically ordered sites on
        # every exact tie, including after earlier users become covered.
        points = [[40, -10], [50, 10], [60, 30]]
        target = scene(points, [1.0, 1.0, 1.0])
        layout, info = _construct(target, VariableConfig(max_beams=3),
                                  "capacity_cluster", 0.5)
        self.assertEqual(layout.centers.tolist(), [[40.0, -10.0], [50.0, 10.0],
                                                  [60.0, 30.0]])
        self.assertEqual(info["candidate_scans"], 3)
        self.assertEqual(info["construction_termination"], "all_covered")
        self.assertEqual(layout.assignment.tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
