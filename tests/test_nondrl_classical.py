import unittest
import numpy as np

from satellite_rl.coverage.nondrl import Problem, solve
from satellite_rl.coverage.nondrl.core import ray_to_location
from satellite_rl.coverage.nondrl.classical import (dominates, nondominated_sort, find_center,
    soft_quota_transport, integral_transport_assignment, solve_pool, square_breakpoints,
    graph_adjacency, canonical_sha, CLASSICAL_METHODS, direct_rule_layout, build_pool)
from satellite_rl.coverage.geometry import satellite_rays_numpy, ray_angle_numpy
from satellite_rl.coverage.scenes import CoverageScene
from satellite_rl.coverage.variable_design import Layout
from test_nondrl_core import small_scene, constant_physical


class NondrlClassicalTests(unittest.TestCase):
    def test_binary_MILP_balances_indivisible_demand_not_aircraft_count(self):
        scene, initial = small_scene()
        problem = Problem(scene, {"online_seconds": 5., "mip_seconds": 1.}, initial, constant_physical)
        result = solve_pool(problem, initial.centers)
        self.assertIsNotNone(result)
        centers, assignment, record = result
        loads = np.bincount(assignment, weights=scene.demands, minlength=2)
        np.testing.assert_array_equal(np.sort(loads), [30., 90.])
        self.assertEqual(len(np.unique(assignment)), 2)
        self.assertGreaterEqual(record["square_approximation_error"], -1e-8)
        self.assertEqual(record["certificate_scope"], "finite_pool_piecewise_square_surrogate_only")

    def test_pool_missing_aircraft_is_not_physical_infeasibility(self):
        scene, initial = small_scene()
        problem = Problem(scene, {"online_seconds": 5.}, initial, constant_physical)
        mask = np.ones((4, 2), dtype=bool)
        mask[3] = False
        self.assertIsNone(solve_pool(problem, initial.centers, edge_mask=mask))
        self.assertEqual(problem.trace[-1]["reason"], "infeasible_pool_coverage")
        self.assertEqual(problem.trace[-1]["missing_members"], [3])

    def test_square_secant_upper_approximation_and_error(self):
        nodes = square_breakpoints(10, 16)
        for point in np.linspace(0, 10, 201):
            linear = max((a + b) * point - a * b for a, b in zip(nodes[:-1], nodes[1:]))
            self.assertGreaterEqual(linear, point ** 2 - 1e-10)
            self.assertLessEqual(linear - point ** 2, max(np.diff(nodes) ** 2) / 4 + 1e-10)

    def test_soft_quota_cannot_invent_edges_or_split_published_aircraft(self):
        demand = np.array([90., 10.])
        legal = np.array([[True, False], [True, True]])
        mass, record = soft_quota_transport(demand, np.zeros((2, 2)), legal)
        np.testing.assert_allclose(mass.sum(axis=1), demand, atol=1e-10)
        self.assertEqual(mass[0, 1], 0.)
        self.assertGreater(record["quota_residual_l1"], 0.79)
        assignment = integral_transport_assignment(demand, mass, legal, [0, 1])
        np.testing.assert_array_equal(assignment, [0, 1])
        np.testing.assert_array_equal(np.bincount(assignment, weights=demand), [90., 10.])

    def test_pairwise_graph_clique_requires_group_footprint(self):
        origin = satellite_rays_numpy(np.array([[45., 13.]]))[0]
        first = np.cross(origin, [0., 0., 1.]); first /= np.linalg.norm(first)
        second = np.cross(origin, first)
        radius = np.deg2rad(0.28)
        rays = [np.cos(radius) * origin + np.sin(radius) *
                (np.cos(t) * first + np.sin(t) * second) for t in (0., 2*np.pi/3, 4*np.pi/3)]
        positions = np.array([ray_to_location(ray) for ray in rays])
        scene = CoverageScene("triangle", positions, np.ones(3), np.empty((0, 2)), ["a", "b", "c"])
        initial = Layout(positions, [0.5]*3, [0, 1, 2], tuple(scene.aircraft_ids))
        problem = Problem(scene, {"online_seconds": 5.}, initial, constant_physical)
        edges, _ = graph_adjacency(problem)
        self.assertEqual(int(edges.sum()), 6)
        self.assertIsNone(find_center(problem, np.arange(3)))
        self.assertEqual(problem.current.assignment.tolist(), [0, 1, 2])

    def test_pareto_equal_vectors_and_example(self):
        points = [[.4, -.8, 50], [.45, -.83, 50], [.5, -.78, 55], [.4, -.8, 50]]
        front = nondominated_sort(points)[0]
        self.assertEqual(set(front), {0, 1, 3})
        self.assertFalse(dominates(points[0], points[3]))

    def test_canonical_hash_ignores_beam_numbering(self):
        _, initial = small_scene()
        other = Layout(initial.centers[::-1], initial.widths[::-1], 1-initial.assignment,
                       initial.aircraft_ids)
        self.assertEqual(canonical_sha(initial), canonical_sha(other))

    def test_all_classical_routes_execute_independent_mechanisms(self):
        scene, initial = small_scene()
        signatures = set()
        for route in ("06", "07", "08", "09", "10", "11", "12"):
            result = solve(scene, route, {"online_seconds": 3., "mip_seconds": .2,
                           "max_rounds": 2, "population": 4, "generations": 2}, initial,
                           physical_evaluator=constant_physical)
            self.assertEqual(result["metrics_all"]["beam_count"], 2)
            self.assertTrue(result["metrics_all"]["coverage_100"])
            self.assertLessEqual(result["metrics_all"]["load_cv_all"],
                                 result["initial_metrics_all"]["load_cv_all"] + 1e-10)
            signatures.add(tuple(row["source"] for row in result["trace"]))
        self.assertEqual(len(signatures), 7)

    def test_declared_ablations_are_executable(self):
        scene, initial = small_scene()
        for route, methods in CLASSICAL_METHODS.items():
            for method in methods:
                if method == "main":
                    continue
                result = solve(scene, route, {"online_seconds": 2., "mip_seconds": .1,
                    "max_rounds": 2, "population": 4, "generations": 2}, initial,
                    method=method, physical_evaluator=constant_physical)
                self.assertEqual(result["method"], method)
                self.assertTrue(result["metrics_all"]["coverage_100"])

    def test_direct_rule_does_not_copy_I0_centers_and_keeps_N(self):
        scene, initial = small_scene()
        initial.centers[:] = [45.025, 13.025]
        problem = Problem(scene, {"online_seconds": 2.}, initial, constant_physical)
        proposal = direct_rule_layout(problem)
        self.assertEqual(len(proposal.centers), 2)
        self.assertFalse(np.array_equal(proposal.centers, initial.centers))
        self.assertEqual(set(proposal.assignment), {0, 1})

    def test_pool_keeps_original_sites_and_caps_total(self):
        scene, initial = small_scene()
        problem = Problem(scene, {"online_seconds": 2., "max_pool": 3}, initial, constant_physical)
        pool, sources = build_pool(problem, expansion=1)
        np.testing.assert_array_equal(pool[:2], initial.centers)
        self.assertLessEqual(len(pool), 3)


if __name__ == "__main__":
    unittest.main()
