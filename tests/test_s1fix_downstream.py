import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from satellite_rl.coverage.s1fix.geom import Layout
from satellite_rl.coverage.s1fix.downstream import (
    REQUEST_BPS, build_problem, evaluate_layout, service_metrics)
from satellite_rl.shared_interface.link import capacity_bps


def example_layout(n=12, counts=None):
    centers = np.column_stack((np.linspace(35, 60, n), np.linspace(-15, 25, n)))
    counts = np.arange(1, n + 1) if counts is None else np.asarray(counts)
    return Layout(centers, np.linspace(.3, 1., n), np.repeat(np.arange(n), counts),
                  100 + np.arange(n))


class DownstreamTests(unittest.TestCase):
    def test_random_power_capacity_matches_shared_link(self):
        problem, initial, link = build_problem(example_layout())
        np.testing.assert_array_equal(problem.demand_bps,
                                      np.arange(1, 13) * REQUEST_BPS)
        rng = np.random.default_rng(51001)
        for _ in range(4):
            power = rng.random(problem.beam_count)
            power *= 2999. / power.sum()
            np.testing.assert_allclose(problem.evaluate(power).capacity_bps,
                                       capacity_bps(link, power), rtol=1e-13)
        self.assertLessEqual(initial.sum(), 3000.)

    def test_groups_rescale_and_reserve_nonoverlapping_slots(self):
        problem, _, _ = build_problem(example_layout(200, np.full(200, 100)))
        self.assertTrue(np.all((problem.slots >= 1) & (problem.slots <= 10)))
        self.assertTrue(np.all(problem.starts + problem.slots <= 100))
        for group in range(8):
            occupied = np.zeros(100, int)
            for i in np.flatnonzero(problem.groups == group):
                occupied[problem.starts[i]:problem.starts[i] + problem.slots[i]] += 1
            self.assertLessEqual(occupied.max(), 1)

    def test_aircraft_weighted_service(self):
        problem = SimpleNamespace(demand_bps=np.array([1, 9]) * REQUEST_BPS)
        evaluation = SimpleNamespace(beam_si=np.array([.2, 1.]), si=.6, sgm=.3)
        metrics = service_metrics(problem, evaluation)
        self.assertAlmostEqual(metrics["aircraft_mean"], .92)
        self.assertAlmostEqual(metrics["aircraft_worst10"], .2)
        self.assertAlmostEqual(metrics["aircraft_ge90"], .9)

    def test_small_sca_smoke(self):
        result = evaluate_layout(example_layout(3),
                                 dict(maxiter=1, max_evaluations=8, time_limit_seconds=1.))
        for phase in ("greedy", "sca"):
            self.assertGreaterEqual(result[f"{phase}_aircraft_mean"], 0.)
            self.assertLessEqual(result[f"{phase}_aircraft_mean"], 1.)

    def test_script_variants_share_actual_n(self):
        from scripts.s1fix.run_C import comparison_job
        centers = np.array([[40., -5.], [50., 10.], [60., 20.]])
        positions = np.repeat(centers, 2, axis=0)
        positions[::2, 0] += .01
        frame = SimpleNamespace(date="2024-01-12", time=1705017720., index=0,
                                positions=positions, icao=np.arange(6, dtype=np.uint32))
        with TemporaryDirectory() as directory:
            rows = comparison_job(("DEV24", frame, 5, directory, "dev", 1, None,
                                   dict(maxiter=1, time_limit_seconds=1.)))
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["feasible"] for row in rows))
            self.assertEqual(len({row["N_used"] for row in rows}), 1)
            self.assertLess(rows[0]["N_used"], 5)
            self.assertTrue(all(row["comparison_N"] == row["N_used"] for row in rows))
            self.assertEqual(len(list((Path(directory) / "C/service_arrays").glob("*.npz"))), 3)


if __name__ == "__main__":
    unittest.main()
