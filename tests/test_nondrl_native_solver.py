"""Hard-clock engineering checks use sleeping children and only tiny real MILP."""
from time import perf_counter
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

from satellite_rl.coverage.nondrl.native_solver import NativeSolver
from satellite_rl.coverage.nondrl.classical import solve_pool, run_pool_mip
from satellite_rl.coverage.nondrl.core import Problem, layout_sha256
from test_nondrl_core import small_scene, constant_physical


class NativeSolverTests(unittest.TestCase):
    def test_parent_native_exit_cannot_leave_cpu_child_running(self):
        import satellite_rl.coverage.nondrl.native_solver as module
        code = "\n".join(("import importlib.util,os,time", "p=" + repr(str(Path(module.__file__).resolve())),
            "spec=importlib.util.spec_from_file_location('owned_native',p)", "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)",
            "solver=m.NativeSolver('controlled_sleep')", "solver.start(time.perf_counter()+3)",
            "print(solver.pid,flush=True)", "os._exit(0)"))
        parent = subprocess.Popen([sys.executable, "-B", "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        raw, error = parent.communicate(timeout=5.)
        self.assertEqual(parent.returncode, 0, error)
        pid = int(raw.strip())
        def running():
            if os.name == "nt":
                from ctypes import wintypes
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.OpenProcess.restype = wintypes.HANDLE
                handle = kernel.OpenProcess(0x1000, False, pid)
                if not handle:
                    return False
                value = wintypes.DWORD()
                kernel.GetExitCodeProcess(handle, ctypes.byref(value)); kernel.CloseHandle(handle)
                return value.value == 259
            status = Path(f"/proc/{pid}/stat")
            return status.exists() and status.read_text().rsplit(")", 1)[1].split()[0] != "Z"
        deadline = perf_counter() + 1.
        while running() and perf_counter() < deadline:
            time.sleep(.01)
        self.assertFalse(running(), "owned native child survived its parent")
    def test_hard_timeout_kills_reaps_and_never_returns_late_solution(self):
        solver = NativeSolver("controlled_sleep")
        try:
            ready = solver.start(perf_counter() + 3.)
            pid = ready["worker_pid"]
            begin = perf_counter()
            receipt = solver.solve({"delay_seconds": 2., "x": [123.], "fun": -999.}, begin + .06)
            elapsed = perf_counter() - begin
            self.assertEqual(receipt["outcome"], "engineering_timeout")
            self.assertFalse(receipt["known"])
            self.assertIsNone(receipt["result"])
            self.assertLess(elapsed, .5)
            self.assertEqual(receipt["cleanup"]["worker_pid"], pid)
            self.assertFalse(receipt["cleanup"]["alive_after"])
            self.assertIsNone(solver.pid)
        finally:
            solver.close()

    def test_successful_calls_reuse_one_child_and_timeout_restarts_clean(self):
        solver = NativeSolver("controlled_sleep")
        try:
            first = solver.solve({"x": [1.]}, perf_counter() + 3.)
            second = solver.solve({"x": [2.]}, perf_counter() + .5)
            self.assertEqual(first["timing"]["worker_pid"], second["timing"]["worker_pid"])
            self.assertTrue(second["timing"]["reused"])
            self.assertEqual(second["result"]["x"], [2.])
            old = solver.pid
            solver.solve({"delay_seconds": 2.}, perf_counter() + .03)
            third = solver.solve({"x": [3.]}, perf_counter() + 3.)
            self.assertTrue(third["known"])
            self.assertNotEqual(third["timing"]["worker_pid"], old)
        finally:
            solver.close()

    def test_timed_out_pool_preserves_existing_improved_best(self):
        scene, initial = small_scene()
        problem = Problem(scene, {"online_seconds": 5., "mip_seconds": .05}, initial, constant_physical)
        improved = problem.preview({"kind": "transfer", "aircraft": 1, "target": 1})
        self.assertTrue(problem.commit(improved))
        best = layout_sha256(problem.current)
        solver = NativeSolver("controlled_sleep")
        try:
            solver.start(perf_counter() + 3.)
            class Stalled:
                def solve(self, payload, deadline):
                    return solver.solve({**payload, "delay_seconds": 2., "x": np.ones(len(payload["objective"]))}, deadline)
            outcome = solve_pool(problem, problem.current.centers, interruptible=True, native_solver=Stalled(), source="10_controlled_test")
            self.assertIsNone(outcome)
            self.assertEqual(layout_sha256(problem.current), best)
            record = problem.trace[-1]
            self.assertEqual(record["status"], "engineering_timeout")
            self.assertFalse(record["known"])
            self.assertFalse(record["infeasible"])
        finally:
            solver.close()

    def test_real_tiny_milp_keeps_same_pool_objective_and_ownership(self):
        scene, initial = small_scene()
        solver = NativeSolver()
        try:
            solver.start(perf_counter() + 5.)
            old_problem = Problem(scene, {"online_seconds": 5., "mip_seconds": 1.}, initial, constant_physical)
            new_problem = Problem(scene, {"online_seconds": 5., "mip_seconds": 1.}, initial, constant_physical)
            old = solve_pool(old_problem, initial.centers)
            new = solve_pool(new_problem, initial.centers, interruptible=True, native_solver=solver)
            self.assertIsNotNone(old); self.assertIsNotNone(new)
            np.testing.assert_array_equal(old[0], new[0])
            np.testing.assert_array_equal(old[1], new[1])
            self.assertEqual(old[2]["pool_sha256"], new[2]["pool_sha256"])
            self.assertAlmostEqual(old[2]["model_objective"], new[2]["model_objective"], places=10)
            self.assertAlmostEqual(old[2]["exact_square_normalized"], new[2]["exact_square_normalized"], places=10)
            self.assertTrue(new[2]["native_process"]["known"])
        finally:
            solver.close()

    def test_other_routes_default_never_start_native_child(self):
        scene, initial = small_scene()
        problem = Problem(scene, {"online_seconds": 5., "mip_seconds": 1.}, initial, constant_physical)
        with patch("satellite_rl.coverage.nondrl.native_solver.owned_solver") as native:
            self.assertIsNotNone(solve_pool(problem, initial.centers))
            native.assert_not_called()


if __name__ == "__main__":
    unittest.main()
