"""No scientific work: mock the workbench to verify bounded idempotent dispatch."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.s2p19 import pilot_jobs as jobs


class PilotJobsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rows = {p: [{"run_id": run, "request_id": f"s2p19-{arm.lower()}-train-51001",
                          "stage": "training", "env_step": 90000}]
                     for arm, (p, run) in jobs.RUNS.items()}
        self.posts = []

    def api(self, port, path, body=None, method="GET", **kwargs):
        if method == "POST":
            self.posts.append((port, path, body))
            return {"run_id": "created"}
        if path == "/api/s2p19/runs":
            return {"runs": self.rows[port]}
        return {"job": next(r for r in self.rows[port] if path.endswith(r["run_id"]))}

    def invoke(self, dispatch):
        with patch.object(jobs, "OUTPUT", Path(self.tmp.name) / "jobs.json"), \
             patch.object(jobs.ops, "api", side_effect=self.api), \
             patch.object(jobs, "controls", return_value={"repeats": 1}):
            return jobs.pass_once(dispatch)

    def test_read_only_does_not_dispatch(self):
        self.invoke(False)
        self.assertEqual(self.posts, [])

    def test_one_pass_submits_only_missing_baseline(self):
        self.rows[20418][0].update(stage="stopped", env_step=100000,
                                 stop_reason="s2p19_pilot_gate_pending")
        self.invoke(True)
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(self.posts[0][2]["request_id"], "s2p19-validation-greedy-direct-51001")

    def test_existing_baseline_uses_one_saved_allocation_for_B(self):
        self.rows[20418].append({"run_id": "baseline", "request_id": "s2p19-validation-greedy-direct-51001",
                                 "stage": "completed"})
        self.invoke(True)
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(self.posts[0][1], "/api/power-refinement/runs")
        self.assertEqual(self.posts[0][2]["source_run_id"], "baseline")

    def test_existing_request_never_duplicates_and_i0_uses_actual_step(self):
        self.rows[20418].extend([
            {"run_id": "baseline", "request_id": "s2p19-validation-greedy-direct-51001", "stage": "completed"},
            {"run_id": "B", "request_id": "s2p19-validation-greedy-B", "stage": "completed"},
            {"run_id": "I0", "request_id": "s2p19-i0-train-51001", "stage": "stopped", "env_step": 54000,
             "stop_reason": "s2p19_pilot_gate_pending"}])
        self.invoke(True)
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(self.posts[0][2]["validation_step"], 54000)
        self.assertEqual(self.posts[0][2]["request_id"], "s2p19-i0-pilot-B")


if __name__ == "__main__":
    unittest.main()
