"""Bounded checks for the independent S2-P19 final-assessment scheduler."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts.s2p19 import four_lane


class FourLaneTest(unittest.TestCase):
    def plan(self, directory: Path) -> Path:
        plan = directory / "plan.json"
        four_lane.save(plan, {"schema": "s2p19.four_lane.v1", "complete_B_budget": 14960,
                              "completed_B_calls": 4400, "remaining_B_calls": 10560,
                              "tasks": [
                                  {"id": "direct", "port": 20424, "kind": "constructive_evaluate",
                                   "request_id": "original-direct", "adopted_run_id": "real-run",
                                   "expected_commit": "a" * 40, "requires": [],
                                   "body": {"repeats": 3}},
                                  {"id": "B", "port": 20424, "kind": "refine",
                                   "request_id": "new-B", "expected_commit": "a" * 40,
                                   "requires": ["direct"],
                                   "body": {"source_run_id": {"$task_run": "direct"},
                                            "config": {"repeats": 3,
                                                       "methods": ["none", "sca", "sca_si_lp"]}}}]})
        return plan

    def test_adopts_direct_and_submits_only_dependent_B(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = self.plan(root)
            calls = []

            def api(port, endpoint, body=None):
                calls.append((port, endpoint, body))
                if body is None:
                    return ({"runs": [{"request_id": "original-direct", "run_id": "real-run",
                                       "git_commit": "a" * 40, "stage": "completed",
                                       "env_step": 2640, "budget_env_steps": 2640}]}
                            if port == 20424 else {"runs": []})
                return {"run_id": "new-run", "git_commit": "a" * 40}

            with patch.object(four_lane, "FINAL_IDS", ("direct", "B")), patch.object(
                    four_lane, "api", side_effect=api):
                report = four_lane.poll(plan, root / "state.json", True)
            self.assertEqual([item["submitted"] for item in report["changes"]], ["B"])
            self.assertEqual(calls[-1][2]["source_run_id"], "real-run")
            self.assertEqual(calls[-1][2]["request_id"], "new-B")

    def test_unknown_post_never_reposts_without_catalog_reconciliation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = self.plan(root)

            def api(port, endpoint, body=None):
                if body is not None:
                    raise ConnectionError("lost reply")
                return ({"runs": [{"request_id": "original-direct", "run_id": "real-run",
                                   "git_commit": "a" * 40, "stage": "completed",
                                   "env_step": 2640, "budget_env_steps": 2640}]}
                        if port == 20424 else {"runs": []})

            with patch.object(four_lane, "FINAL_IDS", ("direct", "B")), patch.object(
                    four_lane, "api", side_effect=api):
                first = four_lane.poll(plan, root / "state.json", True)
                self.assertIn("20424", first["transient_errors"])
                with self.assertRaisesRegex(RuntimeError, "reconcile same request"):
                    four_lane.poll(plan, root / "state.json", True)

    def test_transient_lane_loss_does_not_abort_other_lane(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = self.plan(root)

            def api(port, endpoint, body=None):
                if port == 20473:
                    raise ConnectionError("connection refused")
                return ({"runs": [{"request_id": "original-direct", "run_id": "real-run",
                                   "git_commit": "a" * 40, "stage": "running",
                                   "env_step": 120, "budget_env_steps": 2640}]}
                        if port == 20424 else {"runs": []})

            with patch.object(four_lane, "FINAL_IDS", ("direct", "B")), patch.object(
                    four_lane, "api", side_effect=api):
                report = four_lane.poll(plan, root / "state.json", False)
            self.assertIn("20473", report["transient_errors"])
            self.assertEqual(report["tasks"]["direct"]["run_id"], "real-run")

    def test_completed_direct_requires_full_action_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = self.plan(root)

            def api(port, endpoint, body=None):
                return ({"runs": [{"request_id": "original-direct", "run_id": "real-run",
                                   "git_commit": "a" * 40, "stage": "completed",
                                   "env_step": 2639, "budget_env_steps": 2640}]}
                        if port == 20424 else {"runs": []})

            with patch.object(four_lane, "FINAL_IDS", ("direct", "B")), patch.object(
                    four_lane, "api", side_effect=api):
                with self.assertRaisesRegex(RuntimeError, "2639/2640"):
                    four_lane.poll(plan, root / "state.json", True)

    def test_ssh_banner_failure_is_transient_but_host_key_is_not(self):
        class SSHException(Exception):
            pass

        class Client:
            def set_missing_host_key_policy(self, policy):
                self.policy = policy

            def connect(self, *args, **kwargs):
                raise SSHException("banner EOF")

            def close(self):
                pass

        fake = types.SimpleNamespace(MissingHostKeyPolicy=object, SSHException=SSHException,
                                     SSHClient=Client)
        with patch.dict("sys.modules", {"paramiko": fake}), patch.dict(
                "os.environ", {"BUPT_PASSWORD": "test-secret"}):
            with self.assertRaises(ConnectionError):
                four_lane.api(20418, "/api/s2p19/runs")

        class WrongKeyClient(Client):
            def connect(self, *args, **kwargs):
                self.policy.missing_host_key(self, "host", types.SimpleNamespace(
                    asbytes=lambda: b"wrong-host-key"))

        fake.SSHClient = WrongKeyClient
        with patch.dict("sys.modules", {"paramiko": fake}), patch.dict(
                "os.environ", {"BUPT_PASSWORD": "test-secret"}):
            with self.assertRaises(four_lane.HostKeyMismatch):
                four_lane.api(20473, "/api/s2p19/runs")


if __name__ == "__main__":
    unittest.main()
