"""Transient Windows sharing violations must not discard atomic run records."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from satellite_rl.training import experiment_runner, run_monitor


def windows_error(code):
    error = PermissionError(13, "simulated Windows file lock")
    error.winerror = code
    return error


class AtomicReplaceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_status_json_retries_sharing_violations_then_replaces(self):
        path = self.directory / "status.json"
        path.write_text('{"old": true}', encoding="utf-8")
        original_replace = os.replace
        calls = []

        def temporarily_locked(source, destination):
            calls.append((source, destination))
            if len(calls) == 1:
                raise windows_error(5)
            if len(calls) == 2:
                raise windows_error(32)
            original_replace(source, destination)

        with mock.patch.object(run_monitor.os, "replace", side_effect=temporarily_locked), \
             mock.patch.object(run_monitor.time, "sleep") as sleep:
            run_monitor.atomic_json(path, {"new": 1})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"new": 1})
        self.assertEqual(len(calls), 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.05, 0.1])
        self.assertFalse(list(self.directory.glob("*.tmp")))

    def test_non_transient_error_is_raised_without_retry(self):
        source = self.directory / "candidate"
        target = self.directory / "status.json"
        source.write_text("new", encoding="utf-8")
        target.write_text("old", encoding="utf-8")
        with mock.patch.object(run_monitor.os, "replace", side_effect=windows_error(2)) as replace, \
             mock.patch.object(run_monitor.time, "sleep") as sleep:
            with self.assertRaises(PermissionError):
                run_monitor.replace_with_retry(source, target)
        replace.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(target.read_text(encoding="utf-8"), "old")

    def test_exhausted_retries_raise_and_keep_previous_status(self):
        path = self.directory / "status.json"
        path.write_text('{"old": true}', encoding="utf-8")
        with mock.patch.object(run_monitor.os, "replace", side_effect=windows_error(32)) as replace, \
             mock.patch.object(run_monitor.time, "sleep") as sleep:
            with self.assertRaises(PermissionError):
                run_monitor.atomic_json(path, {"new": 1})
        self.assertEqual(replace.call_count, 8)
        self.assertEqual(sleep.call_count, 7)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"old": True})

    def test_checkpoint_npz_and_actor_use_the_same_retry(self):
        original_replace = os.replace
        failures = set()

        def once_per_output(source, destination):
            if destination not in failures:
                failures.add(destination)
                raise windows_error(5)
            original_replace(source, destination)

        with mock.patch.object(run_monitor.os, "replace", side_effect=once_per_output), \
             mock.patch.object(run_monitor.time, "sleep"):
            arrays_path = self.directory / "scene.npz"
            experiment_runner._atomic_npz(arrays_path, {"rate": np.array([1.0, 2.0])})
            actor_path = self.directory / "actor.pt"
            experiment_runner.save_actor_weights(actor_path, torch.nn.Linear(2, 1))
            checkpoint_path = self.directory / "latest.pt"
            with mock.patch.object(experiment_runner, "_ensure_finite_agent"), \
                 mock.patch.object(experiment_runner, "_checkpoint_payload",
                                   return_value={"env_step": 100}):
                experiment_runner.save_checkpoint(
                    checkpoint_path, object(), object(), {"env_step": 100}, "config")
        with np.load(arrays_path) as arrays:
            np.testing.assert_array_equal(arrays["rate"], [1.0, 2.0])
        self.assertTrue(actor_path.exists())
        self.assertEqual(torch.load(checkpoint_path, map_location="cpu", weights_only=False),
                         {"env_step": 100})
        self.assertEqual(failures, {arrays_path, actor_path, checkpoint_path})


if __name__ == "__main__":
    unittest.main()
