"""Workbench records survive short Windows sharing violations."""

import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np
import torch

from satellite_rl.training import run_monitor
from satellite_rl.workbench import checkpoints, runner


def windows_error(code):
    error = PermissionError(13, "simulated Windows sharing violation")
    error.winerror = code
    return error


class WorkbenchAtomicIOTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def retry_once(self, operation, *, code=5):
        original_replace = os.replace
        calls = []

        def locked_once(source, destination):
            calls.append((Path(source), Path(destination)))
            if len(calls) == 1:
                raise windows_error(code)
            return original_replace(source, destination)

        with mock.patch.object(run_monitor.os, "replace", side_effect=locked_once), \
             mock.patch.object(run_monitor.time, "sleep") as sleep:
            operation()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], calls[1][1])
        sleep.assert_called_once_with(0.05)
        self.assertEqual(list(self.directory.rglob("*.tmp*")), [])

    def test_runner_status_retries_winerror_5_and_32(self):
        for code in (5, 32):
            with self.subTest(winerror=code):
                path = self.directory / "status.json"
                path.write_text('{"stage":"old"}', encoding="utf-8")
                self.retry_once(lambda: runner._atomic_json(path, {"stage": "training", "env_step": 100}), code=code)
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                                 {"stage": "training", "env_step": 100})

    def test_scene_npz_retries_without_changing_values(self):
        path = self.directory / "scene.npz"
        scene = {"demand": np.array([1.5, 2.5]), "order": np.array([2, 1])}
        self.retry_once(lambda: runner._save_scene(path, scene))
        with np.load(path, allow_pickle=False) as saved:
            self.assertEqual(set(saved.files), set(scene))
            for key, value in scene.items():
                np.testing.assert_array_equal(saved[key], value)

    def test_reconcile_retries_and_keeps_only_checkpoint_prefix(self):
        paths = [self.directory / f"{name}.jsonl" for name in ("metrics", "episodes", "updates", "evaluations", "traces")]
        for path in paths:
            path.write_text(''.join(json.dumps(row) + "\n" for row in
                                    ({"env_step": 99}, {"env_step": 100}, {"env_step": 101})), encoding="utf-8")
        original_replace = os.replace
        attempted = set()

        def locked_once_per_file(source, destination):
            if destination not in attempted:
                attempted.add(destination)
                raise windows_error(32)
            return original_replace(source, destination)

        with mock.patch.object(run_monitor.os, "replace", side_effect=locked_once_per_file), \
             mock.patch.object(run_monitor.time, "sleep") as sleep:
            runner._reconcile(self.directory, 100)
        self.assertEqual(attempted, set(paths))
        self.assertEqual(sleep.call_count, len(paths))
        for path in paths:
            self.assertEqual([json.loads(line)["env_step"] for line in path.read_text(encoding="utf-8").splitlines()],
                             [99, 100])
        self.assertEqual(list(self.directory.glob("*.tmp")), [])

    def test_checkpoint_torch_and_json_retry_with_existing_targets(self):
        weight_path = self.directory / "checkpoint.pt"
        metadata_path = self.directory / "model.json"
        torch.save({"step": 1}, weight_path)
        metadata_path.write_text('{"step":1}', encoding="utf-8")
        self.retry_once(lambda: checkpoints._atomic_torch(weight_path, {"step": 2}))
        self.assertEqual(torch.load(weight_path, map_location="cpu", weights_only=False), {"step": 2})
        self.retry_once(lambda: checkpoints._atomic_json(metadata_path, {"step": 2}))
        self.assertEqual(json.loads(metadata_path.read_text(encoding="utf-8")), {"step": 2})

    def test_exhausted_retries_keep_old_targets_and_clean_temp(self):
        status_path = self.directory / "status.json"
        checkpoint_path = self.directory / "checkpoint.pt"
        metadata_path = self.directory / "model.json"
        status_path.write_text('{"stage":"old"}', encoding="utf-8")
        torch.save({"step": 1}, checkpoint_path)
        metadata_path.write_text('{"step":1}', encoding="utf-8")
        old_bytes = {path: path.read_bytes() for path in (status_path, checkpoint_path, metadata_path)}
        operations = (
            lambda: runner._atomic_json(status_path, {"stage": "new"}),
            lambda: checkpoints._atomic_torch(checkpoint_path, {"step": 2}),
            lambda: checkpoints._atomic_json(metadata_path, {"step": 2}),
        )
        for operation in operations:
            with mock.patch.object(run_monitor.os, "replace", side_effect=windows_error(5)) as replace, \
                 mock.patch.object(run_monitor.time, "sleep") as sleep:
                with self.assertRaises(PermissionError):
                    operation()
            self.assertEqual(replace.call_count, 8)
            self.assertEqual(sleep.call_count, 7)
        self.assertEqual({path: path.read_bytes() for path in old_bytes}, old_bytes)
        self.assertEqual(list(self.directory.rglob("*.tmp*")), [])

    @unittest.skipUnless(os.name == "nt", "Windows file sharing check")
    def test_real_windows_reader_lock_is_retried(self):
        import ctypes
        from ctypes import wintypes

        path = self.directory / "status.json"
        path.write_text('{"stage":"old"}', encoding="utf-8")
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                       ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                       wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        # Permit reads/writes but deny DELETE sharing, which os.replace needs.
        handle = kernel.CreateFileW(str(path), 0x80000000, 0x1 | 0x2, None, 3, 0, None)
        self.assertNotEqual(handle, wintypes.HANDLE(-1).value, ctypes.get_last_error())
        released = threading.Event()

        def release_reader():
            time.sleep(0.12)
            kernel.CloseHandle(handle)
            released.set()

        thread = threading.Thread(target=release_reader)
        thread.start()
        try:
            start = time.monotonic()
            runner._atomic_json(path, {"stage": "resumed"})
            self.assertLess(time.monotonic() - start, 2)
        finally:
            if not released.wait(1):
                kernel.CloseHandle(handle)
            thread.join(timeout=1)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"stage": "resumed"})


if __name__ == "__main__":
    unittest.main()
