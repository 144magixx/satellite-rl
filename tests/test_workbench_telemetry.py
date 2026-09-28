"""Ordering, failure and flush contracts for workbench telemetry sinks."""

import json
from pathlib import Path
import random
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np
import torch

from satellite_rl.workbench.telemetry import (
    AsyncTelemetrySink, SyncTelemetrySink, TelemetrySinkError,
)
from satellite_rl.workbench import telemetry


class FakeTensorBoard:
    def __init__(self, block=None, entered=None):
        self.block = block
        self.entered = entered
        self.calls = []
        self.threads = []
        self.closed = False

    def add_scalar(self, tag, value, step):
        self.threads.append(threading.get_ident())
        if self.entered is not None:
            self.entered.set()
        if self.block is not None and not self.block.wait(5):
            raise TimeoutError("test writer remained blocked")
        self.calls.append((tag, value, step))

    def flush(self):
        self.threads.append(threading.get_ident())

    def close(self):
        self.threads.append(threading.get_ident())
        self.closed = True


class TelemetryTests(unittest.TestCase):
    def test_sync_and_async_preserve_json_and_scalar_values(self):
        for sink_type in (SyncTelemetrySink, AsyncTelemetrySink):
            with self.subTest(sink=sink_type.__name__), tempfile.TemporaryDirectory() as tmp:
                writer = FakeTensorBoard()
                sink = sink_type()
                sink.attach_tensorboard(writer)
                log = Path(tmp) / "metrics.jsonl"
                status = Path(tmp) / "status.json"
                item = {"name": "波束", "values": np.array([1, 2], dtype=np.int64),
                        "path": Path(tmp), "flag": np.bool_(True)}
                sink.append_jsonl(log, item)
                sink.write_status(status, {"stage": "running", "step": np.int64(2)})
                sink.record_tensorboard("train", 2, {"si": np.float32(0.75)})
                sink.flush()
                sink.close()
                self.assertEqual(json.loads(log.read_text(encoding="utf-8")),
                                 {"name": "波束", "values": [1, 2], "path": tmp, "flag": True})
                self.assertEqual(json.loads(status.read_text(encoding="utf-8")),
                                 {"stage": "running", "step": 2})
                self.assertEqual(writer.calls, [("train/si", 0.75, 2)])
                self.assertTrue(writer.closed)
                self.assertIsNotNone(sink.health["last_flushed_at"])
                self.assertEqual(sink.health["counts"],
                                 {"append": 1, "status": 1, "tensorboard": 1})
                if sink_type is AsyncTelemetrySink:
                    self.assertEqual(len(set(writer.threads)), 1)
                    self.assertNotEqual(writer.threads[0], threading.get_ident())
                else:
                    self.assertEqual(set(writer.threads), {threading.get_ident()})

    def test_async_captures_mutable_inputs_and_flushes_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            block, entered = threading.Event(), threading.Event()
            writer = FakeTensorBoard(block, entered)
            sink = AsyncTelemetrySink(writer, max_queue=2, flush_interval=2)
            try:
                sink.record_tensorboard("busy", 1, {"x": 1})
                self.assertTrue(entered.wait(2))
                row = {"nested": [1, {"array": np.array([2, 3])}]}
                sink.append_jsonl(Path(tmp) / "rows.jsonl", row)
                row["nested"][1]["array"][0] = 99
                row["nested"].append("late")
                block.set()
                sink.flush()
                actual = json.loads((Path(tmp) / "rows.jsonl").read_text(encoding="utf-8"))
                self.assertEqual(actual, {"nested": [1, {"array": [2, 3]}]})
            finally:
                block.set()
                sink.close()

    def test_async_concurrent_producers_keep_each_stream_order_without_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            sink = AsyncTelemetrySink(max_queue=8, flush_interval=0.05)
            errors = []

            def producer(group):
                try:
                    for index in range(40):
                        sink.append_jsonl(path, {"group": group, "index": index})
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=producer, args=(group,)) for group in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
                self.assertFalse(thread.is_alive())
            sink.flush()
            sink.close()
            self.assertFalse(errors)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 160)
            for group in range(4):
                self.assertEqual([row["index"] for row in rows if row["group"] == group],
                                 list(range(40)))
            self.assertEqual(sink.health["counts"]["append"], 160)

    def test_status_cannot_overtake_prior_buffered_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "updates.jsonl"
            status = Path(tmp) / "status.json"
            sink = AsyncTelemetrySink(flush_interval=2)
            try:
                sink.append_jsonl(log, {"step": 1})
                sink.write_status(status, {"env_step": 1})
                deadline = time.monotonic() + 2
                while not status.exists() and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertTrue(status.exists())
                self.assertEqual(json.loads(log.read_text(encoding="utf-8")), {"step": 1})
            finally:
                sink.close()

    def test_queue_backpressure_and_close_do_not_deadlock(self):
        with tempfile.TemporaryDirectory() as tmp:
            block, entered = threading.Event(), threading.Event()
            sink = AsyncTelemetrySink(FakeTensorBoard(block, entered), max_queue=1,
                                      close_timeout=3)
            path = Path(tmp) / "events.jsonl"
            try:
                sink.record_tensorboard("busy", 1, {"x": 1})
                self.assertTrue(entered.wait(2))
                sink.append_jsonl(path, {"index": 1})
                finished = threading.Event()

                def enqueue_third():
                    sink.append_jsonl(path, {"index": 2})
                    finished.set()

                thread = threading.Thread(target=enqueue_third)
                thread.start()
                time.sleep(0.05)
                self.assertFalse(finished.is_set())
                block.set()
                thread.join(2)
                self.assertFalse(thread.is_alive())
                sink.close()
                self.assertEqual([json.loads(line)["index"] for line in path.read_text().splitlines()],
                                 [1, 2])
                self.assertGreater(sink.health["enqueue_block_seconds"], 0.04)
            finally:
                block.set()
                if not sink.health["closed"]:
                    sink.close()

    def test_async_io_error_reaches_flush_new_calls_and_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent_file = Path(tmp) / "not_a_directory"
            parent_file.write_text("occupied", encoding="utf-8")
            sink = AsyncTelemetrySink(max_queue=1, close_timeout=2)
            sink.append_jsonl(parent_file / "events.jsonl", {"step": 1})
            with self.assertRaises(TelemetrySinkError):
                sink.flush()
            with self.assertRaises(TelemetrySinkError):
                sink.append_jsonl(Path(tmp) / "other.jsonl", {"step": 2})
            self.assertIsNotNone(sink.health["error"])
            with self.assertRaises(TelemetrySinkError):
                sink.close()
            self.assertTrue(sink.health["closed"])

    def test_queued_tensorboard_attach_is_closed_after_prior_io_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            entered, release = threading.Event(), threading.Event()
            original = telemetry._atomic_status

            def hold_status(path, text):
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("test status remained blocked")
                return original(path, text)

            parent_file = Path(tmp) / "not_a_directory"
            parent_file.write_text("occupied", encoding="utf-8")
            writer = FakeTensorBoard()
            with mock.patch.object(telemetry, "_atomic_status", side_effect=hold_status):
                sink = AsyncTelemetrySink(max_queue=2, close_timeout=3)
                try:
                    sink.write_status(Path(tmp) / "status.json", {"step": 1})
                    self.assertTrue(entered.wait(2))
                    sink.append_jsonl(parent_file / "events.jsonl", {"step": 2})
                    sink.attach_tensorboard(writer)
                    release.set()
                    with self.assertRaises(TelemetrySinkError):
                        sink.close()
                    self.assertTrue(writer.closed)
                    self.assertEqual(set(writer.threads), {sink._thread.ident})
                    self.assertTrue(sink.health["closed"])
                finally:
                    release.set()
                    if not sink.health["closed"]:
                        with self.assertRaises(TelemetrySinkError):
                            sink.close()

    def test_stalled_external_writer_times_out_then_can_close(self):
        block, entered = threading.Event(), threading.Event()
        sink = AsyncTelemetrySink(FakeTensorBoard(block, entered),
                                  max_queue=1, close_timeout=0.1)
        try:
            sink.record_tensorboard("stalled", 1, {"value": 1})
            self.assertTrue(entered.wait(2))
            with self.assertRaisesRegex(TelemetrySinkError, "Timed out"):
                sink.flush(timeout=0.1)
            with self.assertRaisesRegex(TelemetrySinkError, "Timed out"):
                sink.close()
        finally:
            block.set()
            sink.close()

    def test_invalid_json_is_rejected_before_enqueue_without_poisoning_sink(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = AsyncTelemetrySink()
            try:
                with self.assertRaises(ValueError):
                    sink.append_jsonl(Path(tmp) / "events.jsonl", {"value": float("nan")})
                self.assertEqual(sink.health["enqueued_count"], 0)
                sink.append_jsonl(Path(tmp) / "events.jsonl", {"value": 1})
                sink.flush()
            finally:
                sink.close()

    def test_sink_operations_do_not_advance_training_random_streams(self):
        random.seed(51001)
        np.random.seed(51001)
        torch.manual_seed(51001)
        before_python = random.getstate()
        before_numpy = np.random.get_state()
        before_torch = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as tmp:
            sink = AsyncTelemetrySink(FakeTensorBoard(), flush_interval=0.01)
            sink.append_jsonl(Path(tmp) / "rows.jsonl", {"value": 1})
            sink.write_status(Path(tmp) / "status.json", {"step": 1})
            sink.record_tensorboard("metric", 1, {"value": 1})
            sink.flush()
            sink.close()
        self.assertEqual(random.getstate(), before_python)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], before_numpy[0])
        np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
        self.assertEqual(after_numpy[2:], before_numpy[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), before_torch))


if __name__ == "__main__":
    unittest.main()
