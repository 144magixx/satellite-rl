"""Ordered workbench telemetry with synchronous and bounded asynchronous sinks.

Callers pass only copied JSON values and scalar metrics. A SummaryWriter may be
constructed on the main thread before training is seeded, then attached here;
only the sink's writer thread calls its methods in asynchronous mode.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
import uuid

import numpy as np

from satellite_rl.training.run_monitor import replace_with_retry


class TelemetrySinkError(RuntimeError):
    """An accepted telemetry operation could not be written safely."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _json_text(value, *, indent=None):
    # Serialization is done by the producer, fixing nested mutable values at
    # call time and reporting NaN/unsupported values before enqueueing.
    return json.dumps(_jsonable(value), ensure_ascii=False,
                      indent=indent, allow_nan=False)


def _scalars(group, step, clean):
    if not isinstance(clean, dict):
        raise TypeError("TensorBoard metrics must be a dictionary")
    values = tuple((f"{group}/{key}", float(value)) for key, value in clean.items())
    if not all(math.isfinite(value) for _, value in values):
        raise ValueError("TensorBoard metrics must be finite")
    return int(step), values


def _atomic_status(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class SyncTelemetrySink:
    """Reference behavior: append, status replacement and TB flush inline."""

    def __init__(self, tensorboard_writer=None):
        self._writer = tensorboard_writer
        self._closed = False
        self._error = None
        self._counts = {"append": 0, "status": 0, "tensorboard": 0}
        self._write_seconds = 0.0
        self._last_flushed_at = None

    def _check(self):
        if self._error is not None:
            raise TelemetrySinkError("Synchronous telemetry write failed") from self._error
        if self._closed:
            raise TelemetrySinkError("Telemetry sink is closed")

    def _execute(self, kind, operation):
        self._check()
        start = time.perf_counter()
        try:
            operation()
        except BaseException as exc:
            self._error = exc
            raise TelemetrySinkError("Synchronous telemetry write failed") from exc
        finally:
            self._write_seconds += time.perf_counter() - start
        self._counts[kind] += 1

    def append_jsonl(self, path, value):
        self._check()
        line = _json_text(value) + "\n"

        def write():
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()

        self._execute("append", write)

    def write_status(self, path, value):
        self._check()
        payload = _json_text(value, indent=2)
        self._execute("status", lambda: _atomic_status(path, payload))

    def record_tensorboard(self, group, step, clean):
        self._check()
        captured_step, values = _scalars(group, step, clean)

        def write():
            if self._writer is not None:
                for tag, value in values:
                    self._writer.add_scalar(tag, value, captured_step)
                self._writer.flush()

        self._execute("tensorboard", write)

    def attach_tensorboard(self, writer):
        self._check()
        if writer is None or self._writer is not None:
            raise ValueError("TensorBoard writer must be new and non-null")
        self._writer = writer

    def flush(self):
        self._check()
        start = time.perf_counter()
        try:
            if self._writer is not None:
                self._writer.flush()
        except BaseException as exc:
            self._error = exc
            raise TelemetrySinkError("Synchronous telemetry flush failed") from exc
        finally:
            self._write_seconds += time.perf_counter() - start
        self._last_flushed_at = _now()

    def close(self):
        if self._closed:
            if self._error is not None:
                raise TelemetrySinkError("Synchronous telemetry failed") from self._error
            return
        if self._writer is not None:
            try:
                self._writer.flush()
            except BaseException as exc:
                self._error = self._error or exc
            try:
                self._writer.close()
            except BaseException as exc:
                self._error = self._error or exc
        self._last_flushed_at = _now()
        self._closed = True
        if self._error is not None:
            raise TelemetrySinkError("Synchronous telemetry failed") from self._error

    @property
    def health(self):
        return {"mode": "sync", "queue_size": 0, "max_queue": 0,
                "enqueued_count": sum(self._counts.values()),
                "processed_count": sum(self._counts.values()),
                "counts": dict(self._counts), "enqueue_block_seconds": 0.0,
                "worker_write_seconds": self._write_seconds,
                "last_flushed_at": self._last_flushed_at,
                "error": repr(self._error) if self._error is not None else None,
                "closed": self._closed}


class AsyncTelemetrySink:
    """One bounded FIFO writer; accepted append records are never coalesced."""

    def __init__(self, tensorboard_writer=None, *, max_queue=2048,
                 flush_interval=2.0, close_timeout=30.0):
        if max_queue < 1 or flush_interval <= 0 or close_timeout <= 0:
            raise ValueError("Queue size and timeouts must be positive")
        self._queue = Queue(maxsize=max_queue)
        self._flush_interval = float(flush_interval)
        self._close_timeout = float(close_timeout)
        self._submission_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._worker_done = threading.Event()
        self._closing = False
        self._closed = False
        self._error = None
        self._writer = None
        self._writer_pending = False
        self._handles = {}
        self._counts = {"append": 0, "status": 0, "tensorboard": 0}
        self._enqueued_count = 0
        self._processed_count = 0
        self._enqueue_block_seconds = 0.0
        self._worker_write_seconds = 0.0
        self._last_flushed_at = None
        self._last_flush_monotonic = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="workbench-telemetry", daemon=True)
        self._thread.start()
        if tensorboard_writer is not None:
            self.attach_tensorboard(tensorboard_writer)

    def _raise_error(self):
        with self._stats_lock:
            error = self._error
        if error is not None:
            raise TelemetrySinkError("Asynchronous telemetry writer failed") from error

    def _record_error(self, error):
        with self._stats_lock:
            if self._error is None:
                self._error = error

    def _offer(self, item, *, deadline=None):
        try:
            self._queue.put_nowait(item)
            return
        except Full:
            pass
        start = time.perf_counter()
        try:
            while True:
                if self._worker_done.is_set():
                    self._raise_error()
                    raise TelemetrySinkError("Telemetry writer stopped unexpectedly")
                timeout = 0.1 if deadline is None else min(0.1, max(0, deadline - time.monotonic()))
                if deadline is not None and timeout == 0:
                    raise TelemetrySinkError("Timed out waiting for telemetry queue")
                try:
                    self._queue.put(item, timeout=timeout)
                    return
                except Full:
                    self._raise_error()
        finally:
            with self._stats_lock:
                self._enqueue_block_seconds += time.perf_counter() - start

    def _submit(self, item, *, deadline=None):
        if deadline is None:
            self._submission_lock.acquire()
        elif not self._submission_lock.acquire(timeout=max(0, deadline - time.monotonic())):
            raise TelemetrySinkError("Timed out waiting for telemetry producers")
        try:
            self._raise_error()
            if self._closing:
                raise TelemetrySinkError("Telemetry sink is closing")
            self._offer(item, deadline=deadline)
            with self._stats_lock:
                self._enqueued_count += 1
            self._raise_error()
        finally:
            self._submission_lock.release()

    def append_jsonl(self, path, value):
        self._raise_error()
        self._submit(("append", Path(path), _json_text(value) + "\n", None))

    def write_status(self, path, value):
        self._raise_error()
        self._submit(("status", Path(path), _json_text(value, indent=2), None))

    def record_tensorboard(self, group, step, clean):
        self._raise_error()
        self._submit(("tensorboard", None, _scalars(group, step, clean), None))

    def attach_tensorboard(self, writer):
        if writer is None:
            raise ValueError("TensorBoard writer must be non-null")
        with self._submission_lock:
            self._raise_error()
            if self._closing:
                raise TelemetrySinkError("Telemetry sink is closing")
            if self._writer_pending:
                raise ValueError("TensorBoard writer already attached")
            self._writer_pending = True
            try:
                self._offer(("attach", None, writer, None))
                with self._stats_lock:
                    self._enqueued_count += 1
                self._raise_error()
            except BaseException:
                self._writer_pending = False
                raise

    def _flush_buffers(self):
        for stream in self._handles.values():
            stream.flush()
            os.fsync(stream.fileno())
        if self._writer is not None:
            self._writer.flush()
        with self._stats_lock:
            self._last_flushed_at = _now()
        self._last_flush_monotonic = time.monotonic()

    def _run(self):
        try:
            while True:
                try:
                    item = self._queue.get(timeout=self._flush_interval)
                except Empty:
                    if self._error is None:
                        start = time.perf_counter()
                        try:
                            self._flush_buffers()
                        except BaseException as exc:
                            self._record_error(exc)
                        finally:
                            with self._stats_lock:
                                self._worker_write_seconds += time.perf_counter() - start
                    continue
                kind, path, payload, signal = item
                start = time.perf_counter()
                try:
                    if self._error is None or kind == "attach":
                        if kind == "append":
                            stream = self._handles.get(path)
                            if stream is None:
                                path.parent.mkdir(parents=True, exist_ok=True)
                                stream = path.open("a", encoding="utf-8")
                                self._handles[path] = stream
                            stream.write(payload)
                        elif kind == "status":
                            # A status must never overtake buffered log/TB rows.
                            self._flush_buffers()
                            _atomic_status(path, payload)
                        elif kind == "tensorboard":
                            if self._writer is not None:
                                step, values = payload
                                for tag, value in values:
                                    self._writer.add_scalar(tag, value, step)
                        elif kind == "attach":
                            if self._writer is not None:
                                raise ValueError("TensorBoard writer already attached")
                            self._writer = payload
                        elif kind in ("flush", "stop"):
                            self._flush_buffers()
                        else:
                            raise ValueError(f"Unknown telemetry operation: {kind}")
                        with self._stats_lock:
                            self._processed_count += 1
                            if kind in self._counts:
                                self._counts[kind] += 1
                except BaseException as exc:
                    self._record_error(exc)
                finally:
                    with self._stats_lock:
                        self._worker_write_seconds += time.perf_counter() - start
                    if signal is not None:
                        signal.set()
                    self._queue.task_done()
                if kind == "stop":
                    break
                if self._error is None and time.monotonic() - self._last_flush_monotonic >= self._flush_interval:
                    start = time.perf_counter()
                    try:
                        self._flush_buffers()
                    except BaseException as exc:
                        self._record_error(exc)
                    finally:
                        with self._stats_lock:
                            self._worker_write_seconds += time.perf_counter() - start
        except BaseException as exc:
            self._record_error(exc)
        finally:
            for stream in self._handles.values():
                try:
                    stream.close()
                except BaseException as exc:
                    self._record_error(exc)
            if self._writer is not None:
                try:
                    self._writer.close()
                except BaseException as exc:
                    self._record_error(exc)
            self._worker_done.set()

    def flush(self, timeout=None):
        if timeout is None:
            timeout = self._close_timeout
        if timeout <= 0:
            raise ValueError("Flush timeout must be positive")
        deadline = time.monotonic() + timeout
        signal = threading.Event()
        self._submit(("flush", None, None, signal), deadline=deadline)
        if not signal.wait(max(0, deadline - time.monotonic())):
            raise TelemetrySinkError("Timed out waiting for telemetry flush")
        self._raise_error()

    def close(self):
        if threading.current_thread() is self._thread:
            raise TelemetrySinkError("Telemetry writer cannot close itself")
        deadline = time.monotonic() + self._close_timeout
        if not self._submission_lock.acquire(timeout=max(0, deadline - time.monotonic())):
            raise TelemetrySinkError("Timed out waiting for telemetry producers to close")
        try:
            if not self._closing:
                self._offer(("stop", None, None, None), deadline=deadline)
                self._closing = True
        finally:
            self._submission_lock.release()
        if not self._worker_done.wait(max(0, deadline - time.monotonic())):
            raise TelemetrySinkError("Timed out closing telemetry writer")
        self._closed = True
        self._raise_error()

    @property
    def health(self):
        with self._stats_lock:
            return {"mode": "async", "queue_size": self._queue.qsize(),
                    "max_queue": self._queue.maxsize,
                    "enqueued_count": self._enqueued_count,
                    "processed_count": self._processed_count,
                    "counts": dict(self._counts),
                    "enqueue_block_seconds": self._enqueue_block_seconds,
                    "worker_write_seconds": self._worker_write_seconds,
                    "last_flushed_at": self._last_flushed_at,
                    "error": repr(self._error) if self._error is not None else None,
                    "closed": self._closed}
