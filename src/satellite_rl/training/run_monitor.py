"""Buffered, run-scoped facts and an atomic progress snapshot for long experiments."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid


def replace_with_retry(source, destination, *, attempts=8, initial_delay=0.05):
    """Retry transient Windows sharing violations while keeping atomic replace semantics."""
    if attempts < 1 or initial_delay < 0:
        raise ValueError("attempts must be positive and initial_delay nonnegative")
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in (5, 32) or attempt == attempts - 1:
                raise
            time.sleep(min(initial_delay * 2 ** attempt, 0.5))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class RunMonitor:
    """Keep canonical records open; flush periodically and on every state transition."""

    def __init__(self, directory, run_id, *, resume=False, flush_interval=5.0):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=resume)
        self.run_id = run_id
        self.flush_interval = flush_interval
        self.last_flush = time.monotonic()
        self.handles = {}
        self.writers = {}
        self.fields = {}
        self.recorded = {"episodes": set(), "evaluations": set()}
        self.status_path = self.directory / "status.json"
        self.status = json.loads(self.status_path.read_text(encoding="utf-8")) if resume else {
            "run_id": run_id, "stage": "initializing", "created_at": utc_now(),
        }
        if self.status["run_id"] != run_id:
            raise ValueError("恢复时 run_id 与状态文件不一致")
        self._open_csv("updates", (
            "run_id", "env_step", "update_step", "episode", "wall_seconds",
            "actor_loss", "critic_1_loss", "critic_2_loss", "alpha_loss",
            "alpha", "entropy", "entropy_freq", "entropy_slots", "entropy_power",
            "q_mean", "target_mean",
        ), resume)
        self._open_csv("episodes", (
            "run_id", "episode", "env_step", "length", "return", "wall_seconds",
            "scene_id", "si_sum", "si_mean", "sgm_mean", "throughput_bps",
            "total_power_w", "served_beams", "total_beams", "power_exceeded",
            "completed", "occupancy_fraction", "zero_slot_rate", "allocated_slots",
            "spectral_efficiency", "interference_beams", "in_range_interference_beams",
            "requested_slots_total", "actual_slots_total", "out_slots_total",
            "occupied_slots_total", "clipped_steps", "termination_reason",
        ), resume)
        self._open_csv("evaluations", (
            "run_id", "checkpoint", "env_step", "scene_id", "action_mode", "action_seed",
            "return", "si_sum", "si_mean", "sgm_mean", "throughput_bps", "total_power_w",
            "served_beams", "total_beams", "power_exceeded", "completed",
            "occupancy_fraction", "zero_slot_rate", "allocated_slots",
            "spectral_efficiency", "interference_beams", "in_range_interference_beams",
            "requested_slots_total", "actual_slots_total", "out_slots_total",
            "occupied_slots_total", "clipped_steps",
        ), resume)
        if resume:
            for name in self.recorded:
                with (self.directory / (name + ".csv")).open(newline="", encoding="utf-8") as stream:
                    for row in csv.DictReader(stream):
                        self.recorded[name].add(self._row_key(name, row))
        self.handles["events"] = (self.directory / "events.jsonl").open(
            "a" if resume else "x", encoding="utf-8", buffering=65536
        )
        self.write_status()

    def _open_csv(self, name, fields, resume):
        path = self.directory / (name + ".csv")
        existed = path.exists()
        handle = path.open("a" if resume else "x", newline="", encoding="utf-8", buffering=65536)
        self.handles[name] = handle
        self.fields[name] = fields
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        self.writers[name] = writer
        if not existed:
            writer.writeheader()

    def row(self, name, values):
        if name in self.recorded:
            key = self._row_key(name, values)
            if key in self.recorded[name]:
                return
            self.recorded[name].add(key)
        self.writers[name].writerow({"run_id": self.run_id, **values})
        self.flush_if_due()

    @staticmethod
    def _row_key(name, values):
        if name == "episodes":
            return str(values["episode"])
        return (str(values["checkpoint"]), str(values["env_step"]),
                str(values["scene_id"]), str(values.get("action_mode") or "sample"),
                str(values["action_seed"]))

    def event(self, kind, **values):
        self.handles["events"].write(json.dumps({
            "at": utc_now(), "run_id": self.run_id, "kind": kind, **values,
        }, ensure_ascii=False, allow_nan=False) + "\n")
        self.flush_if_due()

    def reconcile(self, progress):
        """Discard canonical metric rows beyond a trusted resume checkpoint."""
        boundary_step = int(progress["env_step"])
        boundary_update = int(progress["update_step"])
        removed = {}
        for name in ("episodes", "updates", "evaluations"):
            self.handles[name].close()
            path = self.directory / (name + ".csv")
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            kept = [row for row in rows
                    if int(row["env_step"]) <= boundary_step
                    and (name != "updates" or int(row["update_step"]) <= boundary_update)]
            removed[name] = len(rows) - len(kept)
            if removed[name]:
                temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
                try:
                    with temporary.open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=self.fields[name])
                        writer.writeheader()
                        writer.writerows(kept)
                        stream.flush()
                        os.fsync(stream.fileno())
                    replace_with_retry(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            self.handles[name] = path.open("a", newline="", encoding="utf-8", buffering=65536)
            self.writers[name] = csv.DictWriter(
                self.handles[name], fieldnames=self.fields[name], extrasaction="ignore")
            if name in self.recorded:
                self.recorded[name] = {self._row_key(name, row) for row in kept}
        self.event("canonical_rows_reconciled", checkpoint_env_step=boundary_step,
                   checkpoint_update_step=boundary_update, removed=removed)
        return removed

    def write_status(self, **changes):
        self.status.update(changes)
        self.status["heartbeat_at"] = utc_now()
        atomic_json(self.status_path, self.status)

    def flush_if_due(self):
        if time.monotonic() - self.last_flush >= self.flush_interval:
            self.flush()

    def flush(self):
        for handle in self.handles.values():
            handle.flush()
        self.last_flush = time.monotonic()

    def close(self):
        self.flush()
        for handle in self.handles.values():
            handle.close()
