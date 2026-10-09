"""Opt-in, bounded CPU supply benchmark for D12-A resident workers.

Run as a module with ``python -m tests.test_drl_r04_learning_perf``.  It
uses three fixed historical training scenes and does not perform PPO updates.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import json
import multiprocessing
from pathlib import Path
import pickle
import tempfile
import time

from satellite_rl.coverage.drl_r04_config import DrlR04Config
from satellite_rl.coverage.drl_r04_runner import (
    _fresh_stream, _resident_init, _resident_state, _resident_step)
from satellite_rl.coverage.runner import _sha256
from satellite_rl.coverage.scenes import CoverageDataset
from satellite_rl.coverage.variable_dataset import load_variable_dataset, save_variable_dataset


SOURCE = Path("D:/paper/outputs/experiments/first_stage_s1_v10_20261009/"
              "preflight/throughput_train_input.npz")


def benchmark(workers, dataset, record, core):
    streams = [_fresh_stream(dataset, core, 51001 + 101 + i * 1009)
               for i in range(32)]
    naive_env_roundtrip_bytes_per_step = 2 * len(pickle.dumps(
        [stream["env"] for stream in streams], protocol=5))
    parent_bytes = 0
    child_bytes = 0
    startup = time.monotonic()
    with ExitStack() as stack:
        pools = [stack.enter_context(ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn"),
            initializer=_resident_init,
            initargs=([(i, streams[i]) for i in range(worker, 32, workers)],
                      record, core)))
                 for worker in range(workers)]
        [pool.submit(_resident_state, None).result() for pool in pools]
        startup_seconds = time.monotonic() - startup
        began = time.monotonic()
        observations = [stream["observation"] for stream in streams]
        for turn in range(8):
            # First real split/merge where available, then STOP.  Identical
            # actions and scene streams are sent to both worker counts.
            indices = [1 if turn % 2 == 0 and len(obs["candidate"]) > 1 else 0
                       for obs in observations]
            packets = [(turn * 32,
                        [(i, indices[i]) for i in range(worker, 32, workers)])
                       for worker in range(workers)]
            parent_bytes += sum(len(pickle.dumps(packet, protocol=5)) for packet in packets)
            tasks = [pool.submit(_resident_step, packet)
                     for pool, packet in zip(pools, packets)]
            results = sorted((row for task in tasks for row in task.result()),
                             key=lambda row: row[0])
            child_bytes += len(pickle.dumps(results, protocol=5))
            observations = [row[1] for row in results]
        step_seconds = time.monotonic() - began
        snapshots = [pool.submit(_resident_state, None).result() for pool in pools]
        checkpoint_bytes = len(pickle.dumps(snapshots, protocol=5))
    return {"workers": workers, "interactions": 256,
            "startup_seconds": startup_seconds, "step_seconds": step_seconds,
            "interactions_per_second": 256 / step_seconds,
            "choice_ipc_bytes": parent_bytes, "result_ipc_bytes": child_bytes,
            "checkpoint_ipc_bytes": checkpoint_bytes,
            "naive_env_roundtrip_bytes_per_step": naive_env_roundtrip_bytes_per_step}


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source = load_variable_dataset(SOURCE)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "training" / "tiny.npz"
        subset = CoverageDataset([source.scenes[i] for i in (0, 287, 575)],
                                 {"partition_role": "train", "source_sha256": _sha256(SOURCE)})
        save_variable_dataset(subset, path)
        dataset = load_variable_dataset(path)
        record = {"path": str(path), "sha256": _sha256(path)}
        core = DrlR04Config()
        rows = [benchmark(workers, dataset, record, core) for workers in (8, 16)]
        print(json.dumps({"source_sha256": _sha256(SOURCE),
                          "scene_indices": [0, 287, 575],
                          "actions": "alternate_first_nonstop_and_stop",
                          "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
