"""Opt-in, bounded CPU supply benchmark for D12-A resident workers.

Run as a module with ``python -m tests.test_drl_r04_learning_perf``.  It
uses three fixed historical training scenes and does not perform PPO updates.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import argparse
import json
import multiprocessing
from pathlib import Path
import pickle
import sys
import tempfile
import time

import torch

from satellite_rl.coverage.drl_r04_policy import (
    DrlR04Policy, Transition, collate, ppo_update)

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


def remote_gpu_smoke(dataset, record, core, *, workers=16):
    """Exactly one 2048-action rollout and one real CUDA PPO update."""
    if workers != 16 or not torch.cuda.is_available():
        raise RuntimeError("D12-A supply smoke requires 16 workers and DQN/CUDA")
    torch.set_num_threads(8)
    torch.manual_seed(51001)
    streams = [_fresh_stream(dataset, core, 51001 + 101 + i * 1009)
               for i in range(32)]
    policy = DrlR04Policy().cuda()
    optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    observations = [stream["observation"] for stream in streams]
    transitions = []
    with ExitStack() as stack:
        pools = [stack.enter_context(ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn"),
            initializer=_resident_init,
            initargs=([(i, streams[i]) for i in range(worker, 32, workers)],
                      record, core)))
                 for worker in range(workers)]
        startup = time.monotonic()
        [pool.submit(_resident_state, None).result() for pool in pools]
        startup_seconds = time.monotonic() - startup
        began = time.monotonic()
        completed = 0
        for turn in range(64):
            with torch.no_grad():
                action, logp, value = policy.act(collate(observations, "cuda"))
            choices, logps, values = action.tolist(), logp.tolist(), value.tolist()
            tasks = [pool.submit(_resident_step,
                                 (turn * 32,
                                  [(i, choices[i]) for i in range(worker, 32, workers)]))
                     for worker, pool in enumerate(pools)]
            results = sorted((result for task in tasks for result in task.result()),
                             key=lambda row: row[0])
            if len(results) != 32:
                raise ValueError("D12-A supply smoke lost a resident stream")
            for i, next_observation, reward, done, _, terminal in results:
                transitions.append(Transition(observations[i], choices[i],
                                              logps[i], values[i], reward, done))
                observations[i] = next_observation
                completed += int(terminal is not None)
        rollout_seconds = time.monotonic() - began
        with torch.no_grad():
            bootstrap = policy(collate(observations, "cuda"))[1].cpu().numpy()
        began = time.monotonic()
        diagnostics = ppo_update(policy, optimizer, transitions, bootstrap,
                                 device="cuda", gamma=1.0, gae_lambda=.95,
                                 clip_ratio=.2, entropy_coef=.01,
                                 value_coef=.5, max_grad_norm=.5,
                                 epochs=4, minibatch_size=64, num_envs=32)
        torch.cuda.synchronize()
        update_seconds = time.monotonic() - began
        states = [pool.submit(_resident_state, None).result() for pool in pools]
        checkpoint_bytes = len(pickle.dumps(states, protocol=5))
    changed = any(not torch.equal(old, new) for old, new in
                  zip(before, policy.parameters()))
    if (not changed or diagnostics["rollout_interactions"] != 2048 or
            diagnostics["minibatch_updates"] != 128):
        raise ValueError("D12-A CUDA PPO update did not meet real-update contract")
    return {"workers": workers, "streams": 32,
            "interactions": 2048, "completed_episodes": completed,
            "startup_seconds": startup_seconds,
            "rollout_seconds": rollout_seconds,
            "interactions_per_second": 2048 / rollout_seconds,
            "update_seconds": update_seconds,
            "checkpoint_ipc_bytes": checkpoint_bytes,
            "ppo": diagnostics, "parameters_changed": changed,
            "python_prefix": sys.prefix, "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=SOURCE)
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    source = load_variable_dataset(args.dataset)
    if len(source.scenes) <= 575:
        raise ValueError("D12-A supply check requires frozen 576-mother train input")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "training" / "tiny.npz"
        subset = CoverageDataset([source.scenes[i] for i in (0, 287, 575)],
                                 {"partition_role": "train",
                                  "source_sha256": _sha256(args.dataset)})
        save_variable_dataset(subset, path)
        dataset = load_variable_dataset(path)
        record = {"path": str(path), "sha256": _sha256(path)}
        core = DrlR04Config()
        result = ({"mode": "remote_gpu_smoke",
                   "row": remote_gpu_smoke(dataset, record, core)}
                  if args.gpu_smoke else
                  {"mode": "local_worker_comparison",
                   "actions": "alternate_first_nonstop_and_stop",
                   "rows": [benchmark(workers, dataset, record, core)
                            for workers in (8, 16)]})
        result.update(source_sha256=_sha256(args.dataset),
                      source_path=str(args.dataset),
                      scene_indices=[0, 287, 575],
                      subset_sha256=_sha256(path))
        payload = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)


if __name__ == "__main__":
    main()
