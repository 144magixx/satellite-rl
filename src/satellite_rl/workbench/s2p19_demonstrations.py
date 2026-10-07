"""Frozen, auditable greedy trajectories for the S2-P19 C1 replay stream.

The preparation worker calls the single-scene writer and persists its returned
record after each scene. C1 never regenerates labels during training: it reads
only a manifest whose bytes and every shard are SHA-bound in the job config.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def teacher_contract_sha256(config):
    """Physics, observations and reward only; C0/C1 replay settings may differ."""
    value = {key: config[key] for key in ("environment", "state", "reward", "network")}
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bucket(count):
    if count == 100:
        return "anchor_100"
    if count < 100:
        return "lower"
    if count <= 149:
        return "middle"
    return "upper"


def generate_greedy_trajectory(config, scene_path, scene_sha256, target_path, *, stop=None):
    """Write one complete teacher trajectory atomically; never truncate a shard.

    ``stop`` is an optional no-argument callback returning true to interrupt.
    The caller owns durable per-scene progress and may rerun this idempotently.
    """
    from satellite_rl.workbench.baselines import BaselinePolicy
    from satellite_rl.workbench.environment import WorkbenchEnv

    scene_path, target_path = Path(scene_path), Path(target_path)
    if _sha256(scene_path) != scene_sha256:
        raise ValueError("teacher scene SHA differs from frozen source")
    with np.load(scene_path, allow_pickle=False) as data:
        scene = {key: np.array(data[key], copy=True) for key in data.files}
    env = WorkbenchEnv(config)
    state = env.reset(scene=scene)
    count = int(env.raw_env.beam_num)
    if not 50 <= count <= 200:
        raise ValueError("S2-P19 teacher requires 50..200 target beams")
    policy = BaselinePolicy("greedy", env)
    rows = {key: [] for key in ("state", "action", "reward", "next_state", "done",
                                "mask_fl", "mask_power", "next_fl", "next_power")}
    done = False
    while not done:
        if stop is not None and stop():
            raise InterruptedError("teacher preparation stopped before complete trajectory")
        mask = env.action_mask()
        action = policy.act(state, action_mask=mask)
        next_state, reward, done, _ = env.step(action)
        next_mask = None if done else env.action_mask()
        rows["state"].append(np.asarray(state, dtype=np.float32).copy())
        rows["action"].append(action)
        rows["reward"].append(float(reward))
        rows["next_state"].append(np.asarray(next_state, dtype=np.float32).copy())
        rows["done"].append(done)
        rows["mask_fl"].append(mask["frequency_slots"].copy())
        rows["mask_power"].append(mask["power"].copy())
        rows["next_fl"].append(np.zeros_like(mask["frequency_slots"]) if done else
                               next_mask["frequency_slots"].copy())
        rows["next_power"].append(np.zeros_like(mask["power"]) if done else
                                  next_mask["power"].copy())
        state = next_state
    if len(rows["reward"]) != count:
        raise ValueError("teacher trajectory did not assign every target beam")
    metrics = env.metrics()
    if not np.isclose(sum(rows["reward"]), 100 * metrics["final_si_mean"], atol=1e-5):
        raise ValueError("teacher reward does not telescope to terminal all-beam SI")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp = target_path.with_name(target_path.name + ".incomplete")
    try:
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **{key: np.asarray(value) for key, value in rows.items()},
                                entity_count=np.asarray(count, dtype=np.int32),
                                scene_sha256=np.asarray(scene_sha256),
                                terminal_si_mean=np.asarray(metrics["final_si_mean"], dtype=np.float64))
        temp.replace(target_path)
    finally:
        temp.unlink(missing_ok=True)
    return {"path": target_path.name, "sha256": _sha256(target_path),
            "scene_sha256": scene_sha256, "beam_count": count,
            "transition_count": count, "bucket": _bucket(count)}


PACKED_FIELDS = ("state", "action", "reward", "next_state", "done",
                 "mask_fl", "mask_power", "next_fl", "next_power")


def pack_demonstrations(records, output_dir, *, stop=None, progress=None):
    """One-time immutable memory-mapped corpus; avoids decompressing every update.

    Call after all scene shards exist, before ``write_manifest``. The returned
    SHA records become part of the frozen manifest. ``progress(done,total)``
    can publish bounded preparation progress; interruption leaves .incomplete
    files that are never accepted as a manifest source.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total = sum(int(row["transition_count"]) for row in records)
    if total <= 0:
        raise ValueError("cannot pack an empty teacher dataset")
    first_path = output_dir / records[0]["path"]
    with np.load(first_path, allow_pickle=False) as first:
        specs = {key: (first[key].dtype, (total, *first[key].shape[1:]))
                 for key in PACKED_FIELDS}
    arrays = {}
    for key, (dtype, shape) in specs.items():
        arrays[key] = np.lib.format.open_memmap(
            output_dir / f"packed_{key}.npy.incomplete", mode="w+", dtype=dtype, shape=shape)
    offset = 0
    try:
        for position, record in enumerate(records):
            if stop is not None and stop():
                raise InterruptedError("teacher packing stopped")
            shard = output_dir / record["path"]
            if _sha256(shard) != record["sha256"]:
                raise ValueError("teacher shard changed before packing")
            n = int(record["transition_count"])
            with np.load(shard, allow_pickle=False) as source:
                for key in PACKED_FIELDS:
                    if source[key].shape != (n, *specs[key][1][1:]):
                        raise ValueError("teacher shard shape changed before packing")
                    arrays[key][offset:offset + n] = source[key]
            offset += n
            if progress is not None:
                progress(position + 1, len(records))
        if offset != total:
            raise ValueError("packed teacher length mismatch")
        for value in arrays.values():
            value.flush()
    finally:
        for mapping in arrays.values():
            mapping._mmap.close()
        if "mapping" in locals():
            del mapping
        arrays.clear()
        # Windows refuses to rename an open memmap. The loop variable above
        # otherwise retains the last array after clearing the dictionary.
        if "value" in locals():
            del value
    result = {}
    for key in PACKED_FIELDS:
        name = f"packed_{key}.npy"
        (output_dir / (name + ".incomplete")).replace(output_dir / name)
        result[key] = {"path": name, "sha256": _sha256(output_dir / name),
                       "dtype": str(specs[key][0]), "shape": list(specs[key][1])}
    return result


def write_manifest(config, scene_set_sha256, records, target_path, *, packed):
    """Freeze all shards after preparation; returns manifest SHA for C1 config."""
    target_path = Path(target_path)
    if not records or len({row["scene_sha256"] for row in records}) != len(records):
        raise ValueError("teacher manifest requires distinct complete scenes")
    value = {"format": "s2p19_greedy_trajectory_v1",
             "teacher_contract_sha256": teacher_contract_sha256(config),
             "source_set_sha256": scene_set_sha256, "records": list(records),
             "packed": packed}
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp = target_path.with_name(target_path.name + ".incomplete")
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2),
                        encoding="utf-8")
        temp.replace(target_path)
    finally:
        temp.unlink(missing_ok=True)
    return _sha256(target_path)


def validate_manifest(path, expected_sha256, config, expected_scene_set_sha256):
    """Validate exact source, shape, terminal boundary, legal action and reward."""
    path = Path(path)
    if _sha256(path) != expected_sha256:
        raise ValueError("teacher manifest SHA differs from frozen configuration")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("format") != "s2p19_greedy_trajectory_v1" or
            manifest.get("teacher_contract_sha256") != teacher_contract_sha256(config) or
            manifest.get("source_set_sha256") != expected_scene_set_sha256):
        raise ValueError("teacher manifest source/physics/reward contract mismatch")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("teacher manifest has no trajectories")
    from satellite_rl.workbench.config import expected_state_dim
    state_dim = expected_state_dim(config)
    seen, buckets = set(), set()
    for record in records:
        name = record["path"]
        if Path(name).name != name or name in seen:
            raise ValueError("teacher shard path is not a unique local filename")
        seen.add(name)
        shard = path.parent / name
        if _sha256(shard) != record["sha256"]:
            raise ValueError(f"teacher shard SHA differs: {name}")
        with np.load(shard, allow_pickle=False) as data:
            n = int(data["entity_count"])
            if (n != record["beam_count"] or n != record["transition_count"] or
                    str(data["scene_sha256"]) != record["scene_sha256"] or
                    record["bucket"] != _bucket(n) or not 50 <= n <= 200):
                raise ValueError("teacher shard scene/count/bucket mismatch")
            expected = {"state": (n, state_dim), "next_state": (n, state_dim),
                        "action": (n, 3), "reward": (n,), "done": (n,),
                        "mask_fl": (n, 100, 10), "next_fl": (n, 100, 10),
                        "mask_power": (n, 10), "next_power": (n, 10)}
            if any(data[key].shape != shape for key, shape in expected.items()):
                raise ValueError("teacher shard transition shapes mismatch")
            if (not np.isfinite(data["state"]).all() or not np.isfinite(data["next_state"]).all()
                    or not np.isfinite(data["reward"]).all()):
                raise ValueError("teacher shard has nonfinite transitions")
            done = data["done"].astype(bool)
            if done[:-1].any() or not bool(done[-1]) or not np.array_equal(
                    data["next_state"][:-1], data["state"][1:]):
                raise ValueError("teacher trajectory boundary is incomplete")
            a = data["action"]
            if (np.any(a[:, 0] < 0) or np.any(a[:, 0] >= 100) or
                    np.any(a[:, 1] < 0) or np.any(a[:, 1] >= 10) or
                    np.any(a[:, 2] < 0) or np.any(a[:, 2] >= 10) or
                    not np.all(data["mask_fl"][np.arange(n), a[:, 0], a[:, 1]]) or
                    not np.all(data["mask_power"][np.arange(n), a[:, 2]])):
                raise ValueError("teacher action is not legal under stored masks")
            terminal_si = float(data["terminal_si_mean"])
            if not np.isclose(float(data["reward"].sum()), 100 * terminal_si, atol=1e-5):
                raise ValueError("teacher reward does not match terminal SI")
        buckets.add(record["bucket"])
    if buckets != {"anchor_100", "lower", "middle", "upper"}:
        raise ValueError("teacher set must cover all four training N buckets")
    packed = manifest.get("packed")
    if not isinstance(packed, dict) or set(packed) != set(PACKED_FIELDS):
        raise ValueError("C1 requires complete memory-mapped teacher corpus")
    total = sum(int(record["transition_count"]) for record in records)
    for key, row in packed.items():
        name = row.get("path")
        if name != f"packed_{key}.npy" or _sha256(path.parent / name) != row.get("sha256"):
            raise ValueError("packed teacher array SHA/path mismatch")
        array = np.load(path.parent / name, mmap_mode="r", allow_pickle=False)
        if (array.shape[0] != total or list(array.shape) != row.get("shape") or
                str(array.dtype) != row.get("dtype")):
            raise ValueError("packed teacher array dtype/shape mismatch")
    return manifest


class DemonstrationDataset:
    """Lazy, SHA-verified shard sampler with bounded memory and replayable RNG."""

    def __init__(self, manifest_path, expected_sha256, config, source_set_sha256, seed):
        manifest = validate_manifest(manifest_path, expected_sha256, config, source_set_sha256)
        self.root = Path(manifest_path).parent
        self.records = manifest["records"]
        self.arrays = {key: np.load(self.root / row["path"], mmap_mode="r", allow_pickle=False)
                       for key, row in manifest["packed"].items()}
        self.by_bucket = {key: [] for key in ("anchor_100", "lower", "middle", "upper")}
        self.entity_counts = np.empty(sum(row["transition_count"] for row in self.records), dtype=np.int16)
        offset = 0
        for row in self.records:
            count = int(row["transition_count"])
            self.by_bucket[row["bucket"]].extend(range(offset, offset + count))
            self.entity_counts[offset:offset + count] = int(row["beam_count"])
            offset += count
        self.rng = random.Random(seed)
        self.draw_count = 0
        self.manifest_sha256 = expected_sha256

    def sample(self, count, device, action_dims):
        if count % 4:
            raise ValueError("demonstration sample must split evenly over four N buckets")
        from satellite_rl.agents.workbench_algorithms import pad_states
        from satellite_rl.agents.action_masking import mask_batch
        indices = []
        for bucket in self.by_bucket.values():
            indices.extend(self.rng.choices(bucket, k=count // 4))
        self.rng.shuffle(indices)
        counts = self.entity_counts[indices].tolist()
        s, lengths = pad_states(self.arrays["state"][indices], device,
                                padded_width=self.arrays["state"].shape[1], entity_counts=counts)
        ns, next_lengths = pad_states(self.arrays["next_state"][indices], device,
                                      padded_width=self.arrays["next_state"].shape[1], entity_counts=counts)
        current = mask_batch([{"frequency_slots": self.arrays["mask_fl"][i],
                               "power": self.arrays["mask_power"][i]} for i in indices], action_dims, device)
        following = mask_batch([{"frequency_slots": self.arrays["next_fl"][i],
                                 "power": self.arrays["next_power"][i]} for i in indices], action_dims, device)
        self.draw_count += count
        return (s, torch.as_tensor(np.array(self.arrays["action"][indices], copy=True),
                                   dtype=torch.long, device=device),
                torch.as_tensor(np.array(self.arrays["reward"][indices], copy=True),
                                dtype=torch.float32, device=device),
                ns, torch.as_tensor(np.array(self.arrays["done"][indices], copy=True),
                                    dtype=torch.float32, device=device),
                lengths, next_lengths, current, following)

    def state_dict(self):
        return {"manifest_sha256": self.manifest_sha256, "rng": self.rng.getstate(),
                "draw_count": self.draw_count}

    def load_state_dict(self, state):
        if state.get("manifest_sha256") != self.manifest_sha256:
            raise ValueError("demonstration manifest changed on checkpoint resume")
        self.rng.setstate(state["rng"])
        self.draw_count = int(state["draw_count"])

    def close(self):
        """Release memory maps, including on Windows where open files cannot move."""
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays.clear()


def load_demonstrations(manifest_path, expected_sha256, config, source_set_sha256, seed):
    """Public C1 loader for the worker and preparation smoke checks."""
    return DemonstrationDataset(manifest_path, expected_sha256, config,
                                source_set_sha256, seed)
