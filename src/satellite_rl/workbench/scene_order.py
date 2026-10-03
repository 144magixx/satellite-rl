"""Deterministic ordering of whole, frozen training scenes.

The private random generator never advances the agent or environment RNGs.
An episode number alone reconstructs the same selection after a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import random


def _digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SceneOrder:
    def __init__(self, mode, seed, source_hashes):
        if mode not in ("sequential", "shuffle_each_cycle"):
            raise ValueError("Unknown training scene order")
        if mode == "shuffle_each_cycle" and not source_hashes:
            raise ValueError("每轮打乱要求选择固定训练场景集")
        self.mode = mode
        self.seed = seed
        self.source_hashes = tuple(source_hashes)
        self.source_set_sha256 = _digest(self.source_hashes)
        self._cached_cycle = None
        self._cached_order = None

    def _order(self, cycle):
        if self._cached_cycle != cycle:
            order = list(range(len(self.source_hashes)))
            if self.mode == "shuffle_each_cycle":
                material = f"satellite-rl-scene-order-v1:{self.seed}:{cycle}:{self.source_set_sha256}"
                random.Random(material).shuffle(order)
            self._cached_order = order
            self._cached_cycle = cycle
        return self._cached_order

    def selection(self, episode):
        if not isinstance(episode, int) or episode < 0:
            raise ValueError("Episode number must be nonnegative")
        count = len(self.source_hashes)
        if not count:
            return {"mode": self.mode, "seed": self.seed, "cycle": None,
                    "cycle_position": None, "source_index": None,
                    "source_set_sha256": self.source_set_sha256,
                    "cycle_order_sha256": None}
        cycle, position = divmod(episode, count)
        order = self._order(cycle)
        return {"mode": self.mode, "seed": self.seed, "cycle": cycle,
                "cycle_position": position, "source_index": order[position],
                "source_set_sha256": self.source_set_sha256,
                "cycle_order_sha256": _digest(order)}

    def checkpoint_state(self, next_episode):
        return {"version": 1, "source_count": len(self.source_hashes),
                "next_episode": next_episode,
                "next_selection": self.selection(next_episode)}

    def verify_checkpoint(self, saved, next_episode):
        if saved is None:
            if self.mode == "sequential":
                return  # Pre-scene-order checkpoints followed this exact order.
            raise ValueError("打乱场景断点缺少场景顺序恢复信息")
        if saved != self.checkpoint_state(next_episode):
            raise ValueError("恢复任务的场景顺序、来源SHA或回合位置与断点不一致")


def beam_bucket(beam_count):
    """Scientific V0/V1 replay and scene buckets; held-out counts are excluded."""
    if beam_count == 100:
        return "anchor_100"
    if 50 <= beam_count <= 99 and beam_count not in (60, 90):
        return "lower"
    if 101 <= beam_count <= 149 and beam_count != 140:
        return "middle"
    if 150 <= beam_count <= 200 and beam_count != 180:
        return "upper"
    raise ValueError(f"Beam count {beam_count} is outside the V0/V1 training set")


class VariableBeamSceneOrder:
    """Reproducible bucket/N/mother/load selection over frozen scene records.

    The first four mixed episodes visit all buckets once. Each later four-episode
    cycle does so again. Within a bucket, N cycles evenly and the available
    mother scene and load level are sampled independently with a private seed.
    """

    def __init__(self, mode, seed, source_hashes, scene_records):
        if mode not in ("fixed_100", "mixed_four_buckets"):
            raise ValueError("Unknown variable beam scene mode")
        if len(source_hashes) != len(scene_records) or not source_hashes:
            raise ValueError("Variable beam scenes require frozen metadata")
        self.mode = mode
        self.seed = seed
        self.source_set_sha256 = _digest(tuple(source_hashes))
        self.source_count = len(source_hashes)
        self.buckets = ("anchor_100",) if mode == "fixed_100" else (
            "anchor_100", "lower", "middle", "upper")
        # bucket -> N -> mother -> load -> source index
        self.index = {bucket: {} for bucket in self.buckets}
        for source_index, record in enumerate(scene_records):
            try:
                n = int(record["beam_count"])
                mother = str(record["mother_scene_id"])
                load = float(record["load_multiplier"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Variable beam scene metadata is incomplete") from error
            if not mother or load not in (0.75, 1.25):
                raise ValueError("Invalid variable beam mother/load metadata")
            bucket = beam_bucket(n)  # Reject held-out counts in the full train set.
            if bucket not in self.index:
                continue
            loads = self.index[bucket].setdefault(n, {}).setdefault(mother, {})
            if load in loads:
                raise ValueError("Duplicate N/mother/load training scene")
            loads[load] = source_index
        if any(not self.index[bucket] for bucket in self.buckets):
            raise ValueError("Every active beam bucket needs frozen scenes")
        for bucket in self.buckets:
            for n, mothers in self.index[bucket].items():
                if any(set(loads) != {0.75, 1.25} for loads in mothers.values()):
                    raise ValueError(f"N={n} mother scene lacks one demand load")

    def selection(self, episode):
        if not isinstance(episode, int) or episode < 0:
            raise ValueError("Episode number must be nonnegative")
        bucket_index = episode % len(self.buckets)
        bucket = self.buckets[bucket_index]
        visit = episode // len(self.buckets)
        counts = sorted(self.index[bucket])
        n = counts[visit % len(counts)]
        mothers = sorted(self.index[bucket][n])
        material = f"satellite-rl-variable-scenes-v1:{self.seed}:{episode}:{self.source_set_sha256}"
        rng = random.Random(material)
        mother = rng.choice(mothers)
        load = rng.choice((0.75, 1.25))
        return {"mode": "variable_beams", "seed": self.seed,
                "bucket": bucket, "beam_count": n, "mother_scene_id": mother,
                "load_multiplier": load,
                "source_index": self.index[bucket][n][mother][load],
                "source_set_sha256": self.source_set_sha256,
                "bucket_visit": visit}

    def checkpoint_state(self, next_episode):
        return {"version": 2, "source_count": self.source_count,
                "next_episode": next_episode,
                "next_selection": self.selection(next_episode)}

    def verify_checkpoint(self, saved, next_episode):
        if saved != self.checkpoint_state(next_episode):
            raise ValueError("Variable beam scene order or source differs from checkpoint")
