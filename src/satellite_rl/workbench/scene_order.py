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
