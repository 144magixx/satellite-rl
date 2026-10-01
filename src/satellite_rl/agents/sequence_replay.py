"""Finite-window replay: one sampled decision is one TD target."""

from __future__ import annotations

import random

import numpy as np
import torch

from satellite_rl.agents.action_masking import mask_batch


class SequenceReplay:
    def __init__(self, capacity, seed, window):
        self.capacity = int(capacity)
        self.window = int(window)
        if self.capacity < 1 or self.window < 1:
            raise ValueError("Temporal replay capacity and window must be positive")
        # The extra rows are context only. They are never sampled as TD targets.
        self.storage_capacity = self.capacity + self.window - 1
        self.rows = []
        self.next_index = 0
        self.episode_id = 0
        self.episode_step = 0
        self.rng = random.Random(seed)

    def __len__(self):
        return min(len(self.rows), self.capacity)

    def eligible_rows(self):
        """Chronological TD targets, excluding retained context-only prefix."""
        first = len(self.rows) - len(self)
        return [self._row(index) for index in range(first, len(self.rows))]

    def _row(self, logical_index):
        if len(self.rows) < self.storage_capacity:
            return self.rows[logical_index]
        return self.rows[(self.next_index + logical_index) % self.storage_capacity]

    def add(self, state, action, reward, next_state, done, action_mask,
            next_action_mask, actual_action):
        if not isinstance(actual_action, dict) or any(
                key not in actual_action for key in ("freq", "slots", "power_w")):
            raise ValueError("Temporal replay requires the executed frequency, slots and power")
        actual = {key: float(actual_action[key]) for key in ("freq", "slots", "power_w")}
        expected = np.asarray([actual["freq"] / 99, actual["slots"] / 10,
                               actual["power_w"] / 50, 1], dtype=np.float32)
        next_array = np.asarray(next_state, dtype=np.float32)
        if next_array.ndim != 1 or not np.allclose(next_array[-4:], expected, atol=2e-6):
            raise ValueError("Temporal next state does not contain the executed action")
        row = (np.asarray(state, dtype=np.float32).copy(),
               tuple(int(x) for x in action), float(reward), next_array.copy(), bool(done),
               action_mask, next_action_mask, self.episode_id, self.episode_step, actual)
        if len(self.rows) < self.storage_capacity:
            self.rows.append(row)
        else:
            self.rows[self.next_index] = row
        self.next_index = (self.next_index + 1) % self.storage_capacity
        if done:
            self.episode_id += 1
            self.episode_step = 0
        else:
            self.episode_step += 1

    def _context(self, index):
        target = self._row(index)
        episode_id, step = target[7:9]
        rows = [target]
        for distance in range(1, min(self.window, step + 1)):
            if index - distance < 0:
                break
            prior = self._row(index - distance)
            if prior[7] != episode_id or prior[8] != step - distance:
                break
            rows.append(prior)
        rows.reverse()
        return [row[0] for row in rows]

    def sample(self, count, device, action_dims):
        first_target = len(self.rows) - len(self)
        indices = self.rng.sample(range(first_target, len(self.rows)), count)
        selected = [self._row(index) for index in indices]
        width = selected[0][0].size
        current = np.zeros((count, self.window, width), dtype=np.float32)
        following = np.zeros_like(current)
        lengths, next_lengths = [], []
        for batch_index, (index, row) in enumerate(zip(indices, selected)):
            context = self._context(index)
            length = len(context)
            current[batch_index, :length] = context
            lengths.append(length)
            if row[4]:
                next_lengths.append(1)
            else:
                future = (context + [row[3]])[-self.window:]
                following[batch_index, :len(future)] = future
                next_lengths.append(len(future))
        tensor = lambda value, dtype: torch.as_tensor(value, dtype=dtype, device=device)
        return ((tensor(current, torch.float32), tensor(lengths, torch.long)),
                tensor([row[1] for row in selected], torch.long),
                tensor([row[2] for row in selected], torch.float32),
                (tensor(following, torch.float32), tensor(next_lengths, torch.long)),
                tensor([row[4] for row in selected], torch.float32),
                mask_batch([row[5] for row in selected], action_dims, device),
                mask_batch([row[6] for row in selected], action_dims, device))

    def state_dict(self):
        return {"format": "sequence_replay_v1", "rows": list(self.rows),
                "capacity": self.capacity, "window": self.window,
                "storage_capacity": self.storage_capacity,
                "next_index": self.next_index, "episode_id": self.episode_id,
                "episode_step": self.episode_step, "rng": self.rng.getstate()}

    def load_state_dict(self, data):
        if (data.get("format") != "sequence_replay_v1" or
                data["capacity"] != self.capacity or data["window"] != self.window or
                data.get("storage_capacity") != self.storage_capacity or
                len(data["rows"]) > self.storage_capacity or
                not 0 <= data["next_index"] < self.storage_capacity):
            raise ValueError("Temporal replay capacity, format or window changed")
        self.rows = list(data["rows"])
        self.next_index = int(data["next_index"])
        self.episode_id = int(data["episode_id"])
        self.episode_step = int(data["episode_step"])
        self.rng.setstate(data["rng"])
