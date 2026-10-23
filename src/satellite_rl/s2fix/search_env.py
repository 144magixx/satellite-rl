"""Shared G1-initialized search environment for R, S1, S2 and L1."""
from collections import deque
from dataclasses import dataclass
import random
import time
import numpy as np
from satellite_rl.sac_improvement.environment import ImprovementEnv, STATE_WIDTH
from satellite_rl.power_refinement.physics import _sgm
from satellite_rl.sac_improvement.config import default_config, validate_config

KIND_NAMES = ("move", "boundary", "swap", "window", "revert", "xswap")
ACTION_WIDTH = 4 * 5 + len(KIND_NAMES) + 2


def configuration(**overrides):
    config = default_config()
    config.update(schema_version="s2fix_i0_v1", total_steps=200000, max_episodes=6250)
    config.update(overrides)
    return validate_config(config)


@dataclass(frozen=True)
class Candidate:
    id: str
    kind: str
    changes: tuple
    starts: tuple
    slots: tuple
    groups: tuple


def layout_key(groups, starts, slots):
    return tuple(zip(map(int, groups), map(int, starts), map(int, slots)))


class SearchEnv(ImprovementEnv):
    def __init__(self, config=None):
        super().__init__(configuration(**(config or {})))
        self.stage_input = None

    def reset(self, stage_input, seed=None):
        from .greedy import greedy_allocation, spread_allocation
        start = spread_allocation if self.config["start"] == "spread" else greedy_allocation
        self.stage_input = stage_input
        self.groups, self.current_starts, self.current_slots, self.power = [
            np.asarray(x).copy() for x in start(stage_input)]
        self.demand = np.asarray(stage_input.layout["request_bps_by_beam_int64"], float)
        self.locations = np.asarray(stage_input.layout["centres_lat_lon_deg"], float)
        self.scene = {"beam_location": self.locations}
        self.current_eval = self._evaluate(self.groups, self.current_starts, self.current_slots)
        self.base_eval = self.best_eval = self.current_eval
        self.base_basic = self.current_eval.capacity_bps >= .1 * self.demand
        self.best_groups = self.groups.copy()
        self.best_starts, self.best_slots = self.current_starts.copy(), self.current_slots.copy()
        self.decision_count = self.probe_count = self.invalid_count = self.repeat_count = 0
        self.proposals_generated = 0
        self.done = False
        self.trace = []
        self._candidate_cache = None
        self.candidate_generation_seconds = self.probe_seconds = 0.
        return self.state()

    def _evaluate(self, groups, starts, slots):
        from .evaluate import capacities
        from types import SimpleNamespace
        capacity = np.asarray(capacities(self.stage_input, groups, starts, slots, self.power))
        si = np.minimum(capacity / self.demand, 1.)
        return SimpleNamespace(capacity_bps=capacity, beam_si=si, si=float(si.mean()),
                               sgm=float(_sgm(capacity, self.demand)[0].mean()),
                               W=float(np.minimum(capacity, self.demand).sum() / self.demand.sum()))

    def _legal(self, starts, slots, groups=None):
        groups = self.groups if groups is None else groups
        if np.any(slots < 1) or np.any(slots > 10) or np.any(starts < 0) or np.any(starts + slots > 100):
            return False
        for group in range(8):
            occupied = np.zeros(100, bool)
            for i in np.flatnonzero(groups == group):
                block = slice(int(starts[i]), int(starts[i] + slots[i]))
                if occupied[block].any():
                    return False
                occupied[block] = True
        return True

    def _protected(self, evaluation):
        return (evaluation.W + 1e-12 >= self.base_eval.W and
                np.all(evaluation.capacity_bps[self.base_basic] >= .1 * self.demand[self.base_basic]))

    def _candidate(self, kind, changes):
        groups, starts, slots = self.groups.copy(), self.current_starts.copy(), self.current_slots.copy()
        if kind == "revert":
            groups = self.best_groups.copy()
        elif kind == "xswap":
            i, j = changes[0][0], changes[1][0]
            groups[i], groups[j] = groups[j], groups[i]
        for i, start, count in changes:
            starts[i], slots[i] = start, count
        key = layout_key(groups, starts, slots)
        if not self._legal(starts, slots, groups) or key == layout_key(self.groups, self.current_starts, self.current_slots):
            return None
        identifier = repr((kind, tuple(changes), tuple(int(groups[i]) for i, _, _ in changes)))
        return Candidate(identifier, kind, tuple(changes), tuple(starts), tuple(slots), tuple(groups))

    def candidates(self):
        if self.stage_input is None or self.done:
            return []
        if self._candidate_cache is not None:
            return self._candidate_cache
        started = time.perf_counter()
        n = len(self.demand)
        priority = sorted(range(n), key=lambda i: (float(self.current_eval.beam_si[i]),
                                                   -float(self.demand[i]), i))
        by_kind = {kind: [] for kind in KIND_NAMES}
        seen = set()

        def add(kind, changes):
            self.proposals_generated += 1
            candidate = self._candidate(kind, changes)
            key = None if candidate is None else layout_key(candidate.groups, candidate.starts, candidate.slots)
            if candidate is not None and key not in seen:
                seen.add(key)
                by_kind[kind].append(candidate)

        for i in priority[:min(n, 32)]:
            start, count = int(self.current_starts[i]), int(self.current_slots[i])
            for delta in (-5, -1, 1, 5):
                add("move", ((i, start + delta, count),))
            if len(by_kind["move"]) >= 8:
                break

        # Adjacent boundaries transfer one slot without leaving a gap.
        for group in range(8):
            ordered = sorted(np.flatnonzero(self.groups == group), key=lambda i: int(self.current_starts[i]))
            for left, right in zip(ordered, ordered[1:]):
                a, b = int(self.current_starts[left]), int(self.current_starts[right])
                x, y = int(self.current_slots[left]), int(self.current_slots[right])
                if a + x == b:
                    add("boundary", ((left, a, x + 1), (right, b + 1, y - 1)))
                    add("boundary", ((left, a, x - 1), (right, b - 1, y + 1)))
            if len(by_kind["boundary"]) >= 8:
                break

        # Swap occupied intervals as atomic assignments: works even at 800/800 slots.
        for i in priority[:min(n, 20)]:
            peers = sorted((j for j in np.flatnonzero(self.groups == self.groups[i]) if j != i),
                           key=lambda j: (abs(int(self.current_starts[j] - self.current_starts[i])), j))
            for j in peers[:4]:
                add("swap", ((i, int(self.current_starts[j]), int(self.current_slots[j])),
                             (j, int(self.current_starts[i]), int(self.current_slots[i]))))
            if len(by_kind["swap"]) >= 8:
                break

        for group in range(8):
            ordered = sorted(np.flatnonzero(self.groups == group), key=lambda i: int(self.current_starts[i]))
            for width in (3, 4):
                for offset in range(len(ordered) - width + 1):
                    window = ordered[offset:offset + width]
                    rotated = window[1:] + window[:1]
                    add("window", tuple((i, int(self.current_starts[j]), int(self.current_slots[j]))
                                        for i, j in zip(window, rotated)))
                    if len(by_kind["window"]) >= 8:
                        break
                if len(by_kind["window"]) >= 8:
                    break
            if len(by_kind["window"]) >= 8:
                break

        if layout_key(self.best_groups, self.best_starts, self.best_slots) != layout_key(
                self.groups, self.current_starts, self.current_slots):
            add("revert", tuple((i, int(self.best_starts[i]), int(self.best_slots[i]))
                                for i in range(n) if self.best_starts[i] != self.current_starts[i]
                                or self.best_slots[i] != self.current_slots[i]
                                or self.best_groups[i] != self.groups[i]))

        for i in priority[:min(n, 20)]:
            peers = [j for j in range(n) if self.groups[j] != self.groups[i]
                     and (self.current_slots[j] > self.current_slots[i]
                          or np.linalg.norm(self.locations[j] - self.locations[i]) > 0)]
            peers.sort(key=lambda j: (self.current_slots[j] > self.current_slots[i],
                                      np.linalg.norm(self.locations[j] - self.locations[i])), reverse=True)
            for j in peers[:2]:
                add("xswap", ((i, int(self.current_starts[j]), int(self.current_slots[j])),
                              (j, int(self.current_starts[i]), int(self.current_slots[i]))))

        limit = self.config["candidate_limit"]
        selected = []
        # Round-robin prevents move-only pools from suppressing full-slot swaps.
        while len(selected) < limit and any(by_kind.values()):
            for kind in KIND_NAMES:
                if by_kind[kind] and len(selected) < limit:
                    selected.append(by_kind[kind].pop(0))
        self._candidate_cache = selected
        self.candidate_generation_seconds += time.perf_counter() - started
        return selected


    def state(self):
        result = super().state()
        result[:2800].reshape(200, 14)[:len(self.groups), 4] = self.groups / 7
        d, p = self.config["max_decisions"], self.config["max_probes"]
        result[2801:2805] = (self.decision_count / d, self.probe_count / p,
                             (d - self.decision_count) / d, (p - self.probe_count) / p)
        return result

    def action_feature(self, candidate):
        result = np.zeros(ACTION_WIDTH, np.float32)
        if candidate is None:
            result[-1] = 1
            return result
        for offset, (i, start, count) in enumerate(candidate.changes[:4]):
            result[5*offset:5*offset+5] = (i/200, start/100, count/10,
                (start-self.current_starts[i])/100, (count-self.current_slots[i])/10)
        result[20 + KIND_NAMES.index(candidate.kind)] = 1
        result[-2] = min(len(candidate.changes), 4)/4
        return result

    def probe(self, candidate):
        started = time.perf_counter()
        evaluation = self._evaluate(np.asarray(candidate.groups), np.asarray(candidate.starts), np.asarray(candidate.slots))
        self.probe_seconds += time.perf_counter() - started
        self.probe_count += 1
        gain = 0.
        if self._protected(evaluation) and evaluation.W > self.best_eval.W:
            gain = 100 * (evaluation.W - self.best_eval.W)
            self.best_groups = np.asarray(candidate.groups).copy()
            self.best_starts, self.best_slots = np.asarray(candidate.starts).copy(), np.asarray(candidate.slots).copy()
            self.best_eval = evaluation
        return evaluation, gain

    def move(self, candidate, evaluation):
        self.groups = np.asarray(candidate.groups).copy()
        self.current_starts, self.current_slots = np.asarray(candidate.starts).copy(), np.asarray(candidate.slots).copy()
        self.current_eval = evaluation
        self._candidate_cache = None

    def step(self, candidate_id):
        self.decision_count += 1
        reward = 0.
        kind = "STOP"
        if candidate_id == "STOP":
            self.done = True
        else:
            candidate = next(c for c in self.candidates() if c.id == candidate_id)
            kind = candidate.kind
            evaluation, reward = self.probe(candidate)
            self.move(candidate, evaluation)
        self.done |= (self.decision_count >= self.config["max_decisions"] or
                      self.probe_count >= self.config["max_probes"])
        info = dict(kind=kind, best_W=self.best_eval.W, probes=self.probe_count)
        self.trace.append(info)
        return self.state(), reward, self.done, info

    def allocation(self):
        return tuple(x.copy() for x in (self.best_groups, self.best_starts, self.best_slots, self.power))


def rule_priority(env, candidate):
    benefit = sum((1 - env.current_eval.beam_si[i]) * (count - env.current_slots[i])
                  for i, _, count in candidate.changes)
    return benefit, {"swap":4, "boundary":3, "window":2, "move":1, "revert":0, "xswap":4}[candidate.kind]


def run_search(stage_input, method="R", seed=51001, max_probes=None, return_info=False):
    budget = max_probes if max_probes is not None else (320 if method == "S2" else 32)
    env = SearchEnv(dict(max_probes=budget, max_decisions=budget))
    env.reset(stage_input)
    rng = random.Random(seed)
    tabu = deque([layout_key(env.groups, env.current_starts, env.current_slots)], maxlen=8)
    while not env.done:
        choices = env.candidates()
        if method == "R":
            env.step(rng.choice(choices).id if choices else "STOP")
            continue
        choices = [c for c in choices if layout_key(c.groups, c.starts, c.slots) not in tabu]
        choices = sorted(choices, key=lambda c: rule_priority(env, c), reverse=True)[:min(4, budget-env.probe_count)]
        if not choices:
            break
        outcomes = [(env.probe(c)[0], c) for c in choices]
        evaluation, chosen = max(outcomes, key=lambda pair: pair[0].W)
        env.move(chosen, evaluation)
        tabu.append(layout_key(chosen.groups, chosen.starts, chosen.slots))
        env.decision_count += 1
        env.done = env.probe_count >= budget or env.decision_count >= (budget + 3) // 4
    result = env.allocation()
    return (result, dict(probes=env.probe_count, decisions=env.decision_count)) if return_info else result
