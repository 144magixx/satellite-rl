"""Complete-layout, reversible I0 decisions with exact full-beam physics."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import random
import time

import numpy as np

from satellite_rl.power_refinement.physics import FixedAllocationPhysics
from satellite_rl.workbench.baselines import BaselinePolicy
from satellite_rl.workbench.config import default_config as workbench_default_config
from satellite_rl.workbench.environment import WorkbenchEnv, beam_service_fields

from .config import validate_config


MAX_BEAMS = 200
KIND_NAMES = ("move", "boundary", "swap", "window", "revert")
STATE_WIDTH = MAX_BEAMS * 14 + 12
ACTION_WIDTH = 4 * 5 + len(KIND_NAMES) + 2


def _layout_key(starts, slots):
    return tuple(zip((int(x) for x in starts), (int(x) for x in slots)))


def _sha_layout(starts, slots):
    return sha256(np.column_stack((starts, slots)).astype("<i2").tobytes()).hexdigest()


@dataclass(frozen=True)
class Candidate:
    id: str
    kind: str
    changes: tuple
    starts: tuple
    slots: tuple


class ImprovementEnv:
    """Each non-STOP decision tests exactly one complete candidate allocation.

    An invalid request consumes one decision and zero physical probes. A repeat
    consumes a decision and a real recomputation/probe. STOP consumes one
    decision and zero probes. At decision 32 or probe 32 the
    episode ends. Only a candidate passing all three baseline guards can replace
    the saved output; exploratory current layouts may be worse.
    """

    def __init__(self, config):
        self.config = validate_config(config)
        self.scene = None
        self.initial_beams = None
        self.current_starts = None
        self.current_slots = None
        self.current_eval = None
        self.best_starts = None
        self.best_slots = None
        self.best_eval = None
        self.base_eval = None
        self.power = None
        self.demand = None
        self.base_basic = None
        self.decision_count = 0
        self.probe_count = 0
        self.invalid_count = 0
        self.repeat_count = 0
        self.proposals_generated = 0
        self.done = False
        self.visited = set()
        self.trace = []
        self._candidate_cache = None
        self.initialization_seconds = 0.0
        self.candidate_generation_seconds = 0.0
        self.probe_seconds = 0.0

    def reset(self, scene, seed=None):
        """Generate B1 by executing the existing workbench baseline, not a copy."""
        started = time.perf_counter()
        wc = workbench_default_config()
        wc["environment"].update(profile=self.config["profile"], beam_count_max=200,
                                 power_budget_w=self.config["power_budget_w"],
                                 basic_fraction=self.config["basic_fraction"],
                                 action_mode="legacy_clip", action_mask="joint_feasible")
        env = WorkbenchEnv(wc)
        state = env.reset(scene=scene)
        policy = BaselinePolicy("greedy", env)
        complete = False
        while not complete:
            choice = policy.act(state, deterministic=True, action_mask=env.action_mask())
            state, _, complete, _ = env.step(choice)
        metrics = env.metrics()
        if not metrics["complete"] or metrics["overload"] or metrics["clipped_count"]:
            raise RuntimeError("B1 did not produce a complete, unprojected feasible allocation")
        self.scene = env.scene()
        self.initial_beams = env.beam_results()
        n = len(self.initial_beams)
        self.demand = np.asarray(self.scene["rate"], dtype=float).reshape(n)
        self.power = np.asarray([row["power_w"] for row in self.initial_beams], dtype=float)
        self.current_starts = np.asarray([row["freq"] for row in self.initial_beams], dtype=int)
        self.current_slots = np.asarray([row["slots"] for row in self.initial_beams], dtype=int)
        self.current_eval = self._evaluate(self.current_starts, self.current_slots)
        actual = np.asarray([row["rate_bps"] for row in self.initial_beams])
        if not np.allclose(self.current_eval.capacity_bps, actual, rtol=1e-8, atol=1e-3):
            raise RuntimeError("I0 physics did not reproduce the actual greedy terminal capacities")
        self.base_eval = self.current_eval
        self.base_basic = self.base_eval.capacity_bps >= self.config["basic_fraction"] * self.demand
        self.best_starts = self.current_starts.copy()
        self.best_slots = self.current_slots.copy()
        self.best_eval = self.current_eval
        self.decision_count = self.probe_count = self.invalid_count = self.repeat_count = 0
        self.proposals_generated = 0
        self.done = False
        self.visited = {_layout_key(self.current_starts, self.current_slots)}
        self.trace = []
        self._candidate_cache = None
        self.initialization_seconds = time.perf_counter() - started
        self.candidate_generation_seconds = 0.0
        self.probe_seconds = 0.0
        return self.state()

    def _rows(self, starts, slots):
        return [{**row, "freq": int(starts[i]), "slots": int(slots[i])}
                for i, row in enumerate(self.initial_beams)]

    def _evaluate(self, starts, slots):
        physics = FixedAllocationPhysics.from_scene(
            self.scene, {"environment": {"profile": self.config["profile"],
                                         "power_budget_w": self.config["power_budget_w"]}},
            self._rows(starts, slots))
        return physics.evaluate(self.power)

    def _legal(self, starts, slots):
        if (len(starts) != len(self.demand) or len(slots) != len(self.demand)
                or np.any(slots < 1) or np.any(slots > 10)
                or np.any(starts < 0) or np.any(starts + slots > 100)
                or self.power.sum() > self.config["power_budget_w"] + 1e-6):
            return False
        for group in range(8):
            occupied = np.zeros(100, dtype=bool)
            for i in range(group, len(slots), 8):
                block = slice(int(starts[i]), int(starts[i] + slots[i]))
                if occupied[block].any():
                    return False
                occupied[block] = True
        return True

    def _protected(self, evaluation):
        return (evaluation.si + self.config["si_tolerance"] >= self.base_eval.si
                and evaluation.sgm + self.config["sgm_tolerance"] >= self.base_eval.sgm
                and bool(np.all(evaluation.capacity_bps[self.base_basic]
                                >= self.config["basic_fraction"] * self.demand[self.base_basic])))

    def _candidate(self, kind, changes):
        starts, slots = self.current_starts.copy(), self.current_slots.copy()
        for i, start, length in changes:
            starts[i], slots[i] = int(start), int(length)
        if not self._legal(starts, slots):
            return None
        key = _layout_key(starts, slots)
        if key == _layout_key(self.current_starts, self.current_slots):
            return None
        digest = _sha_layout(starts, slots)
        return Candidate(digest, kind, tuple((int(i), int(f), int(k)) for i, f, k in changes),
                         tuple(map(int, starts)), tuple(map(int, slots)))

    def candidates(self):
        if self.scene is None or self.done:
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
            if candidate is not None and candidate.id not in seen:
                seen.add(candidate.id)
                by_kind[kind].append(candidate)

        for i in priority[:min(n, 32)]:
            start, count = int(self.current_starts[i]), int(self.current_slots[i])
            for delta in (-5, -1, 1, 5):
                add("move", ((i, start + delta, count),))
            if len(by_kind["move"]) >= 8:
                break

        # Adjacent boundaries transfer one slot without leaving a gap.
        for group in range(8):
            ordered = sorted(range(group, n, 8), key=lambda i: int(self.current_starts[i]))
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
            peers = sorted((j for j in range(i % 8, n, 8) if j != i),
                           key=lambda j: (abs(int(self.current_starts[j] - self.current_starts[i])), j))
            for j in peers[:4]:
                add("swap", ((i, int(self.current_starts[j]), int(self.current_slots[j])),
                             (j, int(self.current_starts[i]), int(self.current_slots[i]))))
            if len(by_kind["swap"]) >= 8:
                break

        for group in range(8):
            ordered = sorted(range(group, n, 8), key=lambda i: int(self.current_starts[i]))
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

        if _layout_key(self.best_starts, self.best_slots) != _layout_key(
                self.current_starts, self.current_slots):
            add("revert", tuple((i, int(self.best_starts[i]), int(self.best_slots[i]))
                                for i in range(n) if self.best_starts[i] != self.current_starts[i]
                                or self.best_slots[i] != self.current_slots[i]))

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
        if self.scene is None:
            raise RuntimeError("reset before reading I0 state")
        n = len(self.demand)
        result = np.zeros((MAX_BEAMS, 14), dtype=np.float32)
        locations = np.asarray(self.scene["beam_location"], dtype=float)
        result[:n, 0] = 1
        result[:n, 1] = locations[:, 0] / 90
        result[:n, 2] = locations[:, 1] / 180
        result[:n, 3] = self.demand / 1e9
        result[:n, 4] = np.arange(n) % 8 / 7
        result[:n, 5] = self.current_starts / 100
        result[:n, 6] = self.current_slots / 10
        result[:n, 7] = self.power / 50
        result[:n, 8] = self.current_eval.capacity_bps / self.demand
        result[:n, 9] = self.best_starts / 100
        result[:n, 10] = self.best_slots / 10
        result[:n, 11] = self.best_eval.capacity_bps / self.demand
        result[:n, 12] = self.base_basic.astype(float)
        result[:n, 13] = np.arange(n) / max(n - 1, 1)
        global_part = np.asarray([n / 200, self.decision_count / 32, self.probe_count / 32,
                                  (32 - self.decision_count) / 32, (32 - self.probe_count) / 32,
                                  self.current_eval.si, self.current_eval.sgm,
                                  self.best_eval.si, self.best_eval.sgm,
                                  self.base_eval.si, self.base_eval.sgm,
                                  self.power.sum() / self.config["power_budget_w"]],
                                 dtype=np.float32)
        return np.concatenate((result.reshape(-1), global_part))

    def action_feature(self, candidate):
        result = np.zeros(ACTION_WIDTH, dtype=np.float32)
        if candidate is None:  # STOP
            result[-1] = 1
            return result
        for offset, (i, start, count) in enumerate(candidate.changes[:4]):
            at = 5 * offset
            result[at:at + 5] = (i / 200, start / 100, count / 10,
                                  (start - self.current_starts[i]) / 100,
                                  (count - self.current_slots[i]) / 10)
        result[20 + KIND_NAMES.index(candidate.kind)] = 1
        result[-2] = min(len(candidate.changes), 4) / 4
        return result

    def step(self, candidate_id):
        if self.done:
            raise RuntimeError("I0 episode already complete")
        candidates = {candidate.id: candidate for candidate in self.candidates()}
        self.decision_count += 1
        reward = 0.0
        reason = "candidate"
        candidate = candidates.get(candidate_id)
        if candidate_id == "STOP":
            reason = "stop"
            self.done = True
        elif candidate is None:
            reason = "invalid_or_stale"
            self.invalid_count += 1
        else:
            starts = np.asarray(candidate.starts, dtype=int)
            slots = np.asarray(candidate.slots, dtype=int)
            key = _layout_key(starts, slots)
            if key in self.visited:
                reason = "repeat"
                self.repeat_count += 1
            started = time.perf_counter()
            evaluated = self._evaluate(starts, slots)
            self.probe_seconds += time.perf_counter() - started
            self.probe_count += 1
            self.visited.add(key)
            self.current_starts, self.current_slots, self.current_eval = starts, slots, evaluated
            if self._protected(evaluated) and evaluated.si > self.best_eval.si + 1e-12:
                reward = 100.0 * (evaluated.si - self.best_eval.si)
                self.best_starts, self.best_slots, self.best_eval = starts.copy(), slots.copy(), evaluated
            self._candidate_cache = None
        if self.decision_count >= self.config["max_decisions"] or self.probe_count >= self.config["max_probes"]:
            self.done = True
            if reason == "candidate":
                reason = "decision_or_probe_limit"
        info = {"reason": reason, "candidate_id": candidate_id, "decisions": self.decision_count,
                "probes": self.probe_count, "current_si": self.current_eval.si,
                "best_si": self.best_eval.si, "best_sgm": self.best_eval.sgm,
                "protected_output": self._protected(self.best_eval)}
        self.trace.append(info)
        return self.state(), reward, self.done, info

    def output(self):
        if self.scene is None:
            raise RuntimeError("I0 has no complete allocation")
        beams = []
        for i, old in enumerate(self.initial_beams):
            capacity = float(self.best_eval.capacity_bps[i])
            matching = float(self.best_eval.beam_sgm[i])
            row = dict(old, freq=int(self.best_starts[i]), slots=int(self.best_slots[i]),
                       **beam_service_fields(capacity, float(self.demand[i]), matching))
            row["basic_met"] = bool(capacity >= self.config["basic_fraction"] * self.demand[i])
            beams.append(row)
        return {"beams": beams, "metrics": {"si": self.best_eval.si, "sgm": self.best_eval.sgm,
                                            "initial_si": self.base_eval.si,
                                            "initial_sgm": self.base_eval.sgm,
                                            "basic_met": sum(row["basic_met"] for row in beams),
                                            "power_w": float(self.power.sum()),
                                            "slots_total": int(self.best_slots.sum()),
                                            "decisions": self.decision_count,
                                            "probes": self.probe_count,
                                            "invalid": self.invalid_count,
                                            "repeats": self.repeat_count,
                                            "initialization_seconds": self.initialization_seconds,
                                            "candidate_generation_seconds": self.candidate_generation_seconds,
                                            "probe_seconds": self.probe_seconds,
                                            "proposals_generated": self.proposals_generated},
                "trace": list(self.trace), "layout_sha256": _sha_layout(self.best_starts, self.best_slots)}

    def checkpoint_state(self):
        if self.scene is None:
            raise RuntimeError("Cannot save an uninitialized I0 episode")
        return {"version": 1, "initial_layout_sha256": _sha_layout(
                    [row["freq"] for row in self.initial_beams],
                    [row["slots"] for row in self.initial_beams]),
                "current_starts": self.current_starts.copy(), "current_slots": self.current_slots.copy(),
                "best_starts": self.best_starts.copy(), "best_slots": self.best_slots.copy(),
                "decision_count": self.decision_count, "probe_count": self.probe_count,
                "invalid_count": self.invalid_count, "repeat_count": self.repeat_count,
                "proposals_generated": self.proposals_generated,
                "initialization_seconds": self.initialization_seconds,
                "candidate_generation_seconds": self.candidate_generation_seconds,
                "probe_seconds": self.probe_seconds,
                "visited": list(self.visited), "trace": list(self.trace), "done": self.done}

    def load_checkpoint_state(self, saved):
        """Call after reset on the exact frozen scene; recomputation is audited."""
        if saved["version"] != 1 or saved["initial_layout_sha256"] != _sha_layout(
                [row["freq"] for row in self.initial_beams],
                [row["slots"] for row in self.initial_beams]):
            raise ValueError("I0 resume initial allocation differs")
        current_starts = np.asarray(saved["current_starts"], dtype=int)
        current_slots = np.asarray(saved["current_slots"], dtype=int)
        best_starts = np.asarray(saved["best_starts"], dtype=int)
        best_slots = np.asarray(saved["best_slots"], dtype=int)
        if not self._legal(current_starts, current_slots) or not self._legal(best_starts, best_slots):
            raise ValueError("I0 resume contains an illegal layout")
        current_eval = self._evaluate(current_starts, current_slots)
        best_eval = self._evaluate(best_starts, best_slots)
        if not self._protected(best_eval) or best_eval.si + 1e-12 < self.base_eval.si:
            raise ValueError("I0 resume best output violates baseline guards")
        self.current_starts, self.current_slots, self.current_eval = current_starts, current_slots, current_eval
        self.best_starts, self.best_slots, self.best_eval = best_starts, best_slots, best_eval
        self.decision_count = int(saved["decision_count"])
        self.probe_count = int(saved["probe_count"])
        self.invalid_count = int(saved["invalid_count"])
        self.repeat_count = int(saved["repeat_count"])
        self.proposals_generated = int(saved["proposals_generated"])
        self.initialization_seconds = float(saved["initialization_seconds"])
        self.candidate_generation_seconds = float(saved["candidate_generation_seconds"])
        self.probe_seconds = float(saved["probe_seconds"])
        self.visited = {tuple((int(f), int(k)) for f, k in row) for row in saved["visited"]}
        self.trace = list(saved["trace"])
        self.done = bool(saved["done"])
        if (not 0 <= self.probe_count <= self.decision_count <= self.config["max_decisions"]
                or self.probe_count > self.config["max_probes"]
                or _layout_key(current_starts, current_slots) not in self.visited):
            raise ValueError("I0 resume resource ledger differs")
        self._candidate_cache = None


def run_baseline(scene, method, seed, config):
    """One rule/random rollout using exactly the learner's menu and hard limits."""
    started = time.perf_counter()
    if method not in ("rule", "random", "greedy"):
        raise ValueError("Unknown improvement baseline")
    env = ImprovementEnv(config)
    env.reset(scene, seed=seed)
    rng = random.Random(seed)
    if method != "greedy":
        while not env.done:
            choices = env.candidates()
            if not choices:
                env.step("STOP")
            else:
                if method == "rule":
                    # A fixed, no-lookahead shortage/slot rule. It never reads
                    # candidate physical outcomes before paying for the probe.
                    def priority(candidate):
                        benefit = sum((1 - float(env.current_eval.beam_si[i]))
                                      * (count - int(env.current_slots[i]))
                                      for i, _, count in candidate.changes)
                        kind_order = {"swap": 4, "boundary": 3, "window": 2,
                                      "move": 1, "revert": 0}
                        return (benefit, kind_order[candidate.kind], candidate.id)
                    chosen = max(choices, key=priority)
                else:
                    chosen = rng.choice(choices)
                env.step(chosen.id)
    else:
        env.step("STOP")
    result = env.output()
    result["method"] = method
    result["seed"] = seed
    result["action_seed"] = seed
    result["metrics"]["total_seconds"] = time.perf_counter() - started
    return result
