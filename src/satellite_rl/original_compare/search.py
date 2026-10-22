"""S2-FIX 局部搜索环境（候选菜单、禁忌搜索、I0 输入）原样移植，评价换成原版环境整套重放。

对应 S2-FIX 的 search_env.py / sac_improvement 的 ImprovementEnv.state 与 agent.py。
"""
from collections import deque
from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .physics import demand_of, evaluate

KIND_NAMES = ("move", "boundary", "swap", "window", "revert", "xswap")
ACTION_WIDTH = 4 * 5 + len(KIND_NAMES) + 2
MAX_BEAMS = 200
MAX_ACTIONS = 17
CANDIDATE_LIMIT = 16


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


class SearchEnv:
    def __init__(self, beams, allocation, max_probes, max_decisions=None):
        self.beams = beams
        self.max_probes = max_probes
        self.max_decisions = max_probes if max_decisions is None else max_decisions
        self.groups, self.current_starts, self.current_slots, self.power = [np.asarray(x).copy() for x in allocation]
        self.demand = demand_of(beams)
        self.locations = np.asarray(beams.beam_info['beam_location'], float)
        self.probe_seconds = self.candidate_seconds = 0.
        self.current_eval = self._evaluate(self.groups, self.current_starts, self.current_slots)
        self.base_eval = self.best_eval = self.current_eval
        self.base_basic = self.current_eval.capacity_bps >= .1 * self.demand
        self.best_groups = self.groups.copy()
        self.best_starts, self.best_slots = self.current_starts.copy(), self.current_slots.copy()
        self.decision_count = self.probe_count = 0
        self.done = False
        self._candidate_cache = None

    def _evaluate(self, groups, starts, slots):
        return evaluate(self.beams, groups, starts, slots, self.power)

    def _legal(self, starts, slots, groups):
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
        if self.done:
            return []
        if self._candidate_cache is not None:
            return self._candidate_cache
        started = time.perf_counter()
        n = len(self.demand)
        priority = sorted(range(n), key=lambda i: (float(self.current_eval.beam_si[i]), -float(self.demand[i]), i))
        by_kind = {kind: [] for kind in KIND_NAMES}
        seen = set()

        def add(kind, changes):
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

        selected = []
        while len(selected) < CANDIDATE_LIMIT and any(by_kind.values()):
            for kind in KIND_NAMES:
                if by_kind[kind] and len(selected) < CANDIDATE_LIMIT:
                    selected.append(by_kind[kind].pop(0))
        self._candidate_cache = selected
        self.candidate_seconds += time.perf_counter() - started
        return selected

    # ---- I0 输入（ImprovementEnv.state 与 SearchEnv.state 的组合） ----
    def state(self):
        n = len(self.demand)
        result = np.zeros((MAX_BEAMS, 14), dtype=np.float32)
        result[:n, 0] = 1
        result[:n, 1] = self.locations[:, 0] / 90
        result[:n, 2] = self.locations[:, 1] / 180
        result[:n, 3] = self.demand / 1e9
        result[:n, 4] = self.groups / 7
        result[:n, 5] = self.current_starts / 100
        result[:n, 6] = self.current_slots / 10
        result[:n, 7] = self.power / 50
        result[:n, 8] = self.current_eval.capacity_bps / self.demand
        result[:n, 9] = self.best_starts / 100
        result[:n, 10] = self.best_slots / 10
        result[:n, 11] = self.best_eval.capacity_bps / self.demand
        result[:n, 12] = self.base_basic.astype(float)
        result[:n, 13] = np.arange(n) / max(n - 1, 1)
        d, p = self.max_decisions, self.max_probes
        global_part = np.asarray([n / 200, self.decision_count / d, self.probe_count / p,
                                  (d - self.decision_count) / d, (p - self.probe_count) / p,
                                  self.current_eval.si, self.current_eval.sgm,
                                  self.best_eval.si, self.best_eval.sgm,
                                  self.base_eval.si, self.base_eval.sgm,
                                  self.power.sum() / 3000.], dtype=np.float32)
        return np.concatenate((result.reshape(-1), global_part))

    def action_feature(self, candidate):
        result = np.zeros(ACTION_WIDTH, np.float32)
        if candidate is None:
            result[-1] = 1
            return result
        for offset, (i, start, count) in enumerate(candidate.changes[:4]):
            result[5 * offset:5 * offset + 5] = (i / 200, start / 100, count / 10,
                                                 (start - self.current_starts[i]) / 100,
                                                 (count - self.current_slots[i]) / 10)
        result[20 + KIND_NAMES.index(candidate.kind)] = 1
        result[-2] = min(len(candidate.changes), 4) / 4
        return result

    def probe(self, candidate):
        started = time.perf_counter()
        evaluation = self._evaluate(np.asarray(candidate.groups), np.asarray(candidate.starts), np.asarray(candidate.slots))
        self.probe_seconds += time.perf_counter() - started
        self.probe_count += 1
        if self._protected(evaluation) and evaluation.W > self.best_eval.W:
            self.best_groups = np.asarray(candidate.groups).copy()
            self.best_starts, self.best_slots = np.asarray(candidate.starts).copy(), np.asarray(candidate.slots).copy()
            self.best_eval = evaluation
        return evaluation

    def move(self, candidate, evaluation):
        self.groups = np.asarray(candidate.groups).copy()
        self.current_starts, self.current_slots = np.asarray(candidate.starts).copy(), np.asarray(candidate.slots).copy()
        self.current_eval = evaluation
        self._candidate_cache = None

    def step(self, candidate_id):
        self.decision_count += 1
        if candidate_id == "STOP":
            self.done = True
        else:
            candidate = next(c for c in self.candidates() if c.id == candidate_id)
            self.move(candidate, self.probe(candidate))
        self.done |= self.decision_count >= self.max_decisions or self.probe_count >= self.max_probes
        return self.state()

    def allocation(self):
        return tuple(x.copy() for x in (self.best_groups, self.best_starts, self.best_slots, self.power))


def rule_priority(env, candidate):
    benefit = sum((1 - env.current_eval.beam_si[i]) * (count - env.current_slots[i])
                  for i, _, count in candidate.changes)
    return benefit, {"swap": 4, "boundary": 3, "window": 2, "move": 1, "revert": 0, "xswap": 4}[candidate.kind]


def tabu_search(env, budget):
    """S2-FIX run_search 的禁忌路径：每轮按规则优先级试 4 个，移到最好的那个，最近 8 个布局禁入。"""
    tabu = deque([layout_key(env.groups, env.current_starts, env.current_slots)], maxlen=8)
    while not env.done and env.probe_count < budget:
        choices = [c for c in env.candidates() if layout_key(c.groups, c.starts, c.slots) not in tabu]
        choices = sorted(choices, key=lambda c: rule_priority(env, c), reverse=True)[:min(4, budget - env.probe_count)]
        if not choices:
            break
        evaluation, chosen = max(((env.probe(c), c) for c in choices), key=lambda pair: pair[0].W)
        env.move(chosen, evaluation)
        tabu.append(layout_key(chosen.groups, chosen.starts, chosen.slots))
        env.decision_count += 1
        env.done = env.probe_count >= budget or env.decision_count >= (budget + 3) // 4
    return env.allocation()


# ---------------- I0 策略网络（S2-FIX agent.py 的 CandidateNetwork） ----------------

class CandidateNetwork(nn.Module):
    def __init__(self, state_width, width, action_width):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(state_width, width), nn.LayerNorm(width), nn.ReLU(),
                                   nn.Linear(width, width), nn.ReLU())
        self.action = nn.Sequential(nn.Linear(action_width, width // 2), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(width + width // 2, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, state, features):
        common = self.state(state).unsqueeze(1).expand(-1, features.shape[1], -1)
        return self.head(torch.cat((common, self.action(features)), dim=-1)).squeeze(-1)


def load_i0_actor(path):
    weights = torch.load(path, map_location="cpu", weights_only=True)
    width, state_width = weights["state.0.weight"].shape
    action_width = weights["action.0.weight"].shape[1]
    actor = CandidateNetwork(state_width, width, action_width)
    actor.load_state_dict(weights)
    return actor.eval()


def pack_actions(env):
    choices = env.candidates()
    features = np.zeros((MAX_ACTIONS, ACTION_WIDTH), dtype=np.float32)
    mask = np.zeros(MAX_ACTIONS, dtype=bool)
    identifiers = [None] * MAX_ACTIONS
    for index, choice in enumerate(choices):
        features[index] = env.action_feature(choice)
        mask[index] = True
        identifiers[index] = choice.id
    features[-1] = env.action_feature(None)
    mask[-1] = True
    identifiers[-1] = "STOP"
    return features, mask, identifiers


@torch.no_grad()
def i0_search(env, actor, seed):
    """S2-FIX i0.policy：每步从候选菜单（含 STOP）按策略概率抽一个，探测并移过去。"""
    torch.manual_seed(seed)
    state = env.state()
    while not env.done:
        features, mask, identifiers = pack_actions(env)
        logits = actor(torch.as_tensor(state[None]), torch.as_tensor(features[None]))
        logprob = F.log_softmax(logits.masked_fill(~torch.as_tensor(mask[None]), -torch.inf), dim=-1)
        action = int(torch.multinomial(logprob.exp()[0], 1).item())
        state = env.step(identifiers[action])
    return env.allocation()
