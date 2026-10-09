"""Shared finite-budget R04 executor for learning and rule baselines.

Preview is an explicitly charged operation for offline teachers and search.
The learning policy sees only ``observe``; it never receives preview scores.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from time import perf_counter

import numpy as np

from .drl_r04_config import DrlR04Config
from .drl_r04_observation import candidate_sha256, encode, layout_sha256
from .variable_design import (Layout, VariableAction, apply_action, candidates,
                              geometry_audit, initial_layout)


def r04_objective(layout: Layout, demands, reference_capacity_bps: float,
                  max_beams: int) -> float:
    """Exact R04 objective for certified complete layouts, without a new ray call."""
    owner = np.asarray(layout.assignment)
    demand = np.asarray(demands, dtype=np.float64)
    if len(owner) != len(demand) or np.any(owner < 0):
        raise ValueError("R04 objective requires complete assignment")
    loads = np.bincount(owner, weights=demand, minlength=len(layout.centers))
    ratio = loads / reference_capacity_bps
    pressure = float(np.mean(np.maximum(ratio - 1, 0) ** 2)) if len(ratio) else 0.0
    balance = float(np.var(ratio)) if len(ratio) else 0.0
    return pressure + 0.1 * balance + 0.02 * len(ratio) / max_beams


@dataclass
class Trial:
    layout: Layout
    score: float | None
    info: dict


class DrlR04Env:
    """One scene, shared candidate and geometry ledger, and best-so-far reward."""

    def __init__(self, scene, config: DrlR04Config | None = None):
        self.scene = scene
        self.config = config or DrlR04Config()
        self.cfg = self.config.variable_config()
        self.initial_layout = None
        self.current_layout = None
        self.best_layout = None
        self.initial_sha256 = None
        self.scene_sha256 = None
        self.initial_j = None
        self.current_j = None
        self.best_j = None
        self.failure_reason = None
        self.stop_reason = None
        self.done = False
        self.terminated = False
        self.truncated = False
        self.commits = 0
        self.trials = 0
        self.geometry_calls = 0
        self.matrix_calls = 0
        self.ray_pair_checks = 0
        self.ledger = []
        self.initialization_seconds = 0.0
        self.menu_seconds = 0.0
        self.trial_seconds = 0.0
        self._menu_cache = {}
        self._trial_cache = {}

    def reset(self):
        self.commits = self.trials = self.geometry_calls = 0
        self.matrix_calls = self.ray_pair_checks = 0
        self.ledger = []
        self._menu_cache = {}
        self._trial_cache = {}
        self.failure_reason = None
        self.stop_reason = None
        self.done = self.terminated = self.truncated = False
        payload = {"positions": np.asarray(self.scene.positions, dtype=np.float64).tolist(),
                   "demands": np.asarray(self.scene.demands, dtype=np.float64).tolist(),
                   "aircraft_ids": list(getattr(self.scene, "aircraft_ids", ())) }
        self.scene_sha256 = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                                       separators=(",", ":")).encode()).hexdigest()
        started = perf_counter()
        with geometry_audit() as audit:
            try:
                layout, init = initial_layout(self.scene, self.cfg, "R04")
            except ValueError as exc:
                # An overlarge candidate set or impossible constructor remains
                # a counted scene, with its original aircraft denominator.
                layout = Layout.empty(len(self.scene.demands))
                layout.aircraft_ids = tuple(getattr(self.scene, "aircraft_ids", ()))
                init = {"status": "initialization_exception", "failure_reason": str(exc),
                        "metrics": {"coverage_100": False, "demand_conserved": False,
                                    "aircraft_count": len(self.scene.demands)}}
        self.initialization_seconds = perf_counter() - started
        self.menu_seconds = self.trial_seconds = 0.0
        self.matrix_calls += audit["matrix_calls"]
        self.ray_pair_checks += audit["ray_pair_checks"]
        self.geometry_calls += max(int(init.get("geometry_budget_charges", 0)),
                                   audit["matrix_calls"])
        self.initial_layout = self.current_layout = self.best_layout = layout
        self.initial_sha256 = layout_sha256(layout)
        self.initialization = init
        if not init["metrics"]["coverage_100"] or not init["metrics"]["demand_conserved"]:
            self.failure_reason = init.get("failure_reason") or "initial_layout_incomplete"
            self.stop_reason = "initialization_failed"
            self.done = self.terminated = True
            self.initial_j = self.current_j = self.best_j = None
            return self._failure_observation(), self._info("initialization_failed")
        if len(layout.centers) > self.config.max_beams or not np.allclose(layout.widths, 0.5):
            raise ValueError("initial layout violates A-stage fixed-width cap")
        self.initial_j = self.current_j = self.best_j = r04_objective(
            layout, self.scene.demands, self.cfg.reference_capacity_bps, self.cfg.max_beams)
        if self.geometry_calls >= self.config.max_geometry_calls:
            self.failure_reason = "initial_geometry_budget_exhausted"
            self.stop_reason = "initial_geometry_budget_exhausted"
            self.done = self.truncated = True
            return self._failure_observation(), self._info("initial_geometry_budget_exhausted")
        return self.observe()[0], self._info("ready")

    def _failure_observation(self):
        return {"beam": np.zeros((0, 12), dtype=np.float32),
                "candidate": np.zeros((1, 16), dtype=np.float32),
                "candidate_mask": np.ones(1, dtype=bool),
                "global": np.zeros(14, dtype=np.float32)}

    def _info(self, status, **extra):
        return {"status": status, "failure_reason": self.failure_reason,
                "stop_reason": self.stop_reason,
                "scene_id": getattr(self.scene, "scene_id", None),
                "scene_source_sha256": getattr(self.scene, "source_hash", None),
                "scene_sha256": self.scene_sha256,
                "initial_sha256": self.initial_sha256,
                "current_sha256": layout_sha256(self.current_layout) if self.current_layout is not None else None,
                "best_sha256": layout_sha256(self.best_layout) if self.best_layout is not None else None,
                "initial_j": self.initial_j, "current_j": self.current_j, "best_j": self.best_j,
                "commits": self.commits, "trials": self.trials,
                "geometry_calls": self.geometry_calls, "matrix_calls": self.matrix_calls,
                "ray_pair_checks": self.ray_pair_checks,
                "initialization_seconds": self.initialization_seconds,
                "menu_seconds": self.menu_seconds,
                "trial_seconds": self.trial_seconds, **extra}

    def _charge(self, audit, charge):
        self.matrix_calls += audit["matrix_calls"]
        self.ray_pair_checks += audit["ray_pair_checks"]
        self.geometry_calls += max(int(charge), audit["matrix_calls"])

    def _menu(self, layout):
        key = layout_sha256(layout)
        if key in self._menu_cache:
            return self._menu_cache[key]
        if self.geometry_calls >= self.config.max_geometry_calls:
            menu = [(VariableAction("stop"), np.zeros(16, dtype=np.float32))]
        else:
            started = perf_counter()
            with geometry_audit() as audit:
                raw = candidates(self.scene, layout, self.cfg, mode="local")
            self.menu_seconds += perf_counter() - started
            self._charge(audit, 1)
            menu = [(a, f) for a, f in raw if a.operation in ("stop", "split", "merge")]
        if not menu or menu[0][0].operation != "stop":
            raise ValueError("R04 candidate menu lacks STOP")
        self._menu_cache[key] = menu
        return menu

    @property
    def menu(self):
        return self._menu(self.current_layout) if not self.done else []

    def observe(self):
        if self.current_layout is None:
            raise RuntimeError("reset environment before observe")
        if self.done:
            return self._failure_observation(), []
        menu = self._menu(self.current_layout)
        obs = encode(self.scene, self.current_layout, self.cfg, self.config,
                     current_j=self.current_j, best_j=self.best_j,
                     best_n=len(self.best_layout.centers), commits=self.commits,
                     trials=self.trials, geometry_calls=self.geometry_calls, menu=menu)
        return obs, menu

    def preview(self, action: VariableAction, *, layout: Layout | None = None) -> Trial:
        """Charged real trial, cached for a later commit; never expose to policy input."""
        if self.done:
            raise RuntimeError("episode ended")
        base = layout if layout is not None else self.current_layout
        menu = self._menu(base)
        if action.operation == "stop" or action not in [a for a, _ in menu]:
            raise ValueError("preview requires a menu split/merge action")
        key = (layout_sha256(base), action)
        if key in self._trial_cache:
            return self._trial_cache[key]
        if self.trials >= self.config.max_trials:
            raise RuntimeError("candidate trial budget exhausted")
        # Existing executor makes at most a few ray matrices per split/merge.
        # Leave headroom so observed calls cannot cross the hard ceiling.
        if self.geometry_calls + 8 > self.config.max_geometry_calls:
            raise RuntimeError("geometry budget exhausted")
        self.trials += 1
        started = perf_counter()
        with geometry_audit() as audit:
            proposed, raw = apply_action(self.scene, base, self.cfg, action)
        self.trial_seconds += perf_counter() - started
        self._charge(audit, raw.get("geometry_budget_charges", 0))
        accepted = bool(raw["accepted"])
        score = (r04_objective(proposed, self.scene.demands,
                               self.cfg.reference_capacity_bps, self.cfg.max_beams)
                 if accepted else None)
        info = {**raw, "trial_number": self.trials, "candidate_id": key[0] + ":" +
                hashlib.sha256(json.dumps(action.to_record(), sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest(),
                "base_sha256": key[0], "trial_sha256": layout_sha256(proposed),
                "trial_j": score, "matrix_calls": audit["matrix_calls"],
                "ray_pair_checks": audit["ray_pair_checks"]}
        trial = Trial(proposed, score, info)
        self._trial_cache[key] = trial
        return trial

    def step(self, action_index: int):
        if self.done:
            raise RuntimeError("episode ended")
        obs, menu = self.observe()
        if not isinstance(action_index, (int, np.integer)) or not 0 <= action_index < len(menu):
            raise IndexError("action index outside current menu")
        action = menu[action_index][0]
        old_best = self.best_j
        self.commits += 1
        accepted = action.operation == "stop"
        reason = "stop" if accepted else None
        trial_info = None
        if action.operation == "stop":
            self.stop_reason = "policy_stop"
            self.done = self.terminated = True
        else:
            try:
                trial = self.preview(action)
                trial_info = trial.info
                accepted = bool(trial.info["accepted"])
                reason = trial.info["reason"]
                if accepted:
                    self.current_layout = trial.layout
                    self.current_j = trial.score
                    if trial.score < self.best_j - 1e-10:
                        self.best_layout = trial.layout
                        self.best_j = trial.score
            except RuntimeError as exc:
                accepted = False
                reason = str(exc)
        reward = self.config.reward_scale * (old_best - self.best_j)
        if not self.done and (self.commits >= self.config.max_commits or
                              self.trials >= self.config.max_trials or
                              self.geometry_calls + 8 > self.config.max_geometry_calls):
            self.stop_reason = ("commit_budget" if self.commits >= self.config.max_commits else
                                "trial_budget" if self.trials >= self.config.max_trials else
                                "geometry_budget")
            self.done = self.truncated = True
        row = self._info("step", request_index=int(action_index), action=action.to_record(),
                         accepted=accepted, reason=reason, reward=reward,
                         trial=trial_info, terminated=self.terminated,
                         truncated=self.truncated,
                         menu_sha256=candidate_sha256(menu))
        self.ledger.append(row)
        next_obs = self.observe()[0] if not self.done else self._failure_observation()
        return next_obs, reward, self.terminated, self.truncated, row

    def result(self):
        status = ("failed_initialization" if self.failure_reason else
                  "completed" if self.done else "incomplete")
        return {**self._info(status),
                "initial_layout": self.initial_layout.to_record() if self.initial_layout else None,
                "raw_terminal_layout": self.current_layout.to_record() if self.current_layout else None,
                "best_layout": self.best_layout.to_record() if self.best_layout else None,
                "initialization": self.initialization,
                "ledger": self.ledger, "terminated": self.terminated,
                "truncated": self.truncated,
                "reward_sum": sum(row["reward"] for row in self.ledger)}


def run_i0(env: DrlR04Env):
    env.reset()
    if not env.done:
        env.step(0)
    return env.result()


def run_random(env: DrlR04Env, seed: int):
    rng = np.random.default_rng(seed)
    env.reset()
    while not env.done:
        env.step(int(rng.integers(len(env.menu))))
    return env.result()


def h1_teacher_label(env: DrlR04Env):
    """Offline label and charged expert trials; call only outside policy inference."""
    menu = env.menu
    selected, best = 0, env.best_j
    for index, (action, _) in enumerate(menu[1:], 1):
        if env.trials >= env.config.max_trials or env.geometry_calls + 8 > env.config.max_geometry_calls:
            break
        trial = env.preview(action)
        if trial.score is not None and trial.score < best - 1e-10:
            selected, best = index, trial.score
    return selected


def run_h1(env: DrlR04Env):
    env.reset()
    while not env.done:
        env.step(h1_teacher_label(env))
    return env.result()


def run_h2(env: DrlR04Env):
    """Two-state beam, depth two; all branches debit the same physical ledger."""
    env.reset()
    while not env.done:
        first_menu = env.menu
        first = []
        for i, (action, _) in enumerate(first_menu[1:], 1):
            if env.trials >= env.config.max_trials or env.geometry_calls + 8 > env.config.max_geometry_calls:
                break
            trial = env.preview(action)
            if trial.score is not None:
                first.append((trial.score, i, trial))
        first.sort(key=lambda row: (row[0], row[1]))
        paths = []
        for score, i, trial in first[:2]:
            paths.append((min(env.best_j, score), (i,)))
            if env.config.max_commits - env.commits < 2:
                continue
            branch_menu = env._menu(trial.layout)
            for j, (action, _) in enumerate(branch_menu[1:], 1):
                if env.trials >= env.config.max_trials or env.geometry_calls + 8 > env.config.max_geometry_calls:
                    break
                second = env.preview(action, layout=trial.layout)
                if second.score is not None:
                    paths.append((min(env.best_j, score, second.score), (i, j)))
        # STOP is an explicit zero-length path; ties prefer fewer commits.
        best = min(paths + [(env.best_j, ())], key=lambda row: (row[0], len(row[1]), row[1]))
        if not best[1] or best[0] >= env.best_j - 1e-10:
            env.step(0)
            continue
        for index in best[1]:
            if env.done:
                break
            env.step(index)
    return env.result()
