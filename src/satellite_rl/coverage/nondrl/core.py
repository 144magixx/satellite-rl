"""Common exact-geometry executor and leakage-free learning API.

``solve(scene, route, config, initial=None, policy=None)`` accepts a CoverageScene
and a NondrlConfig/dict. Result Layouts are objects; use ``to_record`` for JSON.
Learning hooks implement ``learning.run_learning(problem, route, policy)`` and
mutate Problem through preview/commit or consider/commit. No teacher scores are
present in observe/menu. Aircraft feature columns: normalized lat, lon, demand,
owner load, legal beam fraction, owner lat, owner lon, zero-demand flag. Beam
columns: lat, lon, load/mean, user fraction, positive-demand fraction, minimum
footprint slack, mean member demand/mean, zero-demand flag. Action columns: source
load/mean, target load/mean, moved demand/mean, ray distance/radius, exact squared
load proxy delta/(N*mean**2), kind(one transfer/two swap/three center), user
latitude/90, longitude/180. Region columns: size/N, aircraft fraction, load/mean,
load CV, max/mean, min/mean, span/radius, boundary legal fraction.

Only this executor publishes layouts. All failure paths leave current intact.
The monotonic clock stops starting work; indivisible calls may finish late and
are explicitly marked. Initial construction and final verification are separate
from the online budget and always included in end-to-end time.
"""
from __future__ import annotations

import hashlib
import json
from time import perf_counter

import numpy as np

from satellite_rl.envs.regions import get_profile
from ..geometry import satellite_rays_numpy, squared_ray_limit, visible_numpy
from ..variable_config import VariableConfig
from ..variable_design import Layout, _check, _legal_matrix, geometry_audit
from .config import NondrlConfig


class BudgetStop(RuntimeError):
    pass


def clone(layout):
    return Layout.from_record(layout.to_record())


def layout_sha256(layout):
    return hashlib.sha256(json.dumps(layout.to_record(), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def variable_config(config):
    return VariableConfig(max_beams=config.max_beams, fixed_width_deg=config.fixed_width_deg,
                          max_geometry_calls=config.max_geometry_calls)


def metrics_all(scene, layout, config=None):
    """Exact whole-layout statistics; all-zero CV is undefined (None)."""
    cfg = NondrlConfig.from_dict(config)
    legal, loads = _check(scene, layout, variable_config(cfg), allow_partial=False)
    total = float(np.sum(scene.demands))
    if not np.isfinite(total) or not np.isfinite(loads).all():
        raise ValueError("aggregate demand cannot be represented as finite float64")
    positive = loads[loads > 0]
    mean = float(loads.mean()) if len(loads) else 0.0
    cv = float(loads.std() / mean) if mean > 0 else None
    cv_pos = float(positive.std() / positive.mean()) if len(positive) else None
    positive_mean = float(positive.mean()) if len(positive) else None
    band = float(np.mean((positive >= 0.8 * positive_mean) &
                         (positive <= 1.2 * positive_mean))) if len(positive) else None
    n = len(loads)
    return {"beam_count": n, "aircraft_count": len(scene.demands), "coverage_100": True,
            "geometry_feasible": True, "unique_assignment": True, "demand_conserved":
            bool(np.isclose(loads.sum(), total, rtol=1e-12, atol=1e-6)),
            "empty_beams": 0, "zero_demand_beams": int(np.sum(loads == 0)),
            "all_zero_demand": total == 0, "load_cv_all": cv, "load_cv_positive": cv_pos,
            "load_cv": cv, "demand_band_fraction": band, "beam_demand_bps": loads.tolist(),
            "total_demand_bps": total, "square_load_objective": float(np.dot(loads, loads)),
            "geometry_version": "exact_satellite_ray_europe_v1",
            "width_physics_version": "idealized_circular_footprint_v1"}


def ray_to_location(ray, profile="europe_v1"):
    """Intersect a unit GEO ray with spherical Earth, selecting near surface."""
    p = get_profile(profile)
    angle = np.deg2rad(p.satellite_longitude_deg)
    sat = p.orbit_radius_km * np.array([np.cos(angle), np.sin(angle), 0.0])
    direction = np.asarray(ray, dtype=float)
    direction = direction / np.linalg.norm(direction)
    dot = float(sat @ direction)
    discriminant = dot * dot - (float(sat @ sat) - p.earth_radius_km ** 2)
    if discriminant < 0:
        return None
    distance = -dot - np.sqrt(discriminant)
    if distance <= 0:
        return None
    ground = sat + distance * direction
    location = np.rad2deg([np.arctan2(ground[2], np.hypot(ground[0], ground[1])),
                          np.arctan2(ground[1], ground[0])])
    return location if bool(visible_numpy(location, profile)) else None


def ray_center(positions, weights=None, profile="europe_v1"):
    if not len(positions):
        return None
    rays = satellite_rays_numpy(positions, profile)
    if weights is None or float(np.sum(weights)) <= 0:
        direction = rays.mean(axis=0)
    else:
        direction = np.average(rays, axis=0, weights=weights)
    return ray_to_location(direction, profile)


class Problem:
    """Single-scene atomic executor shared by optimization, models and teachers."""

    def __init__(self, scene, config=None, initial=None, physical_evaluator=None):
        self.scene = scene
        self.config = NondrlConfig.from_dict(config)
        self.cfg = variable_config(self.config)
        self.positions = np.asarray(scene.positions, dtype=float)
        self.demands = np.asarray(scene.demands, dtype=float)
        self.rays = satellite_rays_numpy(self.positions, self.config.profile)
        self.started = perf_counter()
        self.trace = []
        self.calls = {"candidates": 0, "physical_calls": 0, "geometry_calls": 0,
                      "ray_pair_checks": 0, "solver_calls": 0, "rebuilds": 0,
                      "initial_trials": 0, "initial_geometry_calls": 0,
                      "physical_cache_hits": 0, "allocation_decisions": 0,
                      "independent_physics_checks": 0, "aircraft_link_checks": 0,
                      "uncharged_geometry_calls": 0, "uncharged_ray_pair_checks": 0}
        self.times = {"initial_seconds": 0.0, "initial_physical_seconds": 0.0,
                      "proposal_seconds": 0.0, "physical_seconds": 0.0,
                      "solver_seconds": 0.0, "geometry_seconds": 0.0,
                      "final_verification_seconds": 0.0}
        self._physical_evaluator = physical_evaluator
        self._physical_cache = {}
        self.last_raw = None
        self.last_repaired = None
        self.stop_reason = None
        begin = perf_counter()
        if initial is None:
            from ..drl_r04_config import DrlR04Config
            from ..drl_r04_core import DrlR04Env, run_h1
            env = DrlR04Env(scene, DrlR04Config(max_beams=self.config.max_beams,
                                              max_trials=self.config.initial_trials,
                                              max_geometry_calls=self.config.max_geometry_calls))
            result = run_h1(env)
            initial = result.get("layout", env.best_layout)
            if isinstance(initial, dict):
                initial = Layout.from_record(initial)
            self.calls["initial_trials"] = env.trials
            self.calls["initial_geometry_calls"] = env.matrix_calls
            self.initial_provenance = {"initializer": "drl_r04_h1_bounded",
                                       "trials": env.trials, "geometry_calls": env.matrix_calls}
        else:
            initial = Layout.from_record(initial) if isinstance(initial, dict) else clone(initial)
            self.initial_provenance = {"initializer": "shared_supplied_layout",
                                       "construction_seconds": None}
        self.initial = clone(initial)
        self.current = clone(initial)
        self.initial_sha256 = layout_sha256(initial)
        self.initial_metrics = self._metrics(initial, charge=False)
        self.n = len(initial.centers)
        self.times["initial_seconds"] = perf_counter() - begin
        begin = perf_counter()
        self.initial_physical = self._physical(initial, charge=False)
        self.times["initial_physical_seconds"] = perf_counter() - begin
        self.current_metrics = self.initial_metrics
        self.current_physical = self.initial_physical
        self.online_started = perf_counter()
        self.deadline = self.online_started + self.config.online_seconds

    def available(self):
        reason = None
        if perf_counter() >= self.deadline:
            reason = "online_time_limit"
        elif self.calls["candidates"] >= self.config.max_candidates:
            reason = "candidate_limit"
        elif self.calls["physical_calls"] >= self.config.max_physical_calls:
            reason = "physical_call_limit"
        elif self.calls["geometry_calls"] >= self.config.max_geometry_calls:
            reason = "geometry_call_limit"
        if reason:
            self.stop_reason = reason
        return reason is None

    def guard(self):
        if not self.available():
            raise BudgetStop(self.stop_reason)

    def remaining(self):
        return max(0.0, self.deadline - perf_counter())

    def _geometry(self, function, *args, charge=True, **kwargs):
        if charge:
            if perf_counter() >= self.deadline or self.calls["geometry_calls"] >= self.config.max_geometry_calls:
                self.stop_reason = "online_time_limit" if perf_counter() >= self.deadline else "geometry_call_limit"
                raise BudgetStop(self.stop_reason)
        start = perf_counter()
        with geometry_audit() as audit:
            try:
                result = function(*args, **kwargs)
            finally:
                if charge:
                    self.calls["geometry_calls"] += audit["matrix_calls"]
                    self.calls["ray_pair_checks"] += audit["ray_pair_checks"]
                    self.times["geometry_seconds"] += perf_counter() - start
                else:
                    self.calls["uncharged_geometry_calls"] += audit["matrix_calls"]
                    self.calls["uncharged_ray_pair_checks"] += audit["ray_pair_checks"]
        return result

    def legal(self, centers=None, positions=None):
        centers = self.current.centers if centers is None else np.asarray(centers)
        positions = self.positions if positions is None else np.asarray(positions)
        return self._geometry(_legal_matrix, positions, centers,
                              np.full(len(centers), 0.5), self.cfg)

    def _metrics(self, layout, charge=True):
        return self._geometry(metrics_all, self.scene, layout, self.config, charge=charge)

    def _physical(self, layout, charge=True, force=False):
        sha = layout_sha256(layout)
        if not force and sha in self._physical_cache:
            if charge:
                self.calls["physical_cache_hits"] += 1
            return self._physical_cache[sha]
        if charge:
            if (perf_counter() >= self.deadline or self.calls["physical_calls"] >= self.config.max_physical_calls
                    or self.calls["geometry_calls"] >= self.config.max_geometry_calls):
                self.stop_reason = ("online_time_limit" if perf_counter() >= self.deadline else
                    "physical_call_limit" if self.calls["physical_calls"] >= self.config.max_physical_calls
                    else "geometry_call_limit")
                raise BudgetStop(self.stop_reason)
            self.calls["physical_calls"] += 1
        start = perf_counter()
        if self._physical_evaluator:
            result = self._physical_evaluator(self.scene, layout)
        else:
            from ..variable_downstream import evaluate_layout
            with geometry_audit() as audit:
                result = evaluate_layout(self.scene, layout,
                                         {"coverage": {"variable": {"max_beams": self.config.max_beams}}})
            if charge:
                self.calls["geometry_calls"] += audit["matrix_calls"]
                self.calls["ray_pair_checks"] += audit["ray_pair_checks"]
            else:
                self.calls["uncharged_geometry_calls"] += audit["matrix_calls"]
                self.calls["uncharged_ray_pair_checks"] += audit["ray_pair_checks"]
        elapsed = perf_counter() - start
        if charge:
            self.times["physical_seconds"] += elapsed
        for key in ("allocation_decisions", "independent_physics_checks", "aircraft_link_checks"):
            self.calls[key] += int(result.get("calls", {}).get(key, 0))
        carry = self.carry(result)
        power = result.get("metrics", {}).get("total_power_w", 0.0)
        if (carry is not None and (not np.isfinite(carry) or not 0 <= carry <= 1 + 1e-10)
                or not np.isfinite(power) or power < -1e-10 or power > 3000 + 1e-7):
            raise ValueError("nonfinite or out-of-range physical service/resource result")
        self._physical_cache[sha] = result
        return result

    @staticmethod
    def carry(physical):
        return physical["aircraft_service"]["demand_weighted_carried_fraction"]

    def consider(self, layout, source="proposal", *, allow_nonimproving=False):
        """Charge and evaluate a proposal; return trial, without mutating current."""
        self.guard()
        self.calls["candidates"] += 1
        self.last_raw = clone(layout) if layout is not None else None
        start = perf_counter()
        trial = {"layout": layout, "metrics_all": None, "physical": None, "accepted": False,
                 "reason": "no_proposal", "source": source}
        try:
            if layout is None:
                return trial
            if len(layout.centers) != self.n:
                trial["reason"] = "fixed_N_violation"
                return trial
            metrics = self._metrics(layout)
            trial["metrics_all"] = metrics
            old = self.current_metrics["load_cv_all"]
            new = metrics["load_cv_all"]
            if old is None or new is None:
                trial["reason"] = "all_zero_demand"
                return trial
            if new > self.initial_metrics["load_cv_all"] + self.config.cv_tolerance:
                trial["reason"] = "I0_CV_protection"
                return trial
            if not allow_nonimproving and new >= old - self.config.cv_tolerance:
                trial["reason"] = "no_exact_CV_improvement"
                return trial
            physical = self._physical(layout)
            trial["physical"] = physical
            base_carry, carry = self.carry(self.initial_physical), self.carry(physical)
            if base_carry is not None and (carry is None or
                    carry < base_carry - self.config.carry_tolerance - 1e-12):
                trial["reason"] = "I0_aircraft_carry_protection"
                return trial
            phy_metrics = physical.get("metrics", {})
            if phy_metrics.get("total_power_w", 0) > 3000 + 1e-7:
                trial["reason"] = "power_violation"
                return trial
            trial.update(accepted=True, reason="validated_candidate")
            self.last_repaired = clone(layout)
            return trial
        except ValueError as error:
            trial["reason"] = "validation_rejected: " + str(error)
            return trial
        except BudgetStop as error:
            trial["reason"] = "budget_exhausted_before_full_validation: " + str(error)
            raise
        finally:
            elapsed = perf_counter() - start
            trial["seconds"] = elapsed
            trial["finished_after_deadline"] = perf_counter() > self.deadline
            self.trace.append({k: v for k, v in trial.items() if k not in ("layout", "physical")}
                              | {"layout": layout.to_record() if layout is not None else None,
                                 "layout_sha256": layout_sha256(layout) if layout is not None else None,
                                 "carry": self.carry(trial["physical"]) if trial["physical"] else None})

    def commit(self, trial):
        if not trial.get("accepted"):
            return False
        before = layout_sha256(self.current)
        old_cv = self.current_metrics["load_cv_all"]
        new_cv = trial["metrics_all"]["load_cv_all"]
        if old_cv is not None and (new_cv is None or new_cv > old_cv + self.config.cv_tolerance):
            self.trace.append({"source": "atomic_commit", "committed": False,
                               "reason": "stale_trial_would_worsen_current", "before_sha256": before})
            return False
        self.current = clone(trial["layout"])
        self.current_metrics = trial["metrics_all"]
        self.current_physical = trial["physical"]
        self.trace.append({"source": "atomic_commit", "committed": True,
                           "before_sha256": before, "after_sha256": layout_sha256(self.current),
                           "load_cv_before": old_cv, "load_cv_after": new_cv,
                           "carry_after": self.carry(self.current_physical)})
        return True

    def observe(self):
        legal = self.legal()
        loads = np.bincount(self.current.assignment, weights=self.demands, minlength=self.n)
        mu = max(float(self.demands.sum()) / max(self.n, 1), 1.0)
        owner = self.current.assignment
        counts = np.bincount(owner, minlength=self.n)
        positive = np.bincount(owner, weights=self.demands > 0, minlength=self.n)
        rays = satellite_rays_numpy(self.current.centers, self.config.profile)
        d2 = ((self.rays[:, None] - rays[None]) ** 2).sum(axis=2)
        radius = squared_ray_limit(0.25)
        af = np.column_stack(((self.positions[:, 0] - 52.5) / 27.5,
                              (self.positions[:, 1] - 5) / 45, self.demands / mu,
                              loads[owner] / mu, legal.mean(axis=1),
                              (self.current.centers[owner, 0] - 52.5) / 27.5,
                              (self.current.centers[owner, 1] - 5) / 45,
                              self.demands == 0)) if len(owner) else np.empty((0, 8))
        bf = np.zeros((self.n, 8))
        for b in range(self.n):
            ids = np.flatnonzero(owner == b)
            bf[b] = [(self.current.centers[b, 0] - 52.5) / 27.5,
                     (self.current.centers[b, 1] - 5) / 45, loads[b] / mu,
                     counts[b] / max(len(owner), 1), positive[b] / max(counts[b], 1),
                     1 - float(d2[ids, b].max()) / radius,
                     float(self.demands[ids].mean()) / mu, float(loads[b] == 0)]
        return {"aircraft_features": af.astype(np.float32), "beam_features": bf.astype(np.float32),
                "legal": legal.copy(), "assignment": owner.copy(), "positions": self.positions.copy(),
                "demands": self.demands.copy(), "centers": self.current.centers.copy(),
                "budget_fraction": self.remaining() / self.config.online_seconds,
                "feature_version": "nondrl_features_v1"}

    def menu(self):
        started = perf_counter()
        try:
            return self._menu_impl()
        except BudgetStop:
            self.trace.append({"source": "common_menu_generation", "seconds": perf_counter() - started,
                               "truncated": True, "reason": self.stop_reason,
                               "returned_actions": 0, "unfinished_menu_not_exposed_to_policy": True})
            raise

    def _menu_impl(self):
        started = perf_counter()
        legal = self.legal()
        layout = self.current
        loads = np.bincount(layout.assignment, weights=self.demands, minlength=self.n)
        counts = np.bincount(layout.assignment, minlength=self.n)
        mu = max(float(self.demands.sum()) / max(self.n, 1), 1.0)
        br = satellite_rays_numpy(layout.centers, self.config.profile)
        rows = []
        users, beams = np.nonzero(legal)
        owners = layout.assignment[users]
        cross = np.zeros((self.n, self.n), dtype=bool)
        cross[owners, beams] = True
        keep = (owners != beams) & (counts[owners] > 1)
        users, beams, owners = users[keep], beams[keep], owners[keep]
        demand = self.demands[users]
        delta = 2 * demand * (loads[beams] - loads[owners] + demand)
        indices = np.lexsort((beams, users, delta))[:max(1, self.config.max_menu // 2)]
        for index in indices:
            self.guard()
            u, a, b = int(users[index]), int(owners[index]), int(beams[index])
            features = [loads[a] / mu, loads[b] / mu, self.demands[u] / mu,
                        float(((self.rays[u] - br[b]) ** 2).sum()) / squared_ray_limit(0.25),
                        delta[index] / (max(self.n, 1) * mu ** 2), 1.0,
                        self.positions[u, 0] / 90, self.positions[u, 1] / 180]
            rows.append({"kind": "transfer", "aircraft": u, "source": a,
                         "target": b, "features": features})
        # Cross-beam candidate pairs are bounded before selecting aircraft. This
        # avoids the previous Python U^2 enumeration while retaining real swaps.
        pairs = list(zip(*np.nonzero(np.triu(cross & cross.T, 1))))
        pairs.sort(key=lambda pair: (-abs(loads[pair[0]] - loads[pair[1]]), pair))
        for a, b in pairs[:16]:
            self.guard()
            ua = np.flatnonzero((layout.assignment == a) & legal[:, b])
            vb = np.flatnonzero((layout.assignment == b) & legal[:, a])
            if not len(ua) or not len(vb):
                continue
            ua = ua[np.lexsort((ua, -self.demands[ua]))[:8]]
            vb = vb[np.lexsort((vb, self.demands[vb]))]
            target_difference = (loads[a] - loads[b]) / 2
            positions = np.searchsorted(self.demands[vb], self.demands[ua] - target_difference)
            for u, pos in zip(ua, positions):
                self.guard()
                for v_index in sorted({max(0, min(int(pos), len(vb) - 1)),
                                       max(0, min(int(pos) - 1, len(vb) - 1))}):
                    v = int(vb[v_index])
                    difference = self.demands[u] - self.demands[v]
                    change = 2 * difference * (loads[b] - loads[a] + difference)
                    features = [loads[a] / mu, loads[b] / mu, difference / mu, 0,
                                change / (max(self.n, 1) * mu ** 2), 2,
                                self.positions[u, 0] / 90, self.positions[u, 1] / 180]
                    rows.append({"kind": "swap", "aircraft": int(u), "other": v,
                                 "source": int(a), "target": int(b), "features": features})
        rows.sort(key=lambda row: (row["features"][4], row["kind"], row["aircraft"], row["target"]))
        for index, row in enumerate(rows[:self.config.max_menu]):
            row["id"] = index
        self.trace.append({"source": "common_menu_generation", "seconds": perf_counter() - started,
                           "legal_transfer_edges": len(users), "cross_beam_pairs": len(pairs),
                           "inspected_swap_pairs": min(16, len(pairs)), "max_swap_users_per_pair": 8,
                           "returned_actions": min(len(rows), self.config.max_menu),
                           "finished_after_deadline": perf_counter() > self.deadline})
        return rows[:self.config.max_menu]

    def propose(self, action, layout=None):
        result = clone(self.current if layout is None else layout)
        if action["kind"] == "transfer":
            result.assignment[action["aircraft"]] = action["target"]
        elif action["kind"] == "swap":
            u, v = action["aircraft"], action["other"]
            result.assignment[u], result.assignment[v] = result.assignment[v], result.assignment[u]
        elif action["kind"] == "center":
            result.centers[action["target"]] = action["center"]
        else:
            raise ValueError("unknown action kind")
        return result

    def preview(self, action):
        return self.consider(self.propose(action), source=action["kind"])

    def regions(self):
        if self.n < 2:
            return []
        self.guard()
        loads = np.bincount(self.current.assignment, weights=self.demands, minlength=self.n)
        mu = max(float(loads.mean()), 1.0)
        br = satellite_rays_numpy(self.current.centers, self.config.profile)
        distance = ((br[:, None] - br[None]) ** 2).sum(axis=2)
        legal = self.legal()
        rows, seen, excluded = [], set(), []
        order = np.lexsort((np.arange(self.n), -np.abs(loads - mu)))
        for b in order:
            near = np.argsort(distance[b], kind="stable")
            for size in (2, 3, 4, 6):
                beams = tuple(sorted(map(int, near[:min(size, self.n)])))
                if beams in seen:
                    continue
                seen.add(beams)
                ids = np.flatnonzero(np.isin(self.current.assignment, beams))
                if len(ids) > self.config.local_aircraft_limit:
                    excluded.append({"beams": list(beams), "aircraft": len(ids)})
                    continue
                local = loads[list(beams)]
                features = [len(beams) / self.n, len(ids) / max(len(self.demands), 1),
                            local.sum() / mu, local.std() / max(local.mean(), 1),
                            local.max() / mu, local.min() / mu,
                            distance[np.ix_(beams, beams)].max() / squared_ray_limit(0.25),
                            legal[np.ix_(ids, beams)].mean()]
                rows.append({"id": "r" + "-".join(map(str, beams)), "beams": beams,
                             "features": list(map(float, features))})
                if len(rows) >= self.config.max_regions:
                    self.trace.append({"source": "common_region_generation", "returned_regions": len(rows),
                        "engineering_excluded_regions": len(excluded), "exclusions": excluded,
                        "local_aircraft_limit": self.config.local_aircraft_limit,
                        "reason": "engineering_local_aircraft_limit; not geometric infeasibility"})
                    return rows
        self.trace.append({"source": "common_region_generation", "returned_regions": len(rows),
            "engineering_excluded_regions": len(excluded), "exclusions": excluded,
            "local_aircraft_limit": self.config.local_aircraft_limit,
            "reason": "engineering_local_aircraft_limit; not geometric infeasibility"})
        return rows

    def rebuild(self, region, candidate_pool=None, visible=None, edge_mask=None):
        self.guard()
        if self.calls["rebuilds"] >= self.config.max_rebuilds:
            self.stop_reason = "rebuild_limit"
            return None
        self.calls["rebuilds"] += 1
        from .classical import rebuild_region
        return rebuild_region(self, region, candidate_pool, visible, edge_mask)

    def result(self, route):
        online_seconds = perf_counter() - self.online_started
        begin = perf_counter()
        final_metrics = self._metrics(self.current, charge=False)
        final_phy = self._physical(self.current, charge=False, force=True)
        self.times["final_verification_seconds"] = perf_counter() - begin
        self.calls["final_physical_calls"] = 1
        self.calls["initial_physical_calls"] = 1
        self.times["online_seconds"] = online_seconds
        self.times["proposal_seconds"] = max(0.0, online_seconds - self.times["geometry_seconds"]
                                             - self.times["physical_seconds"] - self.times["solver_seconds"])
        self.times["total_seconds"] = perf_counter() - self.started
        return {"route": route, "layout": clone(self.current), "initial_layout": clone(self.initial),
                "raw_layout": self.last_raw, "repaired_layout": self.last_repaired,
                "final_layout": clone(self.current), "metrics_all": final_metrics,
                "physical": final_phy, "initial_metrics_all": self.initial_metrics,
                "initial_physical": self.initial_physical, "trace": self.trace,
                "timing": self.times.copy(), "calls": self.calls.copy(),
                "initial_sha256": self.initial_sha256, "final_sha256": layout_sha256(self.current),
                "initial_provenance": self.initial_provenance, "config": self.config.to_dict(),
                "stop_reason": self.stop_reason or "method_finished",
                "online_overrun_seconds": max(0.0, online_seconds - self.config.online_seconds),
                "RF_scope": "frozen variable_downstream model; no independent RF reimplementation",
                "timing_scope": {"proposal_seconds": "online remainder after separately timed geometry, solver and full physical calls",
                    "initial_seconds": "construction and validation if generated; supplied Layout copy and validation otherwise",
                    "initial_provenance": "shared construction time must be joined from I0 source by runner"},
                "pareto": getattr(self, "pareto", [])}


def solve(scene, route, config=None, initial=None, policy=None, *, method="main", physical_evaluator=None):
    route = str(route).upper().removeprefix("ND").removeprefix("R")
    if route.isdigit():
        route = f"{int(route):02d}"
    if route not in ("I0", "BH", *(f"{i:02d}" for i in range(1, 13))):
        raise ValueError("unknown non-DRL route")
    if route == "I0" and method != "main":
        raise ValueError("unsupported non-DRL I0 method")
    problem = Problem(scene, config, initial, physical_evaluator)
    try:
        if route == "I0":
            pass
        elif route in ("01", "02", "03", "04", "05"):
            from .learning import run_learning
            run_learning(problem, route, policy, method=method)
        else:
            from .classical import run_classical
            run_classical(problem, route, method=method)
    except BudgetStop:
        pass
    result = problem.result(route)
    result["method"] = method
    return result
