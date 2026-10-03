"""Coverage-first planning from the frozen C1 layout.

This module does not change the historical coverage environment.  All methods
operate on aircraft-level input and keep failed scenes in their output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import hashlib
import json
import math
import time
from typing import Any

import numpy as np

from satellite_rl.coverage.baselines import run_baseline
from satellite_rl.coverage.config import default_config as legacy_default_config
from satellite_rl.coverage.geometry import (
    ray_angle_numpy, satellite_rays_numpy, squared_ray_limit, visible_numpy,
)
from satellite_rl.coverage.scenes import BEAM_COUNT, CoverageScene


RADIUS_DEG = 0.25
CARDINAL = ((-1., 0.), (1., 0.), (0., -1.), (0., 1.))
DIAGONAL = ((-1., -1.), (-1., 1.), (1., -1.), (1., 1.))
ACTION_COUNTS = {"X0": 5, "X1": 9, "X2": 13, "X3": 21, "X4": 37}


def balance_options(config: dict) -> dict:
    """Normalize schema-3 coverage.balance, accepting a full config as input."""
    coverage = config.get("coverage", config)
    raw = coverage.get("balance", coverage)
    defaults = {
        "profile": "coverage_priority_v1", "route": "prepare", "action_menu": "X0",
        "step_angle_deg": 0.025, "max_movement_deg": 0.25,
        "coverage_goal": "strict_100", "min_coverage_demand": 0.999,
        "min_coverage_users": 0.999, "subject_cv_target": 0.2,
        "subject_range_tolerance": 0.2, "max_edge_beams": 10,
        "min_subject_beams_in_range": 90,
        "edge_demand_ratio": 0.5, "max_rounds": 10,
        "max_beam_decisions": 1000, "max_repair_decisions": 200,
        "max_repair_evaluations": 20000, "max_candidate_evaluations": 20000,
        "max_assignment_passes": 20, "max_assignment_attempts": 200000,
        "post_move_transfers": 1,
        "max_refinement_seconds": 60.0,
    }
    out = {**defaults, **raw}
    out["min_coverage_demand"] = coverage.get("min_coverage_demand", out["min_coverage_demand"])
    out["min_coverage_users"] = coverage.get("min_coverage_users", out["min_coverage_users"])
    out["max_assignment_scans"] = out["max_assignment_passes"]
    if out["profile"] != "coverage_priority_v1":
        raise ValueError("unsupported coverage balance profile")
    if out["action_menu"] not in ACTION_COUNTS:
        raise ValueError("unknown balance action menu")
    if out["route"] not in ("prepare", "A0", "A1", "A2", "A3", "X0", "X1", "X2", "X3", "X4", "ppo"):
        raise ValueError("unknown balance route")
    for name in ("step_angle_deg", "max_movement_deg", "max_refinement_seconds"):
        if not math.isfinite(float(out[name])) or float(out[name]) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if out["coverage_goal"] not in ("strict_100", "near_99_9"):
        raise ValueError("coverage_goal must be strict_100 or near_99_9")
    for name in ("min_coverage_demand", "min_coverage_users",
                 "subject_cv_target", "subject_range_tolerance", "edge_demand_ratio"):
        if not math.isfinite(float(out[name])) or not 0 <= float(out[name]) <= 1:
            raise ValueError(f"{name} must be within [0,1]")
    for name in ("max_edge_beams", "max_rounds", "max_beam_decisions", "max_repair_decisions",
                 "max_repair_evaluations", "max_candidate_evaluations", "max_assignment_passes",
                 "min_subject_beams_in_range",
                 "max_assignment_attempts", "post_move_transfers"):
        if isinstance(out[name], bool) or int(out[name]) != out[name] or int(out[name]) < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        out[name] = int(out[name])
    if out["max_beam_decisions"] > BEAM_COUNT * out["max_rounds"]:
        raise ValueError("beam-decision cap cannot exceed scheduled rounds")
    if out["post_move_transfers"] != 1:
        raise ValueError("coverage_priority_v1 fixes post_move_transfers=1")
    return out


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _layout_sha(centers: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(centers, dtype="<f8").tobytes()).hexdigest()


def _association_sha(associations: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(associations, dtype="<i8").tobytes()).hexdigest()


@dataclass
class BalanceStart:
    scene_id: str
    scene_sha256: str
    aircraft_ids: list[str]
    centers: np.ndarray
    associations: np.ndarray
    beam_loads_bps: np.ndarray
    regions: list[dict]
    metrics: dict
    stages: dict = field(default_factory=dict)
    status: str = "completed"
    c1_state: dict = field(default_factory=dict)
    a1_state: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        body = {
            "scene_id": self.scene_id, "scene_sha256": self.scene_sha256,
            "aircraft_ids": list(self.aircraft_ids),
            "centers": np.asarray(self.centers, dtype=np.float64).tolist(),
            "associations": np.asarray(self.associations, dtype=np.int64).tolist(),
            "beam_loads_bps": np.asarray(self.beam_loads_bps, dtype=np.float64).tolist(),
            "regions": self.regions, "metrics": self.metrics, "stages": self.stages,
            "c1_state": self.c1_state, "a1_state": self.a1_state,
            "status": self.status,
            "layout_sha256": _layout_sha(self.centers),
            "association_sha256": _association_sha(self.associations),
            "initial_centers_sha256": self.stages.get("c1", {}).get("initial_layout_sha256"),
        }
        body["start_sha256"] = _canonical_sha(body)
        return body

    @property
    def start_sha256(self) -> str:
        return self.to_record()["start_sha256"]

    @classmethod
    def from_record(cls, scene: CoverageScene, record: dict) -> "BalanceStart":
        body = dict(record)
        expected = body.pop("start_sha256")
        if _canonical_sha(body) != expected:
            raise ValueError("balance start record SHA mismatch")
        if (body["scene_id"] != scene.scene_id or body["scene_sha256"] != scene.source_hash or
                body["aircraft_ids"] != scene.aircraft_ids):
            raise ValueError("balance start scene/aircraft identity mismatch")
        centers = np.asarray(body["centers"], dtype=np.float64)
        associations = np.asarray(body["associations"], dtype=np.int64)
        loads = np.asarray(body["beam_loads_bps"], dtype=np.float64)
        if (centers.shape != (BEAM_COUNT, 2) or associations.shape != (len(scene.demands),) or
                loads.shape != (BEAM_COUNT,) or not np.isfinite(centers).all() or
                not visible_numpy(centers).all()):
            raise ValueError("balance start arrays invalid")
        if (_layout_sha(centers) != body["layout_sha256"] or
                _association_sha(associations) != body["association_sha256"]):
            raise ValueError("balance start geometry/association SHA mismatch")
        if body["initial_centers_sha256"] != _layout_sha(scene.centers):
            raise ValueError("balance start original cluster-center SHA mismatch")
        state = GeometryState(scene, centers, associations)
        if not np.allclose(state.loads, loads, atol=1e-6, rtol=1e-12):
            raise ValueError("balance start demand ledger mismatch")
        for name in ("c1_state", "a1_state"):
            part = body[name]
            old = GeometryState(scene, np.asarray(part["centers"], dtype=np.float64),
                                np.asarray(part["associations"], dtype=np.int64))
            if (part["layout_sha256"] != _layout_sha(old.centers) or
                    part["association_sha256"] != _association_sha(old.associations) or
                    not np.allclose(old.loads, part["beam_loads_bps"], atol=1e-6, rtol=1e-12)):
                raise ValueError(f"balance {name} demand or SHA mismatch")
        return cls(scene.scene_id, scene.source_hash, list(scene.aircraft_ids), centers,
                   associations, loads, list(body["regions"]), dict(body["metrics"]),
                   dict(body["stages"]), body["status"],
                   dict(body["c1_state"]), dict(body["a1_state"]))


def _state_record(state: "GeometryState", metrics: dict) -> dict:
    return {"centers": state.centers.tolist(),
            "associations": state.associations.tolist(),
            "beam_loads_bps": state.loads.tolist(), "metrics": metrics,
            "layout_sha256": _layout_sha(state.centers),
            "association_sha256": _association_sha(state.associations)}


def certify_edge_regions(scene: CoverageScene, options: dict | None = None) -> list[dict]:
    """Conservative continuous-geometry certificate for dedicated light regions.

    An edge joins users whose satellite-view separation is at most a full beam
    width.  Users in different connected components cannot share *any* 0.5°
    beam, even at centers outside the action menus.  A component is exemptible
    only if one legal aircraft-position center covers all its users.
    """
    opt = balance_options(options or {})
    positions = np.asarray(scene.positions, dtype=np.float64)
    demand = np.asarray(scene.demands, dtype=np.float64)
    rays = satellite_rays_numpy(positions)
    parent = np.arange(len(rays), dtype=np.int64)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    limit = squared_ray_limit(2 * RADIUS_DEG) + 1e-14
    # Chunk rows to avoid an unbounded U x U temporary on dense snapshots.
    for lo in range(0, len(rays), 128):
        distance = ((rays[lo:lo + 128, None, :] - rays[None, :, :]) ** 2).sum(-1)
        for local, row in enumerate(distance):
            i = lo + local
            neighbors = np.flatnonzero(row[i + 1:] <= limit) + i + 1
            for j in neighbors:
                a, b = find(i), find(int(j))
                if a != b:
                    parent[b] = a
    groups: dict[int, list[int]] = {}
    for i in range(len(rays)):
        groups.setdefault(find(i), []).append(i)
    threshold = float(demand.sum()) / BEAM_COUNT * opt["edge_demand_ratio"]
    regions = []
    for members in groups.values():
        subtotal = float(demand[members].sum())
        if subtotal <= 0 or subtotal >= threshold:
            continue
        users = rays[members]
        witness = None
        for index in members:
            if np.all(((users - rays[index]) ** 2).sum(-1) <= squared_ray_limit(RADIUS_DEG) - 1e-14):
                witness = index
                break
        if witness is None:
            continue
        regions.append({"aircraft_indices": members,
                        "aircraft_ids": [scene.aircraft_ids[i] for i in members],
                        "demand_bps": subtotal,
                        "witness_center": positions[witness].tolist(),
                        "witness_aircraft_id": scene.aircraft_ids[witness],
                        "proof": "cross-component user separation >0.5deg; one legal center covers component"})
    return sorted(regions, key=lambda r: r["aircraft_ids"][0])


class GeometryState:
    """One scene with cached U x 100 ray distances and explicit unique service."""

    def __init__(self, scene: CoverageScene, centers: np.ndarray,
                 associations: np.ndarray | None = None):
        self.scene = scene
        self.centers = np.asarray(centers, dtype=np.float64).copy()
        if self.centers.shape != (BEAM_COUNT, 2) or not visible_numpy(self.centers).all():
            raise ValueError("100 visible beam centers required")
        self.user_rays = satellite_rays_numpy(scene.positions)
        self.beam_rays = satellite_rays_numpy(self.centers)
        self.distance = ((self.user_rays[:, None, :] - self.beam_rays[None, :, :]) ** 2).sum(-1)
        self.eligible = self.distance <= squared_ray_limit(RADIUS_DEG) + 1e-14
        nearest = self.distance.argmin(1)
        self.geometrically_covered = self.eligible.any(1)
        if associations is None:
            self.associations = np.where(self.geometrically_covered, nearest, -1)
        else:
            self.associations = np.asarray(associations, dtype=np.int64).copy()
        self._validate_assignment()
        self.loads = np.bincount(self.associations[self.associations >= 0],
                                 weights=np.asarray(scene.demands)[self.associations >= 0],
                                 minlength=BEAM_COUNT).astype(np.float64)

    def _validate_assignment(self) -> None:
        if self.associations.shape != (len(self.scene.demands),):
            raise ValueError("association count differs from aircraft")
        valid = self.associations >= 0
        if not np.array_equal(valid, self.geometrically_covered):
            raise ValueError("all and only geometrically covered users need one service beam")
        if np.any(self.associations[valid] >= BEAM_COUNT) or np.any(~self.eligible[np.flatnonzero(valid), self.associations[valid]]):
            raise ValueError("association outside beam footprint")

    def copy(self) -> "GeometryState":
        return GeometryState(self.scene, self.centers, self.associations)

    def proposed(self, changes: dict[int, np.ndarray]) -> "GeometryState":
        proposal = object.__new__(GeometryState)
        proposal.scene = self.scene
        proposal.centers = self.centers.copy()
        proposal.user_rays = self.user_rays
        proposal.beam_rays = self.beam_rays.copy()
        proposal.distance = self.distance.copy()
        for beam, center in changes.items():
            proposal.centers[beam] = center
            proposal.beam_rays[beam] = satellite_rays_numpy(np.asarray(center)[None])[0]
            proposal.distance[:, beam] = ((self.user_rays - proposal.beam_rays[beam]) ** 2).sum(-1)
        proposal.eligible = proposal.distance <= squared_ray_limit(RADIUS_DEG) + 1e-14
        proposal.geometrically_covered = proposal.eligible.any(1)
        nearest = proposal.distance.argmin(1)
        proposal.associations = np.where(proposal.geometrically_covered, nearest, -1)
        valid = proposal.associations >= 0
        proposal.loads = np.bincount(proposal.associations[valid],
                                      weights=np.asarray(self.scene.demands)[valid],
                                      minlength=BEAM_COUNT).astype(np.float64)
        return proposal


def balance_metrics(state: GeometryState, regions: list[dict], options: dict,
                    initial_centers: np.ndarray | None = None) -> dict:
    loads = state.loads
    demand = np.asarray(state.scene.demands, dtype=np.float64)
    covered = state.geometrically_covered
    total = float(demand.sum())
    covered_demand = float(demand[covered].sum())
    coverage_demand = covered_demand / total if total > 0 else 0.0
    coverage_users = float(np.count_nonzero(covered) / len(covered))
    mean = float(loads.mean())
    all_std = float(loads.std())
    exempt = []
    for region in regions:
        ids = np.asarray(region["aircraft_indices"], dtype=np.int64)
        assigned = state.associations[ids]
        if len(assigned) == 0 or assigned[0] < 0 or not np.all(assigned == assigned[0]):
            continue
        beam = int(assigned[0])
        if np.count_nonzero(state.associations == beam) == len(ids):
            exempt.append(beam)
    exempt = sorted(set(exempt))
    subject = np.ones(BEAM_COUNT, dtype=bool)
    subject[exempt] = False
    subject_loads = loads[subject]
    subject_mean = float(subject_loads.mean()) if len(subject_loads) else 0.0
    valid_subject = subject_mean > 0
    subject_cv = float(subject_loads.std() / subject_mean) if valid_subject else None
    band = float(options["subject_range_tolerance"])
    in_band = int(np.count_nonzero((subject_loads >= (1-band) * subject_mean) &
                                   (subject_loads <= (1+band) * subject_mean))) if valid_subject else 0
    movement = (ray_angle_numpy(satellite_rays_numpy(state.centers),
                                satellite_rays_numpy(initial_centers))
                if initial_centers is not None else np.zeros(BEAM_COUNT))
    strict = np.count_nonzero(~covered) == 0
    near = coverage_demand >= 0.999 and coverage_users >= 0.999
    feasible_geometry = bool(visible_numpy(state.centers).all() and np.count_nonzero(loads <= 0) == 0)
    coverage_target = strict if options["coverage_goal"] == "strict_100" else near
    target = bool(coverage_target and feasible_geometry and len(exempt) <= options["max_edge_beams"] and
                  len(regions) <= options["max_edge_beams"] and
                  valid_subject and subject_cv <= options["subject_cv_target"] and
                  in_band >= options["min_subject_beams_in_range"])
    peak_ratio = float(subject_loads.max() / subject_mean) if valid_subject else None
    score = (200.0 * (coverage_demand + coverage_users) + 0.2 * in_band -
             2.0 * subject_cv - 0.1 * peak_ratio - 0.1 * len(exempt) -
             0.01 * float(movement.mean()) / options["max_movement_deg"]
             if valid_subject else -1000.0)
    return {
        "coverage_demand": coverage_demand, "coverage_users": coverage_users,
        "uncovered_demand_bps": total - covered_demand,
        "uncovered_users": int(np.count_nonzero(~covered)),
        "load_cv": all_std / mean if mean > 0 else None,
        "load_std_bps": all_std, "max_load_bps": float(loads.max()),
        "empty_beams": int(np.count_nonzero(loads <= 0)), "active_beams": BEAM_COUNT,
        "subject_cv": subject_cv, "subject_std_bps": float(subject_loads.std()),
        "subject_mean_bps": subject_mean, "subject_beams": int(np.count_nonzero(subject)),
        "subject_in_band_beams": in_band,
        "subject_in_band_fraction_all_100": in_band / BEAM_COUNT,
        "subject_max_mean_ratio": peak_ratio,
        "edge_beams": exempt, "edge_beam_count": len(exempt),
        "certified_edge_region_count": len(regions),
        "edge_limit_met": len(exempt) <= options["max_edge_beams"] and len(regions) <= options["max_edge_beams"],
        "edge_demand_fraction": float(loads[exempt].sum() / total) if total > 0 else 0.0,
        "movement_mean_deg": float(movement.mean()), "movement_max_deg": float(movement.max()),
        "coverage_100": strict, "coverage_99_9": near,
        "feasible": feasible_geometry and near,
        "balanced_target_met": target,
        "score": score,
    }


def _quality(metrics: dict, *, stage: str) -> tuple:
    if stage == "repair":
        return (-int(metrics["uncovered_users"]), -float(metrics["uncovered_demand_bps"]),
                -int(metrics["empty_beams"]))
    return (int(metrics["coverage_100"]), int(metrics["coverage_99_9"]),
            int(metrics["subject_in_band_beams"]),
            -float(metrics["subject_cv"] if metrics["subject_cv"] is not None else math.inf),
            -float(metrics["max_load_bps"]))


def _legal_center(center: np.ndarray, initial: np.ndarray, options: dict) -> bool:
    return bool(visible_numpy(center[None])[0] and
                ray_angle_numpy(satellite_rays_numpy(center[None])[0],
                                satellite_rays_numpy(initial[None])[0]) <=
                options["max_movement_deg"] + 1e-9)


def _towards(center: np.ndarray, target: np.ndarray, angle_deg: float) -> np.ndarray:
    """Move toward a lat/lon target by a fixed satellite-view angular distance."""
    vector = np.asarray(target, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    if np.max(np.abs(vector)) < 1e-12:
        return np.asarray(center, dtype=np.float64).copy()
    start_ray = satellite_rays_numpy(center[None])[0]
    actual = float(ray_angle_numpy(start_ray, satellite_rays_numpy(target[None])[0]))
    if actual <= angle_deg:
        return np.asarray(target, dtype=np.float64).copy()
    low, high = 0.0, 1.0
    for _ in range(28):
        middle = (low + high) / 2
        candidate = center + middle * vector
        distance = float(ray_angle_numpy(start_ray, satellite_rays_numpy(candidate[None])[0]))
        if distance < angle_deg:
            low = middle
        else:
            high = middle
    return np.asarray(center + ((low + high) / 2) * vector, dtype=np.float64)


def _directional(center: np.ndarray, direction: tuple[float, float], angle_deg: float) -> np.ndarray:
    direction_array = np.asarray(direction, dtype=np.float64)
    direction_array /= np.linalg.norm(direction_array)
    # Bisection on a generous local latitude/longitude segment.  Actual angle
    # is measured from the satellite, so diagonal and axial actions are equal.
    return _towards(center, center + direction_array * 10.0, angle_deg)


def _assignment_optimize(state: GeometryState, regions: list[dict], options: dict,
                         *, deadline: float, attempt_budget: int | None = None) -> tuple[GeometryState, dict]:
    """Move whole aircraft demands between legal overlapping beams.

    The pass order is stable in aircraft IDs.  The squared-load objective is
    only a search guide; published subject metrics are recomputed afterward.
    """
    locked = np.zeros(len(state.scene.demands), dtype=bool)
    for region in regions:
        locked[np.asarray(region["aircraft_indices"], dtype=np.int64)] = True
    associations = state.associations.copy()
    loads = state.loads.copy()
    demands = np.asarray(state.scene.demands, dtype=np.float64)
    attempted = moved = scans = 0
    cap = options["max_assignment_attempts"] if attempt_budget is None else min(
        options["max_assignment_attempts"], attempt_budget)
    if cap <= 0:
        return state, {"scans": 0, "attempts": 0, "transfers": 0}
    for _ in range(options["max_assignment_scans"]):
        changed = False
        scans += 1
        for user in range(len(demands)):
            if time.monotonic() >= deadline or attempted >= cap:
                break
            if locked[user] or associations[user] < 0 or demands[user] <= 0:
                continue
            source = int(associations[user])
            value = float(demands[user])
            if loads[source] <= value + 1e-8:
                continue  # Never create an empty positive-demand beam.
            choices = np.flatnonzero(state.eligible[user])
            best_destination, best_change = -1, 0.0
            for destination in choices:
                if destination == source or attempted >= cap:
                    continue
                attempted += 1
                # Exact delta in sum(load^2), with total demand unchanged.
                change = 2 * value * (loads[destination] - loads[source] + value)
                if change < best_change - 1e-7:
                    best_destination, best_change = int(destination), change
            if best_destination >= 0:
                associations[user] = best_destination
                loads[source] -= value
                loads[best_destination] += value
                moved += 1
                changed = True
        if not changed or attempted >= cap or time.monotonic() >= deadline:
            break
    result = GeometryState(state.scene, state.centers, associations)
    if not np.allclose(loads, result.loads, rtol=1e-12, atol=1e-5):
        raise AssertionError("assignment demand ledger drift")
    return result, {"scans": scans, "attempts": attempted, "transfers": moved}


def post_move_assignment(previous: GeometryState, proposal: GeometryState,
                         regions: list[dict], *, attempt_budget: int | None = None) -> tuple[GeometryState, dict]:
    """Frozen per-action rule shared by search and PPO: preserve, then one transfer.

    A2's bounded multi-scan assignment is common preprocessing.  This cheaper
    one-transfer rule runs after every center decision, including no-op.
    """
    associations = previous.associations.copy()
    rows = np.arange(len(associations))
    still_valid = ((associations >= 0) & proposal.geometrically_covered &
                   proposal.eligible[rows, np.maximum(associations, 0)])
    nearest = proposal.distance.argmin(1)
    associations = np.where(still_valid, associations,
                            np.where(proposal.geometrically_covered, nearest, -1))
    current = GeometryState(proposal.scene, proposal.centers, associations)
    locked = np.zeros(len(associations), dtype=bool)
    for region in regions:
        locked[np.asarray(region["aircraft_indices"], dtype=np.int64)] = True
    demands = np.asarray(proposal.scene.demands, dtype=np.float64)
    best = (0.0, -1, -1)
    attempts = 0
    cap = len(demands) * BEAM_COUNT if attempt_budget is None else max(0, int(attempt_budget))
    for user in range(len(demands)):
        if attempts >= cap:
            break
        source = int(current.associations[user])
        value = float(demands[user])
        if locked[user] or source < 0 or value <= 0 or current.loads[source] <= value + 1e-8:
            continue
        for destination in np.flatnonzero(current.eligible[user]):
            if attempts >= cap:
                break
            if destination == source:
                continue
            attempts += 1
            delta = 2 * value * (current.loads[destination] - current.loads[source] + value)
            if delta < best[0] - 1e-7:
                best = (delta, user, int(destination))
    if best[1] >= 0:
        associations[best[1]] = best[2]
        current = GeometryState(proposal.scene, proposal.centers, associations)
    return current, {"attempts": attempts, "transfers": int(best[1] >= 0)}


def _repair_coverage(state: GeometryState, regions: list[dict], options: dict,
                     *, deadline: float) -> tuple[GeometryState, dict]:
    """Bounded, monotone repair with single- and atomic two-beam proposals."""
    protected = state.geometrically_covered.copy()
    initial = np.asarray(state.scene.centers, dtype=np.float64).copy()
    metrics = balance_metrics(state, regions, options, initial)
    attempts = decisions = accepted = pair_attempts = 0
    reason = "no_improving_repair"
    for _ in range(options["max_repair_decisions"]):
        if time.monotonic() >= deadline:
            reason = "time_limit"
            break
        if attempts >= options["max_repair_evaluations"]:
            reason = "repair_evaluation_limit"
            break
        uncovered = np.flatnonzero(~state.geometrically_covered)
        if len(uncovered) == 0:
            reason = "full_coverage"
            break
        decisions += 1
        # Highest-demand missed users first; a deterministic stable tie-break.
        order = uncovered[np.argsort(-np.asarray(state.scene.demands)[uncovered], kind="stable")]
        best = None
        best_metric = metrics
        best_quality = _quality(metrics, stage="repair")
        for user in order[: min(len(order), 12)]:
            if time.monotonic() >= deadline or attempts >= options["max_repair_evaluations"]:
                break
            target = np.asarray(state.scene.positions[user], dtype=np.float64)
            distances = ((state.beam_rays - state.user_rays[user]) ** 2).sum(-1)
            for beam in np.argsort(distances, kind="stable")[: min(BEAM_COUNT, 16)]:
                if time.monotonic() >= deadline or attempts >= options["max_repair_evaluations"]:
                    break
                center = _towards(state.centers[beam], target,
                                  options["max_movement_deg"])
                if not _legal_center(center, initial[beam], options):
                    continue
                changes = {int(beam): center}
                attempts += 1
                trial = state.proposed(changes)
                candidates = [trial]
                # If the moved beam loses an already-covered user, try one
                # compensating beam before testing the atomic final layout.
                lost = np.flatnonzero(protected & ~trial.geometrically_covered)
                if len(lost):
                    for lost_user in lost[:2]:
                        second_dist = ((state.beam_rays - state.user_rays[lost_user]) ** 2).sum(-1)
                        for other in np.argsort(second_dist, kind="stable")[:4]:
                            if other == beam or attempts >= options["max_repair_evaluations"]:
                                continue
                            second = _towards(state.centers[other],
                                              np.asarray(state.scene.positions[lost_user]),
                                              options["max_movement_deg"])
                            if not _legal_center(second, initial[other], options):
                                continue
                            attempts += 1
                            pair_attempts += 1
                            candidates.append(state.proposed({int(beam): center,
                                                              int(other): second}))
                for candidate in candidates:
                    if (not np.all(candidate.geometrically_covered[protected]) or
                            np.any(candidate.loads <= 0)):
                        continue
                    candidate_metric = balance_metrics(candidate, regions, options, initial)
                    quality = _quality(candidate_metric, stage="repair")
                    if quality > best_quality:
                        best, best_quality, best_metric = candidate, quality, candidate_metric
        if best is None:
            break
        state, metrics = best, best_metric
        accepted += 1
    else:
        reason = "repair_decision_limit"
    return state, {"decisions": decisions, "candidate_evaluations": attempts,
                   "pair_evaluations": pair_attempts, "accepted": accepted,
                   "stop_reason": reason,
                   "protected_c1_users": int(protected.sum())}


def _business_positions(state: GeometryState, beam: int, options: dict) -> list[tuple[np.ndarray, bool]]:
    """Eight stable slots; absent evidence is masked, never shifts action IDs."""
    candidates: list[tuple[np.ndarray, bool]] = []
    demand = np.asarray(state.scene.demands, dtype=np.float64)
    positions = np.asarray(state.scene.positions, dtype=np.float64)
    uncovered = np.flatnonzero(~state.geometrically_covered)
    near = ((state.user_rays[uncovered] - state.beam_rays[beam]) ** 2).sum(-1)
    nearest_uncovered = uncovered[np.argsort(near, kind="stable")[:2]]
    for slot in range(2):
        candidates.append((positions[nearest_uncovered[slot]], True)
                          if slot < len(nearest_uncovered) else
                          (state.centers[beam].copy(), False))
    owned = np.flatnonzero(state.associations == beam)
    candidates.append((np.average(positions[owned], axis=0, weights=demand[owned]), True)
                      if len(owned) else (state.centers[beam].copy(), False))
    heavy = [item for item in np.argsort(-state.loads, kind="stable") if item != beam][:2]
    for neighbor in heavy:
        users = np.flatnonzero(state.associations == neighbor)
        top = users[np.argsort(-demand[users], kind="stable")[:2]]
        for slot in range(2):
            candidates.append((positions[top[slot]], True) if slot < len(top)
                              else (state.centers[beam].copy(), False))
        if neighbor == heavy[0]:
            candidates.append((np.average(positions[users], axis=0, weights=demand[users]), True)
                              if len(users) else (state.centers[beam].copy(), False))
    if len(candidates) != 8:
        raise AssertionError("business candidate slots must remain fixed")
    return [(_towards(state.centers[beam], np.asarray(target),
                      4 * options["step_angle_deg"]), valid)
            for target, valid in candidates]


def action_proposals(state: GeometryState, beam: int, menu: str,
                     options: dict) -> list[dict]:
    """Stable menu order with fixed action IDs, padded to the public dimension."""
    if menu not in ACTION_COUNTS:
        raise ValueError("unknown action menu")
    center = state.centers[beam]
    actions = [{"action_id": 0, "changes": {}, "beam_cost": 1, "kind": "keep"}]
    if menu in ("X0", "X4"):
        scales, directions = (1,), CARDINAL
    elif menu == "X1":
        scales, directions = (1,), CARDINAL + DIAGONAL
    else:
        scales, directions = (1, 2, 4), CARDINAL
    for scale in scales:
        for direction in directions:
            new_center = _directional(center, direction,
                                      scale * options["step_angle_deg"])
            actions.append({"action_id": len(actions), "changes": {beam: new_center},
                            "beam_cost": 1, "kind": "single"})
    if menu == "X3":
        for target, valid in _business_positions(state, beam, options):
            if valid:
                valid = not any(
                    np.allclose(target, prior["changes"].get(beam, state.centers[beam]),
                                atol=1e-10, rtol=0)
                    for prior in actions)
            actions.append({"action_id": len(actions), "changes": {beam: target},
                            "beam_cost": 1, "kind": "business", "exists": valid})
    if menu == "X4":
        distance = ((state.beam_rays - state.beam_rays[beam]) ** 2).sum(-1)
        distance[beam] = np.inf
        neighbors = np.argsort(distance, kind="stable")[:4]
        for neighbor in neighbors:
            for direction in CARDINAL:
                first = _directional(center, direction, options["step_angle_deg"])
                same = _directional(state.centers[neighbor], direction,
                                    options["step_angle_deg"])
                actions.append({"action_id": len(actions),
                                "changes": {beam: first, int(neighbor): same},
                                "beam_cost": 2, "kind": "pair_same"})
            for direction in CARDINAL:
                first = _directional(center, direction, options["step_angle_deg"])
                opposite = _directional(state.centers[neighbor],
                                        (-direction[0], -direction[1]),
                                        options["step_angle_deg"])
                actions.append({"action_id": len(actions),
                                "changes": {beam: first, int(neighbor): opposite},
                                "beam_cost": 2, "kind": "pair_opposite"})
    if len(actions) > ACTION_COUNTS[menu]:
        raise AssertionError("action menu overflow")
    return actions


def _center_candidate_ok(state: GeometryState, changes: dict[int, np.ndarray],
                         initial: np.ndarray, options: dict) -> bool:
    return all(_legal_center(np.asarray(center), initial[beam], options)
               for beam, center in changes.items())


def _adjust_centers(state: GeometryState, regions: list[dict], options: dict,
                    menu: str, *, deadline: float, prior_assignment_attempts: int = 0,
                    search: bool = True) -> tuple[GeometryState, dict]:
    """Bounded menu search, with exact acceptance after shared reassociation."""
    initial = np.asarray(state.scene.centers, dtype=np.float64).copy()
    protected = state.geometrically_covered.copy()
    best_state = state.copy()
    best_metrics = balance_metrics(state, regions, options, initial)
    best_quality = _quality(best_metrics, stage="balance")
    requested = accepted = candidate_evaluations = beam_processes = 0
    assignment_attempts = int(prior_assignment_attempts)
    rejected = 0
    trace = []
    stop_reason = "round_limit"
    decision_limit = min(options["max_beam_decisions"], BEAM_COUNT * options["max_rounds"])
    for step in range(decision_limit):
        if time.monotonic() >= deadline:
            stop_reason = "time_limit"
            break
        if beam_processes >= options["max_beam_decisions"]:
            stop_reason = "beam_process_limit"
            break
        if candidate_evaluations >= options["max_candidate_evaluations"]:
            stop_reason = "candidate_evaluation_limit"
            break
        if assignment_attempts >= options["max_assignment_attempts"]:
            stop_reason = "assignment_attempt_limit"
            break
        if best_metrics["balanced_target_met"]:
            stop_reason = ("strict_target_reached" if best_metrics["coverage_100"]
                           else "near_target_reached")
            break
        beam = step % BEAM_COUNT
        actions = action_proposals(state, beam, menu, options)
        requested += 1
        ranked = []
        # Trial ranking uses exact geometry and nearest association.  The
        # frozen load-aware assignment pass is applied to the selected layout;
        # final acceptance checks the post-pass physical state again.
        for action in actions[1:]:
            if not action.get("exists", True):
                continue
            if (candidate_evaluations >= options["max_candidate_evaluations"] or
                    assignment_attempts >= options["max_assignment_attempts"] or
                    time.monotonic() >= deadline):
                break
            if beam_processes + action["beam_cost"] > options["max_beam_decisions"]:
                continue
            if not _center_candidate_ok(state, action["changes"], initial, options):
                continue
            try:
                trial = state.proposed(action["changes"])
            except ValueError:
                continue
            candidate_evaluations += 1
            if not np.all(trial.geometrically_covered[protected]):
                continue
            trial, a_info = post_move_assignment(
                state, trial, regions,
                attempt_budget=options["max_assignment_attempts"] - assignment_attempts)
            assignment_attempts += a_info["attempts"]
            if np.any(trial.loads <= 0):
                continue
            trial_metric = balance_metrics(trial, regions, options, initial)
            ranked.append((_quality(trial_metric, stage="balance"), action, trial))
        ranked.sort(key=lambda item: (item[0], -item[1]["action_id"]), reverse=True)
        accepted_action = None
        if search:
            choices = ranked[:3]
        else:
            choices = ranked[:1]
        for _, action, trial in choices:
            trial_metric = balance_metrics(trial, regions, options, initial)
            trial_quality = _quality(trial_metric, stage="balance")
            current_quality = _quality(balance_metrics(state, regions, options, initial),
                                       stage="balance")
            if trial_quality > current_quality and np.all(trial.geometrically_covered[protected]):
                state = trial
                accepted_action = action
                accepted += 1
                if trial_quality > best_quality:
                    best_state, best_metrics, best_quality = trial.copy(), trial_metric, trial_quality
                break
        if accepted_action is None:
            rejected += 1
            beam_processes += 1
            action_id = 0
            kind = "keep_or_rejected"
        else:
            beam_processes += accepted_action["beam_cost"]
            action_id = accepted_action["action_id"]
            kind = accepted_action["kind"]
        trace.append({"step": step + 1, "beam": int(beam), "action_id": action_id,
                      "kind": kind, "beam_processes": beam_processes,
                      "candidate_evaluations": candidate_evaluations,
                      "coverage_users": best_metrics["coverage_users"],
                      "subject_cv": best_metrics["subject_cv"]})
    else:
        stop_reason = "round_limit" if decision_limit == BEAM_COUNT * options["max_rounds"] else "beam_process_limit"
    return best_state, {"requested": requested, "accepted": accepted,
                        "rejected_or_kept": rejected,
                        "beam_processes": beam_processes,
                        "candidate_evaluations": candidate_evaluations,
                        "assignment_attempts": assignment_attempts - prior_assignment_attempts,
                        "assignment_attempts_cumulative": assignment_attempts,
                        "stop_reason": stop_reason,
                        "raw_terminal_layout_sha256": _layout_sha(state.centers),
                        "best_layout_sha256": _layout_sha(best_state.centers),
                        "trace": trace}


def _row(scene: CoverageScene, state: GeometryState, regions: list[dict], options: dict,
         route: str, stages: dict, elapsed: float, initial_metrics: dict,
         common_start_sha: str | None,
         movement_reference: np.ndarray) -> dict:
    metrics = balance_metrics(state, regions, options, movement_reference)
    uncovered = np.flatnonzero(~state.geometrically_covered)
    return {"status": "completed" if metrics["coverage_99_9"] else "coverage_not_met",
            "scene_id": scene.scene_id, "source_hash": scene.source_hash,
            "input": {"aircraft_ids": scene.aircraft_ids,
                      "positions": np.asarray(scene.positions).tolist(),
                      "demands_bps": np.asarray(scene.demands).tolist(),
                      "initial_centers": np.asarray(scene.centers).tolist(),
                      "movement_reference_centers": np.asarray(movement_reference).tolist(),
                      "metadata": scene.metadata},
            "metrics": metrics, "final_positions": state.centers.tolist(),
            "final_associations": state.associations.tolist(),
            "beam_loads_bps": state.loads.tolist(),
            "uncovered_aircraft_indices": uncovered.tolist(),
            "uncovered_demand_bps_by_aircraft": np.asarray(scene.demands)[uncovered].tolist(),
            "elapsed_seconds": elapsed,
            "balance_diagnostics": {
                "route": route, "action_menu": options["action_menu"],
                "post_move_transfers": options["post_move_transfers"],
                "movement_reference": "cluster_initial_center",
                "common_start_sha256": common_start_sha,
                "initial_metrics": initial_metrics, "final_metrics": metrics,
                "certified_regions": regions,
                "qualified_edge_beam_ids": metrics["edge_beams"],
                "edge_limit_met": metrics["edge_limit_met"],
                "stage_counts": {key: value.get("counts", {}) for key, value in stages.items()},
                "stage_seconds": {key: value.get("seconds", 0.0) for key, value in stages.items()},
                "stages": stages,
                "final_layout_sha256": _layout_sha(state.centers),
                "final_association_sha256": _association_sha(state.associations),
            }}


def _prepare_stages(scene: CoverageScene, config: dict, options: dict,
                    *, device: str, stop_after: str = "A2") -> tuple[
                        GeometryState, GeometryState | None, GeometryState | None,
                        list[dict], dict, dict]:
    started = time.monotonic()
    legacy = legacy_default_config()
    legacy["coverage"].update(initialization="cluster", method="local_search",
                              sweeps=2, coordinate_step_deg=0.1,
                              max_movement_deg=0.25)
    legacy["training"]["device"] = device
    baseline = run_baseline(scene, legacy, method="local_search")
    if baseline["status"] != "completed":
        raise ValueError("C1 initialization failed; retain as a failed scene")
    c1 = GeometryState(scene, np.asarray(baseline["final_positions"], dtype=np.float64),
                       np.asarray(baseline["final_associations"], dtype=np.int64))
    stages = {"c1": {"seconds": time.monotonic() - started,
                     "counts": {"candidate_evaluations": baseline["candidate_checks"],
                                "action_steps": baseline["action_steps"]},
                     "legacy_score": baseline["metrics"]["score"],
                     "initial_layout_sha256": _layout_sha(scene.centers),
                     "legacy_config": {"sweeps": 2, "coordinate_step_deg": 0.1,
                                       "max_movement_deg": 0.25,
                                       "score_weights": [100.0, 1.0, 1.0, 0.1]}}}
    started = time.monotonic()
    regions = certify_edge_regions(scene, options)
    stages["edge_preprocess"] = {"seconds": time.monotonic() - started,
                                 "counts": {"certified_regions": len(regions)}}
    c1_metrics = balance_metrics(c1, regions, options, scene.centers)
    if stop_after == "A0":
        return c1, None, None, regions, stages, c1_metrics
    deadline = time.monotonic() + options["max_refinement_seconds"]
    started = time.monotonic()
    a1, repair_info = _repair_coverage(c1, regions, options, deadline=deadline)
    stages["repair"] = {"seconds": time.monotonic() - started, "counts": repair_info}
    if stop_after == "A1":
        return c1, a1, None, regions, stages, c1_metrics
    started = time.monotonic()
    remaining_attempts = max(0, options["max_assignment_attempts"])
    a2, assignment_info = _assignment_optimize(a1, regions, options,
                                               deadline=deadline,
                                               attempt_budget=remaining_attempts)
    stages["assignment"] = {"seconds": time.monotonic() - started,
                            "counts": assignment_info}
    stages["initial_metrics"] = c1_metrics
    return c1, a1, a2, regions, stages, c1_metrics


def prepare_common_a2(scene: CoverageScene, config: dict, *, device: str = "cpu") -> BalanceStart | dict:
    options = balance_options(config)
    began = time.monotonic()
    try:
        c1, a1, a2, regions, stages, c1_metrics = _prepare_stages(scene, config, options,
                                                                   device=device)
    except ValueError as error:
        if "C1 initialization failed" not in str(error):
            raise
        failure = {"status": "initialization_failed", "scene_id": scene.scene_id,
                   "scene_sha256": scene.source_hash,
                   "aircraft_ids": list(scene.aircraft_ids),
                   "failure_reason": str(error),
                   "stages": {"c1": {"seconds": time.monotonic() - began,
                                     "counts": {"initialization_failed": 1}}}}
        failure["start_sha256"] = _canonical_sha(failure)
        return failure
    metrics = balance_metrics(a2, regions, options, scene.centers)
    return BalanceStart(scene.scene_id, scene.source_hash, list(scene.aircraft_ids),
                        a2.centers, a2.associations, a2.loads, regions, metrics,
                        stages, "completed" if metrics["coverage_99_9"] else "coverage_not_met",
                        _state_record(c1, c1_metrics),
                        _state_record(a1, balance_metrics(a1, regions, options, scene.centers)))


def run_balance_route(scene: CoverageScene, config: dict, route: str,
                      *, common: BalanceStart | dict | None = None,
                      device: str = "cpu") -> dict:
    """Return one archived-result-compatible row for A0-A3 or X0-X4."""
    options = balance_options(config)
    start_time = time.monotonic()
    if route not in ("A0", "A1", "A2", "A3", *ACTION_COUNTS):
        raise ValueError("route must be A0-A3 or X0-X4")
    if common is None:
        raise ValueError("all balance routes require a frozen prepare task artifact")
    if isinstance(common, dict) and common.get("status") == "initialization_failed":
        check = dict(common)
        expected_sha = check.pop("start_sha256")
        if (_canonical_sha(check) != expected_sha or
                check["scene_id"] != scene.scene_id or
                check["scene_sha256"] != scene.source_hash or
                check["aircraft_ids"] != scene.aircraft_ids):
            raise ValueError("failed balance start identity/SHA mismatch")
        return {"status": "initialization_failed", "scene_id": scene.scene_id,
                "source_hash": scene.source_hash,
                "input": {"aircraft_ids": scene.aircraft_ids,
                          "positions": np.asarray(scene.positions).tolist(),
                          "demands_bps": np.asarray(scene.demands).tolist(),
                          "initial_centers": None, "metadata": scene.metadata},
                "metrics": None, "final_positions": None,
                "final_associations": None, "beam_loads_bps": None,
                "uncovered_aircraft_indices": list(range(len(scene.aircraft_ids))),
                "uncovered_demand_bps_by_aircraft": np.asarray(scene.demands).tolist(),
                "elapsed_seconds": float(common.get("stages", {}).get("c1", {}).get("seconds", 0.0)),
                "balance_diagnostics": {"route": route,
                                        "common_start_sha256": expected_sha,
                                        "failure_reason": common["failure_reason"],
                                        "stage_seconds": {"c1": float(common.get("stages", {}).get(
                                            "c1", {}).get("seconds", 0.0))},
                                        "total_pipeline_seconds": float(common.get("stages", {}).get(
                                            "c1", {}).get("seconds", 0.0)),
                                        "common_preparation_reused": True}}
    prepared = (BalanceStart.from_record(scene, common) if isinstance(common, dict)
                else BalanceStart.from_record(scene, common.to_record()))
    stage_record = (prepared.c1_state if route == "A0" else
                    prepared.a1_state if route == "A1" else prepared.to_record())
    state = GeometryState(scene, np.asarray(stage_record["centers"], dtype=np.float64),
                          np.asarray(stage_record["associations"], dtype=np.int64))
    regions, stages = prepared.regions, deepcopy(prepared.stages)
    initial_metrics = prepared.c1_state["metrics"]
    common_sha = prepared.start_sha256
    if route in ("A3", *ACTION_COUNTS):
        menu = options["action_menu"] if route == "A3" else route
        options["action_menu"] = menu
        started = time.monotonic()
        # Common preparation is charged separately; each route owns its own
        # adjustment clock and candidate budget, so methods never contaminate
        # each other's allowance.
        deadline = started + max(0.0, options["max_refinement_seconds"] -
                                  stages.get("repair", {}).get("seconds", 0.0) -
                                  stages.get("assignment", {}).get("seconds", 0.0))
        state, adjust_info = _adjust_centers(
            state, regions, options, menu, deadline=deadline,
            prior_assignment_attempts=int(stages.get("assignment", {}).get(
                "counts", {}).get("attempts", 0)))
        stages["adjustment"] = {"seconds": time.monotonic() - started,
                                "counts": adjust_info}
    stages["final_verification"] = {"seconds": 0.0, "counts": {"verified": 1}}
    checked = time.monotonic()
    # Independent construction rechecks all U x 100 geometry and the demand ledger.
    state = GeometryState(scene, state.centers, state.associations)
    stages["final_verification"]["seconds"] = time.monotonic() - checked
    included = ["c1", "edge_preprocess"]
    if route != "A0":
        included.append("repair")
    if route not in ("A0", "A1"):
        included.append("assignment")
    if route in ("A3", *ACTION_COUNTS):
        included.append("adjustment")
    included.append("final_verification")
    pipeline_seconds = sum(float(stages[name].get("seconds", 0.0)) for name in included)
    movement_reference = scene.centers
    row = _row(scene, state, regions, options, route, stages,
               pipeline_seconds, initial_metrics, common_sha,
               movement_reference)
    row["balance_diagnostics"]["stages_in_route"] = included
    row["balance_diagnostics"]["stage_seconds"] = {
        name: float(stages[name].get("seconds", 0.0)) for name in included}
    row["balance_diagnostics"]["stage_counts"] = {
        name: stages[name].get("counts", {}) for name in included}
    if route == "A0":
        row["metrics"]["legacy_score"] = prepared.stages["c1"]["legacy_score"]
    row["balance_diagnostics"]["evaluation_call_seconds"] = time.monotonic() - start_time
    row["balance_diagnostics"]["total_pipeline_seconds"] = pipeline_seconds
    row["balance_diagnostics"]["common_preparation_reused"] = True
    return row
