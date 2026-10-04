"""Atomic, fixed-width region reconstruction for first-stage coverage.

This module is deliberately independent of PPO.  Every selector sees the same
cheap candidate list and pays for every selected geometric proposal.  Geometry
uses the registered Europe GEO profile and checks each aircraft against the
actual satellite ray after fitting a legal ground-point centre.
"""

from __future__ import annotations

from copy import copy as shallow_copy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from satellite_rl.coverage.geometry import (
    ray_angle_numpy, satellite_rays_numpy, squared_ray_limit, visible_numpy,
)
from satellite_rl.coverage.scenes import BEAM_COUNT, CoverageScene, _initial_centers
from satellite_rl.envs.regions import get_profile


RADIUS_DEG = 0.25  # 0.5 degree full width; this is a footprint, not a movement cap.
RAY_LIMIT = squared_ray_limit(RADIUS_DEG) + 1e-14
OPERATION_ORDER = ("stop", "merge_split", "repartition2", "repartition3", "assign")


@dataclass(frozen=True)
class ReconstructionConfig:
    max_macro_steps: int = 32
    max_geometry_calls_per_action: int = 256
    max_geometry_calls_total: int = 8192
    max_initialization_geometry_calls: int = 8192
    split_templates: tuple[int, ...] = (0, 1, 2, 3)
    neighbor_count: int = 3
    split_target_count: int = 12
    edge_demand_ratio: float = 0.5
    max_edge_beams: int = 10
    subject_cv_target: float = 0.20
    range_tolerance: float = 0.20
    min_subject_beams_in_range: int = 90

    def __post_init__(self) -> None:
        for name in ("max_macro_steps", "max_geometry_calls_per_action",
                     "max_geometry_calls_total", "max_initialization_geometry_calls",
                     "neighbor_count", "split_target_count"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.split_templates or any(t not in range(4) for t in self.split_templates):
            raise ValueError("split_templates must be a nonempty subset of 0..3")


@dataclass(frozen=True, order=True)
class ReconstructionAction:
    operation: str
    beams: tuple[int, ...] = ()
    template: int = 0

    def __post_init__(self) -> None:
        if self.operation not in OPERATION_ORDER:
            raise ValueError("unknown reconstruction operation")
        expected = {"stop": 0, "merge_split": 3, "repartition2": 2,
                    "repartition3": 3, "assign": (2, 3)}[self.operation]
        if len(self.beams) not in ((expected,) if isinstance(expected, int) else expected):
            raise ValueError("wrong number of beam IDs")
        if len(set(self.beams)) != len(self.beams) or any(b < 0 or b >= BEAM_COUNT for b in self.beams):
            raise ValueError("beam IDs must be distinct and in [0,100)")
        if self.template not in range(4):
            raise ValueError("unknown template")
        if self.operation == "stop" and self.template != 0:
            raise ValueError("STOP has no template")


def _geometry(scene: CoverageScene, centers: np.ndarray) -> np.ndarray:
    user_rays = satellite_rays_numpy(scene.positions)
    beam_rays = satellite_rays_numpy(centers)
    return ((user_rays[:, None, :] - beam_rays[None, :, :]) ** 2).sum(axis=-1)


def _edge_regions(scene: CoverageScene, ratio: float) -> list[dict]:
    """Certify isolated light components without any old-centre movement bound."""
    rays = satellite_rays_numpy(scene.positions)
    positions = np.asarray(scene.positions, dtype=np.float64)
    demands = np.asarray(scene.demands, dtype=np.float64)
    parent = np.arange(len(rays))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    # Two users farther apart than the full width cannot share any beam.
    full_limit = squared_ray_limit(2 * RADIUS_DEG) + 1e-14
    for lo in range(0, len(rays), 128):
        distance = ((rays[lo:lo + 128, None] - rays[None]) ** 2).sum(-1)
        for local, row in enumerate(distance):
            i = lo + local
            for j in np.flatnonzero(row[i + 1:] <= full_limit) + i + 1:
                a, b = root(i), root(int(j))
                if a != b:
                    parent[b] = a
    components: dict[int, list[int]] = {}
    for i in range(len(rays)):
        components.setdefault(root(i), []).append(i)
    threshold = float(demands.sum()) / BEAM_COUNT * ratio
    result = []
    for indices in components.values():
        subtotal = float(demands[indices].sum())
        if subtotal <= 0 or subtotal >= threshold:
            continue
        witness = next((i for i in indices if np.all(
            ((rays[indices] - rays[i]) ** 2).sum(-1) <= RAY_LIMIT)), None)
        if witness is None or not bool(visible_numpy(positions[witness:witness + 1])[0]):
            continue
        result.append({"aircraft_indices": list(indices), "demand_bps": subtotal,
                       "witness_center": positions[witness].tolist(),
                       "proof": "isolated full-width ray component; one legal GEO centre covers it"})
    return sorted(result, key=lambda region: region["aircraft_indices"][0])


def _dedicated_edges(regions: list[dict], assignment: np.ndarray) -> frozenset[int]:
    owners = []
    for region in regions:
        indices = np.asarray(region["aircraft_indices"], dtype=np.int64)
        beams = np.unique(assignment[indices])
        if len(beams) == 1 and beams[0] >= 0 and np.count_nonzero(assignment == beams[0]) == len(indices):
            owners.append(int(beams[0]))
    return frozenset(owners)


def _metrics(state: "ReconstructionState", config: ReconstructionConfig) -> dict:
    demands = np.asarray(state.scene.demands, dtype=np.float64)
    served = state.assignment >= 0
    user_fraction = float(np.mean(served))
    total = float(demands.sum())
    demand_fraction = float(demands[served].sum() / total) if total > 0 else 1.0
    strict = bool(np.all(served) and demand_fraction >= 1.0 - 1e-12)
    subject = np.array([b not in state.edge_beam_ids for b in range(BEAM_COUNT)])
    values = state.loads[subject]
    mean = float(values.mean()) if len(values) else 0.0
    if mean > 0:
        normalized = values / mean
        cv2 = float(np.mean((normalized - 1.0) ** 2))
        excess = float(np.mean(np.maximum(np.abs(normalized - 1.0) - config.range_tolerance, 0.0) ** 2))
        j = cv2 + excess
        in_band = int(np.count_nonzero(np.abs(normalized - 1.0) <= config.range_tolerance + 1e-12))
        subject_cv = float(np.sqrt(cv2))
    else:
        j, in_band, subject_cv = float("inf"), 0, float("inf")
    bounded = j / (1.0 + j) if np.isfinite(j) else 1.0
    quality = 2.0 * float(strict) - bounded
    all_mean = float(state.loads.mean())
    load_cv = float(state.loads.std() / all_mean) if all_mean > 0 else float("inf")
    target = (strict and len(state.edge_regions) <= config.max_edge_beams and
              subject_cv <= config.subject_cv_target and
              in_band >= config.min_subject_beams_in_range and
              bool(np.all(state.loads > 0)))
    return {"coverage_users": user_fraction, "coverage_demand": demand_fraction,
            "coverage_100": strict, "uncovered_users": int(np.count_nonzero(~served)),
            "uncovered_demand_bps": float(demands[~served].sum()),
            "j": j, "quality": quality, "subject_cv": subject_cv,
            "load_cv": load_cv, "subject_in_band_beams": in_band,
            "subject_beams": int(subject.sum()), "edge_beam_count": len(state.edge_beam_ids),
            "certified_edge_region_count": len(state.edge_regions),
            "empty_beams": int(np.count_nonzero(state.loads <= 0)),
            "target_met": bool(target), "macro_steps": state.steps,
            "geometry_calls": state.geometry_calls, "attempts": state.attempts}


@dataclass
class ReconstructionState:
    scene: CoverageScene
    centers: np.ndarray
    assignment: np.ndarray
    edge_regions: list[dict]
    edge_beam_ids: frozenset[int]
    steps: int = 0
    geometry_calls: int = 0
    attempts: int = 0
    done: bool = False
    initialization_valid: bool = True
    initialization_source: str = "provided_scene_centers"
    elapsed_fit_s: float = 0.0
    elapsed_validate_s: float = 0.0
    elapsed_repair_s: float = 0.0
    elapsed_initialization_s: float = 0.0
    initialization_candidates: list[dict] = field(default_factory=list)
    fit_candidate_checks: int = 0
    assignment_candidate_checks: int = 0
    ray_pair_checks: int = 0
    best_quality: float = float("-inf")
    best_record: dict | None = None
    loads: np.ndarray = field(init=False, repr=False)
    _candidate_cache: tuple[ReconstructionConfig, list[ReconstructionAction]] | None = field(
        default=None, init=False, repr=False)
    candidate_screen_ray_pairs: int = field(default=0, init=False)
    candidate_builds: int = field(default=0, init=False)
    elapsed_candidate_screen_s: float = field(default=0.0, init=False)
    initialization_repair_geometry_calls: int = field(default=0, init=False)
    initialization_repair_ray_pair_checks: int = field(default=0, init=False)
    initialization_repair_attempts: int = field(default=0, init=False)
    _distance_cache: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.centers = np.asarray(self.centers, dtype=np.float64).copy()
        self.assignment = np.asarray(self.assignment, dtype=np.int64).copy()
        self._validate_layout(_geometry(self.scene, self.centers))

    def _validate_layout(self, distance: np.ndarray) -> None:
        if (self.centers.shape != (BEAM_COUNT, 2) or
                self.assignment.shape != (len(self.scene.demands),) or
                not np.isfinite(self.centers).all() or
                not np.all(visible_numpy(self.centers))):
            raise ValueError("100 legal Europe GEO centres and one entry per aircraft required")
        if np.any((self.assignment < -1) | (self.assignment >= BEAM_COUNT)):
            raise ValueError("invalid aircraft association")
        if distance.shape != (len(self.scene.demands), BEAM_COUNT):
            raise ValueError("invalid aircraft-beam geometry matrix")
        served = self.assignment >= 0
        geometrically_covered = np.min(distance, axis=1) <= RAY_LIMIT
        if not np.array_equal(served, geometrically_covered):
            raise ValueError("geometrically covered aircraft must have exactly one association")
        if np.any(distance[np.flatnonzero(served), self.assignment[served]] > RAY_LIMIT):
            raise ValueError("association outside actual satellite footprint")
        self._distance_cache = distance
        self.loads = np.bincount(self.assignment[served],
                                 weights=np.asarray(self.scene.demands)[served],
                                 minlength=BEAM_COUNT).astype(np.float64)

    @classmethod
    def from_scene(cls, scene: CoverageScene, centers: np.ndarray | None = None,
                   assignment: np.ndarray | None = None,
                   config: ReconstructionConfig | None = None) -> "ReconstructionState":
        cfg = config or ReconstructionConfig()
        chosen = np.asarray(scene.centers if centers is None else centers, dtype=np.float64)
        if assignment is None:
            distances = _geometry(scene, chosen)
            nearest = distances.argmin(axis=1)
            assignment = np.where(distances[np.arange(len(nearest)), nearest] <= RAY_LIMIT,
                                  nearest, -1)
        regions = _edge_regions(scene, cfg.edge_demand_ratio)
        dedicated = _dedicated_edges(regions, np.asarray(assignment))
        for region in regions:
            indices = np.asarray(region["aircraft_indices"], dtype=np.int64)
            beams = np.unique(np.asarray(assignment)[indices])
            region["dedicated_beam_id"] = (int(beams[0]) if len(beams) == 1 and
                                           beams[0] in dedicated and
                                           np.count_nonzero(np.asarray(assignment) == beams[0]) == len(indices)
                                           else None)
        state = cls(scene, chosen, assignment, regions, dedicated)
        state.initialization_valid = bool(np.all(state.loads > 0) and
                                          _metrics(state, cfg)["coverage_users"] >= 0.999 and
                                          _metrics(state, cfg)["coverage_demand"] >= 0.999)
        state.best_quality = _metrics(state, cfg)["quality"]
        state.best_record = {"centers": state.centers.tolist(),
                             "assignment": state.assignment.tolist(),
                             "quality": state.best_quality}
        return state

    def copy(self) -> "ReconstructionState":
        # The source state was already verified. Copying must not perform an
        # uncharged U x 100 ray check before an attempted proposal.
        result = shallow_copy(self)
        result.centers = self.centers.copy()
        result.assignment = self.assignment.copy()
        result.loads = self.loads.copy()
        result._distance_cache = self._distance_cache
        result.initialization_candidates = list(self.initialization_candidates)
        return result

    def to_record(self) -> dict:
        return {"scene_id": self.scene.scene_id, "source_hash": self.scene.source_hash,
                "aircraft_ids": list(self.scene.aircraft_ids),
                "centers": self.centers.tolist(), "assignment": self.assignment.tolist(),
                "edge_regions": self.edge_regions, "edge_beam_ids": sorted(self.edge_beam_ids),
                "steps": self.steps, "geometry_calls": self.geometry_calls,
                "attempts": self.attempts, "done": self.done,
                "initialization_valid": self.initialization_valid,
                "initialization_source": self.initialization_source,
                "elapsed_fit_s": self.elapsed_fit_s,
                "elapsed_validate_s": self.elapsed_validate_s,
                "elapsed_repair_s": self.elapsed_repair_s,
                "elapsed_initialization_s": self.elapsed_initialization_s,
                "initialization_candidates": self.initialization_candidates,
                "fit_candidate_checks": self.fit_candidate_checks,
                "assignment_candidate_checks": self.assignment_candidate_checks,
                "ray_pair_checks": self.ray_pair_checks,
                "candidate_screen_ray_pairs": self.candidate_screen_ray_pairs,
                "candidate_builds": self.candidate_builds,
                "elapsed_candidate_screen_s": self.elapsed_candidate_screen_s,
                "initialization_repair_geometry_calls": self.initialization_repair_geometry_calls,
                "initialization_repair_ray_pair_checks": self.initialization_repair_ray_pair_checks,
                "initialization_repair_attempts": self.initialization_repair_attempts,
                "best_quality": self.best_quality, "best_record": self.best_record}

    @classmethod
    def from_record(cls, scene: CoverageScene, record: dict,
                    config: ReconstructionConfig | None = None) -> "ReconstructionState":
        if (record["scene_id"] != scene.scene_id or
                record["source_hash"] != scene.source_hash or
                record["aircraft_ids"] != scene.aircraft_ids):
            raise ValueError("reconstruction state scene identity mismatch")
        cfg = config or ReconstructionConfig()
        certified = _edge_regions(scene, cfg.edge_demand_ratio)
        stored = record["edge_regions"]
        if [{k: v for k, v in row.items() if k != "dedicated_beam_id"} for row in stored] != certified:
            raise ValueError("edge certificate mismatch")
        certified = stored
        expected_ids = frozenset(row["dedicated_beam_id"] for row in certified
                                 if row.get("dedicated_beam_id") is not None)
        if expected_ids != frozenset(record["edge_beam_ids"]):
            raise ValueError("edge beam identity mismatch")
        state = cls(scene, record["centers"], record["assignment"], certified,
                    frozenset(record["edge_beam_ids"]), record["steps"],
                    record["geometry_calls"], record["attempts"], record["done"],
                    record["initialization_valid"], record["initialization_source"],
                    record["elapsed_fit_s"], record["elapsed_validate_s"],
                    record["elapsed_repair_s"], record.get("elapsed_initialization_s", 0.0),
                    record.get("initialization_candidates", []),
                    record.get("fit_candidate_checks", 0),
                    record.get("assignment_candidate_checks", 0),
                    record.get("ray_pair_checks", 0),
                    record["best_quality"], record["best_record"])
        if not _certificates_preserved(state) or (state.initialization_valid and
                                                   not np.all(state.loads > 0)):
            raise ValueError("restored state violates a fixed certificate or positive loads")
        state.candidate_screen_ray_pairs = record.get("candidate_screen_ray_pairs", 0)
        state.candidate_builds = record.get("candidate_builds", 0)
        state.elapsed_candidate_screen_s = record.get("elapsed_candidate_screen_s", 0.0)
        state.initialization_repair_geometry_calls = record.get(
            "initialization_repair_geometry_calls", 0)
        state.initialization_repair_ray_pair_checks = record.get(
            "initialization_repair_ray_pair_checks", 0)
        state.initialization_repair_attempts = record.get("initialization_repair_attempts", 0)
        return state


def state_metrics(state: ReconstructionState,
                  config: ReconstructionConfig | None = None) -> dict:
    """Public original-terminal metrics; best_record remains diagnostic only."""
    return _metrics(state, config or ReconstructionConfig())


def _certificates_preserved(state: ReconstructionState) -> bool:
    for region in state.edge_regions:
        beam = region.get("dedicated_beam_id")
        if beam is None:
            continue
        indices = np.asarray(region["aircraft_indices"], dtype=np.int64)
        if (not np.all(state.assignment[indices] == beam) or
                not np.array_equal(np.flatnonzero(state.assignment == beam), indices)):
            return False
    return True


def _ray_to_ground(ray: np.ndarray) -> np.ndarray | None:
    p = get_profile("europe_v1")
    lon = np.deg2rad(p.satellite_longitude_deg)
    satellite = p.orbit_radius_km * np.array([np.cos(lon), np.sin(lon), 0.0])
    dot = float(satellite @ ray)
    determinant = dot * dot - (p.orbit_radius_km ** 2 - p.earth_radius_km ** 2)
    if determinant < 0:
        return None
    t = -dot - np.sqrt(determinant)
    if t <= 0:
        return None
    point = satellite + t * ray
    return np.array([np.rad2deg(np.arctan2(point[2], np.hypot(point[0], point[1]))),
                     np.rad2deg(np.arctan2(point[1], point[0]))])


class _BudgetExceeded(Exception):
    pass


def _fit_group(state: ReconstructionState, members: np.ndarray,
               preferred: np.ndarray, counter: list[int], limit: int) -> np.ndarray | None:
    """Bounded deterministic candidate fit; every accepted result gets exact ray QA."""
    if len(members) == 0:
        return None
    positions = np.asarray(state.scene.positions, dtype=np.float64)[members]
    rays = satellite_rays_numpy(positions)
    demand = np.asarray(state.scene.demands, dtype=np.float64)[members]
    candidates = [np.asarray(preferred, dtype=np.float64)]
    candidates.extend(positions)
    # Mean and boundary pairs can find a centre between observed aircraft.
    for weights in (np.ones(len(members)), np.maximum(demand, 1e-30)):
        mean_ray = np.average(rays, axis=0, weights=weights)
        centre = _ray_to_ground(mean_ray / np.linalg.norm(mean_ray))
        if centre is not None:
            candidates.append(centre)
    farthest = int(np.argmax(((rays - rays[0]) ** 2).sum(-1)))
    other = int(np.argmax(((rays - rays[farthest]) ** 2).sum(-1)))
    midpoint = rays[farthest] + rays[other]
    centre = _ray_to_ground(midpoint / np.linalg.norm(midpoint))
    if centre is not None:
        candidates.append(centre)
    # Refinement around the best candidate uses true GEO ray radius, not planar circles.
    best: tuple[float, np.ndarray] | None = None
    for candidate in candidates:
        if counter[0] >= limit:
            raise _BudgetExceeded
        counter[0] += 1
        counter[3] += 1
        counter[1] += len(members)
        if not bool(visible_numpy(candidate[None])[0]):
            continue
        ray = satellite_rays_numpy(candidate[None])[0]
        distances = ((rays - ray) ** 2).sum(-1)
        worst = float(distances.max())
        if worst <= RAY_LIMIT:
            return candidate.copy()
        if best is None or worst < best[0]:
            best = (worst, candidate.copy())
    if best is None:
        return None
    centre = best[1]
    for scale in (0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625):
        improved = False
        for direction in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1),
                          (1, -1), (-1, 1), (-1, -1)):
            if counter[0] >= limit:
                raise _BudgetExceeded
            counter[0] += 1
            counter[3] += 1
            counter[1] += len(members)
            candidate = centre + scale * np.asarray(direction)
            if not bool(visible_numpy(candidate[None])[0]):
                continue
            ray = satellite_rays_numpy(candidate[None])[0]
            worst = float(((rays - ray) ** 2).sum(-1).max())
            if worst <= RAY_LIMIT:
                return candidate
            if worst + 1e-16 < best[0]:
                best, improved = (worst, candidate.copy()), True
        if improved:
            centre = best[1]
    return None


def _partition(state: ReconstructionState, members: np.ndarray, count: int,
               template: int) -> list[np.ndarray]:
    """Finite, deterministic user partitions; templates do not inspect new J."""
    positions = np.asarray(state.scene.positions)[members]
    rays = satellite_rays_numpy(positions)
    demand = np.asarray(state.scene.demands)[members]
    if template == 0:
        axis = rays[:, 0]
    elif template == 1:
        axis = rays[:, 1]
    elif template == 2:
        axis = rays[:, 2]
    else:
        axis = np.sum((rays - rays[np.argmax(demand)]) ** 2, axis=1)
    order = np.lexsort((members, axis))
    sorted_members = members[order]
    sorted_demand = demand[order]
    cumulative = np.cumsum(sorted_demand)
    if cumulative[-1] <= 0:
        cuts = np.round(np.arange(1, count) * len(members) / count).astype(int)
    else:
        cuts = np.searchsorted(cumulative, np.arange(1, count) * cumulative[-1] / count) + 1
    cuts = np.maximum.accumulate(np.maximum(cuts, np.arange(1, count)))
    cuts = np.minimum(cuts, len(members) - np.arange(count - 1, 0, -1))
    return [np.asarray(chunk, dtype=np.int64) for chunk in np.split(sorted_members, cuts)]


def initial_state(scene: CoverageScene,
                  config: ReconstructionConfig | None = None) -> ReconstructionState:
    """Shared coverage-first start, using the scene's old centres only as seeds.

    It does not preserve old beam identity or enforce a distance to those seeds.
    A failed start is returned explicitly for qualification, never hidden.
    """
    cfg = config or ReconstructionConfig()
    start = perf_counter()
    candidates = []
    for name, centres in (("scene_centers_seed", np.asarray(scene.centers)),
                          ("recomputed_cluster", None), ("recomputed_rule", None)):
        t0 = perf_counter()
        if centres is None:
            method = "cluster" if name == "recomputed_cluster" else "rule"
            centres, detail = _initial_centers(scene.positions, scene.demands, method)
        else:
            detail = {"initializer": "scene_centers_candidate"}
        if centres is None or not np.isfinite(centres).all():
            candidates.append({"source": name, "valid": False,
                               "seconds": perf_counter() - t0, "detail": detail})
            continue
        candidate = ReconstructionState.from_scene(scene, centres, config=cfg)
        metric = _metrics(candidate, cfg)
        candidates.append({"source": name, "valid": True,
                           "seconds": perf_counter() - t0,
                           "covered_users": metric["coverage_users"],
                           "covered_demand": metric["coverage_demand"],
                           "positive_beams": BEAM_COUNT - metric["empty_beams"],
                           "detail": detail, "state": candidate})
    viable = [candidate for candidate in candidates if candidate["valid"]]
    if not viable:
        raise ValueError("no legal 100-centre initialization candidate")
    selected = max(viable, key=lambda candidate: (
        candidate["covered_users"], candidate["covered_demand"],
        candidate["positive_beams"]))
    state = selected["state"]
    state.initialization_source = selected["source"] + "+coverage_repair"
    for candidate in candidates:
        candidate.pop("state", None)
    state.initialization_candidates = candidates
    state.elapsed_initialization_s = perf_counter() - start
    if _metrics(state, cfg)["coverage_100"] and state.initialization_valid:
        return state
    repair_start = perf_counter()
    working = state.copy()
    repair_calls = repair_pairs = attempts = 0
    user_rays = satellite_rays_numpy(scene.positions)

    def try_reassign(beam: int, center: np.ndarray,
                     incoming: int | None = None) -> bool:
        nonlocal working, repair_calls, repair_pairs, attempts
        if repair_calls >= cfg.max_initialization_geometry_calls:
            return False
        repair_calls += 1
        repair_pairs += len(scene.demands)
        candidate = working.copy()
        candidate.centers[beam] = center
        assignment = candidate.assignment
        if incoming is not None:
            assignment[incoming] = beam
        distance = working._distance_cache.copy()
        ray = satellite_rays_numpy(center[None])[0]
        distance[:, beam] = ((user_rays - ray) ** 2).sum(-1)
        missing = np.flatnonzero(assignment < 0)
        if len(missing):
            covered = distance[missing].min(axis=1) <= RAY_LIMIT
            picked = missing[covered]
            if len(picked):
                assignment[picked] = distance[picked].argmin(axis=1)
        try:
            candidate._validate_layout(distance)
        except ValueError:
            return False
        if not _certificates_preserved(candidate):
            return False
        working = candidate
        return True

    # First repair uncovered aircraft while preserving all existing service.
    missing_order = np.flatnonzero(working.assignment < 0)
    missing_order = missing_order[np.argsort(-np.asarray(scene.demands)[missing_order],
                                          kind="stable")]
    for user in missing_order:
        if repair_calls >= cfg.max_initialization_geometry_calls:
            break
        if working.assignment[user] >= 0:
            continue
        ranked = np.argsort(working._distance_cache[user], kind="stable")
        for beam in ranked:
            beam = int(beam)
            if beam in working.edge_beam_ids:
                continue
            attempts += 1
            members = np.flatnonzero(working.assignment == beam)
            if len(members) == 0:
                fitted = np.asarray(scene.positions[user], dtype=np.float64)
            else:
                counter = [0, 0, 0, 0]
                limit = max(0, cfg.max_initialization_geometry_calls - repair_calls - 1)
                try:
                    fitted = _fit_group(working, np.append(members, user),
                                        working.centers[beam], counter, limit)
                except _BudgetExceeded:
                    fitted = None
                repair_calls += counter[0]
                repair_pairs += counter[1]
            if fitted is not None and try_reassign(beam, fitted, int(user)):
                break
            if repair_calls >= cfg.max_initialization_geometry_calls:
                break

    # A complete layout also needs all 100 beams to carry positive demand.
    for beam in np.flatnonzero(working.loads <= 0):
        if repair_calls >= cfg.max_initialization_geometry_calls:
            break
        donors = np.argsort(-working.loads, kind="stable")
        for donor in donors:
            members = np.flatnonzero(working.assignment == donor)
            positive = members[np.asarray(scene.demands)[members] > 0]
            if len(positive) < 2 or donor in working.edge_beam_ids:
                continue
            attempts += 1
            user = int(positive[np.argmin(np.asarray(scene.demands)[positive])])
            # The donor keeps its old centre; the new beam covers the moved user.
            if try_reassign(int(beam), np.asarray(scene.positions[user]), user):
                break
    working.initialization_valid = bool(np.all(working.loads > 0) and
                                        _metrics(working, cfg)["coverage_users"] >= .999 and
                                        _metrics(working, cfg)["coverage_demand"] >= .999)
    working.initialization_source = state.initialization_source
    working.initialization_candidates = candidates
    working.initialization_repair_geometry_calls = repair_calls
    working.initialization_repair_ray_pair_checks = repair_pairs
    working.initialization_repair_attempts = attempts
    working.elapsed_repair_s += perf_counter() - repair_start
    working.elapsed_initialization_s = perf_counter() - start
    working.best_quality = _metrics(working, cfg)["quality"]
    working.best_record = {"centers": working.centers.tolist(),
                           "assignment": working.assignment.tolist(),
                           "quality": working.best_quality}
    return working


def action_candidates(state: ReconstructionState,
                      config: ReconstructionConfig | None = None) -> list[ReconstructionAction]:
    """Cheap shared menu; proximity is necessary only, never a feasibility claim.

    No proposed layout or resulting J is computed here.  A selector may choose
    a worse move.  Stable beam IDs are bookkeeping, never geographic labels.
    """
    cfg = config or ReconstructionConfig()
    if state.done or state.steps >= cfg.max_macro_steps or not state.initialization_valid:
        return []
    if state._candidate_cache is not None and state._candidate_cache[0] == cfg:
        return state._candidate_cache[1]
    screen_start = perf_counter()
    result = [ReconstructionAction("stop")]
    rays = satellite_rays_numpy(state.centers)
    distance = ((rays[:, None] - rays[None]) ** 2).sum(-1)
    available = [b for b in range(BEAM_COUNT) if b not in state.edge_beam_ids]
    # Load and geographic rank only narrow the k head; no result quality is used.
    def location_key(beam: int) -> tuple[float, float]:
        return (float(state.centers[beam, 0]), float(state.centers[beam, 1]))

    # The policy must not infer geography from a stable beam ID. Equal loads
    # and equal angular distances resolve by physical centre coordinates.
    heavy_key = lambda b: (-float(state.loads[b]), *location_key(b))
    heavy = sorted(available, key=heavy_key)
    heavy_limit = max(0, cfg.split_target_count - 4)
    heavy_targets = heavy[:heavy_limit]
    if heavy_targets:
        cutoff = heavy_key(heavy_targets[-1])
        heavy_targets.extend(b for b in heavy[heavy_limit:] if heavy_key(b) == cutoff)
    geographic = []
    if available:
        coordinates = state.centers[available]
        seed = available[int(np.lexsort((coordinates[:, 1], coordinates[:, 0]))[0])]
        geographic.extend(b for b in available if location_key(b) == location_key(seed))
        while len(geographic) < min(4, len(available)):
            ranked = [b for b in available if b not in geographic]
            def spread_key(beam: int) -> tuple[float, float, float]:
                return (float(distance[beam, geographic].min()),
                        -location_key(beam)[0], -location_key(beam)[1])
            best_key = max(spread_key(b) for b in ranked)
            geographic.extend(b for b in ranked if spread_key(b) == best_key)
    targets = list(dict.fromkeys(heavy_targets + geographic))
    for i in available:
        neighbor_key = lambda b: (float(distance[i, b]), *location_key(b))
        ranked = [int(j) for j in sorted(available, key=neighbor_key)
                  if j != i and distance[i, j] <=
                  squared_ray_limit(2 * RADIUS_DEG) + 1e-14]
        neighbors = ranked[:cfg.neighbor_count]
        if neighbors:
            cutoff = neighbor_key(neighbors[-1])
            neighbors.extend(b for b in ranked[cfg.neighbor_count:]
                             if neighbor_key(b) == cutoff)
        for j in neighbors:
            result.extend(ReconstructionAction("repartition2", (i, j), t)
                          for t in cfg.split_templates)
            result.extend(ReconstructionAction("assign", (i, j), t)
                          for t in cfg.split_templates)
            for k in targets:
                if k not in (i, j):
                    result.extend(ReconstructionAction("merge_split", (i, j, k), t)
                                  for t in cfg.split_templates)
            for k in neighbors:
                if k not in (i, j):
                    result.extend(ReconstructionAction("repartition3", (i, j, k), t)
                                  for t in cfg.split_templates)
                    result.extend(ReconstructionAction("assign", (i, j, k), t)
                                  for t in cfg.split_templates)
    rank = {name: i for i, name in enumerate(OPERATION_ORDER)}
    ordered = sorted(result, key=lambda a: (rank[a.operation], a.beams, a.template))
    state._candidate_cache = (cfg, ordered)
    state.candidate_screen_ray_pairs += BEAM_COUNT * BEAM_COUNT
    state.candidate_builds += 1
    state.elapsed_candidate_screen_s += perf_counter() - screen_start
    return ordered


def _propose(state: ReconstructionState, action: ReconstructionAction,
             cfg: ReconstructionConfig, counter: list[int], limit: int
             ) -> tuple[np.ndarray, np.ndarray]:
    centres = state.centers.copy()
    assignment = state.assignment.copy()
    beams = action.beams
    groups = [np.flatnonzero(assignment == b) for b in beams]
    if action.operation == "merge_split":
        merged = np.concatenate((groups[0], groups[1]))
        split = _partition(state, groups[2], 2, action.template)
        pieces = (merged, split[0], split[1])
    elif action.operation in ("repartition2", "repartition3"):
        pieces = _partition(state, np.concatenate(groups), len(beams), action.template)
    elif action.operation == "assign":
        # Keep centres fixed; transfer each member to an eligible selected beam.
        members = np.concatenate(groups)
        distances = state._distance_cache[np.ix_(members, beams)]
        counter[2] += len(members) * len(beams)
        eligible = distances <= RAY_LIMIT
        loads = state.loads.copy()
        positive_counts = np.bincount(assignment[assignment >= 0],
                                      weights=(np.asarray(state.scene.demands)[assignment >= 0] > 0),
                                      minlength=BEAM_COUNT).astype(np.int64)
        order = (np.argsort(-np.asarray(state.scene.demands)[members], kind="stable")
                 if action.template == 0 else np.arange(len(members)))
        for row in order:
            user = int(members[row])
            choices = np.flatnonzero(eligible[row])
            if len(choices) < 2:
                continue
            source = int(assignment[user])
            demand = float(state.scene.demands[user])
            if demand > 0 and positive_counts[source] <= 1:
                choices = choices[np.asarray(beams)[choices] == source]
            if not len(choices):
                continue
            if action.template == 0:
                projected = loads[np.asarray(beams)[choices]] + demand
                projected[np.asarray(beams)[choices] == source] -= demand
                selected = choices[np.argmin(projected)]
            elif action.template == 1:
                selected = choices[np.argmax(state.loads[np.asarray(beams)[choices]])]
            elif action.template == 2:
                selected = choices[np.argmin(distances[row, choices])]
            else:
                selected = choices[-1]
            target = beams[int(selected)]
            if target != source:
                loads[source] -= demand
                loads[target] += demand
                if demand > 0:
                    positive_counts[source] -= 1
                    positive_counts[target] += 1
                assignment[user] = target
        return centres, assignment
    else:
        raise ValueError("STOP is handled outside the proposal")
    for beam, members in zip(beams, pieces):
        fitted = _fit_group(state, members, centres[beam], counter,
                            limit)
        if fitted is None:
            raise ValueError("no_legal_center_for_group")
        centres[beam] = fitted
        assignment[members] = beam
    return centres, assignment


def apply_action(state: ReconstructionState, action: ReconstructionAction,
                 config: ReconstructionConfig | None = None
                 ) -> tuple[ReconstructionState, dict, float, bool, dict]:
    cfg = config or ReconstructionConfig()
    if state.done or not state.initialization_valid:
        raise ValueError("cannot act on terminated or failed-initialization state")
    if state.steps >= cfg.max_macro_steps:
        raise ValueError("macro-action budget exhausted")
    if action not in action_candidates(state, cfg):
        raise ValueError("action absent from frozen cheap candidate menu")
    before = _metrics(state, cfg)
    next_state = state.copy()
    next_state.steps += 1
    next_state.attempts += 1
    count = [0, 0, 0, 0]  # geometry calls, ray pairs, assignment checks, fit candidates
    accepted, reason = False, "stop" if action.operation == "stop" else "rejected"
    fit_time = validate_time = 0.0
    if action.operation != "stop":
        if state.geometry_calls >= cfg.max_geometry_calls_total:
            reason = "episode_geometry_budget_exhausted"
        else:
            fit_start = perf_counter()
            phase = "fit"
            try:
                limit = min(cfg.max_geometry_calls_per_action,
                            cfg.max_geometry_calls_total - state.geometry_calls)
                centres, assignment = _propose(state, action, cfg, count, limit)
                fit_time = perf_counter() - fit_start
                validate_start = perf_counter()
                phase = "validate"
                distance = state._distance_cache.copy()
                user_rays = satellite_rays_numpy(state.scene.positions)
                for beam in action.beams:
                    if np.array_equal(centres[beam], state.centers[beam]):
                        continue
                    if count[0] >= limit:
                        raise _BudgetExceeded
                    count[0] += 1
                    count[1] += len(state.scene.demands)
                    ray = satellite_rays_numpy(centres[beam:beam + 1])[0]
                    distance[:, beam] = ((user_rays - ray) ** 2).sum(-1)
                # New centres can pick up previously uncovered aircraft. They
                # receive one legal owner before the global invariant is checked.
                missing = np.flatnonzero(assignment < 0)
                if len(missing):
                    new_coverage = distance[missing].min(axis=1) <= RAY_LIMIT
                    picked = missing[new_coverage]
                    if len(picked):
                        assignment[picked] = distance[picked].argmin(axis=1)
                    count[2] += len(missing) * BEAM_COUNT
                proposal = state.copy()
                proposal.centers = centres
                proposal.assignment = assignment
                proposal._validate_layout(distance)
                old_served = state.assignment >= 0
                if np.any(old_served & (proposal.assignment < 0)):
                    reason = "covered_aircraft_lost"
                elif np.any(proposal.loads <= 0):
                    reason = "empty_or_nonpositive_beam"
                elif not np.isclose(proposal.loads.sum(),
                                    np.asarray(state.scene.demands)[proposal.assignment >= 0].sum(),
                                    rtol=1e-12, atol=1e-6):
                    reason = "demand_not_conserved"
                elif not _certificates_preserved(proposal):
                    reason = "edge_certificate_changed"
                elif np.array_equal(proposal.assignment, state.assignment) and np.array_equal(
                        proposal.centers, state.centers):
                    reason = "no_change"
                else:
                    next_state.centers = proposal.centers
                    next_state.assignment = proposal.assignment
                    next_state.loads = proposal.loads
                    next_state._candidate_cache = None
                    accepted, reason = True, "accepted"
                validate_time = perf_counter() - validate_start
            except _BudgetExceeded:
                if phase == "fit":
                    fit_time = perf_counter() - fit_start
                else:
                    validate_time = perf_counter() - validate_start
                reason = "action_geometry_budget_exhausted"
            except ValueError as exc:
                if phase == "fit":
                    fit_time = perf_counter() - fit_start
                else:
                    validate_time = perf_counter() - validate_start
                reason = str(exc)
    next_state.geometry_calls += count[0]
    next_state.fit_candidate_checks += count[3]
    next_state.assignment_candidate_checks += count[2]
    next_state.ray_pair_checks += count[1] + count[2]
    next_state.elapsed_fit_s += fit_time
    next_state.elapsed_validate_s += validate_time
    after = _metrics(next_state, cfg)
    next_state.done = bool(action.operation == "stop" or
                           next_state.steps >= cfg.max_macro_steps or
                           next_state.geometry_calls >= cfg.max_geometry_calls_total or
                           after["target_met"])
    reward = float(after["quality"] - before["quality"])
    if after["quality"] > next_state.best_quality:
        next_state.best_quality = after["quality"]
        next_state.best_record = {"centers": next_state.centers.tolist(),
                                  "assignment": next_state.assignment.tolist(),
                                  "quality": after["quality"]}
    info = {"accepted": accepted, "reason": reason, "operation": action.operation,
            "beams": action.beams, "template": action.template,
            "actual_action": ({"operation": action.operation, "beams": action.beams,
                               "template": action.template} if accepted else None),
            "geometry_calls": count[0], "geometry_calls_total": next_state.geometry_calls,
            "fit_candidate_checks": count[3],
            "assignment_candidate_checks": count[2],
            "ray_pair_checks": count[1] + count[2],
            "candidate_screen_ray_pairs_total": next_state.candidate_screen_ray_pairs,
            "candidate_builds_total": next_state.candidate_builds,
            "elapsed_candidate_screen_s_total": next_state.elapsed_candidate_screen_s,
            "attempts": next_state.attempts, "macro_steps": next_state.steps,
            "fit_seconds": fit_time, "validate_seconds": validate_time,
            "repair_seconds": 0.0, "terminal_reason": ("stop" if action.operation == "stop" else
            "target_met" if after["target_met"] else
            "macro_budget" if next_state.steps >= cfg.max_macro_steps else
            "geometry_budget" if next_state.geometry_calls >= cfg.max_geometry_calls_total else None)}
    return next_state, info, reward, next_state.done, after
