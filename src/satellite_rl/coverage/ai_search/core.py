"""Immutable layouts and strict per-beam satellite-view geometry.

All widths are full angles. Geometry is a projected circular footprint, not a
radio-frequency gain, capacity, QoS, or downstream resource model.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np
from satellite_rl.envs.regions import get_profile
from ..geometry import satellite_rays_numpy, visible_numpy, ray_angle_numpy
from .config import GeometryConfig


def _immutable(value, dtype):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Scene:
    positions: np.ndarray
    demands: np.ndarray
    aircraft_ids: tuple = ()
    profile: str = "europe_v1"

    def __post_init__(self):
        p, d = _immutable(self.positions, float), _immutable(self.demands, float)
        if not p.size:
            p = p.reshape(0, 2)
        if (d.ndim != 1 or p.shape != (len(d), 2) or not np.isfinite(p).all()
                or not np.isfinite(d).all() or (d < 0).any() or not np.isfinite(d.sum())):
            raise ValueError("finite positions and nonnegative conserved aircraft requests required")
        ids = tuple(str(x) for x in self.aircraft_ids) if self.aircraft_ids else tuple(str(i) for i in range(len(d)))
        if len(ids) != len(d) or len(set(ids)) != len(ids) or any(not x for x in ids):
            raise ValueError("unique aircraft IDs must match scene order")
        if self.profile != "europe_v1" or not visible_numpy(p, self.profile).all():
            raise ValueError("aircraft must belong to declared Europe GEO visibility")
        object.__setattr__(self, "positions", p)
        object.__setattr__(self, "demands", d)
        object.__setattr__(self, "aircraft_ids", ids)
        object.__setattr__(self, "rays", _immutable(satellite_rays_numpy(p, self.profile), float))
        identity = {"positions": p.tolist(), "demands": d.tolist(), "aircraft_ids": list(ids), "profile": self.profile}
        object.__setattr__(self, "input_sha256", hashlib.sha256(json.dumps(identity, sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode()).hexdigest())


@dataclass(frozen=True)
class Layout:
    centers: np.ndarray
    widths: np.ndarray
    assignment: np.ndarray
    beam_ids: tuple = ()

    def __post_init__(self):
        c = _immutable(self.centers, float).reshape(-1, 2)
        w = _immutable(self.widths, float)
        raw = np.asarray(self.assignment)
        if raw.ndim != 1 or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
            raise ValueError("ownership must be a one-dimensional integer vector")
        a = _immutable(raw, np.int64)
        ids = tuple(str(x) for x in self.beam_ids) if self.beam_ids else tuple(f"b{i}" for i in range(len(c)))
        if len(ids) != len(c) or len(set(ids)) != len(ids) or any(not x for x in ids):
            raise ValueError("unique beam IDs must match centers")
        object.__setattr__(self, "centers", c)
        object.__setattr__(self, "widths", w)
        object.__setattr__(self, "assignment", a)
        object.__setattr__(self, "beam_ids", ids)

    def to_record(self):
        return {"centers": self.centers.tolist(), "widths_deg": self.widths.tolist(),
                "assignment": self.assignment.tolist(), "beam_ids": list(self.beam_ids)}

    @property
    def widths_deg(self):
        return self.widths

    @classmethod
    def from_record(cls, record):
        return cls(record["centers"], record.get("widths_deg", record.get("widths")), record["assignment"], tuple(record.get("beam_ids", ())))


def layout_sha256(layout):
    return hashlib.sha256(json.dumps(layout.to_record(), sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def legal_matrix(scene, centers, widths, geometry=GeometryConfig(), budget=None):
    centers, widths = np.asarray(centers, float).reshape(-1, 2), np.asarray(widths, float)
    if widths.shape != (len(centers),) or not np.isfinite(widths).all() or not np.isfinite(centers).all():
        raise ValueError("one finite full width per finite center required")
    if budget is not None:
        budget.geometry(len(scene.demands) * len(centers))
    rays = satellite_rays_numpy(centers, scene.profile)
    chord = np.sum((scene.rays[:, None] - rays[None]) ** 2, axis=-1)
    limit = 4 * np.sin(np.deg2rad(widths / 2) / 2) ** 2
    return ((chord <= limit[None] + geometry.tolerance)
            & visible_numpy(centers, scene.profile)[None])


def validate_layout(scene, layout, geometry=GeometryConfig(), budget=None):
    n, a = len(layout.centers), layout.assignment
    reasons = []
    if scene.profile != geometry.profile:
        reasons.append("profile_mismatch")
    if n > geometry.max_beams or (len(scene.demands) and n < 1):
        reasons.append("beam_limit")
    if layout.widths.shape != (n,):
        reasons.append("width_length")
    if not np.isfinite(layout.centers).all():
        reasons.append("nonfinite_center")
    if (not np.isfinite(layout.widths).all() or
            (layout.widths < geometry.width_min_deg).any() or (layout.widths > geometry.width_max_deg).any()):
        reasons.append("width_bounds")
    valid_ids = len(a) == len(scene.demands) and bool(((a >= 0) & (a < n)).all())
    if not valid_ids:
        reasons.append("unassigned_or_invalid_owner")
    if n and not visible_numpy(layout.centers, scene.profile).all():
        reasons.append("center_visibility")
    loads, min_slack = np.zeros(n), None
    if valid_ids and not reasons:
        if len(a):
            legal = legal_matrix(scene, layout.centers, layout.widths, geometry, budget)
            if not legal[np.arange(len(a)), a].all():
                reasons.append("geometric_coverage")
            loads = np.bincount(a, weights=scene.demands, minlength=n)
            if (np.bincount(a, minlength=n) == 0).any():
                reasons.append("empty_beam")
            brays = satellite_rays_numpy(layout.centers, scene.profile)
            limits = 4 * np.sin(np.deg2rad(layout.widths[a] / 2) / 2) ** 2
            min_slack = float(np.min(limits - np.sum((scene.rays - brays[a]) ** 2, axis=1)))
        elif n:
            reasons.append("empty_beam")
    conserved = valid_ids and bool(np.isclose(loads.sum(), scene.demands.sum(), rtol=1e-12, atol=1e-6))
    if not conserved:
        reasons.append("demand_conservation")
    return {"valid": not reasons, "reasons": reasons, "beam_count": n,
            "unique_assignment": valid_ids, "demand_conserved": conserved,
            "coverage_100": valid_ids and not reasons, "loads": loads.tolist(),
            "minimum_squared_ray_slack": min_slack, "geometry_version": "ai17_per_beam_ray_v1"}


def metrics(scene, layout, geometry=GeometryConfig(), evidence=None, budget=None):
    evidence = evidence if evidence is not None else validate_layout(scene, layout, geometry, budget)
    if not evidence["valid"]:
        raise ValueError(evidence["reasons"])
    loads, total = np.asarray(evidence["loads"]), float(scene.demands.sum())
    n = len(loads)
    peak = float(loads.max()) if n else 0.
    tail = float(np.sort(loads)[-max(1, math.ceil(n * .1)):].mean()) if n else 0.
    mean = total / n if n else 0.
    return {"N": n, "beam_count": n, "aircraft_count": len(scene.demands),
            "total_demand": total, "maxload": peak, "top10_mean": tail,
            "P": peak / mean if mean else None, "T": tail / mean if mean else None,
            "Q": float(np.mean((loads / mean) ** 2)) if mean else None,
            "CV": float(loads.std() / mean) if mean else None,
            "solid_angle_sum_sr": float(np.sum(2 * np.pi * (1 - np.cos(np.deg2rad(layout.widths / 2))))),
            "width_min_deg": float(layout.widths.min()) if n else None,
            "width_max_deg": float(layout.widths.max()) if n else None,
            "beam_loads": loads.tolist(), "widths": layout.widths.tolist(), "evidence": evidence}


def minimum_enclosing_width(scene, members, center, geometry=GeometryConfig(), budget=None):
    """Minimum allowed full width about this given center, not global optimality.

    Float64 outward rounding makes the strict independent footprint replay safe;
    no geometric tolerance is spent to shrink a physically needed angle.
    """
    members = np.asarray(members, dtype=int)
    if not len(members) or not visible_numpy(np.asarray(center).reshape(1, 2), scene.profile).all():
        return None
    if budget is not None:
        budget.geometry(len(members))
    ray = satellite_rays_numpy(np.asarray(center).reshape(1, 2), scene.profile)[0]
    required = 2 * float(ray_angle_numpy(scene.rays[members], ray).max())
    if required > geometry.width_max_deg + 1e-12:
        return None
    return float(min(geometry.width_max_deg, max(geometry.width_min_deg, np.nextafter(required, np.inf))))


def ray_center(scene, members):
    """Finite mean-ray ground intersection; not a minimum enclosing cap solver."""
    ray = scene.rays[np.asarray(members, int)].mean(0)
    norm = np.linalg.norm(ray)
    if norm < 1e-12:
        return None
    return ray_location(scene, ray / norm)


def ray_location(scene, ray):
    ray = np.asarray(ray, float)
    norm = np.linalg.norm(ray)
    if not np.isfinite(ray).all() or norm < 1e-12:
        return None
    ray = ray / norm
    p = get_profile(scene.profile)
    angle = np.deg2rad(p.satellite_longitude_deg)
    sat = p.orbit_radius_km * np.array([np.cos(angle), np.sin(angle), 0.])
    dot = float(sat @ ray)
    disc = dot * dot - float(sat @ sat - p.earth_radius_km ** 2)
    if disc < 0:
        return None
    ground = sat + (-dot - np.sqrt(disc)) * ray
    center = np.rad2deg([np.arctan2(ground[2], np.hypot(ground[0], ground[1])),
                        np.arctan2(ground[1], ground[0])])
    return center if visible_numpy(center[None], scene.profile).all() else None


def initial_layout(scene, geometry=GeometryConfig(), width_deg=0.5, budget=None):
    if not geometry.width_min_deg <= width_deg <= geometry.width_max_deg:
        raise ValueError("initial full width outside declared bounds")
    if not len(scene.demands):
        return Layout([], [], [])
    legal = legal_matrix(scene, scene.positions, np.full(len(scene.demands), width_deg), geometry, budget)
    remaining = np.ones(len(scene.demands), bool)
    centers, owners = [], np.full(len(scene.demands), -1)
    # Scene order preserves input identity. ID explicitly resolves site ties.
    id_rank = {s: i for i, s in enumerate(sorted(scene.aircraft_ids))}
    while remaining.any():
        if budget is not None:
            budget.check()
        counts = legal[remaining].sum(0)
        chosen = min(range(len(counts)), key=lambda i: (-int(counts[i]), id_rank[scene.aircraft_ids[i]]))
        members = remaining & legal[:, chosen]
        if not members.any() or len(centers) >= geometry.max_beams:
            raise ValueError("bounded initializer found no complete cover; no infeasibility certificate")
        owners[members] = len(centers)
        centers.append(scene.positions[chosen])
        remaining[members] = False
    result = Layout(centers, np.full(len(centers), width_deg), owners)
    if not validate_layout(scene, result, geometry, budget)["valid"]:
        raise ValueError("invalid initialized geometry")
    return result
