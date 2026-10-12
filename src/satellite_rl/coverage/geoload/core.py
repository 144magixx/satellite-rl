"""Atomic, geometry-only ownership executor. No capacity or service model."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from ..geometry import satellite_rays_numpy, squared_ray_limit, visible_numpy
from .config import GeometryConfig

@dataclass(frozen=True)
class Scene:
    positions: np.ndarray
    demands: np.ndarray
    profile: str = "europe_v1"

    def __post_init__(self):
        p = np.array(self.positions, dtype=float, copy=True)
        d = np.array(self.demands, dtype=float, copy=True)
        if p.shape != (len(d), 2) or d.ndim != 1 or not np.isfinite(p).all() or not np.isfinite(d).all() or (d < 0).any():
            raise ValueError("finite positions and nonnegative per-aircraft demand required")
        if not np.isfinite(d.sum()):
            raise ValueError("aggregate demand overflow")
        p.setflags(write=False); d.setflags(write=False)
        object.__setattr__(self, "positions", p); object.__setattr__(self, "demands", d)

@dataclass(frozen=True)
class Layout:
    centers: np.ndarray
    assignment: np.ndarray

    def __post_init__(self):
        c = np.array(self.centers, dtype=float, copy=True).reshape(-1, 2)
        raw = np.asarray(self.assignment)
        if raw.ndim != 1 or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
            raise ValueError("ownership must be a one-dimensional integer vector")
        a = np.array(raw, dtype=np.int64, copy=True)
        c.setflags(write=False); a.setflags(write=False)
        object.__setattr__(self, "centers", c); object.__setattr__(self, "assignment", a)

    def to_record(self):
        return {"centers": self.centers.tolist(), "assignment": self.assignment.tolist()}

    @classmethod
    def from_record(cls, record):
        return cls(record["centers"], record["assignment"])

def layout_sha256(layout):
    return hashlib.sha256(json.dumps(layout.to_record(), sort_keys=True, allow_nan=False).encode()).hexdigest()

def legal_matrix(scene, centers, config=GeometryConfig()):
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    u = satellite_rays_numpy(scene.positions, scene.profile)
    b = satellite_rays_numpy(centers, scene.profile)
    return ((np.sum((u[:, None] - b[None, :])**2, axis=-1) <=
             squared_ray_limit(config.width_deg / 2) + config.tolerance) &
            visible_numpy(scene.positions, scene.profile)[:, None] &
            visible_numpy(centers, scene.profile)[None, :])

def validate_layout(scene, layout, config=GeometryConfig()):
    n = len(layout.centers); a = layout.assignment
    reasons = []
    if n > config.max_beams: reasons.append("beam_limit")
    if len(a) != len(scene.demands): reasons.append("ownership_length")
    if not np.isfinite(layout.centers).all(): reasons.append("nonfinite_center")
    valid_ids = len(a) == len(scene.demands) and bool(((a >= 0) & (a < n)).all())
    if not valid_ids: reasons.append("unassigned_or_invalid_owner")
    loads = np.zeros(n)
    min_slack = None
    if valid_ids and np.isfinite(layout.centers).all():
        if len(a):
            legal = legal_matrix(scene, layout.centers, config)
            if not legal[np.arange(len(a)), a].all(): reasons.append("geometric_coverage")
            loads = np.bincount(a, weights=scene.demands, minlength=n)
            counts = np.bincount(a, minlength=n)
            if (counts == 0).any(): reasons.append("empty_beam")
            rays = satellite_rays_numpy(scene.positions, scene.profile)
            brays = satellite_rays_numpy(layout.centers, scene.profile)
            min_slack = float(np.min(squared_ray_limit(config.width_deg / 2) - np.sum((rays - brays[a])**2, axis=1)))
        elif n: reasons.append("empty_beam")
    conserved = valid_ids and bool(np.isclose(loads.sum(), scene.demands.sum(), rtol=1e-12, atol=1e-6))
    if not conserved: reasons.append("demand_conservation")
    return {"valid": not reasons, "reasons": reasons, "beam_count": n,
            "unique_assignment": valid_ids, "demand_conserved": conserved,
            "coverage_100": "geometric_coverage" not in reasons and "nonfinite_center" not in reasons and valid_ids,
            "minimum_squared_ray_slack": min_slack, "loads": loads.tolist()}

def metrics(scene, layout, config=GeometryConfig(), *, evidence=None):
    evidence = validate_layout(scene, layout, config) if evidence is None else evidence
    if not evidence["valid"]: raise ValueError(evidence["reasons"])
    loads = np.asarray(evidence["loads"]); total = float(scene.demands.sum()); n = len(loads)
    tail = float(np.sort(loads)[-max(1, int(np.ceil(n * .1))):].mean()) if n else 0.0
    peak = float(loads.max()) if n else 0.0
    return {"N": n, "beam_count": n, "total_demand": total, "maxload": peak,
            "P": peak / (total / n) if total and n else None,
            "T": tail / (total / n) if total and n else None,
            "Q": float(np.mean((loads / (total / n))**2)) if total and n else None,
            "top10_mean": tail, "peak_share": peak / total if total else 0.0,
            "upper_tail_share": tail / total if total else 0.0,
            "CV": float(loads.std() / loads.mean()) if total else None,
            "load_cv": float(loads.std() / loads.mean()) if total else None,
            "beam_loads": loads.tolist(), "evidence": evidence}

def objective_key(values, objective="lex"):
    cv = values["CV"] if values["CV"] is not None else 0.0
    if objective == "legacyCV": return (cv,)
    if objective == "lex": return tuple(values[k] if values[k] is not None else 0.0 for k in ("P", "T", "Q"))
    raise ValueError(objective)

def initial_layout(scene, config=GeometryConfig()):
    """Deterministic finite aircraft-center cover; failure is not infeasibility."""
    if not len(scene.demands): return Layout([], [])
    all_legal = legal_matrix(scene, scene.positions, config)
    remaining = np.ones(len(scene.demands), bool); centers = []; a = np.full(len(remaining), -1)
    while remaining.any():
        scores = np.sum(all_legal[remaining], axis=0)
        chosen = int(np.argmax(scores))
        members = remaining & all_legal[:, chosen]
        if not members.any() or len(centers) == config.max_beams:
            raise ValueError("bounded initializer found no complete cover; no infeasibility certificate")
        a[members] = len(centers); centers.append(scene.positions[chosen]); remaining[members] = False
    result = Layout(centers, a)
    if not validate_layout(scene, result, config)["valid"]: raise ValueError("invalid initial geometry")
    return result

def evaluate_case(case, method, config=None):
    """JSON worker adapter; inputs contain locations and requested demand only."""
    from .classical import run_classical
    from .config import SearchConfig
    from time import perf_counter
    started = perf_counter()
    config = dict(config or {})
    geometry = GeometryConfig(**config.pop("geometry", {}))
    scene = Scene(case["positions"], case["demands"], case.get("profile", "europe_v1"))
    initial = Layout.from_record(case["initial"]) if "initial" in case else initial_layout(scene, geometry)
    result = run_classical(scene, initial, method, SearchConfig(**config), geometry).to_record()
    result["initial_sha256"] = layout_sha256(initial)
    result["initial_metrics"] = metrics(scene, initial, geometry)
    result["end_to_end_seconds"] = perf_counter() - started
    return result
