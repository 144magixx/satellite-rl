"""Variable-count, aircraft-conserving GEO beam layouts.

The reference capacity is an explicit planning surrogate, never a measured
interference-aware capacity.  All accepted layouts are independently checked
using satellite rays.  Search failures leave the last published layout intact.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
from time import perf_counter

import numpy as np

from .geometry import satellite_rays_numpy, squared_ray_limit, visible_numpy
from .variable_config import CORE_ROUTES, VariableConfig


OPERATIONS = ("stop", "add", "split", "merge", "move", "assign", "resize")
_GEOMETRY_AUDIT = ContextVar("variable_geometry_audit", default=None)


@contextmanager
def geometry_audit():
    """Count actual vectorized footprint matrices and user-beam ray pairs.

    Pass the returned mutable dictionary around one route/evaluation.  Nested
    calls to initial_layout, candidates and apply_action share this scope.
    """
    audit = {"matrix_calls": 0, "ray_pair_checks": 0}
    token = _GEOMETRY_AUDIT.set(audit)
    try:
        yield audit
    finally:
        _GEOMETRY_AUDIT.reset(token)


@dataclass
class Layout:
    centers: np.ndarray
    widths: np.ndarray
    assignment: np.ndarray
    aircraft_ids: tuple[str, ...] = ()
    beam_ids: tuple[str, ...] = ()
    next_beam_serial: int = 0

    def __post_init__(self) -> None:
        self.centers = np.asarray(self.centers, dtype=np.float64).reshape(-1, 2).copy()
        self.widths = np.asarray(self.widths, dtype=np.float64).reshape(-1).copy()
        self.assignment = np.asarray(self.assignment, dtype=np.int64).reshape(-1).copy()
        self.aircraft_ids = tuple(self.aircraft_ids)
        if self.aircraft_ids and len(self.aircraft_ids) != len(self.assignment):
            raise ValueError("layout aircraft identity length mismatch")
        self.beam_ids = tuple(self.beam_ids) or tuple(f"b{i}" for i in range(len(self.centers)))
        if (len(self.beam_ids) != len(self.centers) or
                any(not isinstance(s, str) or not s for s in self.beam_ids) or
                len(set(self.beam_ids)) != len(self.beam_ids)):
            raise ValueError("beam identities must be unique and complete")
        derived = max((int(s[1:]) for s in self.beam_ids
                       if s.startswith("b") and s[1:].isdigit()), default=-1) + 1
        self.next_beam_serial = max(int(self.next_beam_serial), derived)

    @classmethod
    def empty(cls, user_count: int) -> "Layout":
        return cls(np.empty((0, 2)), np.empty(0), np.full(user_count, -1))

    def to_record(self) -> dict:
        return {"centers": self.centers.tolist(), "widths": self.widths.tolist(),
                "assignment": self.assignment.tolist(), "aircraft_ids": list(self.aircraft_ids),
                "beam_ids": list(self.beam_ids), "next_beam_serial": self.next_beam_serial}

    @classmethod
    def from_record(cls, record: dict) -> "Layout":
        return cls(record["centers"], record["widths"], record["assignment"],
                   tuple(record.get("aircraft_ids", ())), tuple(record.get("beam_ids", ())),
                   int(record.get("next_beam_serial", 0)))


@dataclass(frozen=True, order=True)
class VariableAction:
    operation: str
    beams: tuple[int, ...] = ()
    target: int = -1
    width_deg: float = 0.5

    def __post_init__(self) -> None:
        if self.operation not in OPERATIONS:
            raise ValueError("unknown variable-layout action")
        if self.operation in ("add", "split", "move", "assign") and self.target < 0:
            raise ValueError("action requires an aircraft target")
        count = {"stop": 0, "add": 0, "split": 1, "merge": 2,
                 "move": 1, "assign": 1, "resize": 1}[self.operation]
        if len(self.beams) != count or len(set(self.beams)) != count or any(b < 0 for b in self.beams):
            raise ValueError("wrong or repeated beam IDs")

    def to_record(self) -> dict:
        return {"operation": self.operation, "beams": list(self.beams),
                "target": self.target, "width_deg": self.width_deg}

    @classmethod
    def from_record(cls, record: dict) -> "VariableAction":
        return cls(record["operation"], tuple(record.get("beams", ())),
                   int(record.get("target", -1)), float(record.get("width_deg", 0.5)))


def _scene_arrays(scene, cfg: VariableConfig) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(scene.positions, dtype=np.float64)
    demands = np.asarray(scene.demands, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or demands.shape != (len(positions),):
        raise ValueError("scene needs one 2D position and demand per aircraft")
    if not np.isfinite(positions).all() or not np.isfinite(demands).all() or np.any(demands < 0):
        raise ValueError("scene contains nonfinite coordinates or invalid demand")
    if len(positions) and not bool(np.all(visible_numpy(positions, cfg.profile))):
        raise ValueError("aircraft outside registered Europe GEO visibility")
    ids = getattr(scene, "aircraft_ids", None)
    if ids is not None and (len(ids) != len(positions) or len(set(ids)) != len(ids)):
        raise ValueError("aircraft identities are not unique and complete")
    return positions, demands


def _legal_matrix(positions: np.ndarray, centers: np.ndarray,
                  widths: np.ndarray, cfg: VariableConfig) -> np.ndarray:
    audit = _GEOMETRY_AUDIT.get()
    if audit is not None:
        audit["matrix_calls"] += 1
        audit["ray_pair_checks"] += int(len(positions) * len(centers))
    if not len(centers):
        return np.zeros((len(positions), 0), dtype=bool)
    users = satellite_rays_numpy(positions, cfg.profile)
    beams = satellite_rays_numpy(centers, cfg.profile)
    chord2 = ((users[:, None] - beams[None, :]) ** 2).sum(axis=2)
    limit = np.array([squared_ray_limit(w / 2) for w in widths])
    return chord2 <= limit[None, :] + 1e-14


def _check(scene, layout: Layout, cfg: VariableConfig, *, allow_partial: bool = True) -> tuple[np.ndarray, np.ndarray]:
    positions, demands = _scene_arrays(scene, cfg)
    n = len(layout.centers)
    if n > cfg.max_beams or layout.widths.shape != (n,) or layout.assignment.shape != (len(positions),):
        raise ValueError("beam count, widths or aircraft assignment length invalid")
    if layout.aircraft_ids and tuple(getattr(scene, "aircraft_ids", ())) != layout.aircraft_ids:
        raise ValueError("layout aircraft IDs differ from current scene order")
    if not np.isfinite(layout.centers).all() or not np.isfinite(layout.widths).all():
        raise ValueError("nonfinite layout")
    if n and not np.all(visible_numpy(layout.centers, cfg.profile)):
        raise ValueError("beam centre outside registered visibility")
    if np.any(layout.widths <= 0) or np.any((layout.assignment < -1) | (layout.assignment >= n)):
        raise ValueError("invalid width or aircraft owner")
    allowed = cfg.width_choices_deg if np.any(np.abs(layout.widths - cfg.fixed_width_deg) > 1e-12) else (cfg.fixed_width_deg,)
    if any(not any(abs(float(w) - x) <= 1e-12 for x in allowed) for w in layout.widths):
        raise ValueError("width outside declared physical version")
    legal = _legal_matrix(positions, layout.centers, layout.widths, cfg)
    assigned = layout.assignment >= 0
    if np.any(~legal[np.flatnonzero(assigned), layout.assignment[assigned]]):
        raise ValueError("aircraft assigned outside exact satellite footprint")
    if not allow_partial and not np.all(assigned):
        raise ValueError("incomplete aircraft coverage")
    if n and np.any(np.bincount(layout.assignment[assigned], minlength=n) == 0):
        raise ValueError("empty artificial beam is not a valid layout")
    loads = np.bincount(layout.assignment[assigned], weights=demands[assigned], minlength=n).astype(np.float64)
    return legal, loads


def metrics(scene, layout: Layout, cfg: VariableConfig) -> dict:
    _, loads = _check(scene, layout, cfg)
    _, demands = _scene_arrays(scene, cfg)
    covered = layout.assignment >= 0
    total = float(demands.sum())
    positive = loads[loads > 0]
    processing_order = sorted(range(len(loads)), key=lambda b: (-loads[b], layout.beam_ids[b]))
    processing_index = [0] * len(loads)
    for i, b in enumerate(processing_order):
        processing_index[b] = i
    cv = float(positive.std() / positive.mean()) if len(positive) and positive.mean() > 0 else 0.0
    cap = cfg.reference_capacity_bps
    geometry_feasible = bool(np.all(covered))
    overloaded = int(np.count_nonzero(loads > cap + 1e-6))
    result = {
        "beam_count": len(layout.centers), "active_business_beams": len(positive),
        "empty_beams": int(np.count_nonzero(np.bincount(layout.assignment[covered], minlength=len(loads)) == 0)),
        "aircraft_count": len(demands), "covered_users": int(covered.sum()),
        "coverage_users": float(covered.mean()) if len(covered) else 1.0,
        "coverage_demand": float(demands[covered].sum() / total) if total else 1.0,
        "coverage_100": geometry_feasible,
        "geometry_feasible": geometry_feasible,
        "reference_overloaded_beams": overloaded,
        "reference_capacity_feasible": bool(geometry_feasible and overloaded == 0),
        "reference_capacity_scope": "planning_proxy_per_beam_only; excludes interference, slots, power and user TDMA",
        "uncovered_aircraft_indices": np.flatnonzero(~covered).tolist(),
        "uncovered_demand_bps": float(demands[~covered].sum()),
        "total_demand_bps": total, "beam_demand_bps": loads.tolist(),
        "beam_ids": list(layout.beam_ids), "processing_order": processing_order,
        "processing_index_by_beam": processing_index,
        "processing_group_by_beam": [i % 8 for i in processing_index],
        "load_cv": cv, "peak_load_ratio": float(loads.max() / cap) if len(loads) else 0.0,
        "reference_capacity_bps": cap,
        "reference_unserved_bps": float(np.maximum(loads - cap, 0).sum() + demands[~covered].sum()),
        "demand_conserved": bool(np.isclose(loads.sum() + demands[~covered].sum(), total, rtol=1e-12, atol=1e-6)),
        "geometry_version": "exact_satellite_ray_europe_v1",
        "width_physics_version": "idealized_circular_footprint_v1",
    }
    return result


def _assign(positions, demands, centers, widths, cfg, *, mode="nearest"):
    legal = _legal_matrix(positions, centers, widths, cfg)
    result = np.full(len(positions), -1, dtype=np.int64)
    if not len(centers):
        return result
    ur = satellite_rays_numpy(positions, cfg.profile)
    br = satellite_rays_numpy(centers, cfg.profile)
    distance = ((ur[:, None] - br[None, :]) ** 2).sum(axis=2)
    loads = np.zeros(len(centers))
    order = np.lexsort((np.arange(len(demands)), -demands)) if mode == "balanced" else np.arange(len(demands))
    for u in order:
        choices = np.flatnonzero(legal[u])
        if not len(choices):
            continue
        if mode == "balanced":
            # Integral assignment: demand is never split across beams.
            score = np.maximum((loads[choices] + demands[u]) / cfg.reference_capacity_bps - 1, 0) ** 2
            score += 0.05 * loads[choices] / cfg.reference_capacity_bps
            score += distance[u, choices]
            b = int(choices[np.argmin(score)])
        else:
            b = int(choices[np.argmin(distance[u, choices])])
        result[u] = b
        loads[b] += demands[u]
    return result


def _compact(layout: Layout) -> Layout:
    if not len(layout.centers):
        return layout
    used = np.unique(layout.assignment[layout.assignment >= 0])
    remap = np.full(len(layout.centers), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return Layout(layout.centers[used], layout.widths[used],
                  np.where(layout.assignment >= 0, remap[np.maximum(layout.assignment, 0)], -1),
                  layout.aircraft_ids, tuple(layout.beam_ids[b] for b in used),
                  layout.next_beam_serial)


def _candidate_sites(positions, cfg):
    sites = np.unique(positions, axis=0)
    if len(sites) > cfg.max_initial_candidates:
        raise ValueError(f"{len(sites)} distinct candidate sites exceed max_initial_candidates={cfg.max_initial_candidates}; no aircraft truncated")
    return sites


def _construct(scene, cfg, method: str, width: float) -> tuple[Layout, dict]:
    positions, demands = _scene_arrays(scene, cfg)
    if not len(positions):
        return Layout.empty(0), {"candidate_sites": 0, "selection_method": method, "geometry_calls": 0}
    sites = _candidate_sites(positions, cfg)
    cover = _legal_matrix(positions, sites, np.full(len(sites), width), cfg)
    uncovered = np.ones(len(positions), dtype=bool)
    chosen = []
    calls = 1  # One vectorized exact user-by-site ray matrix.
    scans = 0
    no_candidate = False
    while np.any(uncovered) and len(chosen) < cfg.max_beams:
        scans += 1
        # Preserve aircraft order while skipping rows already covered.  This
        # avoids rescanning the full user×site matrix after every chosen beam.
        new = cover[uncovered]
        counts = new.sum(axis=0)
        if method == "capacity_cluster":
            mass = (new * demands[uncovered, None]).sum(axis=0)
            score = counts + np.minimum(mass / cfg.reference_capacity_bps, 1.0) - 2 * np.maximum(mass / cfg.reference_capacity_bps - 1, 0)
        elif method == "set_cover":
            score = counts + 1e-6 * (new * demands[uncovered, None]).sum(axis=0) / max(float(demands.sum()), 1.0)
        elif method == "power_transport":
            mass = (new * demands[uncovered, None]).sum(axis=0)
            score = np.minimum(mass, cfg.reference_capacity_bps) / cfg.reference_capacity_bps + 0.01 * counts
        else:
            raise ValueError("unknown constructor")
        score[counts == 0] = -np.inf
        if chosen:
            score[chosen] = -np.inf
        b = int(np.argmax(score))
        if not np.isfinite(score[b]):
            no_candidate = True
            break
        chosen.append(b)
        uncovered &= ~cover[:, b]
    centers = sites[chosen]
    widths = np.full(len(chosen), width)
    assignment = _assign(positions, demands, centers, widths, cfg,
                         mode="balanced" if method != "set_cover" else "nearest")
    layout = _compact(Layout(centers, widths, assignment, tuple(getattr(scene, "aircraft_ids", ()))))
    info = {"candidate_sites": len(sites), "selection_method": method,
            "geometry_calls": calls, "candidate_scans": scans,
            "ray_pair_checks": int(len(positions) * len(sites)),
            "selected_sites": len(chosen), "uncovered_after_selection": int(uncovered.sum()),
            "construction_termination": ("all_covered" if not np.any(uncovered) else
                                         "beam_cap_reached" if len(chosen) >= cfg.max_beams else
                                         "no_candidate" if no_candidate else "unknown")}
    return layout, info


def _set_cover_mip(scene, seed: Layout, cfg: VariableConfig) -> tuple[Layout, dict]:
    """Binary candidate-set cover, with a finite solver budget and incumbent gate.

    Its certificate applies only to the screened discrete candidate set.
    Capacity and unique indivisible assignment are repaired after selection.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_matrix

    positions, demands = _scene_arrays(scene, cfg)
    sites = _candidate_sites(positions, cfg)
    if not len(sites):
        return seed, {"solver": "scipy.optimize.milp", "status": "empty_scene", "selected_sites": 0}
    seed_sites = np.unique(seed.centers, axis=0)
    legal_all = _legal_matrix(positions, sites, np.full(len(sites), cfg.fixed_width_deg), cfg)
    coverage_count = legal_all.sum(axis=0)
    ranks = np.argsort(-coverage_count, kind="stable")
    mandatory = [int(np.flatnonzero(np.all(sites == s, axis=1))[0]) for s in seed_sites]
    selected = list(dict.fromkeys(mandatory))
    # Give each aircraft a chance to enter the restricted MIP even when the
    # greedy seed failed.  These are candidate sites, never dropped users.
    for u in range(len(positions)):
        if selected and np.any(legal_all[u, selected]):
            continue
        if len(selected) >= max(cfg.max_mip_sites, len(mandatory)):
            break
        own = int(np.flatnonzero(np.all(sites == positions[u], axis=1))[0])
        if own not in selected:
            selected.append(own)
    for j in ranks:
        if len(selected) >= max(cfg.max_mip_sites, len(mandatory)):
            break
        if int(j) not in selected:
            selected.append(int(j))
    cover = legal_all[:, selected]
    if not np.all(cover.any(axis=1)):
        return seed, {"solver": "scipy.optimize.milp", "status": "candidate_subset_incomplete",
                      "candidate_sites_total": len(sites), "candidate_sites_mip": len(selected),
                      "fallback": "greedy_seed"}
    constraints = LinearConstraint(csc_matrix(cover.astype(float)),
                                   lb=np.ones(len(positions)), ub=np.full(len(positions), np.inf))
    result = milp(c=np.ones(len(selected)), integrality=np.ones(len(selected)),
                  bounds=Bounds(np.zeros(len(selected)), np.ones(len(selected))),
                  constraints=constraints,
                  options={"time_limit": cfg.mip_time_limit_s, "mip_rel_gap": cfg.mip_rel_gap})
    info = {"solver": "scipy.optimize.milp", "status": int(result.status),
            "message": str(result.message), "candidate_sites_total": len(sites),
            "candidate_sites_mip": len(selected), "time_limit_s": cfg.mip_time_limit_s,
            "relative_gap_limit": cfg.mip_rel_gap,
            "candidate_site_indices": selected,
            "mip_gap": float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else None,
            "mip_dual_bound": float(result.mip_dual_bound) if getattr(result, "mip_dual_bound", None) is not None else None}
    if result.x is None:
        info["fallback"] = "greedy_seed_no_incumbent"
        return seed, info
    chosen = np.flatnonzero(result.x > 0.5)
    if not len(chosen) or not np.all(cover[:, chosen].any(axis=1)) or len(chosen) > cfg.max_beams:
        info["fallback"] = "greedy_seed_invalid_or_over_cap_incumbent"
        return seed, info
    centers = sites[np.asarray(selected)[chosen]]
    widths = np.full(len(centers), cfg.fixed_width_deg)
    assignment = _assign(positions, demands, centers, widths, cfg, mode="balanced")
    proposal = _compact(Layout(centers, widths, assignment, tuple(getattr(scene, "aircraft_ids", ()))))
    if not metrics(scene, proposal, cfg)["coverage_100"]:
        info["fallback"] = "greedy_seed_integer_repair_failed"
        return seed, info
    info.update(selected_sites=len(chosen), integer_repair_beams=len(proposal.centers),
                selected_candidate_columns=chosen.tolist(),
                fallback=None, certificate_scope="screened_discrete_candidate_sites_only")
    return proposal, info


def _transport_lp(scene, seed: Layout, cfg: VariableConfig) -> tuple[Layout, dict]:
    """Continuous demand-mass transport, then indivisible legal assignment.

    The LP may split an aircraft's mass.  Its output is only a relaxation;
    the returned layout assigns each whole aircraft to one certified beam.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    positions, demands = _scene_arrays(scene, cfg)
    n = len(seed.centers)
    if not n or not len(positions):
        return seed, {"solver": "scipy.optimize.linprog/highs", "status": "empty_layout"}
    legal = _legal_matrix(positions, seed.centers, seed.widths, cfg)
    rays = satellite_rays_numpy(positions, cfg.profile)
    centers = satellite_rays_numpy(seed.centers, cfg.profile)
    distance = ((rays[:, None] - centers[None, :]) ** 2).sum(axis=2)
    user_edge = []
    beam_edge = []
    for u in range(len(positions)):
        options = np.flatnonzero(legal[u])
        order = options[np.argsort(distance[u, options], kind="stable")]
        # Keep the previous integral owner among bounded options so the LP
        # candidate graph never loses the feasible seed assignment.
        picked = list(order[:cfg.max_ot_options_per_user])
        old = int(seed.assignment[u])
        if old >= 0 and old not in picked:
            picked[-1] = old
        for b in picked:
            user_edge.append(u)
            beam_edge.append(int(b))
    edge_count = len(user_edge)
    if edge_count == 0 or any(u not in user_edge for u in range(len(positions))):
        return seed, {"solver": "scipy.optimize.linprog/highs", "status": "candidate_graph_incomplete",
                      "fallback": "integral_seed"}
    user_edge = np.asarray(user_edge, dtype=int)
    beam_edge = np.asarray(beam_edge, dtype=int)
    supply = demands / cfg.reference_capacity_bps
    # Variables are normalized mass on each user->beam edge plus nonnegative
    # per-beam overload.  Equalities conserve every aircraft's full demand.
    eq = coo_matrix((np.ones(edge_count), (user_edge, np.arange(edge_count))),
                    shape=(len(positions), edge_count + n)).tocsr()
    ub_rows = np.concatenate((beam_edge, np.arange(n)))
    ub_cols = np.concatenate((np.arange(edge_count), edge_count + np.arange(n)))
    ub_data = np.concatenate((np.ones(edge_count), -np.ones(n)))
    upper = coo_matrix((ub_data, (ub_rows, ub_cols)), shape=(n, edge_count + n)).tocsr()
    target_mass = float(supply.sum()) / n
    cost = np.concatenate((distance[user_edge, beam_edge] * 100,
                           np.ones(n)))
    result = linprog(cost, A_ub=upper, b_ub=np.full(n, target_mass),
                     A_eq=eq, b_eq=supply, bounds=(0, None), method="highs",
                     options={"time_limit": cfg.ot_time_limit_s})
    info = {"solver": "scipy.optimize.linprog/highs", "status": int(result.status),
            "message": str(result.message), "candidate_edges": edge_count,
            "time_limit_s": cfg.ot_time_limit_s, "target_mass_per_beam_ratio": target_mass,
            "fractional_objective": float(result.fun) if result.fun is not None else None}
    if result.x is None:
        info["fallback"] = "integral_seed_no_fractional_solution"
        return seed, info
    flow = result.x[:edge_count]
    recovered = np.bincount(user_edge, weights=flow, minlength=len(positions))
    if not np.allclose(recovered, supply, rtol=1e-7, atol=1e-8):
        info["fallback"] = "integral_seed_fractional_mass_not_conserved"
        return seed, info
    owner = seed.assignment.copy()
    split_users = 0
    for u in range(len(positions)):
        edges = np.flatnonzero(user_edge == u)
        positive = edges[flow[edges] > 1e-8]
        split_users += len(positive) > 1
        if len(edges):
            owner[u] = beam_edge[edges[np.argmax(flow[edges])]]
    # Integral repair: each move preserves exact footprint and whole demand.
    loads = np.bincount(owner, weights=demands, minlength=n).astype(float)
    for _ in range(2):
        moved = False
        for u in np.argsort(-demands, kind="stable"):
            old = owner[u]
            old_score = np.maximum(loads / cfg.reference_capacity_bps - 1, 0) ** 2
            best, best_delta = old, 0.0
            for b in np.flatnonzero(legal[u]):
                if b == old:
                    continue
                new_old = max((loads[old] - demands[u]) / cfg.reference_capacity_bps - 1, 0) ** 2
                new_other = max((loads[b] + demands[u]) / cfg.reference_capacity_bps - 1, 0) ** 2
                delta = new_old + new_other - old_score[old] - old_score[b]
                if delta < best_delta - 1e-12:
                    best, best_delta = int(b), float(delta)
            if best != old:
                owner[u] = best
                loads[old] -= demands[u]
                loads[best] += demands[u]
                moved = True
        if not moved:
            break
    proposal = _compact(Layout(seed.centers, seed.widths, owner, seed.aircraft_ids,
                               seed.beam_ids, seed.next_beam_serial))
    if not metrics(scene, proposal, cfg)["coverage_100"]:
        info["fallback"] = "integral_seed_repair_failed"
        return seed, info
    fractional_loads = np.bincount(beam_edge, weights=flow * cfg.reference_capacity_bps, minlength=n)
    repaired_loads = np.bincount(owner, weights=demands, minlength=n)
    info.update(fractionally_split_users=int(split_users),
                integer_repair_changed_users=int(np.count_nonzero(owner != seed.assignment)),
                fractional_to_integer_load_l1_bps=float(np.abs(fractional_loads - repaired_loads).sum()),
                fractional_flow_edges=[[int(u), int(b), float(mass * cfg.reference_capacity_bps)]
                                       for u, b, mass in zip(user_edge, beam_edge, flow) if mass > 0],
                fractional_loads_bps=fractional_loads.tolist(),
                repaired_loads_bps=repaired_loads.tolist(),
                repaired_beams=len(proposal.centers), fallback=None)
    return proposal, info


def _action_features(scene, layout: Layout, cfg: VariableConfig, action: VariableAction,
                     legal: np.ndarray, loads: np.ndarray) -> np.ndarray:
    """Sixteen pre-action quantities; no candidate is executed to build these."""
    _, demands = _scene_arrays(scene, cfg)
    f = np.zeros(16, dtype=np.float32)
    f[OPERATIONS.index(action.operation)] = 1.0  # 0:7 operation one-hot
    n = len(layout.centers)
    f[7] = n / cfg.max_beams
    f[8] = float(np.mean(layout.assignment >= 0)) if len(demands) else 1.0
    f[9] = float(loads[action.beams[0]] / cfg.reference_capacity_bps) if action.beams else 0.0
    f[10] = float(loads[action.beams[-1]] / cfg.reference_capacity_bps) if len(action.beams) > 1 else 0.0
    f[11] = float(demands[action.target] / cfg.reference_capacity_bps) if action.target >= 0 else 0.0
    f[12] = float(legal[action.target].sum() / max(n, 1)) if action.target >= 0 else 0.0
    f[13] = (cfg.max_beams - n) / cfg.max_beams
    f[14] = float(action.width_deg)
    f[15] = float(demands.sum() / (cfg.reference_capacity_bps * max(n, 1)))
    return f


def candidates(scene, layout: Layout, cfg: VariableConfig, mode: str = "local",
               budget: dict | None = None) -> list[tuple[VariableAction, np.ndarray]]:
    """Stable, bounded action menu.  Features describe only the current state."""
    if mode not in ("local", "construct", "width"):
        raise ValueError("unknown candidate mode")
    if budget is not None:
        if budget.get("geometry_calls", 0) + 1 > budget.get("max_geometry_calls", cfg.max_geometry_calls):
            return []
        budget["geometry_calls"] = budget.get("geometry_calls", 0) + 1
    positions, demands = _scene_arrays(scene, cfg)
    legal, loads = _check(scene, layout, cfg)
    n = len(layout.centers)
    actions = [VariableAction("stop")]
    if mode == "construct":
        if n < cfg.max_beams:
            uncovered = np.flatnonzero(layout.assignment < 0)
            order = sorted(uncovered, key=lambda u: (-demands[u], int(u)))
            actions += [VariableAction("add", target=int(u), width_deg=cfg.fixed_width_deg)
                        for u in order[:max(0, cfg.max_candidates - 1)]]
    elif n:
        heavy = sorted(range(n), key=lambda b: (-loads[b], b))[:min(4, n)]
        for b in heavy:
            users = np.flatnonzero(layout.assignment == b)
            if len(users) > 1 and n < cfg.max_beams:
                target = sorted(users, key=lambda u: (-demands[u], int(u)))[0]
                actions.append(VariableAction("split", (b,), int(target), cfg.fixed_width_deg))
            if len(users):
                target = int(users[np.argmax(demands[users])])
                actions.append(VariableAction("move", (b,), target))
            if mode == "width":
                for width in cfg.width_choices_deg:
                    if abs(width - layout.widths[b]) > 1e-12:
                        actions.append(VariableAction("resize", (b,), width_deg=width))
        light = sorted(range(n), key=lambda b: (loads[b], b))[:min(4, n)]
        for i, a in enumerate(light):
            for b in light[i + 1:]:
                actions.append(VariableAction("merge", (a, b)))
        for u in sorted(range(len(demands)), key=lambda i: (-demands[i], i))[:16]:
            owner = layout.assignment[u]
            if owner < 0:
                continue
            for b in np.flatnonzero(legal[u]):
                if b != owner:
                    actions.append(VariableAction("assign", (int(b),), int(u)))
        if n < cfg.max_beams:
            for u in np.flatnonzero(layout.assignment < 0)[:4]:
                actions.append(VariableAction("add", target=int(u)))
    # Deduplication and ranking are deterministic; STOP retains slot zero.
    distinct = list(dict.fromkeys(actions))[:cfg.max_candidates]
    return [(a, _action_features(scene, layout, cfg, a, legal, loads)) for a in distinct]


def _apply_action_impl(scene, layout: Layout, cfg: VariableConfig, action: VariableAction,
                       budget: dict | None = None) -> tuple[Layout, dict]:
    """Execute one proposal in a copy, certify it, then commit or reject atomically."""
    t0 = perf_counter()
    info = {"action": action.to_record(), "accepted": False, "reason": None,
            "geometry_calls": 0, "elapsed_s": 0.0}
    # Charge a conservative fixed action-specific bound before evaluating a
    # proposal, including the incumbent and final exact geometry checks.
    charge = 1 if action.operation == "stop" else 2 + int(action.operation in
                                                             ("add", "split", "merge", "move", "assign"))
    if budget is not None and budget.get("geometry_calls", 0) + charge > budget.get("max_geometry_calls", cfg.max_geometry_calls):
        info["reason"] = "geometry_budget_exhausted"
        return layout, info
    if budget is not None:
        budget["geometry_calls"] = budget.get("geometry_calls", 0) + charge
    info["geometry_calls"] = charge
    positions, demands = _scene_arrays(scene, cfg)
    before = metrics(scene, layout, cfg)
    info["before_coverage_100"] = before["coverage_100"]
    if action.operation == "stop":
        info.update(accepted=before["coverage_100"], reason="stop" if before["coverage_100"] else "stop_before_full_coverage",
                    elapsed_s=perf_counter() - t0)
        return layout, info
    n = len(layout.centers)
    if any(b >= n for b in action.beams) or action.target >= len(positions):
        info["reason"] = "action_index_out_of_range"
        return layout, info
    centers, widths, owner = layout.centers.copy(), layout.widths.copy(), layout.assignment.copy()
    beam_ids = list(layout.beam_ids)
    next_serial = layout.next_beam_serial
    reason = None
    if action.operation in ("add", "split"):
        if n >= cfg.max_beams:
            reason = "beam_cap_reached"
        elif action.operation == "split" and owner[action.target] != action.beams[0]:
            reason = "split_target_not_member"
        elif not any(abs(action.width_deg - x) <= 1e-12 for x in
                     (cfg.fixed_width_deg, *cfg.width_choices_deg)):
            reason = "invalid_width"
        else:
            centers = np.vstack((centers, positions[action.target]))
            widths = np.append(widths, action.width_deg)
            beam_ids.append(f"b{next_serial}")
            next_serial += 1
            new = n
            legal = _legal_matrix(positions, centers[-1:], widths[-1:], cfg)[:, 0]
            if action.operation == "split":
                source = action.beams[0]
                members = np.flatnonzero((owner == source) & legal)
                # Demand-aware integral partition, with one guaranteed child.
                source_load = float(demands[owner == source].sum())
                new_load = 0.0
                for u in sorted(members, key=lambda k: (-demands[k], int(k))):
                    if u == action.target or (new_load < source_load / 2 and u != action.target):
                        owner[u] = new
                        new_load += demands[u]
                owner[action.target] = new
            else:
                owner[(owner < 0) & legal] = new
                # Constructor may rebalance users only when it lowers overload.
                for u in np.flatnonzero(legal & (owner >= 0) & (owner != new)):
                    old = owner[u]
                    old_load = demands[owner == old].sum()
                    new_load = demands[owner == new].sum()
                    if old_load > cfg.reference_capacity_bps and new_load + demands[u] < old_load:
                        owner[u] = new
    elif action.operation == "merge":
        a, b = action.beams
        members = np.flatnonzero((owner == a) | (owner == b))
        sites = np.unique(np.vstack((centers[[a, b]], positions[members])), axis=0)
        possible = _legal_matrix(positions[members], sites,
                                 np.full(len(sites), cfg.fixed_width_deg), cfg)
        good = np.flatnonzero(np.all(possible, axis=0))
        if not len(good):
            reason = "merge_has_no_legal_candidate_centre"
        else:
            centers[a] = sites[good[0]]
            widths[a] = cfg.fixed_width_deg
            owner[members] = a
            keep = np.arange(n) != b
            centers, widths = centers[keep], widths[keep]
            del beam_ids[b]
            owner[owner > b] -= 1
    elif action.operation == "move":
        centers[action.beams[0]] = positions[action.target]
        owner = _assign(positions, demands, centers, widths, cfg, mode="balanced")
    elif action.operation == "assign":
        b = action.beams[0]
        if not _legal_matrix(positions[action.target:action.target + 1],
                             centers[b:b + 1], widths[b:b + 1], cfg)[0, 0]:
            reason = "assignment_outside_footprint"
        else:
            owner[action.target] = b
    elif action.operation == "resize":
        if not any(abs(action.width_deg - x) <= 1e-12 for x in cfg.width_choices_deg):
            reason = "invalid_width"
        else:
            widths[action.beams[0]] = action.width_deg
    if reason is None:
        proposal = _compact(Layout(centers, widths, owner, layout.aircraft_ids,
                                   tuple(beam_ids), next_serial))
        try:
            after = metrics(scene, proposal, cfg)
            if before["coverage_100"] and not after["coverage_100"]:
                reason = "full_coverage_would_be_lost"
            elif after["covered_users"] < before["covered_users"]:
                reason = "coverage_would_decrease"
            elif not after["demand_conserved"]:
                reason = "demand_not_conserved"
            elif len(positions) and len(proposal.centers) == 0:
                reason = "positive_scene_without_beam"
            else:
                info.update(accepted=True, reason="committed", after_metrics=after,
                            elapsed_s=perf_counter() - t0)
                return proposal, info
        except ValueError as exc:
            reason = f"geometry_or_layout_invalid: {exc}"
    info.update(reason=reason, elapsed_s=perf_counter() - t0)
    return layout, info


def apply_action(scene, layout: Layout, cfg: VariableConfig, action: VariableAction,
                 budget: dict | None = None) -> tuple[Layout, dict]:
    """Atomic action with both charged and observed exact geometry costs."""
    audit = _GEOMETRY_AUDIT.get()
    own = audit is None
    token = None
    if own:
        audit = {"matrix_calls": 0, "ray_pair_checks": 0}
        token = _GEOMETRY_AUDIT.set(audit)
    before_calls = audit["matrix_calls"]
    before_pairs = audit["ray_pair_checks"]
    try:
        proposal, info = _apply_action_impl(scene, layout, cfg, action, budget)
        info["actual_geometry_calls"] = audit["matrix_calls"] - before_calls
        info["actual_ray_pair_checks"] = audit["ray_pair_checks"] - before_pairs
        info["geometry_budget_charges"] = info["geometry_calls"]
        return proposal, info
    finally:
        if own:
            _GEOMETRY_AUDIT.reset(token)


def _robust_demand_patterns(scene, cfg: VariableConfig) -> tuple[list[np.ndarray], list[str]]:
    """Four fixed demand perturbations keyed by stable aircraft identity.

    Hashes cover the *normalized bps vectors by identity*, so a mere input-row
    permutation has exactly the same evidence key.  Without aircraft IDs,
    identical coordinates receive identical factors rather than row-based ones.
    """
    positions, demands = _scene_arrays(scene, cfg)
    ids = getattr(scene, "aircraft_ids", None)
    keys = ([str(x) for x in ids] if ids is not None else
            [f"{float(p[0]).hex()}|{float(p[1]).hex()}" for p in positions])
    canonical_order = sorted(range(len(demands)),
                             key=lambda u: (keys[u], float(demands[u]).hex()))
    patterns, hashes = [], []
    for mode in range(4):
        coefficients = np.empty(len(demands), dtype=np.float64)
        for u, key in enumerate(keys):
            digest = hashlib.sha256(f"s1_variable_r11_v1|{mode}|{key}".encode("utf-8")).digest()
            coefficients[u] = 2.0 * (int.from_bytes(digest[:8], "big") / 2 ** 64) - 1.0
        perturbed = demands * (1.0 + cfg.robust_perturbation * coefficients)
        total = float(np.sum(demands[canonical_order]))
        if total:
            perturbed *= total / float(np.sum(perturbed[canonical_order]))
        rows = sorted((key, float(original).hex(), float(value).hex())
                      for key, original, value in zip(keys, demands, perturbed))
        payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        hashes.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())
        patterns.append(perturbed)
    return patterns, hashes


def _objective(scene, layout: Layout, cfg: VariableConfig, *, route: str,
               previous: Layout | None = None) -> float:
    m = metrics(scene, layout, cfg)
    if not m["coverage_100"]:
        return 1e9 + m["uncovered_demand_bps"] / cfg.reference_capacity_bps + len(m["uncovered_aircraft_indices"])
    loads = np.asarray(m["beam_demand_bps"])
    if len(loads):
        ratio = loads / cfg.reference_capacity_bps
        pressure = float(np.mean(np.maximum(ratio - 1, 0) ** 2))
        balance = float(np.std(ratio) ** 2)
    else:
        pressure = balance = 0.0
    value = pressure + 0.1 * balance + 0.02 * len(loads) / cfg.max_beams
    if route == "R05":
        # Wider footprint is a planning penalty; physical gain/interference is
        # assessed downstream under this separately named idealized version.
        value += float(np.mean(layout.widths ** 2)) * 0.05 if len(loads) else 0
    if route == "R11":
        patterns, _ = _robust_demand_patterns(scene, cfg)
        if len(layout.assignment):
            scenario_values = []
            for perturbed in patterns:
                p_load = np.bincount(layout.assignment, weights=perturbed, minlength=len(loads))
                scenario_values.append(float(np.mean(np.maximum(p_load / cfg.reference_capacity_bps - 1, 0) ** 2)))
            value += max(scenario_values)
    if route == "R12" and previous is not None:
        before_ids, after_ids = set(previous.beam_ids), set(layout.beam_ids)
        value += cfg.switch_beam_cost * len(before_ids ^ after_ids)
        shared = sorted(before_ids & after_ids)
        if shared:
            before_index = {s: i for i, s in enumerate(previous.beam_ids)}
            after_index = {s: i for i, s in enumerate(layout.beam_ids)}
            rays_now = satellite_rays_numpy(layout.centers[[after_index[s] for s in shared]], cfg.profile)
            rays_before = satellite_rays_numpy(previous.centers[[before_index[s] for s in shared]], cfg.profile)
            value += cfg.switch_pointing_cost * float(np.linalg.norm(rays_now - rays_before, axis=1).sum())
        if previous.aircraft_ids and layout.aircraft_ids:
            old_owner = {u: previous.beam_ids[b] if b >= 0 else None
                         for u, b in zip(previous.aircraft_ids, previous.assignment)}
            new_owner = {u: layout.beam_ids[b] if b >= 0 else None
                         for u, b in zip(layout.aircraft_ids, layout.assignment)}
            shared_users = old_owner.keys() & new_owner.keys()
            value += cfg.switch_assignment_cost * sum(old_owner[u] != new_owner[u] for u in shared_users)
        elif len(layout.assignment) == len(previous.assignment):
            old_owner = [previous.beam_ids[b] if b >= 0 else None for b in previous.assignment]
            new_owner = [layout.beam_ids[b] if b >= 0 else None for b in layout.assignment]
            value += cfg.switch_assignment_cost * sum(a != b for a, b in zip(old_owner, new_owner))
    return value


def _lns_step(scene, layout: Layout, cfg: VariableConfig, budget: dict):
    """Destroy three adjacent beams and rebuild their integral user region."""
    if len(layout.centers) < 3 or budget["geometry_calls"] >= cfg.max_geometry_calls:
        return layout, {"accepted": False, "reason": "lns_unavailable", "geometry_calls": 0}
    positions, demands = _scene_arrays(scene, cfg)
    _, loads = _check(scene, layout, cfg)
    anchor = int(np.argmax(loads))
    rays = satellite_rays_numpy(layout.centers, cfg.profile)
    distances = ((rays - rays[anchor]) ** 2).sum(axis=1)
    group = np.argsort(distances, kind="stable")[:3]
    members = np.flatnonzero(np.isin(layout.assignment, group))
    if not len(members):
        return layout, {"accepted": False, "reason": "lns_empty_region", "geometry_calls": 0}
    sites = np.unique(positions[members], axis=0)
    if len(sites) > cfg.max_initial_candidates:
        return layout, {"accepted": False, "reason": "lns_candidate_limit", "geometry_calls": 0}
    local = _legal_matrix(positions[members], sites,
                          np.full(len(sites), cfg.fixed_width_deg), cfg)
    budget["geometry_calls"] += 1
    remaining = np.ones(len(members), dtype=bool)
    chosen = []
    while np.any(remaining) and len(chosen) <= 4:
        cover = local & remaining[:, None]
        mass = (cover * demands[members, None]).sum(axis=0)
        score = cover.sum(axis=0) + np.minimum(mass / cfg.reference_capacity_bps, 1)
        if chosen:
            score[chosen] = -1
        s = int(np.argmax(score))
        if score[s] <= 0:
            break
        chosen.append(s)
        remaining &= ~local[:, s]
    if np.any(remaining) or not 1 <= len(chosen) <= 4 or len(layout.centers) - 3 + len(chosen) > cfg.max_beams:
        return layout, {"accepted": False, "reason": "lns_rebuild_failed", "geometry_calls": 1}
    keep = np.flatnonzero(~np.isin(np.arange(len(layout.centers)), group))
    centers = np.vstack((layout.centers[keep], sites[chosen]))
    widths = np.concatenate((layout.widths[keep], np.full(len(chosen), cfg.fixed_width_deg)))
    owner = _assign(positions, demands, centers, widths, cfg, mode="balanced")
    next_id = layout.next_beam_serial
    beam_ids = tuple(layout.beam_ids[b] for b in keep) + tuple(f"b{next_id + i}" for i in range(len(chosen)))
    proposal = _compact(Layout(centers, widths, owner, layout.aircraft_ids,
                               beam_ids, next_id + len(chosen)))
    budget["geometry_calls"] += 1
    try:
        valid = metrics(scene, proposal, cfg)["coverage_100"]
    except ValueError:
        valid = False
    if not valid:
        return layout, {"accepted": False, "reason": "lns_full_validation_failed", "geometry_calls": 2}
    return proposal, {"accepted": True, "reason": "lns_region_rebuilt", "geometry_calls": 2,
                      "destroyed_beams": group.tolist(), "rebuilt_beams": len(chosen)}


def _solve_impl(scene, cfg: VariableConfig, route: str, previous: Layout | None = None):
    """Run a deterministic, budgeted route and return layout plus full audit.

    R08/R09 learning policies and R10 downstream feedback live in separate
    modules.  This dispatcher does not claim those distinct methods were run.
    """
    if route not in CORE_ROUTES:
        raise ValueError(f"{route} is not implemented by the non-learning core")
    start = perf_counter()
    layout, init = initial_layout(scene, cfg, route, previous)
    budget = {"geometry_calls": int(init.get("geometry_calls", 0)),
              "max_geometry_calls": cfg.max_geometry_calls}
    log = []
    best = layout
    best_score = _objective(scene, best, cfg, route=route, previous=previous)

    if route == "R02":
        # Candidate-set local integer selection: remove a selected beam only
        # if all of its indivisible users can be assigned to the remaining set.
        for b in reversed(range(len(layout.centers))):
            if budget["geometry_calls"] >= cfg.max_geometry_calls:
                break
            positions, demands = _scene_arrays(scene, cfg)
            centers = np.delete(layout.centers, b, axis=0)
            widths = np.delete(layout.widths, b)
            owner = _assign(positions, demands, centers, widths, cfg, mode="balanced")
            budget["geometry_calls"] += 1
            keep_ids = tuple(s for i, s in enumerate(layout.beam_ids) if i != b)
            trial = _compact(Layout(centers, widths, owner, layout.aircraft_ids,
                                    keep_ids, layout.next_beam_serial))
            good = metrics(scene, trial, cfg)["coverage_100"]
            log.append({"operation": "set_cover_column_delete", "beam": b, "accepted": good})
            if good:
                layout = trial
        best = layout
        best_score = _objective(scene, best, cfg, route=route)
    elif route == "R03":
        # Demand assignment is optimized with centres held exactly fixed.
        log.append({"operation": "integral_demand_reassignment",
                    "accepted": init.get("reassignment") != "skipped_geometry_budget"})
    elif route == "R06":
        # Power-price OT relaxation and integer repair happen in initialization.
        transport = init.get("transport", {})
        log.append({"operation": "mass_transport_lp_then_integer_repair",
                    "accepted": transport.get("fallback") is None,
                    "solver_status": transport.get("status")})
    elif route == "R07":
        for _ in range(cfg.max_actions):
            trial, record = _lns_step(scene, layout, cfg, budget)
            if not record["accepted"]:
                log.append(record)
                break
            score = _objective(scene, trial, cfg, route=route)
            record["accepted"] = score < best_score - 1e-10
            if record["accepted"]:
                layout = best = trial
                best_score = score
            log.append(record)
            if not record["accepted"]:
                break
    else:
        # R01 capacity split, R04 split/merge, R05 width search, R11 robust
        # scenario search, R12 warm-start search have distinct menus/objectives.
        for _ in range(cfg.max_actions):
            if budget["geometry_calls"] >= cfg.max_geometry_calls:
                break
            menu = candidates(scene, layout, cfg,
                              mode="width" if route == "R05" else "local", budget=budget)
            operations = ({"split"} if route == "R01" else
                          {"split", "merge"} if route == "R04" else
                          {"resize", "merge", "split"} if route == "R05" else
                          {"split", "merge", "assign", "move"})
            selected = None
            selected_score = best_score
            selected_record = None
            for action, _ in menu:
                if action.operation not in operations:
                    continue
                if route == "R01" and action.operation == "split":
                    load = metrics(scene, layout, cfg)["beam_demand_bps"][action.beams[0]]
                    if load <= cfg.reference_capacity_bps:
                        continue
                trial, record = apply_action(scene, layout, cfg, action, budget)
                log.append(record)
                if record["accepted"]:
                    score = _objective(scene, trial, cfg, route=route, previous=previous)
                    if score < selected_score - 1e-10:
                        selected, selected_score, selected_record = trial, score, record
                if budget["geometry_calls"] >= cfg.max_geometry_calls:
                    break
            if selected is None:
                break
            layout = best = selected
            best_score = selected_score
            selected_record["selected_for_commit"] = True
    final = metrics(scene, best, cfg)
    status = "geometry_feasible" if final["coverage_100"] else "failed_full_coverage"
    audit = _GEOMETRY_AUDIT.get()
    route_details = {}
    if route == "R11":
        _, hashes = _robust_demand_patterns(scene, cfg)
        route_details["robust_perturbation_sha256_by_mode"] = hashes
        route_details["robust_perturbation_identity"] = "aircraft_id_or_stable_coordinate"
    if route == "R12":
        old = set(previous.beam_ids) if previous is not None else set()
        new = set(best.beam_ids)
        route_details.update(created_beam_ids=sorted(new - old),
                             retired_beam_ids=sorted(old - new),
                             warm_start_fallback=bool(init.get("warm_start_fallback", False)),
                             switch_policy="switch_cost_only_no_hysteresis_v1",
                             hysteresis_implemented=False)
    return best, {"route": route, "status": status,
                  "status_scope": "exact_geometry_and_unique_aircraft_assignment_only",
                  "failure_reason": None if final["coverage_100"] else init.get("failure_reason", "search_budget_exhausted"),
                  "initialization": init, "metrics": final, "objective": best_score,
                  "geometry_calls": audit["matrix_calls"] if audit is not None else budget["geometry_calls"],
                  "ray_pair_checks": audit["ray_pair_checks"] if audit is not None else None,
                  "geometry_budget_charges": budget["geometry_calls"],
                  "action_attempts": len(log),
                  "accepted_actions": sum(bool(row.get("selected_for_commit", False)) for row in log),
                  "action_log": log, "elapsed_s": perf_counter() - start,
                  "output_kind": "best_certified_layout", **route_details}


def solve(scene, cfg: VariableConfig, route: str, previous: Layout | None = None):
    """Exact matrix-call audit wraps the budgeted search, including final checks."""
    audit = {"matrix_calls": 0, "ray_pair_checks": 0}
    token = _GEOMETRY_AUDIT.set(audit)
    try:
        return _solve_impl(scene, cfg, route, previous)
    finally:
        _GEOMETRY_AUDIT.reset(token)


def _rebalance(scene, layout, cfg, *, transport=False):
    positions, demands = _scene_arrays(scene, cfg)
    legal = _legal_matrix(positions, layout.centers, layout.widths, cfg)
    if not len(layout.centers):
        return layout
    assignment = layout.assignment.copy()
    loads = np.bincount(assignment[assignment >= 0], weights=demands[assignment >= 0],
                        minlength=len(layout.centers)).astype(float)
    counts = np.bincount(assignment[assignment >= 0], minlength=len(layout.centers))
    # R06 uses a power-diagram dual price to approximate demand-mass OT, then
    # returns to indivisible aircraft and legal footprints for integer repair.
    prices = np.zeros(len(layout.centers))
    rays = satellite_rays_numpy(positions, cfg.profile)
    beams = satellite_rays_numpy(layout.centers, cfg.profile)
    distance = ((rays[:, None] - beams[None, :]) ** 2).sum(axis=2)
    for _ in range(5 if transport else 1):
        changed = False
        for u in np.argsort(-demands, kind="stable"):
            options = np.flatnonzero(legal[u])
            if not len(options):
                continue
            old = assignment[u]
            if old >= 0:
                loads[old] -= demands[u]
            cost = np.maximum((loads[options] + demands[u]) / cfg.reference_capacity_bps - 1, 0) ** 2
            cost += (prices[options] if transport else 0) + distance[u, options]
            new = int(options[np.argmin(cost)])
            if old >= 0 and new != old and counts[old] == 1:
                new = int(old)  # Keep every fixed centre occupied.
            loads[new] += demands[u]
            if new != old:
                if old >= 0:
                    counts[old] -= 1
                counts[new] += 1
            assignment[u] = new
            changed |= new != old
        if transport:
            prices += 0.1 * (loads / cfg.reference_capacity_bps - np.mean(loads / cfg.reference_capacity_bps))
        if not changed:
            break
    return _compact(Layout(layout.centers, layout.widths, assignment, layout.aircraft_ids,
                           layout.beam_ids, layout.next_beam_serial))


def _initial_layout_impl(scene, cfg: VariableConfig, route: str = "R01", previous: Layout | None = None):
    positions, demands = _scene_arrays(scene, cfg)
    if route == "R09":
        layout = Layout.empty(len(positions))
        layout.aircraft_ids = tuple(getattr(scene, "aircraft_ids", ()))
        return layout, {"status": "constructing", "initialization": "empty",
                        "geometry_calls": 0, "metrics": metrics(scene, layout, cfg)}
    if route not in CORE_ROUTES and route not in ("R08", "R10"):
        raise ValueError("unsupported initializer route")
    warm_reject = None
    if route == "R12" and previous is not None:
        if not previous.aircraft_ids and len(previous.assignment) != len(positions):
            raise ValueError("warm start without aircraft IDs needs identical aircraft count/order")
        if np.any(np.abs(previous.widths - cfg.fixed_width_deg) > 1e-12):
            warm_reject = "previous_width_physics_differs_from_fixed_r12"
        else:
            layout = Layout(previous.centers, previous.widths, np.full(len(positions), -1),
                            tuple(getattr(scene, "aircraft_ids", ())), previous.beam_ids,
                            previous.next_beam_serial)
            # Re-associate the new timestamp; old per-user assignments are not
            # copied when flights arrive, depart or change input row order.
            layout.assignment = _assign(positions, demands, layout.centers, layout.widths, cfg, mode="balanced")
            layout = _compact(layout)
            warm_metrics = metrics(scene, layout, cfg)
            if warm_metrics["coverage_100"]:
                old_ids, new_ids = set(previous.beam_ids), set(layout.beam_ids)
                return layout, {"status": "geometry_feasible", "initialization": "previous_revalidated",
                                "status_scope": "exact_geometry_and_unique_aircraft_assignment_only",
                                "geometry_calls": 1, "metrics": warm_metrics,
                                "warm_start_fallback": False, "created_beam_ids": sorted(new_ids - old_ids),
                                "retired_beam_ids": sorted(old_ids - new_ids),
                                "switch_policy": "switch_cost_only_no_hysteresis_v1",
                                "hysteresis_implemented": False}
            warm_reject = "previous_layout_does_not_cover_current_aircraft"
    method = "set_cover" if route == "R02" else "capacity_cluster"
    width = 0.7 if route == "R05" else cfg.fixed_width_deg
    layout, info = _construct(scene, cfg, method, width)
    if route == "R12":
        if previous is not None:
            # Rebuilt pointing has no certified continuity with an old beam.
            # Retire its identities and allocate fresh IDs monotonically.
            start_serial = previous.next_beam_serial
            new_ids = tuple(f"b{start_serial + i}" for i in range(len(layout.centers)))
            layout = Layout(layout.centers, layout.widths, layout.assignment,
                            layout.aircraft_ids, new_ids, start_serial + len(new_ids))
        old_ids = set(previous.beam_ids) if previous is not None else set()
        new_ids = set(layout.beam_ids)
        info.update(warm_start_fallback=previous is not None,
                    warm_start_fallback_reason=warm_reject,
                    created_beam_ids=sorted(new_ids - old_ids),
                    retired_beam_ids=sorted(old_ids - new_ids),
                    switch_policy="switch_cost_only_no_hysteresis_v1",
                    hysteresis_implemented=False)
    if route == "R02":
        if info["geometry_calls"] < cfg.max_geometry_calls:
            layout, mip = _set_cover_mip(scene, layout, cfg)
            info["mip"] = mip
            info["geometry_calls"] += 1
        else:
            info["mip"] = {"status": "skipped_geometry_budget", "fallback": "greedy_seed"}
    if route == "R03":
        if info["geometry_calls"] < cfg.max_geometry_calls:
            layout = _rebalance(scene, layout, cfg)
            info["geometry_calls"] += 1
        else:
            info["reassignment"] = "skipped_geometry_budget"
    if route == "R06" and metrics(scene, layout, cfg)["coverage_100"]:
        if info["geometry_calls"] < cfg.max_geometry_calls:
            layout, transport = _transport_lp(scene, layout, cfg)
            info["transport"] = transport
            info["geometry_calls"] += 1
        else:
            info["transport"] = {"status": "skipped_geometry_budget", "fallback": "integral_seed"}
    result = metrics(scene, layout, cfg)
    info.update(status="geometry_feasible" if result["coverage_100"] else "budget_or_candidate_failure",
                status_scope="exact_geometry_and_unique_aircraft_assignment_only",
                initialization=method, failure_reason=None if result["coverage_100"] else
                info["construction_termination"],
                metrics=result)
    return layout, info


def initial_layout(scene, cfg: VariableConfig, route: str = "R01", previous: Layout | None = None):
    """Initialize with exact GEO ray-call audit and explicit route method."""
    audit = _GEOMETRY_AUDIT.get()
    own = audit is None
    token = None
    if own:
        audit = {"matrix_calls": 0, "ray_pair_checks": 0}
        token = _GEOMETRY_AUDIT.set(audit)
    before_calls = audit["matrix_calls"]
    before_pairs = audit["ray_pair_checks"]
    try:
        layout, info = _initial_layout_impl(scene, cfg, route, previous)
        actual_calls = audit["matrix_calls"] - before_calls
        info["actual_geometry_calls"] = actual_calls
        info["actual_ray_pair_checks"] = audit["ray_pair_checks"] - before_pairs
        info["geometry_budget_charges"] = max(info.get("geometry_calls", 0), actual_calls)
        info["geometry_calls"] = actual_calls
        return layout, info
    finally:
        if own:
            _GEOMETRY_AUDIT.reset(token)
