"""Seven genuinely different bounded optimization routes and common BH.

All methods use the exact GEO ray gate and atomic Problem publication. SciPy
MILP solves a finite pool with binary aircraft ownership and a convex secant
upper approximation to squared load. Its certificate concerns that surrogate
only. OT is a row-conserving soft-quota transport relaxation, never a final
split-aircraft allocation. Graph edges are search hints, never footprint proofs.
"""
from __future__ import annotations

import hashlib
import json
from time import perf_counter

import numpy as np

from ..geometry import satellite_rays_numpy, squared_ray_limit, visible_numpy
from ..variable_design import Layout
from .core import BudgetStop, clone, layout_sha256, ray_center


def _new_layout(problem, centers, assignment, *, original=None):
    original = problem.current if original is None else original
    return Layout(centers, np.full(len(centers), 0.5), assignment,
                  tuple(getattr(problem.scene, "aircraft_ids", ())),
                  original.beam_ids if len(centers) == len(original.centers) else (),
                  original.next_beam_serial)


def center_candidates(problem, members, old=None, limit=None):
    """Real ray-centroid directions with checked line-search and boundary points.

    This is bounded feasible-center search, not an infeasibility certificate.
    """
    limit = problem.config.centers_per_beam if limit is None else limit
    positions = problem.positions[members]
    demands = problem.demands[members]
    candidates = []
    if old is not None:
        candidates.append(np.asarray(old))
    target = ray_center(positions, demands, problem.config.profile)
    plain = ray_center(positions, None, problem.config.profile)
    for center in (target, plain):
        if center is not None:
            candidates.append(center)
            if old is not None:
                candidates.extend(np.asarray(old) + f * (center - old) for f in (0.5, 0.25))
    if len(members):
        # The farthest member and highest demand member are geometric candidates.
        mid = positions.mean(axis=0)
        idx = int(np.argmax(np.sum((positions - mid) ** 2, axis=1)))
        candidates.extend((positions[idx], positions[int(np.argmax(demands))]))
    result, seen = [], set()
    for center in candidates:
        problem.guard()
        key = tuple(np.round(center, 11))
        if key in seen or not bool(visible_numpy(center, problem.config.profile)):
            continue
        seen.add(key)
        # Preserve candidate center, even when full group cannot fit: MILP may
        # reassign some members. Caller find_center requires all members fit.
        result.append(np.asarray(center).copy())
        if len(result) >= limit:
            break
    return result


def find_center(problem, members, old=None, limit=24):
    """Return a true whole-group footprint center or bounded-search failure."""
    pool = center_candidates(problem, members, old, min(limit, problem.config.centers_per_beam))
    if not pool:
        return None
    legal = problem.legal(pool, problem.positions[members])
    possible = np.flatnonzero(legal.all(axis=0))
    if not len(possible):
        return None
    # Largest minimum actual ray slack, deterministic tie by candidate order.
    br = satellite_rays_numpy(np.asarray(pool), problem.config.profile)
    d2 = ((problem.rays[members, None] - br[None]) ** 2).sum(axis=2)
    chosen = min(possible, key=lambda j: (float(d2[:, j].max()), int(j)))
    return np.asarray(pool[chosen]).copy()


def build_pool(problem, beams=None, expansion=0):
    """Keep each original site (including coincident sites), then bounded extras."""
    beams = tuple(range(problem.n)) if beams is None else tuple(beams)
    pool = [problem.current.centers[b].copy() for b in beams]
    sources = [{"kind": "original", "beam": b} for b in beams]
    seen = {tuple(np.round(center, 11)) for center in pool}
    maximum = min(problem.config.max_pool, len(beams) * problem.config.centers_per_beam)
    if len(pool) > maximum:
        raise ValueError("pool bound excludes required original sites")
    for b in beams:
        problem.guard()
        members = np.flatnonzero(problem.current.assignment == b)
        sites = center_candidates(problem, members, problem.current.centers[b])
        if expansion:
            nearby = np.argsort(np.sum((problem.positions - problem.current.centers[b]) ** 2, axis=1),
                                kind="stable")[:problem.config.centers_per_beam]
            sites.extend(problem.positions[nearby])
        added_for_beam = 0
        for center in sites:
            if len(pool) >= maximum or added_for_beam >= problem.config.centers_per_beam - 1:
                break
            key = tuple(np.round(center, 11))
            if key in seen:
                continue
            seen.add(key)
            pool.append(np.asarray(center).copy())
            added_for_beam += 1
            sources.append({"kind": "member_ray_centroid_or_boundary", "beam": b,
                            "expansion": expansion})
    return np.asarray(pool).reshape(-1, 2), sources


def region_pool(problem, region):
    """Teacher-before-label pool; first k slots match region beam order exactly."""
    return build_pool(problem, tuple(region["beams"]))[0]


def square_breakpoints(count, segments):
    """Common frozen nodes in load/scene-mean units, including the full range."""
    upper = max(float(count), 2.0)
    return np.unique(np.r_[np.linspace(0, 2, segments + 1),
                           np.geomspace(2, upper, segments + 1)])


def solve_pool(problem, centers, *, members=None, count=None, edge_mask=None,
               visible=None, time_limit=None, source="pool_milp"):
    """Finite-pool MILP: selected nonempty sites + unique indivisible ownership.

    ``visible`` is a whole-scene owner array with -1 masked. Nonnegative entries
    fix ownership to corresponding original site slots. ``edge_mask`` has local
    member-by-site shape and can only remove truly legal edges. No warm-start is
    claimed: SciPy's milp API has no incumbent injection.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    problem.guard()
    started = perf_counter()
    members = np.arange(len(problem.demands)) if members is None else np.asarray(members, dtype=int)
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    count = problem.n if count is None else int(count)
    legal = problem.legal(centers, problem.positions[members])
    if edge_mask is not None:
        mask = np.asarray(edge_mask, dtype=bool)
        if mask.shape != legal.shape:
            raise ValueError("edge mask shape differs from local candidate pool")
        legal &= mask
    if visible is not None:
        fixed = np.asarray(visible, dtype=int)[members]
        for row, owner in enumerate(fixed):
            if owner >= 0:
                if owner >= len(centers):
                    raise ValueError("visible ownership outside candidate pool")
                keep = bool(legal[row, owner])
                legal[row] = False
                legal[row, owner] = keep
    pool_sha = hashlib.sha256(centers.tobytes() + legal.tobytes()).hexdigest()
    if len(centers) < count or np.any(~legal.any(axis=1)):
        problem.trace.append({"source": source, "reason": "infeasible_pool_coverage",
                              "pool_sha256": pool_sha, "missing_members": members[~legal.any(axis=1)].tolist()})
        return None
    ei, ej = np.nonzero(legal)
    edge_count, site_count = len(ei), len(centers)
    y0, l0, z0 = edge_count, edge_count + site_count, edge_count + 2 * site_count
    variable_count = edge_count + 3 * site_count
    scale = max(float(problem.demands.sum()) / max(problem.n, 1), 1.0)
    demand = problem.demands[members] / scale
    upper_load = max(float(demand.sum()), 1.0)
    nodes = square_breakpoints(max(problem.n, upper_load), problem.config.square_segments)
    entries_r, entries_c, entries_v, lower, upper = [], [], [], [], []
    def add(columns, values, lo, hi):
        row = len(lower)
        entries_r.extend([row] * len(columns))
        entries_c.extend(columns)
        entries_v.extend(values)
        lower.append(lo)
        upper.append(hi)
    # Row conservation, aircraft cannot be split.
    for i in range(len(members)):
        indices = np.flatnonzero(ei == i).tolist()
        add(indices, [1.0] * len(indices), 1, 1)
    for e, j in enumerate(ej):
        add([e, y0 + int(j)], [1, -1], -np.inf, 0)
    for j in range(site_count):
        edges = np.flatnonzero(ej == j)
        add(edges.tolist() + [y0 + j], [1.0] * len(edges) + [-1], 0, np.inf)
        add(edges.tolist() + [l0 + j], demand[ei[edges]].tolist() + [-1], 0, 0)
        # z >= (left+right)*L - left*right*y for every secant.
        for left, right in zip(nodes[:-1], nodes[1:]):
            add([z0 + j, l0 + j, y0 + j], [1, -(left + right), left * right], 0, np.inf)
    add(list(range(y0, y0 + site_count)), [1.0] * site_count, count, count)
    matrix = coo_matrix((entries_v, (entries_r, entries_c)),
                        shape=(len(lower), variable_count)).tocsc()
    bounds_upper = np.r_[np.ones(edge_count + site_count),
                          np.full(site_count, upper_load), np.full(site_count, upper_load ** 2 + 1)]
    objective = np.r_[np.zeros(edge_count + 2 * site_count), np.ones(site_count)]
    integrality = np.r_[np.ones(edge_count + site_count), np.zeros(2 * site_count)].astype(int)
    problem.guard()
    available = min(problem.remaining(), problem.config.mip_seconds if time_limit is None else time_limit)
    if available <= 0:
        raise BudgetStop("online_time_limit")
    problem.calls["solver_calls"] += 1
    solver_start = perf_counter()
    result = milp(objective, integrality=integrality,
                  bounds=Bounds(np.zeros(variable_count), bounds_upper),
                  constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
                  options={"time_limit": available, "mip_rel_gap": problem.config.mip_relative_gap})
    solver_seconds = perf_counter() - solver_start
    problem.times["solver_seconds"] += solver_seconds
    record = {"source": source, "pool_sha256": pool_sha, "pool_centers": centers.tolist(),
              "sites": site_count, "binary_edges": edge_count, "variables": variable_count,
              "status_code": int(result.status), "message": str(result.message),
              "status": "no_incumbent", "model_gap": getattr(result, "mip_gap", None),
              "model_dual_bound": getattr(result, "mip_dual_bound", None),
              "model_objective": float(result.fun) if result.fun is not None else None,
              "square_breakpoints_load_over_scene_mean": nodes.tolist(),
              "square_max_error_bound_normalized": float(np.max(np.diff(nodes) ** 2) / 4),
              "time_limit_seconds": available, "solver_seconds": solver_seconds,
              "construction_and_solver_seconds": perf_counter() - started,
              "finished_after_deadline": perf_counter() > problem.deadline,
              "warm_start": False, "certificate_scope": "finite_pool_piecewise_square_surrogate_only"}
    problem.trace.append(record)
    if result.x is None:
        record["status"] = "infeasible_pool" if result.status == 2 else "no_incumbent"
        return None
    x = result.x[:edge_count]
    used = np.flatnonzero(result.x[y0:y0 + site_count] > 0.5)
    if len(used) != count or np.any(np.abs(x - np.rint(x)) > 1e-5):
        record["status"] = "invalid_incumbent"
        return None
    assignment = np.full(len(members), -1, dtype=int)
    for i, j, value in zip(ei, ej, x):
        if value > 0.5:
            if assignment[i] != -1:
                record["status"] = "invalid_duplicate_owner"
                return None
            assignment[i] = int(j)
    if np.any(assignment < 0) or not np.all(np.isin(assignment, used)):
        record["status"] = "invalid_incomplete_owner"
        return None
    if any(not np.any(assignment == j) for j in used):
        record["status"] = "invalid_empty_site"
        return None
    remap = np.full(site_count, -1, dtype=int)
    remap[used] = np.arange(count)
    loads = np.bincount(remap[assignment], weights=problem.demands[members], minlength=count) / scale
    record.update(status="optimal_surrogate" if result.status == 0 else "feasible_timeout",
                  exact_square_normalized=float(loads @ loads),
                  square_approximation_error=float(result.fun - loads @ loads),
                  selected_sites=used.tolist(), raw_assignment=assignment.tolist())
    return centers[used].copy(), remap[assignment], record


def rebuild_region(problem, region, candidate_pool=None, visible=None, edge_mask=None):
    beams = tuple(region["beams"])
    if not 2 <= len(beams) <= 6 or any(b < 0 or b >= problem.n for b in beams):
        raise ValueError("region must contain 2..6 existing beams")
    members = np.flatnonzero(np.isin(problem.current.assignment, beams))
    if len(members) > 64:
        problem.trace.append({"source": "local_rebuild", "reason": "local_aircraft_limit",
                              "beams": list(beams), "aircraft": len(members)})
        return None
    pool = build_pool(problem, beams)[0] if candidate_pool is None else np.asarray(candidate_pool)
    local_visible = None
    if visible is not None:
        local_visible = np.asarray(visible).copy()
        mapping = {beam: i for i, beam in enumerate(beams)}
        for member in members:
            if local_visible[member] >= 0:
                local_visible[member] = mapping[int(local_visible[member])]
    result = solve_pool(problem, pool, members=members, count=len(beams), visible=local_visible,
                        edge_mask=edge_mask, source="local_rebuild_milp")
    if result is None:
        return None
    centers, assignment, record = result
    layout = clone(problem.current)
    layout.centers[list(beams)] = centers
    layout.assignment[members] = np.asarray(beams)[assignment]
    return layout


def repair_layout(problem, layout):
    """Common fixed-N exact integer repair; model proposals stay separately logged."""
    if layout is None or len(layout.centers) != problem.n:
        return None
    pool = np.asarray(layout.centers)
    valid = visible_numpy(pool, problem.config.profile)
    pool = np.where(valid[:, None], pool, problem.initial.centers)
    # Add original sites as backstop; a finite-pool incumbent is externally gated.
    pool = np.concatenate((pool, problem.initial.centers))
    result = solve_pool(problem, pool, source="common_repair_milp")
    if result is None:
        return None
    centers, assignment, _ = result
    return _new_layout(problem, centers, assignment)


def direct_rule_layout(problem):
    """Aircraft-only farthest-ray construction for 03 H/HR, independent of I0.

    Select N real aircraft seed directions by farthest-first demand weighting;
    preserve one anchor per site and greedily allocate remaining whole aircraft
    on legal edges. Uncovered proposals retain -1 and are never silently hidden.
    The common repair_layout is applied separately by HR/DR.
    """
    problem.guard()
    if not len(problem.demands):
        return Layout.empty(0)
    demand = problem.demands
    positive_mean = max(float(demand.mean()), 1.0)
    first = int(np.argmax(demand))
    selected = [first]
    distance = np.sum((problem.rays - problem.rays[first]) ** 2, axis=1)
    while len(selected) < problem.n:
        problem.guard()
        weights = 1 + demand / positive_mean
        score = distance * weights
        score[selected] = -np.inf
        following = int(np.argmax(score))
        selected.append(following)
        distance = np.minimum(distance, np.sum((problem.rays - problem.rays[following]) ** 2, axis=1))
    centers = problem.positions[selected].copy()
    legal = problem.legal(centers)
    assignment = np.full(len(demand), -1, dtype=int)
    assignment[selected] = np.arange(problem.n)
    loads = demand[selected].copy()
    br = problem.rays[selected]
    costs = ((problem.rays[:, None] - br[None]) ** 2).sum(axis=2)
    for user in np.lexsort((np.arange(len(demand)), -demand)):
        if assignment[user] >= 0:
            continue
        choices = np.flatnonzero(legal[user])
        if not len(choices):
            continue
        score = (loads[choices] + demand[user]) ** 2 - loads[choices] ** 2
        chosen = min(choices, key=lambda b: (float(score[np.flatnonzero(choices == b)[0]]),
                                            float(costs[user, b]), int(b)))
        assignment[user] = chosen
        loads[chosen] += demand[user]
    return _new_layout(problem, centers, assignment)


def run_bh(problem, *, centers=True):
    """Deterministic transfer/swap/center-replacement greedy from identical I0."""
    for round_index in range(problem.config.max_rounds):
        improved = False
        for action in problem.menu():
            problem.guard()
            if problem.commit(problem.preview(action)):
                improved = True
                break
        if not improved and centers:
            # A feasible center replacement alone cannot improve CV. Jointly
            # inspect its newly enabled best transfer under the same real gate.
            for b in range(problem.n):
                problem.guard()
                members = np.flatnonzero(problem.current.assignment == b)
                center = find_center(problem, members, problem.current.centers[b])
                if center is None or np.allclose(center, problem.current.centers[b], atol=1e-12):
                    continue
                internal = clone(problem.current)
                internal.centers[b] = center
                legal = problem.legal(internal.centers)
                loads = np.bincount(internal.assignment, weights=problem.demands, minlength=problem.n)
                for u in np.flatnonzero(legal[:, b] & (internal.assignment != b)):
                    a = int(internal.assignment[u])
                    if np.sum(internal.assignment == a) <= 1 or loads[a] <= loads[b] + problem.demands[u]:
                        continue
                    candidate = clone(internal)
                    candidate.assignment[u] = b
                    if problem.commit(problem.consider(candidate, "BH_center_then_transfer")):
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break


def run_demand_clustering(problem, *, move_centers=True):
    """06: binary demand assignment, ray-centroid center update, alternating."""
    stale = 0
    for iteration in range(problem.config.max_rounds):
        problem.guard()
        regions = problem.regions() if problem.n >= 2 else []
        improved = False
        for region in regions[:16]:
            problem.guard()
            beams = tuple(region["beams"])
            members = np.flatnonzero(np.isin(problem.current.assignment, beams))
            if len(members) > 64:
                problem.trace.append({"source": "06_local_assignment", "reason": "local_aircraft_limit",
                                      "aircraft": len(members), "beams": list(beams)})
                continue
            proposal = clone(problem.current)
            if move_centers:
                for b in beams:
                    group = np.flatnonzero(proposal.assignment == b)
                    center = find_center(problem, group, proposal.centers[b])
                    if center is not None:
                        proposal.centers[b] = center
            # Fixed count local demand model, outside squared loads are constant.
            result = solve_pool(problem, proposal.centers[list(beams)], members=members,
                                count=len(beams), source="06_demand_binary_assignment")
            if result is None:
                continue
            centers, assignment, _ = result
            proposal.centers[list(beams)] = centers
            proposal.assignment[members] = np.asarray(beams)[assignment]
            changed = problem.commit(problem.consider(proposal, "06_alternating_centroid_assignment"
                                     if move_centers else "06_fixed_centers_assignment_only"))
            improved |= changed
            if changed:
                break
        stale = 0 if improved else stale + 1
        if stale >= 2:
            break


def run_footprint_projection(problem):
    """07: proposed regrouping first, then find whole-group feasible centers."""
    stale = 0
    for iteration in range(problem.config.max_rounds):
        problem.guard()
        loads = np.bincount(problem.current.assignment, weights=problem.demands, minlength=problem.n)
        br = satellite_rays_numpy(problem.current.centers, problem.config.profile)
        distance = ((br[:, None] - br[None]) ** 2).sum(axis=2)
        proposals = []
        for a in np.argsort(-loads, kind="stable"):
            members = np.flatnonzero(problem.current.assignment == a)
            if len(members) <= 1:
                continue
            for b in np.argsort(distance[a], kind="stable")[:min(problem.n, 8)]:
                if b == a:
                    continue
                for u in members[np.argsort(-problem.demands[members], kind="stable")[:8]]:
                    delta = 2 * problem.demands[u] * (loads[b] - loads[a] + problem.demands[u])
                    if delta < -1e-6:
                        proposals.append((delta, int(u), int(a), int(b)))
        proposals.sort()
        improved = False
        for delta, u, a, b in proposals[:16]:
            problem.guard()
            candidate = clone(problem.current)
            candidate.assignment[u] = b
            success = True
            for beam in (a, b):
                members = np.flatnonzero(candidate.assignment == beam)
                center = find_center(problem, members, candidate.centers[beam])
                if center is None:
                    success = False
                    break
                candidate.centers[beam] = center
            if not success:
                problem.consider(None, "07_bounded_center_search_no_feasible_candidate")
                continue
            if problem.commit(problem.consider(candidate, "07_regroup_then_exact_footprint_projection")):
                improved = True
                break
        stale = 0 if improved else stale + 1
        if stale >= 2:
            break


def soft_quota_transport(demands, costs, legal, iterations=100, quota_penalty=4.0,
                         entropy=0.05, guard=None):
    """Projected exponentiated-gradient row-conserving soft-quota transport.

    Empty quota columns are allowed; impossible equal mass is never claimed
    feasible. Rows sum to whole demand, only legal edges receive mass.
    """
    demands = np.asarray(demands, dtype=float)
    costs = np.asarray(costs, dtype=float)
    legal = np.asarray(legal, dtype=bool)
    if legal.shape != costs.shape or costs.shape[0] != len(demands) or not legal.any(axis=1).all():
        raise ValueError("transport rows require legal edges")
    total = float(demands.sum())
    if total <= 0:
        return np.zeros_like(costs), {"iterations": 0, "all_zero_demand": True,
                                      "row_residual": 0.0, "quota_residual_l1": None}
    masses = demands / total
    quota = 1.0 / costs.shape[1]
    q = legal.astype(float)
    q *= masses[:, None] / q.sum(axis=1, keepdims=True)
    previous = q.copy()
    completed = 0
    for step in range(iterations):
        if guard:
            guard()
        gradient = costs + 2 * quota_penalty * (q.sum(axis=0) - quota)[None]
        gradient += entropy * np.log(np.maximum(q / np.maximum(masses[:, None], 1e-300), 1e-300))
        logits = np.log(np.maximum(q, 1e-300)) - 0.5 * gradient
        logits[~legal] = -np.inf
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.where(legal, np.exp(logits), 0.0)
        q = weights * (masses / weights.sum(axis=1))[:, None]
        completed = step + 1
        if np.max(np.abs(q - previous)) < 1e-10:
            break
        previous = q.copy()
    split = int(np.sum(np.sum(q > 1e-8, axis=1) > 1))
    return q * total, {"iterations": completed, "row_residual": float(np.max(np.abs(q.sum(axis=1) - masses))),
                       "quota_residual_l1": float(np.sum(np.abs(q.sum(axis=0) - quota))),
                       "soft_split_aircraft": split, "soft_objective": float(np.sum(costs * q) +
                           quota_penalty * np.sum((q.sum(axis=0) - quota) ** 2)),
                       "model": "row_conserving_entropy_soft_quota_v1"}


def integral_transport_assignment(demands, q, legal, original):
    """Soft-share guided indivisible rounding with original nonempty anchors."""
    demands, q, legal = np.asarray(demands), np.asarray(q), np.asarray(legal)
    n = legal.shape[1]
    assignment = np.asarray(original, dtype=int).copy()
    counts = np.bincount(assignment, minlength=n)
    loads = np.bincount(assignment, weights=demands, minlength=n)
    mu = max(float(demands.sum()) / max(n, 1), 1.0)
    for u in np.lexsort((np.arange(len(demands)), -demands)):
        a = assignment[u]
        if counts[a] <= 1:
            continue
        choices = np.flatnonzero(legal[u])
        # Soft confidence contributes only a proposal score; exact publication
        # recomputes full CV and cannot leak fractional demand into loads.
        delta = 2 * demands[u] * (loads[choices] - loads[a] + demands[u])
        delta[choices == a] = 0
        shares = q[u, choices] / max(float(demands[u]), 1.0)
        score = delta / (mu ** 2) - 0.05 * shares
        b = int(choices[np.argmin(score)])
        if b != a:
            assignment[u] = b
            counts[a] -= 1
            counts[b] += 1
            loads[a] -= demands[u]
            loads[b] += demands[u]
    return assignment


def run_ot(problem, *, method="main"):
    """08: soft quota mass partition -> indivisible rounding -> geometry repair."""
    for iteration in range(min(problem.config.max_rounds, 4)):
        problem.guard()
        legal = problem.legal()
        br = satellite_rays_numpy(problem.current.centers, problem.config.profile)
        costs = ((problem.rays[:, None] - br[None]) ** 2).sum(axis=2) / squared_ray_limit(0.25)
        if method == "main":
            q, record = soft_quota_transport(problem.demands, costs, legal,
                                             problem.config.ot_iterations, guard=problem.guard)
        elif method == "voronoi":
            nearest = np.argmin(np.where(legal, costs, np.inf), axis=1)
            q = np.zeros_like(costs)
            q[np.arange(len(problem.demands)), nearest] = problem.demands
            record = {"model": "demand_weighted_nearest_voronoi", "iterations": 0}
        elif method == "integer":
            result = solve_pool(problem, problem.current.centers, source="08_integer_only")
            if result is None:
                break
            _, assignment, _ = result
            q = np.zeros_like(costs)
            q[np.arange(len(problem.demands)), assignment] = problem.demands
            record = {"model": "integer_assignment_without_transport", "iterations": 0}
        else:
            raise ValueError("unknown OT method")
        problem.trace.append({"source": "08_transport", **record})
        assignment = integral_transport_assignment(problem.demands, q, legal, problem.current.assignment)
        centers = problem.current.centers.copy()
        for b in range(problem.n):
            members = np.flatnonzero(assignment == b)
            center = find_center(problem, members, centers[b])
            if center is not None:
                centers[b] = center
        candidate = _new_layout(problem, centers, assignment)
        if not problem.commit(problem.consider(candidate, "08_OT_integerized_geometric_candidate")):
            # Small exact integer repair on the same centers, separate from OT.
            result = solve_pool(problem, centers, source="08_integer_repair")
            if result is None:
                break
            centers, assignment, _ = result
            if not problem.commit(problem.consider(_new_layout(problem, centers, assignment),
                                                     "08_OT_then_integer_repair")):
                break


def graph_adjacency(problem, neighbours=12):
    distance = ((problem.rays[:, None] - problem.rays[None]) ** 2).sum(axis=2)
    edges = np.zeros(distance.shape, dtype=bool)
    for i in range(len(distance)):
        near = np.argsort(distance[i], kind="stable")[1:neighbours + 1]
        edges[i, near] = distance[i, near] <= squared_ray_limit(0.5) + 1e-14
    edges |= edges.T
    np.fill_diagonal(edges, False)
    return edges, distance


def run_graph(problem, *, method="main"):
    """09: graph-guided connected group growth; true common center hard gate."""
    problem.guard()
    adjacency, distance = graph_adjacency(problem)
    problem.trace.append({"source": "09_graph", "edges": int(adjacency.sum() // 2),
                          "neighbours": 12, "edge_scope": "search_hint_not_common_footprint_proof"})
    for iteration in range(problem.config.max_rounds):
        loads = np.bincount(problem.current.assignment, weights=problem.demands, minlength=problem.n)
        rows = []
        for u, v in zip(*np.nonzero(np.triu(adjacency, 1))):
            a, b = int(problem.current.assignment[u]), int(problem.current.assignment[v])
            if a == b:
                continue
            for aircraft, source, target in ((u, a, b), (v, b, a)):
                if np.sum(problem.current.assignment == source) <= 1:
                    continue
                demand = problem.demands[aircraft]
                delta = 2 * demand * (loads[target] - loads[source] + demand)
                # Frozen three-level attribution: edge distance, demand-aware
                # order, and exact common-footprint pruning before evaluation.
                priority = float(distance[u, v]) if method == "distance" else delta
                rows.append((priority, delta, float(distance[u, v]), int(aircraft), source, target))
        rows.sort()
        improved = False
        for priority, delta, _, u, a, b in rows[:min(60, problem.config.max_menu)]:
            problem.guard()
            if delta >= -1e-6:
                continue
            candidate = clone(problem.current)
            candidate.assignment[u] = b
            new_members = np.flatnonzero(candidate.assignment == b)
            if method in ("distance", "demand"):
                center = ray_center(problem.positions[new_members], problem.demands[new_members],
                                    problem.config.profile)
                # External full-scene gate rejects footprint failures. These
                # ablations omit geometry pruning, never the publication gate.
            else:
                center = find_center(problem, new_members, candidate.centers[b])
            if center is None:
                problem.consider(None, "09_graph_group_no_common_footprint_found")
                continue
            candidate.centers[b] = center
            if problem.commit(problem.consider(candidate, "09_graph_connected_region_growth")):
                improved = True
                break
        if not improved:
            break


def run_pool_mip(problem, *, expand=True):
    """10: whole-scene finite candidate pool, joint selected sites/ownership."""
    for expansion in range((problem.config.pool_expansions if expand else 0) + 1):
        problem.guard()
        centers, sources = build_pool(problem, expansion=expansion)
        problem.trace.append({"source": "10_pool_generation", "expansion": expansion,
                              "sources": sources, "pool_count": len(centers)})
        result = solve_pool(problem, centers, time_limit=problem.remaining() if not expand else
                            min(problem.remaining(), max(problem.config.mip_seconds, problem.remaining() / 2)),
                            source="10_joint_site_ownership_milp")
        if result is not None:
            centers, assignment, _ = result
            problem.commit(problem.consider(_new_layout(problem, centers, assignment),
                                           "10_externally_checked_incumbent"))


def run_lns(problem, *, selector="rule"):
    """11: rule region choice; exactly the same rebuilding kernel used by 05."""
    failures, tried = 0, set()
    rng = np.random.default_rng(problem.config.seed)
    while problem.available() and problem.calls["rebuilds"] < problem.config.max_rebuilds:
        regions = [row for row in problem.regions() if row["id"] not in tried]
        if not regions:
            break
        if selector == "random":
            region = regions[int(rng.integers(len(regions)))]
        else:
            if selector == "single":
                regions = [row for row in regions if len(row["beams"]) == 2]
                if not regions:
                    break
            regions.sort(key=lambda row: (-row["features"][3], -row["features"][7], row["id"]))
            region = regions[0]
        tried.add(region["id"])
        candidate = problem.rebuild(region)
        if problem.commit(problem.consider(candidate, "11_rule_region_rebuild:" + region["id"])):
            failures, tried = 0, set()
        else:
            failures += 1
        if failures >= 2:
            break


def dominates(a, b, tolerance=1e-10):
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a <= b + tolerance) and np.any(a < b - tolerance))


def nondominated_sort(objectives, tolerance=1e-10):
    """NSGA-II fronts; equal objectives never mutually dominate."""
    n = len(objectives)
    dominated, counts, first = [[] for _ in range(n)], np.zeros(n, dtype=int), []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objectives[i], objectives[j], tolerance):
                dominated[i].append(j)
            elif dominates(objectives[j], objectives[i], tolerance):
                counts[i] += 1
        if counts[i] == 0:
            first.append(i)
    fronts, current = [], first
    while current:
        fronts.append(current)
        following = []
        for i in current:
            for j in dominated[i]:
                counts[j] -= 1
                if counts[j] == 0:
                    following.append(j)
        current = following
    return fronts


def crowding_distance(objectives, front):
    distance = {index: 0.0 for index in front}
    if len(front) <= 2:
        return {index: np.inf for index in front}
    for axis in range(len(objectives[0])):
        order = sorted(front, key=lambda i: (objectives[i][axis], i))
        span = objectives[order[-1]][axis] - objectives[order[0]][axis]
        if span <= 1e-12:
            continue
        distance[order[0]] = distance[order[-1]] = np.inf
        for k in range(1, len(order) - 1):
            distance[order[k]] += (objectives[order[k + 1]][axis] -
                                   objectives[order[k - 1]][axis]) / span
    return distance


def canonical_sha(layout):
    """Deduplicate beam permutations, including coincident centers by members."""
    order = sorted(range(len(layout.centers)), key=lambda b:
                   (tuple(np.flatnonzero(layout.assignment == b)), tuple(layout.centers[b])))
    remap = np.empty(len(order), dtype=int)
    remap[order] = np.arange(len(order))
    value = {"centers": layout.centers[order].tolist(), "widths": layout.widths[order].tolist(),
             "assignment": remap[layout.assignment].tolist(), "aircraft_ids": layout.aircraft_ids}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run_evolution(problem, *, single_objective=False):
    """12: NSGA-II nondomination/crowding with closed-beam crossover/mutations."""
    rng = np.random.default_rng(problem.config.seed)
    base = {"layout": clone(problem.current), "metrics_all": problem.current_metrics,
            "physical": problem.current_physical, "accepted": True}
    population, seen = [base], {canonical_sha(base["layout"])}
    problem._evolution_population = population
    proposed = 0
    def objectives(row):
        return [row["metrics_all"]["load_cv_all"]] if single_objective else [
            row["metrics_all"]["load_cv_all"], -problem.carry(row["physical"])]
    if base["metrics_all"]["load_cv_all"] is None:
        return
    saved = clone(problem.current)
    for generation in range(problem.config.generations):
        offspring = []
        for k in range(problem.config.population):
            problem.guard()
            if proposed >= problem.config.evolution_proposals:
                break
            proposed += 1
            parent = population[int(rng.integers(len(population)))]
            child = clone(parent["layout"])
            # Crossover transfers complete beam groups only if both parent
            # aircraft sets are identical in those beam slots (closed region).
            if len(population) > 1 and rng.random() < 0.5:
                second = population[int(rng.integers(len(population)))]["layout"]
                beams = sorted(rng.choice(problem.n, min(2, problem.n), replace=False).tolist())
                ids = np.isin(child.assignment, beams)
                if np.array_equal(ids, np.isin(second.assignment, beams)):
                    child.centers[beams] = second.centers[beams]
                    child.assignment[ids] = second.assignment[ids]
            legal = problem.legal(child.centers)
            u = int(rng.integers(len(problem.demands)))
            choices = np.flatnonzero(legal[u])
            old = int(child.assignment[u])
            if len(choices) and np.sum(child.assignment == old) > 1:
                target = int(rng.choice(choices))
                if rng.random() < 0.5:
                    child.assignment[u] = target
                else:
                    others = np.flatnonzero((child.assignment == target) & legal[:, old])
                    if len(others):
                        v = int(rng.choice(others))
                        child.assignment[u], child.assignment[v] = child.assignment[v], child.assignment[u]
            if problem.n and rng.random() < 0.35:
                beam = int(rng.integers(problem.n))
                members = np.flatnonzero(child.assignment == beam)
                if len(members):
                    center = find_center(problem, members, child.centers[beam])
                    if center is not None:
                        child.centers[beam] = center
            sha = canonical_sha(child)
            if sha in seen:
                problem.consider(None, "12_duplicate_generation")
                continue
            seen.add(sha)
            trial = problem.consider(child, "12_generation", allow_nonimproving=True)
            if trial["accepted"]:
                offspring.append(trial)
                problem._evolution_population = population + offspring
        combined = population + offspring
        scores = [objectives(row) for row in combined]
        new_population = []
        for front in nondominated_sort(scores):
            distance = crowding_distance(scores, front)
            order = sorted(front, key=lambda i: (-distance[i], canonical_sha(combined[i]["layout"])))
            new_population.extend(combined[i] for i in order[:problem.config.population - len(new_population)])
            if len(new_population) >= problem.config.population:
                break
        population = new_population
        problem._evolution_population = population
        problem.trace.append({"source": "12_NSGA_II_generation", "generation": generation,
                              "proposals_total": proposed, "population": len(population),
                              "new_feasible_discoveries": len(offspring), "objectives": scores})
        if proposed >= problem.config.evolution_proposals or not problem.available():
            break
    publish_evolution(problem, population)


def publish_evolution(problem, population):
    """Publish fully validated known incumbents even if a later call hits budget."""
    if not population or population[0]["metrics_all"]["load_cv_all"] is None:
        return
    # Single fixed publish rule: lowest exact CV, then carry, then canonical SHA.
    selected = min(population, key=lambda row: (row["metrics_all"]["load_cv_all"],
                   -problem.carry(row["physical"]), canonical_sha(row["layout"])))
    if selected["metrics_all"]["load_cv_all"] <= problem.current_metrics["load_cv_all"] + 1e-10:
        problem.commit(selected)
    scores = [[row["metrics_all"]["load_cv_all"], -problem.carry(row["physical"])] for row in population]
    first = nondominated_sort(scores)[0]
    problem.pareto = [{"layout": population[i]["layout"].to_record(), "objectives": scores[i],
                       "N": problem.n, "scope": "finite_seen_candidates_fixed_N"} for i in first]


CLASSICAL_METHODS = {"BH": ("main",), "06": ("main", "assign_only"),
    "07": ("main", "assign_only"), "08": ("main", "voronoi", "integer"),
    "09": ("main", "distance", "demand", "geometric"), "10": ("main", "initial_pool"),
    "11": ("main", "random", "single"), "12": ("main", "single_objective")}


def run_classical(problem, route, method="main"):
    if method not in CLASSICAL_METHODS.get(route, ()):
        raise ValueError(f"unsupported non-DRL method {route}/{method}")
    if route == "BH":
        run_bh(problem)
    elif route == "06":
        run_demand_clustering(problem, move_centers=method != "assign_only")
    elif route == "07":
        run_bh(problem, centers=False) if method == "assign_only" else run_footprint_projection(problem)
    elif route == "08":
        run_ot(problem, method=method)
    elif route == "09":
        run_graph(problem, method=method)
    elif route == "10":
        run_pool_mip(problem, expand=method != "initial_pool")
    elif route == "11":
        run_lns(problem, selector="rule" if method == "main" else method)
    elif route == "12":
        try:
            run_evolution(problem, single_objective=method == "single_objective")
        finally:
            publish_evolution(problem, getattr(problem, "_evolution_population", []))
