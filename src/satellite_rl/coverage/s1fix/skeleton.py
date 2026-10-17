"""FLOW-AO: exact load assignment alternating with geometric structure moves."""

from dataclasses import dataclass
from time import perf_counter
import warnings

import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.spatial import cKDTree

from .geom import (Layout, angles, cone, envelope, metrics, neighbor_lists,
                   ray_to_latlon, rays, validate_layout)


@dataclass
class Result:
    layout: Layout | None
    N0: int
    feasible: bool
    t_init: float
    t_opt: float
    t_check: float
    t_total: float
    over_cap_beams: int = 0
    reserve: int = 0


def _flow(n, m, aircraft, beam, peak):
    source, sink = n + m, n + m + 1
    rows = np.concatenate((np.full(n, source), aircraft, n + np.arange(m)))
    cols = np.concatenate((np.arange(n), n + beam, np.full(m, sink)))
    capacity = np.concatenate((np.ones(n + len(aircraft), dtype=np.int64),
                               np.full(m, peak, dtype=np.int64)))
    graph = coo_matrix((capacity, (rows, cols)), shape=(sink + 1, sink + 1)).tocsr()
    flow = maximum_flow(graph, source, sink)
    if flow.flow_value != n:
        return None
    assigned = flow.flow[:n, n:n + m].tocoo()
    positive = assigned.data > 0
    owner = np.empty(n, dtype=np.int64)
    owner[assigned.row[positive]] = assigned.col[positive]
    return owner


def coverage_radii(centers, spacing=True, fixed_width=None, stack_id=None):
    if fixed_width is not None:
        return np.full(len(centers), fixed_width / 2)
    if not spacing or len(centers) < 2:
        return np.full(len(centers), .5)
    distance = angles(centers, centers)
    np.fill_diagonal(distance, np.inf)
    if stack_id is not None:
        stacks = np.asarray(stack_id)
        distance[(stacks[:, None] >= 0) & (stacks[:, None] == stacks[None, :])] = np.inf
    return np.minimum(.5, distance.min(axis=1))


def centers_separated(centers, spacing=True, fixed_width=None):
    distance = angles(centers, centers)
    np.fill_diagonal(distance, np.inf)
    minimum = (.15 if fixed_width is None else fixed_width / 2) if spacing else 1e-10
    return bool(np.all(distance >= minimum - (1e-9 if spacing else 0)))


def _tolerant(radii):
    """Eligibility slack must stay below validate_layout's 1e-8 spacing slack."""
    return np.asarray(radii) + 1e-9


def _edges(centers, members, radii):
    rows = cKDTree(members).query_ball_point(
        centers, 2 * np.sin(np.deg2rad(_tolerant(radii)) / 2))
    aircraft = np.concatenate(rows).astype(np.int64)
    beam = np.repeat(np.arange(len(centers)), [len(row) for row in rows])
    order = np.lexsort((beam, aircraft))
    return aircraft[order], beam[order]


def _at_peak(centers, members, peak, radii, per_beam_cap=None):
    aircraft, beam = _edges(centers, members, radii)
    if not np.bincount(aircraft, minlength=len(members)).all():
        return None
    return _flow(len(members), len(centers), aircraft, beam,
                 peak if per_beam_cap is None else min(peak, per_beam_cap))


def _nonempty(owner, aircraft, beam, m):
    """Fill usable empty centers by alternating paths without raising the peak."""
    owner = owner.copy()
    loads = np.bincount(owner, minlength=m)
    adjacency = coo_matrix((np.ones(len(beam)), (aircraft, beam)),
                           shape=(len(owner), m)).tocsc()
    for empty in np.flatnonzero(loads == 0):
        queue = [int(empty)]
        parent = {int(empty): None}
        filled = False
        for target in queue:
            for a in adjacency.indices[adjacency.indptr[target]:adjacency.indptr[target + 1]]:
                donor = int(owner[a])
                if donor in parent:
                    continue
                if loads[donor] == 1:
                    parent[donor] = (target, a)
                    queue.append(donor)
                    continue
                owner[a] = target
                loads[donor] -= 1
                loads[target] += 1
                while target != empty:
                    previous, a = parent[target]
                    owner[a] = previous
                    loads[target] -= 1
                    loads[previous] += 1
                    target = previous
                filled = True
                break
            if filled:
                break
    return owner


def assign(centers_rays, member_rays, mode="peak", prev_owner=None, kappa=0.,
           eligible_mask=None, radii=None, per_beam_cap=None, objective=None,
           current_peak=None, deadline=None):
    """Exact minimum peak; optional integral convex-cost balance at that peak.

    ``eligible_mask`` has aircraft rows and beam columns. ``prev_owner`` uses
    the current beam column indices, with -1 for aircraft without a prior beam.
    """
    n, m = len(member_rays), len(centers_rays)
    if deadline is not None and perf_counter() >= deadline:
        return None
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if m == 0:
        return None
    radii = np.full(m, .5) if radii is None else np.asarray(radii)
    aircraft, beam = _edges(centers_rays, member_rays, radii)
    if eligible_mask is not None:
        keep = eligible_mask[aircraft, beam]
        aircraft, beam = aircraft[keep], beam[keep]
    if not np.bincount(aircraft, minlength=n).all():
        return None
    distances = np.linalg.norm(member_rays[aircraft] - centers_rays[beam], axis=1)
    order = np.lexsort((beam, distances, aircraft))
    nearest = beam[order[np.r_[True, np.diff(aircraft[order]) != 0]]]
    low, high = int(np.ceil(n / m)), int(np.bincount(nearest, minlength=m).max())
    best = nearest
    if objective == "narrow":
        mode = "narrow"
    if mode == "narrow":
        low = high if current_peak is None else int(current_peak)
        if per_beam_cap is not None: low = min(low, per_beam_cap)
        best = _flow(n, m, aircraft, beam, low)
        if best is None: return None
    else:
        if per_beam_cap is not None:
            high = min(high, per_beam_cap)
            best = _flow(n, m, aircraft, beam, high)
            if best is None: return None
        while low < high:
            if deadline is not None and perf_counter() >= deadline: return None
            middle = (low + high) // 2
            candidate = _flow(n, m, aircraft, beam, middle)
            if candidate is None:
                low = middle + 1
            else:
                high, best = middle, candidate
        if np.bincount(best, minlength=m).max() > low:
            best = _flow(n, m, aircraft, beam, low)
    if mode == "peak" and not (kappa and prev_owner is not None):
        return _nonempty(best, aircraft, beam, m)

    # Network LP: assignment edges followed by unit-capacity marginal load arcs.
    # Total unimodularity gives integer solutions without integer programming.
    e = len(beam)
    if mode == "balance":
        slot_beam = np.repeat(np.arange(m), low)
        slot_cost = np.tile(2 * np.arange(1, low + 1) - 1, m).astype(float)
        slot_bounds = (0, 1)
    elif mode in ("peak", "narrow"):
        slot_beam = np.arange(m)
        slot_cost = np.zeros(m)
        slot_bounds = (0, low)
    else:
        raise ValueError(f"Unknown assignment mode: {mode}")
    s = len(slot_beam)
    rows = np.concatenate((aircraft, n + beam, n + slot_beam))
    cols = np.concatenate((np.arange(e), np.arange(e), e + np.arange(s)))
    values = np.concatenate((np.ones(e), np.ones(e), -np.ones(s)))
    equality = coo_matrix((values, (rows, cols)), shape=(n + m, e + s)).tocsr()
    # Across all n choices the angular tie-break contributes strictly less than 1.
    cost = distances * (.1 / (max(n, 1) * 2 * np.sin(np.deg2rad(.5) / 2)))
    if mode == "narrow":
        cost = np.rad2deg(2 * np.arcsin(np.minimum(distances / 2, 1)))
    if prev_owner is not None and kappa:
        cost += kappa * ((prev_owner[aircraft] >= 0) & (beam != prev_owner[aircraft]))
    bounds = [(0, 1)] * e + [slot_bounds] * s
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected.*", category=OptimizeWarning)
        options = {"threads": 1}
        if deadline is not None:
            if perf_counter() >= deadline: return None
            options["time_limit"] = max(.001, deadline - perf_counter())
        result = linprog(np.concatenate((cost, slot_cost)), A_eq=equality,
                         b_eq=np.concatenate((np.ones(n), np.zeros(m))), bounds=bounds,
                         method="highs", options=options)
    if not result.success:
        return None
    owner = np.empty(n, dtype=np.int64)
    chosen = result.x[:e] > .5
    owner[aircraft[chosen]] = beam[chosen]
    return owner


def initial_cover(member_rays, radius=.5, rng=None):
    """Greedy radius net: only uncovered aircraft can become new centers."""
    if len(member_rays) == 0:
        return np.empty((0, 3))
    neighbors = neighbor_lists(member_rays, member_rays, radius)
    uncovered = np.ones(len(member_rays), dtype=bool)
    order = np.arange(len(member_rays)) if rng is None else rng.permutation(len(member_rays))
    chosen = []
    while uncovered.any():
        scores = np.array([np.count_nonzero(uncovered[row]) for row in neighbors])
        candidates = order[uncovered[order]]
        index = candidates[np.argmax(scores[candidates])]
        chosen.append(index)
        uncovered[neighbors[index]] = False
    return np.asarray(member_rays)[chosen].copy()


def adaptive_net(member_rays, k, fixed_width=None):
    """Open sparse-region centers first, using each aircraft's kth-neighbor radius."""
    tree = cKDTree(member_rays)
    chord = tree.query(member_rays, k=[min(k + 1, len(member_rays))])[0][:, 0]
    radius = np.rad2deg(2 * np.arcsin(np.minimum(chord / 2, 1)))
    minimum, maximum = (.15, .5) if fixed_width is None else (fixed_width / 2, fixed_width / 2)
    radius = np.clip(radius, minimum, maximum)
    covered = np.zeros(len(member_rays), dtype=bool)
    chosen = []
    for index in np.argsort(-radius, kind="stable"):
        if not covered[index]:
            chosen.append(index)
            covered[tree.query_ball_point(member_rays[index],
                2 * np.sin(np.deg2rad(radius[index] + 1e-10) / 2))] = True
    return np.asarray(member_rays)[chosen].copy()


def _adaptive_centers(members, budget, fixed_width):
    low, high = 1, len(members)
    while low < high:
        middle = (low + high) // 2
        if len(adaptive_net(members, middle, fixed_width)) <= budget:
            high = middle
        else:
            low = middle + 1
    return adaptive_net(members, low, fixed_width)


def _nearest(centers, members, radii):
    if len(members) == 0:
        return np.empty(0, dtype=np.int64)
    distance = angles(members, centers)
    distance[distance > _tolerant(radii)] = np.inf
    if not np.isfinite(distance.min(axis=1)).all():
        return None
    return distance.argmin(axis=1)


def _layout(centers, members, owner, ids, fixed_width=None, recenter=False):
    keep = np.flatnonzero(np.bincount(owner, minlength=len(centers)))
    remap = np.full(len(centers), -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    owner, centers, ids = remap[owner], centers[keep].copy(), np.asarray(ids)[keep]
    if recenter:
        centers = np.array([cone(members[owner == j])[0] for j in range(len(centers))])
    widths = envelope(centers, members, owner) if fixed_width is None else np.full(len(centers), fixed_width)
    return Layout(ray_to_latlon(centers), widths, owner, ids)


def _key(layout):
    m = metrics(layout)
    return m["peak"], m["top10"], m["CV"]


def _split(members, rng=None):
    if len(members) < 2:
        return None
    first = 0 if rng is None else int(rng.integers(len(members)))
    first = np.linalg.norm(members - members[first], axis=1).argmax()
    second = np.linalg.norm(members - members[first], axis=1).argmax()
    centers = members[[first, second]].copy()
    if np.linalg.norm(centers[0] - centers[1]) < 1e-12:
        return None
    for _ in range(10):
        owner = np.linalg.norm(members[:, None] - centers[None], axis=2).argmin(axis=1)
        if len(np.unique(owner)) < 2:
            return None
        updated = np.array([cone(members[owner == j])[0] for j in (0, 1)])
        if np.max(np.linalg.norm(updated - centers, axis=1)) < 1e-10:
            break
        centers = updated
    return centers


def _spread_split(split, others, minimum):
    """Yield separated pairs along the children's outward great circle."""
    midpoint, tangent = split.sum(axis=0), split[1] - split[0]
    midpoint /= np.linalg.norm(midpoint)
    tangent /= np.linalg.norm(tangent)
    half = angles(split[:1], split[1:])[0, 0] / 2
    for extra in np.arange(0, .255, .005):
        theta = np.deg2rad(half + extra)
        candidate = split if extra == 0 else (midpoint * np.cos(theta)
                    + np.array([-1, 1])[:, None] * tangent * np.sin(theta))
        if (angles(candidate[:1], candidate[1:])[0, 0] >= minimum - 1e-9
                and np.all(angles(candidate, others) >= minimum - 1e-9)):
            yield candidate


def _accept(layout, members, budget, spacing, fixed):
    return layout if (centers_separated(rays(layout.centers_latlon), spacing, fixed)
                      and validate_layout(layout, members, budget, spacing, fixed)) else None


def _add_beams(layout, members, budget, spacing, fixed, next_id, rng=None,
               nearest=False, deadline=np.inf, per_beam_cap=None):
    while len(layout.widths) < budget and _key(layout)[0] > 1 and perf_counter() < deadline:
        centers = rays(layout.centers_latlon)
        loads = np.bincount(layout.owner, minlength=len(centers))
        order = np.argsort(-loads, kind="stable")
        if rng is not None:
            # Randomize among equal-load beams while keeping the heaviest first.
            order = np.lexsort((rng.random(len(loads)), -loads))
        accepted = False
        for j in order:
            split = _split(members[layout.owner == j], rng)
            if split is None:
                continue
            candidates = (_spread_split(split, np.delete(centers, j, axis=0),
                                          .15 if fixed is None else fixed / 2)
                          if spacing else [split])
            for split in candidates:
                if perf_counter() >= deadline:
                    break
                proposed = np.concatenate((centers.copy(), split[1:]))
                proposed[j] = split[0]
                if not spacing and not centers_separated(proposed, False):
                    continue
                radii = coverage_radii(proposed, spacing, fixed)
                if nearest:
                    owner = _nearest(proposed, members, radii)
                else:
                    owner = _at_peak(proposed, members, int(loads.max()), radii, per_beam_cap)
                    if owner is not None:
                        aircraft, beam = _edges(proposed, members, radii)
                        owner = _nonempty(owner, aircraft, beam, len(proposed))
                if owner is None or np.bincount(owner).max() > loads.max():
                    continue
                candidate = _layout(proposed, members, owner, np.append(layout.beam_ids, next_id), fixed)
                if len(candidate.widths) > len(layout.widths):
                    if not nearest:
                        owner = assign(proposed, members, radii=radii, per_beam_cap=per_beam_cap)
                        candidate = _layout(proposed, members, owner, np.append(layout.beam_ids, next_id), fixed)
                    next_id += 1
                    layout, accepted = candidate, True
                    break
            if accepted or perf_counter() >= deadline:
                break
        if not accepted:
            break
    return layout, next_id


def _single(frame, N_budget, spacing, effort, deadline, fixed_width,
            init_layout, balance, max_iters, rng, nearest=False, preserve_N=False, objective=None, per_beam_cap=None):
    def score(value):
        if objective == "narrow":
            center_rays = rays(value.centers_latlon)
            return float(angles(members,center_rays)[np.arange(len(members)),value.owner].sum())
        return _key(value)

    start = perf_counter()
    members = rays(frame.positions)
    if len(members) == 0:
        empty = Layout(np.empty((0, 2)), np.empty(0), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
        return Result(empty, 0, True, 0., 0., 0., perf_counter() - start)
    N0 = len(initial_cover(members))
    if objective is not None and perf_counter() >= deadline:
        return Result(None, N0, False, 0., 0., 0., perf_counter() - start)
    centers = (_adaptive_centers(members, N_budget, fixed_width) if init_layout is None
               else rays(init_layout.centers_latlon))
    if objective is not None and perf_counter() >= deadline:
        return Result(None, N0, False, 0., 0., 0., perf_counter() - start)
    initial_count = len(centers)
    if initial_count > N_budget:
        return Result(None, N0, False, perf_counter() - start, 0., 0., perf_counter() - start)
    ids = np.arange(initial_count) if init_layout is None else init_layout.beam_ids.copy()
    radii = coverage_radii(centers, spacing, fixed_width)
    owner = _nearest(centers, members, radii) if nearest else assign(centers, members, radii=radii, objective=objective, per_beam_cap=per_beam_cap,
            deadline=deadline if objective is not None else None)
    if owner is None:
        return Result(None, N0, False, perf_counter() - start, 0., 0., perf_counter() - start)
    layout = _layout(centers, members, owner, ids, fixed_width)
    layout = _accept(layout, members, N_budget, spacing, fixed_width)
    if preserve_N and init_layout is not None and (layout is None or len(layout.widths) != initial_count):
        layout = init_layout
    if layout is None:
        return Result(None, N0, False, perf_counter() - start, 0., 0., perf_counter() - start)
    next_id = int(ids.max()) + 1
    growth_budget = initial_count if preserve_N else N_budget
    if not nearest:
        layout, next_id = _add_beams(layout, members, growth_budget, spacing, fixed_width,
                                     next_id, rng, False, deadline, per_beam_cap)
    init_end = perf_counter()
    best, stale = layout, 0
    for _ in range(0 if nearest else (30 * effort if max_iters is None else max_iters)):
        if perf_counter() >= deadline:
            break
        centers = rays(layout.centers_latlon)
        radii = coverage_radii(centers, spacing, fixed_width)
        owner = assign(centers, members, radii=radii, objective=objective, per_beam_cap=per_beam_cap,
            current_peak=int(np.bincount(layout.owner).max()), deadline=deadline if objective is not None else None)
        if owner is not None and balance and objective != "narrow":
            owner = assign(centers, members, mode="balance", radii=radii, per_beam_cap=per_beam_cap)
        if owner is None:
            break
        assigned = _layout(centers, members, owner, layout.beam_ids, fixed_width)
        if (score(assigned) < score(layout)
                and (not preserve_N or len(assigned.widths) == len(layout.widths))):
            layout = assigned
            centers, owner = rays(layout.centers_latlon), layout.owner
        # Tighten one beam at a time: a blocked neighbor must not veto all moves.
        # Keeping the owners also admits load-neutral moves that enable a split.
        centers = rays(layout.centers_latlon)
        loads = np.bincount(layout.owner, minlength=len(centers))
        for j in np.argsort(-loads, kind="stable"):
            recentered = centers.copy()
            recentered[j] = cone(members[layout.owner == j])[0]
            candidate = _layout(recentered, members, layout.owner, layout.beam_ids, fixed_width)
            if _accept(candidate, members, N_budget, spacing, fixed_width) is not None:
                layout, centers = candidate, recentered
        owner = assign(centers, members, radii=coverage_radii(centers, spacing, fixed_width), objective=objective, per_beam_cap=per_beam_cap,
            current_peak=int(np.bincount(layout.owner).max()), deadline=deadline if objective is not None else None)
        if owner is not None:
            candidate = _layout(centers, members, owner, layout.beam_ids, fixed_width)
            if (score(candidate) < score(layout)
                    and (not preserve_N or len(candidate.widths) == len(layout.widths))):
                layout = candidate
        # Delete a dispensable light beam, then reuse its slot on the busiest.
        centers = rays(layout.centers_latlon)
        loads = np.bincount(layout.owner, minlength=len(centers))
        for j in np.argsort(loads, kind="stable"):
            if len(centers) <= 1 or perf_counter() >= deadline:
                break
            reduced = np.delete(centers, j, axis=0)
            owner = _at_peak(reduced, members, int(loads.max()),
                             coverage_radii(reduced, spacing, fixed_width), per_beam_cap)
            if owner is None or np.bincount(owner).max() > loads.max():
                continue
            proposal = _layout(reduced, members, owner, np.delete(layout.beam_ids, j), fixed_width)
            proposal = _accept(proposal, members, N_budget, spacing, fixed_width)
            if proposal is None:
                continue
            proposal, next_id = _add_beams(proposal, members, min(N_budget, len(layout.widths)),
                                          spacing, fixed_width, next_id, rng, False, deadline, per_beam_cap)
            if score(proposal) < score(layout) and (not preserve_N or len(proposal.widths) == len(layout.widths)):
                layout = proposal
            break
        previous_count = len(layout.widths)
        layout, next_id = _add_beams(layout, members, growth_budget, spacing, fixed_width,
                                     next_id, rng, False, deadline, per_beam_cap)
        if len(layout.widths) > previous_count or score(layout) < score(best):
            best, stale = layout, 0
        else:
            stale += 1
            if score(layout) == score(best):
                best = layout
        if stale >= 3:
            break
    opt_end = perf_counter()
    feasible = validate_layout(best, members, N_budget, spacing, fixed_width)
    end = perf_counter()
    return Result(best if feasible else None, N0, feasible, init_end - start,
                  opt_end - init_end, end - opt_end, end - start)


def solve(frame, N_budget, spacing=True, effort=1, time_limit=None,
          fixed_width=None, init_layout=None, balance=True, max_iters=None,
          preserve_N=False, per_beam_cap=None, objective=None, stacking=False):
    if per_beam_cap is not None or objective is not None or stacking:
        return _solve_v2(frame, N_budget, spacing, effort, time_limit, fixed_width,
                         init_layout, balance, max_iters, preserve_N, per_beam_cap, objective, stacking)
    start = perf_counter()
    deadline = np.inf if time_limit is None else start + time_limit
    results = []
    for restart in range(effort):
        if restart and perf_counter() >= deadline:
            break
        rng = None if restart == 0 else np.random.default_rng(51001 + restart)
        results.append(_single(frame, N_budget, spacing, effort, deadline, fixed_width,
                               init_layout, balance, max_iters, rng, preserve_N=preserve_N))
    feasible = [result for result in results if result.feasible]
    best = min(feasible, key=lambda result: _key(result.layout)) if feasible else results[0]
    return Result(best.layout, best.N0, best.feasible,
                  sum(result.t_init for result in results), sum(result.t_opt for result in results),
                  sum(result.t_check for result in results), perf_counter() - start)


def init_baseline(frame, N_budget, spacing=True, fixed_width=None, time_limit=None, **kwargs):
    deadline = np.inf if time_limit is None else perf_counter() + time_limit
    return _single(frame, N_budget, spacing, 1, deadline, fixed_width,
                   None, False, 0, None, nearest=True)


def _narrow_layout(layout, members, spacing, fixed_width, deadline, max_iters, cap):
    """Distance-cost flow followed by minimum enclosing-cone recentering."""
    for _ in range(30 if max_iters is None else max_iters):
        if perf_counter() >= deadline:
            break
        centers = rays(layout.centers_latlon)
        owner = assign(centers, members, radii=coverage_radii(centers, spacing, fixed_width),
                       per_beam_cap=cap, objective="narrow",
                       current_peak=int(np.bincount(layout.owner).max()), deadline=deadline)
        if owner is None:
            break
        candidate = _layout(centers, members, owner, layout.beam_ids, fixed_width)
        centers = rays(candidate.centers_latlon)
        for j in range(len(centers)):
            proposed = centers.copy()
            proposed[j] = cone(members[candidate.owner == j])[0]
            moved = _layout(proposed, members, candidate.owner, candidate.beam_ids, fixed_width)
            if _accept(moved, members, 200, spacing, fixed_width) is not None:
                candidate, centers = moved, proposed
        old_cost = angles(members, rays(layout.centers_latlon))[np.arange(len(members)), layout.owner].sum()
        new_cost = angles(members, rays(candidate.centers_latlon))[np.arange(len(members)), candidate.owner].sum()
        if new_cost >= old_cost - 1e-10:
            break
        layout = candidate
    return layout


def _stack_layout(layout, members, cap, budget):
    """Duplicate overloaded sites, preserving their exact geometry and eligibility."""
    loads = np.bincount(layout.owner, minlength=len(layout.widths))
    copies = np.maximum(1, (loads + cap - 1) // cap)
    if copies.sum() > budget:
        return layout, int(np.count_nonzero(loads > cap))
    source = np.repeat(np.arange(len(copies)), copies)
    centers = layout.centers_latlon[source]
    widths = layout.widths[source]
    stacks = np.full(len(source), -1, dtype=np.int64)
    owner = np.empty(len(members), dtype=np.int64)
    for j in range(len(copies)):
        columns = np.flatnonzero(source == j)
        aircraft = np.flatnonzero(layout.owner == j)
        # The local flow only contains the original beam's members and copies.
        assigned = _flow(len(aircraft), len(columns),
                         np.repeat(np.arange(len(aircraft)), len(columns)),
                         np.tile(np.arange(len(columns)), len(aircraft)), cap)
        owner[aircraft] = columns[assigned]
        if len(columns) > 1:
            stacks[columns] = j
    return Layout(centers, widths, owner, np.arange(len(source)), stacks), 0


def _solve_v2(frame, N_budget, spacing, effort, time_limit, fixed_width,
              init_layout, balance, max_iters, preserve_N, cap, objective, stacking):
    start = perf_counter()
    deadline = np.inf if time_limit is None else start + time_limit
    members = rays(frame.positions)
    reserves = range(0, min(50, N_budget - 1) + 1, 5) if stacking else (0,)
    last = None
    reserves = list(reserves)
    for trial, reserve in enumerate(reserves):
        if perf_counter() >= deadline: break
        trial_deadline = deadline if time_limit is None else perf_counter() + max(0., deadline - perf_counter()) / (len(reserves) - trial)
        base = _single(frame, N_budget - reserve, spacing, effort, trial_deadline, fixed_width,
                       init_layout, balance, max_iters, None, preserve_N=preserve_N, objective=objective,
                       per_beam_cap=None if stacking else cap)
        if base.layout is None:
            continue
        layout = base.layout
        if objective == "narrow":
            # Before stacking, an overloaded site must retain enough capacity.
            assignment_cap = cap if not stacking else None
            layout = _narrow_layout(layout, members, spacing, fixed_width,
                                    trial_deadline, max_iters, assignment_cap)
        if stacking:
            layout, over = _stack_layout(layout, members, cap, N_budget)
        else:
            over = int(np.count_nonzero(np.bincount(layout.owner) > cap)) if cap else 0
        if layout.stack_id is None:
            layout.stack_id = np.full(len(layout.widths), -1, dtype=np.int64)
        legal = validate_layout(layout, members, N_budget, spacing, fixed_width)
        if not legal: continue
        last = Result(layout, base.N0, legal,
                      base.t_init, perf_counter() - start - base.t_init, 0.,
                      perf_counter() - start, over, reserve)
        if legal and over == 0:
            return last
    if last is None:
        return Result(None, len(initial_cover(members)), False, 0., 0., 0., perf_counter() - start)
    return last
