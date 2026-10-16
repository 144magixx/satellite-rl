"""FLOW-AO: exact load assignment alternating with geometric structure moves."""

from dataclasses import dataclass
from time import perf_counter
import warnings

import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow

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


def _at_peak(centers, members, peak, radius):
    neighbors = neighbor_lists(centers, members, radius)
    if any(not row for row in neighbors):
        return None
    aircraft = np.repeat(np.arange(len(members)), [len(row) for row in neighbors])
    beam = np.concatenate(neighbors).astype(np.int64)
    return _flow(len(members), len(centers), aircraft, beam, peak)


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
           eligible_mask=None, radius=.5):
    """Exact minimum peak; optional integral convex-cost balance at that peak.

    ``eligible_mask`` has aircraft rows and beam columns. ``prev_owner`` uses
    the current beam column indices, with -1 for aircraft without a prior beam.
    """
    n, m = len(member_rays), len(centers_rays)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if m == 0:
        return None
    neighbors = neighbor_lists(centers_rays, member_rays, radius)
    if eligible_mask is not None:
        neighbors = [[j for j in row if eligible_mask[i, j]] for i, row in enumerate(neighbors)]
    if any(not row for row in neighbors):
        return None
    aircraft = np.repeat(np.arange(n), [len(row) for row in neighbors])
    beam = np.concatenate(neighbors).astype(np.int64)
    distances = np.linalg.norm(member_rays[aircraft] - centers_rays[beam], axis=1)
    nearest = np.array([row[np.argmin(np.linalg.norm(centers_rays[row] - member_rays[i], axis=1))]
                        for i, row in enumerate(neighbors)])
    low, high = int(np.ceil(n / m)), int(np.bincount(nearest, minlength=m).max())
    best = nearest
    while low < high:
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
    elif mode == "peak":
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
    cost = distances * (.1 / (max(n, 1) * 2 * np.sin(np.deg2rad(radius) / 2)))
    if prev_owner is not None and kappa:
        cost += kappa * ((prev_owner[aircraft] >= 0) & (beam != prev_owner[aircraft]))
    bounds = [(0, 1)] * e + [slot_bounds] * s
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected.*", category=OptimizeWarning)
        result = linprog(np.concatenate((cost, slot_cost)), A_eq=equality,
                         b_eq=np.concatenate((np.ones(n), np.zeros(m))), bounds=bounds,
                         method="highs", options={"threads": 1})
    if not result.success:
        return None
    owner = np.empty(n, dtype=np.int64)
    chosen = result.x[:e] > .5
    owner[aircraft[chosen]] = beam[chosen]
    return owner


def initial_cover(member_rays, radius=.5, rng=None):
    """Greedy set cover using every aircraft ray as a candidate center."""
    if len(member_rays) == 0:
        return np.empty((0, 3))
    neighbors = neighbor_lists(member_rays, member_rays, radius)
    uncovered = np.ones(len(member_rays), dtype=bool)
    order = np.arange(len(member_rays)) if rng is None else rng.permutation(len(member_rays))
    chosen = []
    while uncovered.any():
        scores = np.array([np.count_nonzero(uncovered[row]) for row in neighbors])
        index = order[np.argmax(scores[order])]
        chosen.append(index)
        uncovered[neighbors[index]] = False
    return np.asarray(member_rays)[chosen].copy()


def _nearest(centers, members, radius):
    if len(members) == 0:
        return np.empty(0, dtype=np.int64)
    distance = angles(members, centers)
    owner = distance.argmin(axis=1)
    # Coincident centers are legal when spacing is disabled. Spread exact
    # nearest-distance ties so a valid split is not discarded as an empty beam.
    ties = distance <= distance[np.arange(len(members)), owner, None] + 1e-12
    loads = np.zeros(len(centers), dtype=int)
    for i, row in enumerate(ties):
        choices = np.flatnonzero(row)
        owner[i] = choices[loads[choices].argmin()]
        loads[owner[i]] += 1
    return owner if np.all(distance[np.arange(len(members)), owner] <= radius + 1e-9) else None


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
        return centers
    for _ in range(10):
        owner = np.linalg.norm(members[:, None] - centers[None], axis=2).argmin(axis=1)
        if len(np.unique(owner)) < 2:
            return None
        updated = np.array([cone(members[owner == j])[0] for j in (0, 1)])
        if np.max(np.linalg.norm(updated - centers, axis=1)) < 1e-10:
            break
        centers = updated
    return centers


def repair_spacing(layout, member_rays, N_budget, fixed_width=None):
    """Resolve conflicting beam pairs; return None if the local repairs fail.

    All replacements use fresh arrays. Existing beam IDs survive deletion and
    movement, so callers can simply retain their previous layout on failure.
    """
    radius = .5 if fixed_width is None else fixed_width / 2
    current = layout
    seen = set()
    for _ in range(max(8, 4 * len(layout.widths))):
        if validate_layout(current, member_rays, N_budget, True, fixed_width):
            return current
        centers = rays(current.centers_latlon)
        distance = angles(centers, centers)
        required = np.maximum(current.widths[:, None], current.widths[None, :]) / 2
        violations = np.triu(distance + 1e-8 < required, 1)
        pairs = np.argwhere(violations)
        if not len(pairs):
            return None
        i, j = pairs[0]
        loads = np.bincount(current.owner, minlength=len(centers))
        if loads[i] < loads[j]:
            i, j = j, i
        pair_id = (int(current.beam_ids[i]), int(current.beam_ids[j]))
        if pair_id in seen:
            return None
        seen.add(pair_id)
        peak = int(loads.max())
        combined = (current.owner == i) | (current.owner == j)
        center, half = cone(member_rays[combined])
        if half <= radius + 1e-9 and loads[i] + loads[j] <= peak:
            merged = centers.copy()
            merged[i] = center
            owner = current.owner.copy()
            owner[owner == j] = i
            current = _layout(merged, member_rays, owner, current.beam_ids, fixed_width)
            continue
        remaining = np.delete(centers, j, axis=0)
        owner = _at_peak(remaining, member_rays, peak, radius)
        if owner is not None and np.bincount(owner).max() <= peak:
            current = _layout(remaining, member_rays, owner, np.delete(current.beam_ids, j), fixed_width)
            continue
        # Move the lighter center away on the great circle through this pair.
        tangent = centers[j] - (centers[j] @ centers[i]) * centers[i]
        norm = np.linalg.norm(tangent)
        if norm < 1e-12:
            tangent = np.cross(centers[i], np.eye(3)[np.argmin(abs(centers[i]))])
            norm = np.linalg.norm(tangent)
        tangent /= norm
        moved = False
        for extra in (1e-7, .02, .05, .1):
            target = np.deg2rad(required[i, j] + extra)
            candidate = centers.copy()
            candidate[j] = centers[i] * np.cos(target) + tangent * np.sin(target)
            owner = _at_peak(candidate, member_rays, peak, radius)
            if owner is not None and np.bincount(owner).max() <= peak:
                proposal = _layout(candidate, member_rays, owner, current.beam_ids, fixed_width)
                # Envelope changes with reassignment. Accept only if this pair
                # is now separated; subsequent pairs are processed next.
                new_centers = rays(proposal.centers_latlon)
                a = np.flatnonzero(proposal.beam_ids == current.beam_ids[i])
                b = np.flatnonzero(proposal.beam_ids == current.beam_ids[j])
                if not len(a) or not len(b) or angles(new_centers[a], new_centers[b])[0, 0] + 1e-8 >= max(proposal.widths[a[0]], proposal.widths[b[0]]) / 2:
                    current, moved = proposal, True
                    break
        if not moved:
            return None
    return None


def _accept(layout, members, budget, spacing, fixed):
    if spacing:
        return repair_spacing(layout, members, budget, fixed)
    return layout if validate_layout(layout, members, budget, False, fixed) else None


def _add_beams(layout, members, budget, spacing, fixed, next_id, rng=None,
               nearest=False, deadline=np.inf):
    radius = .5 if fixed is None else fixed / 2
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
            proposed = np.concatenate((centers.copy(), split[1:]))
            proposed[j] = split[0]
            owner = _nearest(proposed, members, radius) if nearest else assign(proposed, members, radius=radius)
            if owner is None:
                continue
            candidate = _layout(proposed, members, owner, np.append(layout.beam_ids, next_id), fixed)
            candidate = _accept(candidate, members, budget, spacing, fixed)
            if candidate is not None and len(candidate.widths) > len(layout.widths):
                next_id += 1
                layout, accepted = candidate, True
                break
            if perf_counter() >= deadline:
                break
        if not accepted:
            break
    return layout, next_id


def _single(frame, N_budget, spacing, effort, deadline, fixed_width,
            init_layout, balance, max_iters, rng, nearest=False, preserve_N=False):
    start = perf_counter()
    members = rays(frame.positions)
    radius = .5 if fixed_width is None else fixed_width / 2
    if len(members) == 0:
        empty = Layout(np.empty((0, 2)), np.empty(0), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
        return Result(empty, 0, True, 0., 0., 0., perf_counter() - start)
    centers = initial_cover(members, radius, rng) if init_layout is None else rays(init_layout.centers_latlon)
    N0 = len(centers)
    if N0 > N_budget:
        return Result(None, N0, False, perf_counter() - start, 0., 0., perf_counter() - start)
    ids = np.arange(N0) if init_layout is None else init_layout.beam_ids.copy()
    owner = _nearest(centers, members, radius) if nearest else assign(centers, members, radius=radius)
    if owner is None:
        return Result(None, N0, False, perf_counter() - start, 0., 0., perf_counter() - start)
    layout = _layout(centers, members, owner, ids, fixed_width, recenter=True)
    layout = _accept(layout, members, N_budget, spacing, fixed_width)
    if preserve_N and init_layout is not None and (layout is None or len(layout.widths) != N0):
        layout = init_layout
    if layout is None:
        return Result(None, N0, False, perf_counter() - start, 0., 0., perf_counter() - start)
    next_id = int(ids.max()) + 1
    layout, next_id = _add_beams(layout, members, N_budget, spacing, fixed_width,
                                 next_id, rng, nearest, deadline)
    init_end = perf_counter()
    best, stale = layout, 0
    for _ in range(0 if nearest else (30 * effort if max_iters is None else max_iters)):
        if perf_counter() >= deadline:
            break
        centers = rays(layout.centers_latlon)
        owner = assign(centers, members, radius=radius)
        if balance:
            owner = assign(centers, members, mode="balance", radius=radius)
        proposal = _layout(centers, members, owner, layout.beam_ids, fixed_width, recenter=True)
        candidate = _accept(proposal, members, N_budget, spacing, fixed_width)
        if candidate is not None and (not preserve_N or len(candidate.widths) == len(layout.widths)):
            layout = candidate
        # Delete a dispensable light beam, then reuse its slot on the busiest.
        centers = rays(layout.centers_latlon)
        loads = np.bincount(layout.owner, minlength=len(centers))
        for j in np.argsort(loads, kind="stable"):
            if len(centers) <= 1 or perf_counter() >= deadline:
                break
            reduced = np.delete(centers, j, axis=0)
            owner = _at_peak(reduced, members, int(loads.max()), radius)
            if owner is None or np.bincount(owner).max() > loads.max():
                continue
            proposal = _layout(reduced, members, owner, np.delete(layout.beam_ids, j), fixed_width, recenter=True)
            proposal = _accept(proposal, members, N_budget, spacing, fixed_width)
            if proposal is None:
                continue
            proposal, next_id = _add_beams(proposal, members, min(N_budget, len(layout.widths)),
                                          spacing, fixed_width, next_id, rng, False, deadline)
            if _key(proposal) < _key(layout) and (not preserve_N or len(proposal.widths) == len(layout.widths)):
                layout = proposal
            break
        if _key(layout) < _key(best):
            best, stale = layout, 0
        else:
            stale += 1
        if stale >= 3:
            break
    opt_end = perf_counter()
    feasible = validate_layout(best, members, N_budget, spacing, fixed_width)
    end = perf_counter()
    return Result(best if feasible else None, N0, feasible, init_end - start,
                  opt_end - init_end, end - opt_end, end - start)


def solve(frame, N_budget, spacing=True, effort=1, time_limit=None,
          fixed_width=None, init_layout=None, balance=True, max_iters=None,
          preserve_N=False):
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
