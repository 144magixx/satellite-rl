"""Bounded lightweight task ranking followed by exact geometry decoding.

No exact candidate quality is computed before task ranking. A task changes a
complete local region; all its aircraft remain assigned, including zero-demand
and edge aircraft. Pareto tradeoffs never use cross-N normalized P ranking.
"""
from dataclasses import dataclass, field, replace
from itertools import islice, zip_longest
from functools import cmp_to_key
import hashlib
import json
from time import perf_counter
import numpy as np
from ..geometry import satellite_rays_numpy
from .config import GeometryConfig, SearchConfig
from .core import (Scene, Layout, legal_matrix, validate_layout, metrics,
                   initial_layout, minimum_enclosing_width, ray_center, ray_location, layout_sha256)


class BudgetStop(RuntimeError):
    pass


class Budget:
    """One monotonic wall/candidate/geometry ledger, including initialization."""
    def __init__(self, config=SearchConfig(), stop=None, clock=None):
        self.config = config
        self.clock = clock or perf_counter
        self.start = self.clock()
        self.deadline = self.start + config.seconds
        self.attempt_limit = config.max_attempts
        self.attempts = self.geometry_calls = self.ray_pair_checks = 0
        self.late_candidates = 0
        self.reason = None
        self.stop = stop

    @property
    def elapsed(self):
        return self.clock() - self.start

    def check(self):
        if self.stop is not None and self.stop():
            self.reason = "stop_requested"
        elif self.clock() >= self.deadline:
            self.reason = "time_budget"
        elif self.attempts >= self.attempt_limit:
            self.reason = "candidate_budget"
        elif self.geometry_calls >= self.config.max_geometry_calls:
            self.reason = "geometry_budget"
        else:
            return
        raise BudgetStop(self.reason)

    def geometry(self, pairs):
        self.check()
        self.geometry_calls += 1
        self.ray_pair_checks += int(pairs)

    def attempt(self):
        self.check()
        self.attempts += 1

    def record(self):
        return {"seconds": self.elapsed, "seconds_limit": self.config.seconds,
                "attempts": self.attempts, "candidate_limit": self.config.max_attempts,
                "geometry_calls": self.geometry_calls, "ray_pair_checks": self.ray_pair_checks,
                "late_candidates": self.late_candidates, "stop_reason": self.reason,
                "overrun_seconds": max(0., self.elapsed - self.config.seconds)}


@dataclass(frozen=True)
class Task:
    kind: str
    region_ids: tuple
    target_k: int
    widths: tuple
    width_policy: str = "menu"
    variant: int = 0
    task_id: str = ""
    cheap_features: dict = field(default_factory=dict, compare=False)
    parent_layout_sha256: str = ""
    width_mode: str = ""
    preference_index: int = 0

    def __post_init__(self):
        object.__setattr__(self, "region_ids", tuple(int(x) for x in self.region_ids))
        object.__setattr__(self, "widths", tuple(float(x) for x in self.widths))
        if (not self.region_ids or len(set(self.region_ids)) != len(self.region_ids)
                or min(self.region_ids) < 0 or self.target_k < 1 or len(self.widths) != self.target_k
                or self.width_policy not in ("menu", "shrink") or self.variant < 0):
            raise ValueError("invalid finite region task")
        if not self.task_id:
            record = {k: getattr(self, k) for k in ("kind", "region_ids", "target_k", "widths", "width_policy", "variant")}
            object.__setattr__(self, "task_id", hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest())
        if not self.width_mode:
            mode = str(self.widths[0]) if len(set(self.widths)) == 1 else "mixed"
            object.__setattr__(self, "width_mode", mode)

    def to_record(self):
        return {"kind": self.kind, "region_ids": list(self.region_ids), "target_k": self.target_k,
                "widths": list(self.widths), "width_policy": self.width_policy, "variant": self.variant,
                "task_id": self.task_id, "cheap_features": self.cheap_features,
                "parent_layout_sha256": self.parent_layout_sha256, "width_mode": self.width_mode,
                "preference_index": self.preference_index}

    @classmethod
    def from_record(cls, record):
        return cls(**record)


@dataclass
class DecodeResult:
    layout: Layout | None
    reason: str
    details: dict
    candidates: list = field(default_factory=list)

    @property
    def status(self):
        if self.reason in ("time_budget", "candidate_budget", "geometry_budget", "stop_requested", "region_user_limit"):
            return "censored"
        return "completed" if self.layout is not None else "no_feasible_found_under_pool"

    @property
    def metrics(self):
        return self.details.get("best_metrics")

    @property
    def budget_delta(self):
        return {k:self.details.get(k,0) for k in ("seconds", "attempts", "geometry_calls", "ray_pair_checks")}

    def to_record(self):
        return {"layout": self.layout.to_record() if self.layout is not None else None,
                "status": self.status, "metrics": self.metrics, "budget_delta": self.budget_delta,
                "reason": self.reason, "details": self.details, "candidates": [
                    {**x, "layout": x["layout"].to_record()} for x in self.candidates]}


def _regions(scene, current, config, budget=None):
    n = len(current.centers)
    loads = np.bincount(current.assignment, weights=scene.demands, minlength=n)
    brays = satellite_rays_numpy(current.centers, scene.profile)
    if budget is not None:
        budget.geometry(len(scene.demands))
    limits = 4 * np.sin(np.deg2rad(current.widths / 2) / 2) ** 2
    slack = limits[current.assignment] - np.sum((scene.rays - brays[current.assignment]) ** 2, axis=1)
    tight = [float(slack[current.assignment == b].min()) for b in range(n)]
    counts = np.bincount(current.assignment, minlength=n)
    groups = [sorted(range(n), key=lambda b: (-loads[b], current.beam_ids[b])),
              sorted(range(n), key=lambda b: (tight[b], current.beam_ids[b])),
              sorted(range(n), key=lambda b: (counts[b], loads[b], current.beam_ids[b])),
              sorted(range(n), key=lambda b: (current.centers[b, 0], current.centers[b, 1], current.beam_ids[b]))]
    anchors = list(dict.fromkeys(b for batch in zip(*groups) for b in batch))
    yielded = set()
    # Interleaving sizes prevents low task caps from hiding larger regions.
    for rank in range(len(anchors)):
        anchor = anchors[rank]
        nearest = sorted(range(n), key=lambda b: (float(np.sum((brays[b] - brays[anchor]) ** 2)), current.beam_ids[b]))
        for size in config.region_sizes:
            if size > n:
                continue
            region = tuple(sorted(nearest[:size]))
            if region in yielded:
                continue
            yielded.add(region)
            yield region
            if len(yielded) >= config.max_regions:
                return


def _tasks_for_region(scene, current, region, config, geometry):
    members = np.flatnonzero(np.isin(current.assignment, region))
    k, n = len(region), len(current.centers)
    loads = np.bincount(current.assignment, weights=scene.demands, minlength=n)
    subloads = loads[list(region)]
    cheap = {"region_size": k, "region_users": len(members), "region_request": float(scene.demands[members].sum()),
             "region_peak": float(subloads.max()), "region_mean": float(subloads.mean()),
             "region_cv": float(subloads.std() / subloads.mean()) if subloads.mean() else 0.,
             "current_widths": current.widths[list(region)].tolist(), "current_N": n,
             "region_user_fraction": len(members) / max(1, len(scene.demands)),
             "region_overflow": len(members) > config.max_region_users}
    # k-1/k/k+1 are interleaved by width; finite tasks have no decoder outcomes.
    for width in geometry.width_menu_deg:
        for target in (k - 1, k, k + 1):
            if target < 1 or target > len(members) or n - k + target > geometry.max_beams:
                continue
            for policy in ("menu", "shrink"):
                kind = "merge" if target < k else "split" if target > k else "joint_reconstruct"
                yield Task(kind, region, target, (width,) * target, policy,
                           cheap_features={**cheap, "target_k": target, "proposed_width": width},
                           parent_layout_sha256=layout_sha256(current))
    # An explicit mixed-width option, not only post hoc width shrinking.
    if k > 1:
        yield Task("joint_reconstruct", region, k,
                   (geometry.width_min_deg,) + (geometry.width_max_deg,) * (k - 1), "menu",
                   cheap_features={**cheap, "target_k": k, "proposed_width": geometry.width_max_deg},
                   parent_layout_sha256=layout_sha256(current))


def generate_tasks(scene, current, config=SearchConfig(), geometry=GeometryConfig(), budget=None):
    """Yield cheap tasks. Region requests/geometry do not use teacher metrics."""
    regions = list(_regions(scene, current, config, budget))
    streams = []
    for i, region in enumerate(regions):
        tasks = list(_tasks_for_region(scene, current, region, config, geometry))
        # Deterministic diagonal starts expose width/count alternatives in a
        # finite first pool rather than hiding all broad/mixed tasks at its end.
        shift = (i * len(tasks) // max(1, len(regions))) if tasks else 0
        streams.append(iter(tasks[shift:] + tasks[:shift]))
    for group in zip_longest(*streams):
        for task in group:
            if task is not None:
                yield task


def _site_pool(scene, current, region, members, config, budget):
    budget.check()
    sites = [current.centers[b].copy() for b in region]
    mean = ray_center(scene, members)
    if mean is not None:
        sites.append(mean)
    average = scene.rays[members].mean(0)
    average /= np.linalg.norm(average)
    boundary = sorted(members.tolist(), key=lambda u: (-float(np.sum((scene.rays[u] - average) ** 2)), scene.aircraft_ids[u]))[:8]
    # Boundary-ray midpoint sites can cover a pair whose original sites cannot.
    for i, u in enumerate(boundary):
        for v in boundary[i + 1:]:
            budget.check()
            center = ray_location(scene, scene.rays[u] + scene.rays[v])
            if center is not None:
                sites.append(center)
            if len(sites) >= config.max_centers_per_task:
                break
        if len(sites) >= config.max_centers_per_task:
            break
    for u in sorted(members.tolist(), key=lambda i: scene.aircraft_ids[i]):
        if len(sites) >= config.max_centers_per_task:
            break
        sites.append(scene.positions[u].copy())
    distinct = []
    for center in sites:
        if not any(np.array_equal(center, old) for old in distinct):
            distinct.append(center)
    return np.asarray(distinct[:config.max_centers_per_task])


def _seed_centers(scene, current, region, members, target, variant, config, sites):
    # Alternate current sites with farthest-point aircraft sites; all finite.
    if target == 1:
        return sites[variant % len(sites)][None]
    if variant % 2 == 0:
        old = list(region)
        shift = (variant // 2) % len(old)
        old = old[shift:] + old[:shift]
        centers = [current.centers[b].copy() for b in old[:target]]
    else:
        centers = [sites[(len(region) + variant // 2) % len(sites)].copy()]
    while len(centers) < target:
        crays = satellite_rays_numpy(np.asarray(centers), scene.profile)
        distance = np.sum((scene.rays[members, None] - crays[None]) ** 2, axis=-1).min(1)
        selected = min(range(len(members)), key=lambda i: (-distance[i], scene.aircraft_ids[members[i]]))
        proposed = scene.positions[members[selected]]
        if any(np.array_equal(proposed, center) for center in centers):
            return None
        centers.append(proposed.copy())
    return np.asarray(centers)


def _assign_region(scene, members, centers, widths, config, geometry, budget, variant):
    legal = legal_matrix(scene, centers, widths, geometry, budget)[members]
    if not legal.any(1).all():
        return None
    k = len(centers)
    owners = np.full(len(members), -1)
    loads = np.zeros(k)
    # Geometrically constrained aircraft first, then requests. Identity ties.
    order = sorted(range(len(members)), key=lambda i: (int(legal[i].sum()),
                   -scene.demands[members[i]] if variant % 2 == 0 else scene.demands[members[i]], scene.aircraft_ids[members[i]]))
    for i in order:
        budget.check()
        allowed = np.flatnonzero(legal[i])
        owner = min(allowed, key=lambda b: (loads[b], (int(b) - variant) % k))
        owners[i] = owner
        loads[owner] += scene.demands[members[i]]
    # Repair empty slots without losing any aircraft or a legal owner.
    counts = np.bincount(owners, minlength=k)
    def fill_empty(empty, depth, seen_beams):
        budget.check()
        eligible = sorted((i for i in range(len(members)) if legal[i, empty] and owners[i] != empty
                           and owners[i] not in seen_beams), key=lambda i: (scene.demands[members[i]], scene.aircraft_ids[members[i]]))
        for i in eligible:
            source = int(owners[i])
            if counts[source] > 1:
                counts[source] -= 1; owners[i] = empty; counts[empty] += 1
                return True
            if depth > 0:
                before_owners, before_counts = owners.copy(), counts.copy()
                counts[source] -= 1; owners[i] = empty; counts[empty] += 1
                if fill_empty(source, depth - 1, seen_beams | {empty, source}):
                    return True
                owners[:], counts[:] = before_owners, before_counts
        return False
    for empty in np.flatnonzero(counts == 0):
        if not fill_empty(int(empty), config.max_assignment_augment_depth, {int(empty)}):
            return None
    return owners


def _merge_region(scene, current, region, members, centers, widths, owners, task):
    if len(region) == len(centers):
        all_centers, all_widths, assignment = current.centers.copy(), current.widths.copy(), current.assignment.copy()
        all_centers[list(region)] = centers
        all_widths[list(region)] = widths
        assignment[members] = np.asarray(region)[owners]
        return Layout(all_centers, all_widths, assignment, current.beam_ids)
    keep = [b for b in range(len(current.centers)) if b not in region]
    remap = np.full(len(current.centers), -1)
    remap[keep] = np.arange(len(keep))
    assignment = remap[current.assignment].copy()
    assignment[members] = len(keep) + owners
    ids = [current.beam_ids[b] for b in keep]
    # Deterministic new identities bind a task, not a global mutable serial.
    new_ids = [f"r{task.task_id[:16]}_{i}" for i in range(len(centers))]
    while set(ids).intersection(new_ids):
        new_ids = ["r" + x for x in new_ids]
    return Layout(np.vstack((current.centers[keep], centers)), np.concatenate((current.widths[keep], widths)),
                  assignment, tuple(ids + new_ids))


def _quality_key(values, preference, fields):
    if values.get("CV") is None:
        fields = tuple(x for x in fields if x in ("N", "solid_angle_sum_sr"))
        preference = preference if preference in fields else fields[0]
    order = (preference,) + tuple(x for x in fields if x != preference)
    return tuple(float(values[x]) if values[x] is not None else 0. for x in order)


def metric_tolerance(field, first, second, config=SearchConfig()):
    if field == "N":
        return 0.
    return dict(config.metric_abs_tolerances)[field] + config.metric_relative_tolerance * max(abs(first), abs(second))


def compare_metrics(first, second, preference=0, config=SearchConfig()):
    """-1 preferred, 0 equivalent, +1 worse under a frozen metric lex order."""
    fields = config.pareto_fields
    if first.get("CV") is None or second.get("CV") is None:
        fields = tuple(x for x in fields if x in ("N", "solid_angle_sum_sr"))
    leading = config.pareto_fields[preference % len(config.pareto_fields)] if isinstance(preference, int) else preference
    leading = leading if leading in fields else fields[0]
    order = (leading,) + tuple(x for x in fields if x != leading)
    for field in order:
        x, y = float(first[field]), float(second[field])
        tol = metric_tolerance(field, x, y, config)
        if x < y - tol:
            return -1
        if x > y + tol:
            return 1
    return 0


def _best_row(rows, preference, config):
    def compare(a, b):
        decision = compare_metrics(a["metrics"], b["metrics"], preference, config)
        return decision or ((a["sha256"] > b["sha256"]) - (a["sha256"] < b["sha256"]))
    return min(rows, key=cmp_to_key(compare))


def decode_task(scene, current, task, budget=None, geometry=GeometryConfig(), config=SearchConfig(), on_candidate=None):
    """Try at most task_candidate_limit complete candidates after task ranking.

    Failed construction effort/geometry is charged even without a complete
    candidate. Every complete candidate records time and full independent
    validation. A late completion is evidence only, never publishable.
    """
    budget = budget or Budget(config)
    start, attempts_before, geometry_before, pairs_before = budget.elapsed, budget.attempts, budget.geometry_calls, budget.ray_pair_checks
    candidates, best = [], None
    reason = "no_feasible_candidate"
    region = task.region_ids
    if task.parent_layout_sha256 and task.parent_layout_sha256 != layout_sha256(current):
        raise ValueError("task parent SHA differs")
    if max(region) >= len(current.centers) or any(not geometry.width_min_deg <= w <= geometry.width_max_deg for w in task.widths):
        raise ValueError("task not compatible with current layout/geometry")
    members = np.flatnonzero(np.isin(current.assignment, region))
    if len(members) > config.max_region_users:
        return DecodeResult(None, "region_user_limit", {"region_users": len(members), "fixed_excess_users": 0})
    if task.target_k > len(members) or len(current.centers) - len(region) + task.target_k > geometry.max_beams:
        return DecodeResult(None, "beam_count_limit", {})
    seen = set()
    try:
        sites = _site_pool(scene, current, region, members, config, budget)
        for variant in range(task.variant, task.variant + config.variants):
            budget.check()
            centers = _seed_centers(scene, current, region, members, task.target_k, variant, config, sites)
            if centers is None:
                continue
            widths = np.asarray(task.widths, float)
            owners = _assign_region(scene, members, centers, widths, config, geometry, budget, variant)
            if owners is None:
                continue
            for iteration in range(config.center_iterations + 1):
                budget.check()
                candidate_widths = widths.copy()
                if task.width_policy == "shrink":
                    for b in range(task.target_k):
                        needed = minimum_enclosing_width(scene, members[owners == b], centers[b], geometry, budget)
                        if needed is None or needed > widths[b] + 1e-12:
                            candidate_widths = None
                            break
                        candidate_widths[b] = min(widths[b], needed)
                if candidate_widths is not None:
                    candidate = _merge_region(scene, current, region, members, centers, candidate_widths, owners, task)
                    sha = layout_sha256(candidate)
                    if sha not in seen:
                        seen.add(sha)
                        began = budget.elapsed
                        # Candidate count increments after validation to avoid
                        # rejecting the allowed final candidate at its boundary.
                        budget.check()
                        evidence = validate_layout(scene, candidate, geometry, budget)
                        values = metrics(scene, candidate, geometry, evidence) if evidence["valid"] else None
                        budget.attempts += 1
                        ended = budget.elapsed
                        late = budget.clock() > budget.deadline
                        if late:
                            budget.late_candidates += 1
                        row = {"layout": candidate, "sha256": sha, "metrics": values, "evidence": evidence,
                               "started_seconds": began, "completed_seconds": ended, "late": late,
                               "variant": variant, "center_iteration": iteration, "attempt": budget.attempts}
                        if on_candidate is not None:
                            on_candidate(row)
                        candidates.append(row)
                        if values is not None and not row["late"]:
                            preference = config.pareto_fields[task.preference_index % len(config.pareto_fields)]
                            if best is None or compare_metrics(values, best["metrics"], preference, config) < 0:
                                best = row
                            reason = "completed"
                        if len(candidates) >= config.task_candidate_limit:
                            raise BudgetStop("task_candidate_budget")
                if iteration < config.center_iterations:
                    moved = centers.copy()
                    for b in range(task.target_k):
                        center = ray_center(scene, members[owners == b])
                        if center is not None:
                            moved[b] = center
                    proposal = _assign_region(scene, members, moved, widths, config, geometry, budget, variant)
                    if proposal is None:
                        break
                    centers, owners = moved, proposal
    except BudgetStop as error:
        if str(error) != "task_candidate_budget":
            reason = str(error)
    details = {"region_users": len(members), "fixed_excess_users": 0, "target_k": task.target_k,
               "complete_candidates": len(candidates), "seconds": budget.elapsed - start,
               "attempts": budget.attempts - attempts_before, "geometry_calls": budget.geometry_calls - geometry_before,
               "ray_pair_checks": budget.ray_pair_checks - pairs_before,
               "feasible_candidates": sum(x["metrics"] is not None and not x["late"] for x in candidates),
               "best_metrics": best["metrics"] if best else None,
               "parent_sha256": layout_sha256(current), "result_sha256": best["sha256"] if best else None}
    return DecodeResult(best["layout"] if best else None, reason, details, candidates)


def dominates(a, b, fields, tolerance=1e-9, config=None):
    # Zero-request load quality is undefined; it stays in a separate geometry
    # archive rather than being silently ranked as a perfect zero-load policy.
    if any(a[x] is None or b[x] is None for x in fields):
        fields = tuple(x for x in fields if x in ("N", "solid_angle_sum_sr"))
    av = [0. if a[x] is None else float(a[x]) for x in fields]
    bv = [0. if b[x] is None else float(b[x]) for x in fields]
    tolerances = [metric_tolerance(f,x,y,config) if config is not None else tolerance for f,x,y in zip(fields,av,bv)]
    return all(x <= y + t for x, y,t in zip(av, bv,tolerances)) and any(x < y - t for x, y,t in zip(av, bv,tolerances))


def _prune(archive, config):
    if len(archive) <= config.archive_limit:
        return archive
    fields = config.pareto_fields if archive[0]["metrics"].get("CV") is not None else ("N", "solid_angle_sum_sr")
    extremes = set()
    for f in fields:
        best = _best_row(archive,f,config)
        extremes.add(next(i for i,r in enumerate(archive) if r is best))
    if config.archive_limit < len(extremes):
        raise ValueError("archive cap cannot preserve all objective extremes")
    crowd = np.zeros(len(archive))
    for f in fields:
        values = np.array([0. if r["metrics"][f] is None else r["metrics"][f] for r in archive], float)
        order = sorted(range(len(archive)), key=lambda i: (values[i], archive[i]["sha256"]))
        span = values[order[-1]] - values[order[0]]
        if span <= 0:
            continue
        crowd[order[0]] = crowd[order[-1]] = np.inf
        for j in range(1, len(order) - 1):
            crowd[order[j]] += (values[order[j + 1]] - values[order[j - 1]]) / span
    rest = sorted((i for i in range(len(archive)) if i not in extremes), key=lambda i: (-crowd[i], archive[i]["sha256"]))
    chosen = extremes.union(rest[:config.archive_limit - len(extremes)])
    return [archive[i] for i in sorted(chosen, key=lambda i: archive[i]["sha256"])]


def add_archive(archive, candidate, config):
    if any(r["sha256"] == candidate["sha256"] for r in archive):
        return archive, False
    if any(dominates(r["metrics"], candidate["metrics"], config.pareto_fields, config=config) for r in archive):
        return archive, False
    result = [r for r in archive if not dominates(candidate["metrics"], r["metrics"], config.pareto_fields, config=config)]
    result = _prune(result + [candidate], config)
    retained = any(r["sha256"] == candidate["sha256"] for r in result)
    return result, retained


@dataclass
class SearchResult:
    layout: Layout | None
    metrics: dict | None
    pareto: list
    journal: list
    budgets: dict
    initial_archive: list = field(default_factory=list)
    nested_reference: list = field(default_factory=list)
    slice_reference: dict | None = None
    reference_nodes: list = field(default_factory=list)

    def to_record(self):
        def rows(values):
            return [{**r, "layout": r["layout"].to_record()} for r in values]
        return {"layout": self.layout.to_record() if self.layout is not None else None,
                "metrics": self.metrics, "pareto": rows(self.pareto), "journal": self.journal,
                "budgets": self.budgets, "initial_archive": rows(self.initial_archive),
                "nested_reference": rows(self.nested_reference),
                "slice_reference": ({**self.slice_reference, "layout": self.slice_reference["layout"].to_record()}
                                    if self.slice_reference is not None else None),
                "reference_nodes": [{**r, "archive": rows(r["archive"])} for r in self.reference_nodes],
                "outcome": "completed" if self.layout is not None else "not_found"}


def layout_delta(parent, candidate):
    """Compact complete replay against a shared parent, including N changes."""
    id_to_new = {identity: i for i, identity in enumerate(candidate.beam_ids)}
    remap = np.array([id_to_new.get(identity, -1) for identity in parent.beam_ids], dtype=int)
    base = remap[parent.assignment]
    users = np.flatnonzero(base != candidate.assignment)
    return {"centers": candidate.centers.tolist(), "widths_deg": candidate.widths.tolist(),
            "beam_ids": list(candidate.beam_ids), "users": users.tolist(),
            "owners": candidate.assignment[users].tolist(), "parent_sha256": layout_sha256(parent),
            "result_sha256": layout_sha256(candidate)}


def apply_delta(parent, delta):
    if layout_sha256(parent) != delta["parent_sha256"]:
        raise ValueError("delta parent SHA differs")
    ids = tuple(delta["beam_ids"])
    index = {identity: i for i, identity in enumerate(ids)}
    remap = np.array([index.get(identity, -1) for identity in parent.beam_ids], dtype=int)
    owners = remap[parent.assignment]
    owners[np.asarray(delta["users"], int)] = np.asarray(delta["owners"], int)
    result = Layout(delta["centers"], delta["widths_deg"], owners, ids)
    if layout_sha256(result) != delta["result_sha256"]:
        raise ValueError("delta reconstructed SHA differs")
    return result


def run_search(scene, initial=None, config=SearchConfig(), geometry=GeometryConfig(), orderer=None, stop=None, method="RULE", budget=None):
    if isinstance(config, dict):
        config = SearchConfig(**config)
    if isinstance(geometry, dict):
        geometry = GeometryConfig(**geometry)
    if method not in ("RULE", "RANDOM", "AI", "INIT", "N_ONLY", "W_ONLY", "POST_SHRINK"):
        raise ValueError("unknown AI17 method")
    if method in ("RANDOM", "AI") and orderer is None:
        raise ValueError("RANDOM/AI require an explicit task orderer")
    budget = budget if budget is not None else Budget(config, stop)
    archive, initial_archive, journal, nested = [], [], [], []
    tasks_evaluated = pools = 0
    slice_reference = None
    checkpoint_pairs = list(dict.fromkeys([(config.nested_seconds, config.nested_max_attempts)] +
                                         list(zip(config.checkpoint_seconds, config.checkpoint_attempts)) +
                                         [(config.seconds, config.max_attempts)]))
    checkpoint_archives = {pair: [] for pair in checkpoint_pairs}
    checkpoint_closed = set()
    checkpoint_close_ledger = {}
    def update_checkpoints(row):
        for pair in checkpoint_pairs:
            seconds_limit, candidate_limit = pair
            if pair not in checkpoint_closed and row["completed_seconds"] <= seconds_limit and row["attempt"] <= candidate_limit:
                checkpoint_archives[pair], _ = add_archive(checkpoint_archives[pair], row, config)
    def close_checkpoints():
        for pair in checkpoint_pairs:
            if pair not in checkpoint_closed and (budget.elapsed >= pair[0] or budget.attempts >= pair[1]):
                checkpoint_closed.add(pair)
                checkpoint_close_ledger[pair] = (budget.elapsed, budget.attempts)
    def reference_nodes():
        return [{"kind": "paired", "seconds_limit": s, "candidate_limit": a,
                 "archive": list(checkpoint_archives[(s,a)]),
                 "actual_seconds": checkpoint_close_ledger.get((s,a),(budget.elapsed,budget.attempts))[0],
                 "actual_attempts": checkpoint_close_ledger.get((s,a),(budget.elapsed,budget.attempts))[1],
                 "terminated_before_node": (s,a) not in checkpoint_closed,
                 "trace_complete": method != "POST_SHRINK",
                 "scope": "initialization_only_old_fixed_search_has_no_shared_trace" if method == "POST_SHRINK" else "full_online_trajectory"}
                for s,a in checkpoint_pairs]
    nested_closed = False
    def nested_check():
        nonlocal nested_closed, nested
        if not nested_closed and (budget.elapsed >= config.nested_seconds or budget.attempts >= config.nested_max_attempts):
            nested_closed = True
    if initial is not None:
        # Intended for bounded diagnostics only; caller must report its cost.
        ev = validate_layout(scene, initial, geometry, budget)
        if not ev["valid"]:
            raise ValueError("invalid caller-provided initial layout")
        row = {"layout": initial, "metrics": metrics(scene, initial, geometry, ev), "sha256": layout_sha256(initial),
               "completed_seconds": budget.elapsed, "attempt": budget.attempts}
        archive = initial_archive = [row]
        if row["completed_seconds"] <= config.nested_seconds and row["attempt"] <= config.nested_max_attempts:
            nested = [row]
        update_checkpoints(row)
    else:
        final_deadline, final_limit = budget.deadline, budget.attempt_limit
        budget.deadline = min(final_deadline, budget.start + config.initial_seconds)
        budget.attempt_limit = min(final_limit, config.initial_max_attempts)
        initial_widths = (0.5,) if method in ("N_ONLY", "W_ONLY", "POST_SHRINK") else geometry.width_menu_deg
        for width in initial_widths:
            try:
                budget.check()
                init = initial_layout(scene, geometry, width, budget)
                raw_evidence = validate_layout(scene, init, geometry, budget)
                raw_values = metrics(scene, init, geometry, raw_evidence)
                if width == .5:
                    slice_reference = {"layout": init, "metrics": raw_values, "sha256": layout_sha256(init),
                                       "note": "unshrunk .5 initial only; fixed resource slice reference"}
                raw_layout = init
                if method not in ("N_ONLY", "POST_SHRINK") and len(init.centers):
                    widths = [minimum_enclosing_width(scene, np.flatnonzero(init.assignment == b), init.centers[b], geometry, budget)
                              for b in range(len(init.centers))]
                    init = Layout(init.centers, widths, init.assignment, init.beam_ids)
                evidence = validate_layout(scene, init, geometry, budget)
                values = metrics(scene, init, geometry, evidence)
                budget.attempts += 1
                late = budget.clock() > budget.deadline
                journal.append({"phase": "initial", "width_deg": width, "late": late, "seconds": budget.elapsed,
                                "sha256": layout_sha256(init), "layout": init.to_record(), "metrics": values})
                journal[-1]["unshrunk_layout"] = raw_layout.to_record()
                journal[-1]["unshrunk_metrics"] = raw_values
                if late:
                    budget.late_candidates += 1
                else:
                    row = {"layout": init, "metrics": values, "sha256": layout_sha256(init),
                           "completed_seconds": budget.elapsed, "attempt": budget.attempts}
                    prior_archive, prior_nested = archive, nested
                    prior_checkpoints = dict(checkpoint_archives)
                    updated_archive, _ = add_archive(archive, row, config)
                    updated_nested, _ = add_archive(nested, row, config)
                    update_checkpoints(row)
                    completed = budget.elapsed
                    journal[-1]["seconds"] = completed
                    row["completed_seconds"] = completed
                    if budget.clock() > budget.deadline:
                        checkpoint_archives.update(prior_checkpoints)
                        journal[-1]["late"] = True
                        budget.late_candidates += 1
                    else:
                        initial_archive.append(row)
                        archive = updated_archive
                        if completed <= config.nested_seconds and row["attempt"] <= config.nested_max_attempts:
                            nested = updated_nested
                        for pair in checkpoint_pairs:
                            if completed > pair[0] or row["attempt"] > pair[1]:
                                checkpoint_archives[pair] = prior_checkpoints[pair]
            except ValueError as error:
                journal.append({"phase": "initial", "width_deg": width, "outcome": "not_found", "error": str(error), "seconds": budget.elapsed})
            except BudgetStop as error:
                journal.append({"phase": "initial", "width_deg": width, "outcome": "budget_stop", "error": str(error), "seconds": budget.elapsed})
                break
        budget.deadline, budget.attempt_limit = final_deadline, final_limit
        if budget.reason != "stop_requested":
            budget.reason = None
    nested_check()
    close_checkpoints()
    if method == "POST_SHRINK" and archive:
        from ..geoload.core import Scene as OldScene, Layout as OldLayout
        from ..geoload.classical import run_classical
        from ..geoload.config import SearchConfig as OldSearch
        parent = archive[0]["layout"]
        remaining_seconds = max(0., config.seconds - budget.elapsed - config.post_shrink_reserve_seconds)
        # The legacy executor separately validates its initial layout once;
        # charge that real matrix before lending it the remaining call budget.
        try:
            budget.geometry(len(scene.demands)*len(parent.centers))
        except BudgetStop:
            return SearchResult(archive[0]["layout"],archive[0]["metrics"],archive,journal,
                {**budget.record(),"method":method,"nested_trace_complete":False},initial_archive,nested,slice_reference,reference_nodes())
        result = run_classical(OldScene(scene.positions, scene.demands), OldLayout(parent.centers, parent.assignment),
                               "JOINT", OldSearch(seconds=remaining_seconds, max_attempts=max(0, config.max_attempts-budget.attempts),
                                                  max_geometry_calls=max(0, config.max_geometry_calls-budget.geometry_calls)))
        budget.attempts += result.budgets["attempts"]
        budget.geometry_calls += result.budgets["geometry_calls"]
        fixed = Layout(result.layout.centers, np.full(len(result.layout.centers), .5), result.layout.assignment)
        journal.append({"phase": "post_shrink_fixed_search", "result": result.to_record()})
        try:
            budget.check()
            widths = [minimum_enclosing_width(scene, np.flatnonzero(fixed.assignment == b), fixed.centers[b], geometry, budget)
                      for b in range(len(fixed.centers))]
            candidate = Layout(fixed.centers, widths, fixed.assignment, fixed.beam_ids)
            evidence = validate_layout(scene, candidate, geometry, budget)
            values = metrics(scene, candidate, geometry, evidence)
            budget.attempts += 1
            if budget.clock() <= budget.deadline:
                row = {"layout": candidate, "metrics": values, "sha256": layout_sha256(candidate),
                       "completed_seconds": budget.elapsed, "attempt": budget.attempts}
                updated, _ = add_archive(archive, row, config)
                if budget.clock() <= budget.deadline:
                    row["completed_seconds"] = budget.elapsed
                    archive = updated
                else:
                    budget.late_candidates += 1
            else:
                budget.late_candidates += 1
        except BudgetStop:
            pass
    if method in ("INIT", "POST_SHRINK") or not len(scene.demands):
        close_checkpoints()
        chosen = _best_row(archive, "maxload", config) if archive else None
        return SearchResult(chosen["layout"] if chosen else None, chosen["metrics"] if chosen else None,
                            archive, journal, {**budget.record(), "method": method, "initialization_included": initial is None,
                                              "tasks_evaluated": 0, "task_pools": 0,
                                              "nested_trace_complete": method != "POST_SHRINK",
                                              "legacy_ray_pair_count_unknown": method == "POST_SHRINK",
                                              "zero_total_demand": float(scene.demands.sum()) == 0.,
                                              "quality_scope": "geometry_only" if float(scene.demands.sum()) == 0. else "geometry_and_request"},
                            initial_archive, list(nested), slice_reference, reference_nodes())
    visited = set()
    preference_index = 0
    exhausted_states = set()
    try:
        for restart in range(config.max_restarts):
            budget.check()
            if not archive:
                budget.reason = "initial_not_found"
                break
            preference = config.pareto_fields[preference_index % len(config.pareto_fields)]
            preference_index += 1
            available = [r for r in archive if r["sha256"] not in exhausted_states]
            if not available:
                budget.reason = "exhausted_tasks"
                break
            current_row = _best_row(available, preference, config)
            current = current_row["layout"]
            iterator = (t for t in generate_tasks(scene, current, config, geometry, budget) if (current_row["sha256"], t.task_id) not in visited)
            if method == "N_ONLY":
                iterator = (t for t in iterator if t.width_policy == "menu" and all(w == .5 for w in t.widths))
            elif method == "W_ONLY":
                iterator = (t for t in iterator if t.target_k == len(t.region_ids))
            pool = [replace(t, preference_index=(preference_index-1) % len(config.pareto_fields)) for t in islice(iterator, config.pool_size)]
            if not pool:
                exhausted_states.add(current_row["sha256"])
                continue
            budget.check()
            pools += 1
            if orderer is not None:
                if hasattr(orderer, "set_budget"):
                    orderer.set_budget(budget)
                if hasattr(orderer, "set_deadline"):
                    orderer.set_deadline(budget.deadline)
                if hasattr(orderer, "set_preference"):
                    orderer.set_preference((preference_index - 1) % len(config.pareto_fields))
                ordered = list(orderer(scene, current, pool))
                if len(ordered) != len(pool) or sorted(map(id, ordered)) != sorted(map(id, pool)):
                    raise ValueError("task orderer must permute exactly the shared predecode task objects")
            else:
                ordered = pool
            budget.check()
            for task in ordered:
                budget.check()
                visited.add((current_row["sha256"], task.task_id))
                def commit_candidate(trial):
                    nonlocal archive, nested
                    if trial["metrics"] is None or trial["late"]:
                        trial["archive_retained_at_trial"] = False
                        return
                    row = {"layout": trial["layout"], "metrics": trial["metrics"], "sha256": trial["sha256"],
                           "completed_seconds": trial["completed_seconds"], "attempt": trial["attempt"]}
                    updated, retained = add_archive(archive, row, config)
                    nested_updated = None
                    if row["completed_seconds"] <= config.nested_seconds and trial["attempt"] <= config.nested_max_attempts:
                        nested_updated, _ = add_archive(nested, row, config)
                    prior_checkpoints = dict(checkpoint_archives)
                    update_checkpoints(row)
                    completed = budget.elapsed
                    if budget.clock() > budget.deadline:
                        checkpoint_archives.update(prior_checkpoints)
                        trial["late"] = True
                        budget.late_candidates += 1
                        trial["archive_retained_at_trial"] = False
                        return
                    row["completed_seconds"] = completed
                    archive = updated
                    trial["completed_seconds"] = completed
                    trial["archive_retained_at_trial"] = retained
                    if completed <= config.nested_seconds and trial["attempt"] <= config.nested_max_attempts:
                        nested = nested_updated if nested_updated is not None else nested
                    for pair in checkpoint_pairs:
                        if completed > pair[0] or trial["attempt"] > pair[1]:
                            checkpoint_archives[pair] = prior_checkpoints[pair]
                decoded = decode_task(scene, current, task, budget, geometry, config, commit_candidate)
                tasks_evaluated += 1
                trials = []
                for trial in decoded.candidates:
                    # Nested checkpoint admits only candidates completed within
                    # both nested limits, not merely those decoded in its task.
                    trials.append({k: v for k, v in trial.items() if k != "layout"})
                    trials[-1]["delta"] = layout_delta(current, trial["layout"])
                journal.append({"phase": "task", "pool": pools, "preference": preference, "current_sha256": current_row["sha256"],
                                "task": task.to_record(), "reason": decoded.reason, "details": decoded.details, "candidates": trials})
                nested_check()
                close_checkpoints()
                if decoded.reason in ("time_budget", "candidate_budget", "geometry_budget", "stop_requested"):
                    raise BudgetStop(decoded.reason)
        else:
            budget.reason = "restart_budget"
    except BudgetStop as error:
        budget.reason = str(error)
    if not nested_closed:
        nested = list(archive)
    chosen = _best_row(archive, "maxload", config) if archive else None
    record = {**budget.record(), "method": method, "tasks_evaluated": tasks_evaluated, "task_pools": pools, "archive_size": len(archive),
              "initialization_included": initial is None, "initial_seconds_limit": config.initial_seconds,
              "initial_candidate_limit": config.initial_max_attempts, "nested_seconds_limit": config.nested_seconds,
              "nested_candidate_limit": config.nested_max_attempts, "nested_closed": nested_closed,
              "published_layout_rule": "peak-first descriptive member; full Pareto archive is scientific result",
              "zero_total_demand": float(scene.demands.sum()) == 0.,
              "quality_scope": "geometry_only" if float(scene.demands.sum()) == 0. else "geometry_and_request"}
    return SearchResult(chosen["layout"] if chosen else None, chosen["metrics"] if chosen else None,
                        archive, journal, record, initial_archive, nested, slice_reference, reference_nodes())
