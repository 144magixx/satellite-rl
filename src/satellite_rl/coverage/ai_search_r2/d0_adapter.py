"""Explicit fbd3a743 R1 assignment/repair adapter; its tie rules preserved.
Differences: local qualification, owner-only full validation, precharge accounting,
shared R2 precomputed seed descriptors, base/envelope forms, stable surviving
beam IDs, construction records, and strict R2 finite task caps. This is not a
re-execution of fbd's task/seed generator.
No runtime source patching or edits to R1 globals.
"""
import numpy as np
from ..ai_search.core import (Layout,metrics,ray_center,ray_location,layout_sha256)
from .envelope import minimum_enclosing_width
from ..ai_search.config import GeometryConfig,SearchConfig
from ..ai_search.search import (BudgetStop,DecodeResult,compare_metrics)
from ..geometry import satellite_rays_numpy
from .validation import validate_owned as validate_layout

def _assign_region(scene, members, centers, widths, config, geometry, budget, variant):
    from .decoder import _qualification
    legal = _qualification(scene, members, centers, widths, geometry, budget)
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
        budget.work('assignment_updates',1)
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
                budget.work('assignment_updates',1)
                counts[source] -= 1; owners[i] = empty; counts[empty] += 1
                return True
            if depth > 0:
                budget.work('assignment_updates',1)
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


def decode_task(scene, current, task, budget=None, geometry=GeometryConfig(), config=SearchConfig(), on_candidate=None):
    """Try at most task_candidate_limit complete candidates after task ranking.

    Failed construction effort/geometry is charged even without a complete
    candidate. Every complete candidate records time and full independent
    validation. A late completion is evidence only, never publishable.
    """
    if budget is None:raise ValueError('R2 requires explicit Ledger')
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
        seeds=task.cheap_features['center_seed_sets']
        for variant in range(task.variant, min(len(seeds),task.variant + config.variants)):
            budget.check()
            seed=seeds[variant]
            if seed['status']!='available':
                continue
            centers=np.asarray(seed['centers'],float)
            widths = np.asarray(task.widths, float)
            owners = _assign_region(scene, members, centers, widths, config, geometry, budget, variant)
            if owners is None:
                continue
            for iteration in range(config.center_iterations + 1):
                budget.check()
                for width_form in ("menu", "shrink"):
                    candidate_widths = widths.copy()
                    if width_form == "shrink":
                        for b in range(task.target_k):
                            needed = minimum_enclosing_width(scene, members[owners == b], centers[b], geometry, budget)
                            if needed is None or needed > widths[b] + 1e-12:
                                candidate_widths = None
                                break
                            candidate_widths[b] = min(widths[b], needed)
                    if candidate_widths is not None:
                        from .decoder import assemble_region
                        candidate = assemble_region(scene, current, region, members, centers, candidate_widths, owners, task)
                        sha = layout_sha256(candidate)
                        if sha not in seen:
                            seen.add(sha)
                            began = budget.elapsed
                            # R2 reserves candidate work before final validation.
                            budget.check()
                            budget.attempt()
                            evidence = validate_layout(scene, candidate, geometry, budget)
                            values = metrics(scene, candidate, geometry, evidence) if evidence["valid"] else None

                            ended = budget.elapsed
                            late = budget.clock() > budget.deadline
                            if late:
                                budget.late_candidates += 1
                            row = {"layout": candidate, "sha256": sha, "metrics": values, "evidence": evidence,
                                   "started_seconds": began, "completed_seconds": ended, "late": late,
                                   "variant": variant, "center_iteration": iteration, "attempt": budget.attempts,"width_form":width_form}
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
                        users=members[owners == b]
                        center = budget.construct('mean_ray_update',int(len(users)),ray_center,scene,users)
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
