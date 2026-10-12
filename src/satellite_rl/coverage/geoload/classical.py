"""Finite deterministic neighborhoods with atomic whole-layout publication."""
from dataclasses import dataclass
from itertools import combinations, product, islice, zip_longest
from time import perf_counter
import numpy as np
from satellite_rl.envs.regions import get_profile
from ..geometry import satellite_rays_numpy
from .config import GeometryConfig, SearchConfig
from .core import Layout, metrics, objective_key, validate_layout

METHODS = ("BH", "07", "09-CV", "09-LEX", "09+SWAP", "JOINT", "QUOTA", "VARIABLE")

@dataclass
class SearchResult:
    layout: Layout
    metrics: dict
    journal: list
    budgets: dict

    def to_record(self):
        return {"layout": self.layout.to_record(), "metrics": self.metrics,
                "journal": self.journal, "budgets": self.budgets,
                "reference_nodes": self.budgets.get("checkpoints", [])}

def _ray_center(scene, members):
    ray = satellite_rays_numpy(scene.positions[members], scene.profile).mean(axis=0)
    ray /= np.linalg.norm(ray)
    p = get_profile(scene.profile); angle = np.deg2rad(p.satellite_longitude_deg)
    sat = p.orbit_radius_km * np.array([np.cos(angle), np.sin(angle), 0.])
    dot = sat @ ray; disc = dot**2 - (sat @ sat - p.earth_radius_km**2)
    if disc < 0: return None
    ground = sat + (-dot - np.sqrt(disc)) * ray
    return np.rad2deg([np.arctan2(ground[2], np.hypot(ground[0], ground[1])), np.arctan2(ground[1], ground[0])])

def center_candidates(scene, members, old=None, limit=8):
    choices = [] if old is None else [np.asarray(old)]
    mean = _ray_center(scene, members)
    if mean is not None: choices.append(mean)
    choices.extend(scene.positions[members[:limit]])
    result = []
    for c in choices:
        if not any(np.array_equal(c, prior) for prior in result): result.append(c)
        if len(result) == limit: break
    return result

def _compact(centers, a):
    used = np.unique(a); mapping = np.full(len(centers), -1); mapping[used] = np.arange(len(used))
    return Layout(np.asarray(centers)[used], mapping[a])

def _ordinary_candidates(scene, current, method, cfg):
    """Candidate order depends only on input/current and frozen limits."""
    a = current.assignment; n = len(current.centers); users = len(a)
    loads = np.bincount(a, weights=scene.demands, minlength=n)
    order = np.argsort(-loads, kind="stable")
    if method in ("BH", "09-CV", "09-LEX", "09+SWAP", "JOINT"):
        for src in order:
            for u in np.flatnonzero(a == src):
                for dst in order[::-1]:
                    if src == dst or np.sum(a == src) == 1: continue
                    b = a.copy(); b[u] = dst
                    yield "transfer", Layout(current.centers, b), {"users": [int(u)], "beams": [int(src), int(dst)]}
                    if method != "BH":
                        centers = current.centers.copy()
                        center = _ray_center(scene, np.flatnonzero(b == dst))
                        if center is not None:
                            centers[dst] = center
                            yield "transfer_center", Layout(centers, b), {"users": [int(u)], "beams": [int(src), int(dst)]}
    if method in ("09+SWAP", "JOINT"):
        for u, v in combinations(range(users), 2):
            if a[u] == a[v]: continue
            b = a.copy(); b[u], b[v] = b[v], b[u]
            yield "swap", Layout(current.centers, b), {"users": [u, v]}
    if method in ("07", "JOINT"):
        for beam in order:
            members = np.flatnonzero(a == beam)
            for center in center_candidates(scene, members, current.centers[beam], cfg.centers_per_beam):
                centers = current.centers.copy(); centers[beam] = center
                if method == "07":
                    rays = satellite_rays_numpy(scene.positions, scene.profile)
                    beams = satellite_rays_numpy(centers, scene.profile)
                    ownership = np.argmin(np.sum((rays[:, None] - beams[None])**2, axis=-1), axis=1)
                    yield "cluster_center_reassign", Layout(centers, ownership), {"beam": int(beam)}
                else:
                    yield "center", Layout(centers, a), {"beam": int(beam)}
    if method == "JOINT-ONLY":
        for region_index in range(cfg.max_joint_regions):
            size = 2 + region_index % 3
            if size > n: continue
            anchor = int(order[(region_index // 3) % n])
            distances = np.sum((current.centers - current.centers[anchor])**2, axis=1)
            beams = tuple(np.argsort(distances, kind="stable")[:size].tolist())
            if len(beams) == size:
                members = np.flatnonzero(np.isin(a, beams))
                # A bounded subset; unsampled aircraft retain their ownership.
                selected = members[np.argsort(-scene.demands[members], kind="stable")[:cfg.max_joint_users]]
                for owners in islice(product(beams, repeat=len(selected)), cfg.joint_assignments_per_region):
                    b = a.copy(); b[selected] = owners
                    if np.array_equal(a, b) or any(not np.any(b == k) for k in beams): continue
                    yield "joint", Layout(current.centers, b), {"beams": list(beams), "users": selected.tolist()}
                    centers = current.centers.copy()
                    for beam in beams:
                        center = _ray_center(scene, np.flatnonzero(b == beam))
                        if center is not None: centers[beam] = center
                    yield "joint_center", Layout(centers, b), {"beams": list(beams), "users": selected.tolist()}
    if method in ("QUOTA", "VARIABLE"):
        # Split candidates partition all members, including zero-demand edges.
        splits = []
        for beam in order:
            members = np.flatnonzero(a == beam)
            if len(members) < 2: continue
            for axis in (0, 1):
                ranked = members[np.argsort(scene.positions[members, axis], kind="stable")]
                cut = len(ranked) // 2
                centers = current.centers.tolist(); centers.append(scene.positions[ranked[cut]].tolist())
                b = a.copy(); b[ranked[cut:]] = n
                center = _ray_center(scene, ranked[:cut])
                if center is not None: centers[beam] = center.tolist()
                center = _ray_center(scene, ranked[cut:])
                if center is not None: centers[-1] = center.tolist()
                split = Layout(centers, b); splits.append((beam, split))
                if method == "VARIABLE": yield "split", split, {"beam": int(beam)}
        for first, second in combinations(range(n), 2):
            b = a.copy(); b[b == second] = first
            centers = current.centers.copy(); center = _ray_center(scene, np.flatnonzero(b == first))
            if center is not None: centers[first] = center
            merged = _compact(centers, b)
            if method == "VARIABLE": yield "merge", merged, {"beams": [first, second]}
            else:
                for split_beam, split in splits:
                    if split_beam in (first, second): continue
                    b = split.assignment.copy(); b[b == second] = first
                    centers = split.centers.copy()
                    center = _ray_center(scene, np.flatnonzero(b == first))
                    if center is not None: centers[first] = center
                    yield "quota_merge_split", _compact(centers, b), {"merge": [first, second], "split": int(split_beam)}

def candidates(scene, current, method, cfg):
    if method != "JOINT":
        yield from _ordinary_candidates(scene, current, method, cfg)
        return
    # Interleave joint sizes with ordinary moves so candidate caps cannot hide
    # all multi-beam moves behind a long sequence of rejected single moves.
    for ordinary, joint in zip_longest(_ordinary_candidates(scene, current, "JOINT", cfg),
                                      _ordinary_candidates(scene, current, "JOINT-ONLY", cfg)):
        if ordinary is not None: yield ordinary
        if joint is not None: yield joint

def run_classical(scene, initial, method, config=SearchConfig(), geometry=GeometryConfig(), candidate_order=None):
    method = {"09CV": "09-CV", "09LEX": "09-LEX", "09SWAP": "09+SWAP"}.get(method, method)
    if isinstance(config, dict): config = SearchConfig(**config)
    if method not in METHODS: raise ValueError(method)
    if method == "VARIABLE" and not config.allow_variable_n: raise ValueError("variable N requires explicit config")
    initialization_start = perf_counter()
    initial_metrics = metrics(scene, initial, geometry); current = initial; values = initial_metrics
    initial_validation_seconds = perf_counter() - initialization_start
    objective = "legacyCV" if method == "09-CV" else config.objective
    start = perf_counter(); attempts = calls = accepted = 0; journal = []; stop = "exhausted"
    checkpoints = []; seen_checkpoints = set(); pools = generated = 0
    accepted_history = [(0., 0, current, values)]
    def checkpoint():
        elapsed = perf_counter() - start
        for kind, limits in (("seconds", config.checkpoint_seconds), ("attempts", config.checkpoint_attempts)):
            actual = elapsed if kind == "seconds" else attempts
            for limit in limits:
                key = (kind, limit)
                if actual >= limit and key not in seen_checkpoints:
                    seen_checkpoints.add(key)
                    eligible = [h for h in accepted_history if (h[0] if kind == "seconds" else h[1]) <= limit]
                    state = eligible[-1]
                    checkpoints.append({"kind": kind, "limit": limit, "actual_seconds": elapsed,
                                        "actual_attempts": attempts, "state_seconds": state[0],
                                        "state_attempts": state[1], "layout": state[2].to_record(), "metrics": state[3]})
    while True:
        improved = False
        iterator = candidates(scene, current, method, config)
        if candidate_order is not None:
            pool = list(islice(iterator, config.pool_size)); pools += 1; generated += len(pool)
            ordered = list(candidate_order(scene, current, pool))
            if len(ordered) != len(pool) or sorted(map(id, ordered)) != sorted(map(id, pool)):
                raise ValueError("candidate_order must permute the exact shared pool")
            iterator = iter(ordered)
        for kind, candidate, details in iterator:
            checkpoint()
            elapsed = perf_counter() - start
            if elapsed >= config.seconds or attempts >= config.max_attempts or calls >= config.max_geometry_calls:
                stop = "time_budget" if elapsed >= config.seconds else "candidate_budget" if attempts >= config.max_attempts else "geometry_budget"
                break
            attempts += 1; calls += 1
            evidence = validate_layout(scene, candidate, geometry)
            reason = "invalid_geometry"; candidate_values = None
            if evidence["valid"]:
                candidate_values = metrics(scene, candidate, geometry, evidence=evidence)
                before = objective_key(values, objective); after = objective_key(candidate_values, objective)
                win = False
                for x, y in zip(after, before):
                    if x < y - config.objective_tolerance: win = True; break
                    if x > y + config.objective_tolerance: break
                if method == "VARIABLE" and config.target_n is not None:
                    win = abs(candidate_values["N"] - config.target_n) < abs(values["N"] - config.target_n) or (candidate_values["N"] == values["N"] and win)
                if method != "VARIABLE" and candidate_values["N"] != initial_metrics["N"]: win = False
                reason = "accepted" if win else "objective_reject"
                if win:
                    current = candidate; values = candidate_values; improved = True; accepted += 1
                    accepted_history.append((perf_counter() - start, attempts, current, values))
            journal.append({"attempt": attempts, "kind": kind, "details": details, "reason": reason,
                            "elapsed_seconds": perf_counter() - start, "coverage_evidence": evidence,
                            "candidate_metrics": candidate_values})
            if improved: break
        else:
            if improved: continue
            break
        if not improved: break
    elapsed = perf_counter() - start
    checkpoint()
    for kind, limits in (("seconds", config.checkpoint_seconds), ("attempts", config.checkpoint_attempts)):
        for limit in limits:
            if (kind, limit) not in seen_checkpoints:
                checkpoints.append({"kind": kind, "limit": limit, "actual_seconds": elapsed,
                                    "actual_attempts": attempts, "state_seconds": accepted_history[-1][0],
                                    "state_attempts": accepted_history[-1][1], "layout": current.to_record(),
                                    "metrics": values, "terminated_before_node": True, "stop_reason": stop})
    return SearchResult(current, values, journal, {"attempts": attempts, "geometry_calls": calls,
        "candidate_pools": pools, "generated_pool_candidates": generated, "checkpoints": checkpoints,
        "initial_validation_calls": 1, "initial_validation_seconds": initial_validation_seconds,
        "accepted": accepted, "seconds": elapsed, "seconds_limit": config.seconds,
        "candidate_limit": config.max_attempts, "stop_reason": stop,
        "deadline_overrun_seconds": max(0., elapsed - config.seconds),
        "reference_scope": "bounded_finite_neighborhood_no_global_certificate"})
