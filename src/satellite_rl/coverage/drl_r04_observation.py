"""Cheap pre-action features for the fixed-width R04 improvement task."""

import hashlib
import json

import numpy as np

from .variable_design import Layout


BEAM_FEATURES = 12
CANDIDATE_FEATURES = 16
GLOBAL_FEATURES = 14


def layout_sha256(layout: Layout) -> str:
    payload = json.dumps(layout.to_record(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def candidate_sha256(menu) -> str:
    payload = json.dumps([action.to_record() for action, _ in menu],
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def encode(scene, layout, cfg, contract, *, current_j, best_j, best_n,
           commits, trials, geometry_calls, menu):
    """No candidate execution, objective lookup or exact ray matrix occurs here."""
    positions = np.asarray(scene.positions, dtype=np.float64)
    demands = np.asarray(scene.demands, dtype=np.float64)
    owner = np.asarray(layout.assignment, dtype=np.int64)
    n = len(layout.centers)
    valid = owner >= 0
    loads = np.bincount(owner[valid], weights=demands[valid], minlength=n)
    counts = np.bincount(owner[valid], minlength=n)
    centroid = np.zeros((n, 2), dtype=np.float64)
    if n:
        np.add.at(centroid, owner[valid], positions[valid] * demands[valid, None])
        centroid /= np.maximum(loads[:, None], 1e-9)
    beam = np.zeros((n, BEAM_FEATURES), dtype=np.float32)
    if n:
        beam[:, 0:2] = np.asarray(layout.centers, dtype=np.float32) / (90.0, 180.0)
        beam[:, 2] = np.asarray(layout.widths) / cfg.fixed_width_deg
        beam[:, 3] = loads / cfg.reference_capacity_bps
        beam[:, 4] = np.log1p(loads / cfg.reference_capacity_bps)
        beam[:, 5] = counts / max(len(demands), 1)
        beam[:, 6:8] = (centroid - layout.centers) / (10.0, 10.0)
        beam[:, 8] = (loads > cfg.reference_capacity_bps).astype(np.float32)
        beam[:, 9] = loads / max(float(demands.sum()), 1e-9)
        beam[:, 10] = np.arange(n) / max(n, 1)
        beam[:, 11] = 1.0
    ratio = loads / cfg.reference_capacity_bps
    cv = float(loads.std() / loads.mean()) if n and loads.mean() > 0 else 0.0
    global_features = np.asarray([
        n / cfg.max_beams, float(valid.mean()) if len(valid) else 1.0,
        float(demands[valid].sum() / demands.sum()) if demands.sum() else 1.0,
        cv, float(ratio.max()) if n else 0.0,
        commits / contract.max_commits, geometry_calls / contract.max_geometry_calls,
        np.log1p(float(demands.sum()) / cfg.reference_capacity_bps),
        float(current_j), float(best_j),
        (contract.max_commits - commits) / contract.max_commits,
        (contract.max_trials - trials) / contract.max_trials,
        (contract.max_geometry_calls - geometry_calls) / contract.max_geometry_calls,
        best_n / cfg.max_beams,
    ], dtype=np.float32)
    features = np.stack([np.asarray(f, dtype=np.float32) for _, f in menu]) if menu else np.zeros((1, CANDIDATE_FEATURES), dtype=np.float32)
    mask = np.ones(len(menu), dtype=bool) if menu else np.ones(1, dtype=bool)
    if not np.isfinite(beam).all() or not np.isfinite(features).all() or not np.isfinite(global_features).all():
        raise ValueError("nonfinite R04 observation")
    return {"beam": beam, "candidate": features, "candidate_mask": mask,
            "global": global_features}
