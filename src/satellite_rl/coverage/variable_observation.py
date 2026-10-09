"""Current-layout geometry adapter for variable-set PPO."""
from __future__ import annotations
import hashlib
import json
import numpy as np
from .variable_design import candidates, metrics
from .variable_ppo import BEAM_FEATURES, CANDIDATE_FEATURES, GLOBAL_FEATURES

def candidate_digest(menu):
    identities = [action.to_record() for action, _ in menu]
    return hashlib.sha256(json.dumps(identities, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def observation(scene, layout, cfg, *, route, steps, geometry_calls,
                budget=None, value_only=False):
    """Encode all current beams and user ownership without a fixed-N slice."""
    if route not in ("R08", "R09"):
        raise ValueError("Learning observation requires R08 or R09")
    positions = np.asarray(scene.positions, dtype=np.float64)
    demand = np.asarray(scene.demands, dtype=np.float64)
    owner = np.asarray(layout.assignment, dtype=np.int64)
    n = len(layout.centers)
    if len(owner) != len(demand) or n > 200:
        raise ValueError("Malformed variable-layout observation")
    valid = owner >= 0
    loads = np.bincount(owner[valid], weights=demand[valid], minlength=n)
    counts = np.bincount(owner[valid], minlength=n)
    weighted = np.zeros((n, 2), dtype=np.float64)
    if n:
        np.add.at(weighted, owner[valid], positions[valid] * demand[valid, None])
    centroid = weighted / np.maximum(loads[:, None], 1e-9)
    beam = np.zeros((n, BEAM_FEATURES), dtype=np.float32)
    if n:
        beam[:, 0:2] = np.asarray(layout.centers, dtype=np.float32) / (90.0, 180.0)
        beam[:, 2] = np.asarray(layout.widths) / max(cfg.fixed_width_deg, 1e-9)
        beam[:, 3] = loads / cfg.reference_capacity_bps
        beam[:, 4] = np.log1p(loads / cfg.reference_capacity_bps)
        beam[:, 5] = counts / max(len(demand), 1)
        beam[:, 6:8] = (centroid - layout.centers) / (10.0, 10.0)
        beam[:, 8] = (loads > cfg.reference_capacity_bps).astype(np.float32)
        beam[:, 9] = loads / max(float(demand.sum()), 1e-9)
        beam[:, 10] = np.arange(n) / max(n, 1)
        beam[:, 11] = 1.0
    current = metrics(scene, layout, cfg)
    step_limit = 200 if route == "R09" else cfg.max_actions
    global_features = np.asarray([
        n / cfg.max_beams, current["coverage_users"], current["coverage_demand"],
        current["load_cv"], current["peak_load_ratio"],
        steps / max(step_limit, 1), geometry_calls / max(cfg.max_geometry_calls, 1),
        np.log1p(float(demand.sum()) / cfg.reference_capacity_bps),
    ], dtype=np.float32)
    if value_only:
        return {"beam": beam, "global": global_features,
                "candidate": np.zeros((1, CANDIDATE_FEATURES), dtype=np.float32),
                "candidate_mask": np.ones(1, dtype=bool)}, []
    menu = candidates(scene, layout, cfg, mode="construct" if route == "R09" else "local",
                      budget=budget)
    if not menu or menu[0][0].operation != "stop":
        raise ValueError("Candidate menu must begin with explicit STOP")
    features = np.stack([np.asarray(feature, dtype=np.float32) for _, feature in menu])
    if features.shape[1] != CANDIDATE_FEATURES or not np.isfinite(features).all():
        raise ValueError("Candidate features are malformed")
    allowed = np.ones(len(menu), dtype=bool)
    if not current["coverage_100"]:
        allowed[0] = False
    if not allowed.any():
        # An impossible full-coverage continuation is a scientific failure.
        raise ValueError("No legal non-STOP action before full coverage")
    return {"beam": beam, "global": global_features, "candidate": features,
            "candidate_mask": allowed}, menu


