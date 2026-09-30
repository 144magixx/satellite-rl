"""No-adjustment and deterministic local-search coverage baselines."""

from __future__ import annotations

import hashlib
import time

import numpy as np
import torch

from satellite_rl.coverage.environment import CoverageBatchEnv
from satellite_rl.coverage.scenes import CoverageDataset


def _layout_hash(centers):
    return hashlib.sha256(np.asarray(centers, dtype="<f8").tobytes()).hexdigest()


def run_baseline(scene, config, method="none"):
    """Return raw terminal evidence, including failures, for one original scene.

    Local search checks all legal actions at every fixed beam turn, cloning the
    complete environment state before each candidate. The selected action is
    the smallest action ID among equal scores. It has more candidate checks
    than a PPO rollout and that work is reported explicitly.
    """
    if method not in ("none", "local_search"):
        raise ValueError("baseline method must be 'none' or 'local_search'")
    start = time.perf_counter()
    common = {"method": method, "scene_id": scene.scene_id,
              "source_hash": scene.source_hash,
              "initialization_valid": scene.metadata.get("initialization_valid"),
              "initialization_error": scene.metadata.get("initialization_error"),
              "initial_layout_hash": _layout_hash(scene.centers),
              "candidate_checks": 0}
    if not np.isfinite(scene.centers).all():
        return {**common, "status": "initialization_failed",
                "initialization_error": scene.metadata.get("initialization_error"),
                "final_layout_hash": None, "final_positions": None,
                "final_associations": None, "beam_loads_bps": None,
                "uncovered_aircraft_indices": None, "uncovered_demand_bps_by_aircraft": None,
                "elapsed_seconds": time.perf_counter() - start}
    dataset = CoverageDataset([scene], {"baseline": True})
    device = config.get("training", {}).get("device", "cpu") if isinstance(config, dict) else "cpu"
    env = CoverageBatchEnv(dataset, config, num_envs=1, device=device,
                           seed=0, auto_reset=False)
    env.reset(scene_indices=[0])
    checks = 0
    if method == "local_search":
        for _ in range(env.horizon):
            saved = env.state_dict()
            legal = env._action_mask()[0].nonzero(as_tuple=True)[0].tolist()
            choice, best = 0, -float("inf")
            for action in legal:
                env.load_state_dict(saved)
                _, _, _, _ = env.step(torch.tensor([action], device=env.device))
                score = float(env.metrics()["score"][0].item())
                checks += 1
                if score > best + 1e-12:
                    best, choice = score, action
            env.load_state_dict(saved)
            env.step(torch.tensor([choice], device=env.device))
    # The no-adjustment baseline is the untouched initial layout. It does not
    # pretend to have spent 200 actions or candidate evaluations.
    metrics = {key: value[0].item() for key, value in env.metrics().items()}
    final_positions = env.layouts()[0].cpu().numpy()
    associations, mask = env.associations()
    final_associations = associations[0][mask[0]].cpu().numpy()
    uncovered = np.flatnonzero(final_associations < 0)
    return {**common, "status": "completed", "metrics": metrics,
            "final_positions": final_positions.tolist(),
            "final_associations": final_associations.tolist(),
            "beam_loads_bps": env.loads[0].cpu().tolist(),
            "uncovered_aircraft_indices": uncovered.tolist(),
            "uncovered_demand_bps_by_aircraft": scene.demands[uncovered].tolist(),
            "final_layout_hash": _layout_hash(final_positions),
            "candidate_checks": checks,
            "action_steps": env.horizon if method == "local_search" else 0,
            "elapsed_seconds": time.perf_counter() - start}
