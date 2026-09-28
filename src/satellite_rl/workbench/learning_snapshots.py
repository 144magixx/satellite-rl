"""Small, read-only SAC learning evidence at each complete checkpoint boundary.

These files support offline Critic and terminal TD diagnosis. They are not
recoverable training checkpoints and deliberately omit optimizer/replay state.
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import torch

from satellite_rl.training.run_monitor import replace_with_retry
from satellite_rl.workbench.checkpoints import config_hash


FORMAT = "satellite_rl_sac_learning_snapshot_v1"
SAMPLE_SEED = 51001
SAMPLE_PER_STRATUM = 64
NETWORKS = ("actor", "critic1", "critic2", "target1", "target2")


def _cpu_state(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _cpu_mask(mask):
    if mask is None:
        return None
    return {key: torch.as_tensor(np.asarray(value, dtype=np.bool_)).clone()
            for key, value in mask.items()}


def _sample_replay(replay):
    rows = replay.rows
    rng = np.random.default_rng(SAMPLE_SEED)
    samples = []
    index_groups = {}
    for label, terminal in (("terminal", True), ("nonterminal", False)):
        available = np.fromiter((index for index, row in enumerate(rows)
                                 if bool(row[4]) is terminal), dtype=np.int64)
        count = min(SAMPLE_PER_STRATUM, len(available))
        selected = sorted(int(index) for index in rng.choice(available, size=count, replace=False))
        index_groups[label] = selected
        for index in selected:
            row = rows[index]
            if len(row) not in (5, 7):
                raise ValueError("Unsupported SAC replay transition shape")
            samples.append({"replay_index": index,
                            "state": torch.as_tensor(np.asarray(row[0], dtype=np.float32)).clone(),
                            "action": torch.as_tensor(row[1], dtype=torch.int64).clone(),
                            "reward": float(row[2]),
                            "next_state": torch.as_tensor(np.asarray(row[3], dtype=np.float32)).clone(),
                            "done": bool(row[4]),
                            "action_mask": _cpu_mask(row[5]) if len(row) == 7 else None,
                            "next_action_mask": _cpu_mask(row[6]) if len(row) == 7 else None})
    return {"seed": SAMPLE_SEED, "method": "independent_numpy_rng_stratified_without_replacement",
            "replay_size": len(rows), "terminal_indices": index_groups["terminal"],
            "nonterminal_indices": index_groups["nonterminal"], "transitions": samples}


def _same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _same(value, right[key]) for key, value in left.items())
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right))
    return left == right


def save_learning_snapshot(directory, agent, config, provenance, env_step):
    """Save one immutable step file without sampling from any training RNG."""
    if config["algorithm"] != "sac":
        return None
    if any(not hasattr(agent, name) for name in NETWORKS) or not hasattr(agent, "replay"):
        raise ValueError("SAC learning snapshot requires actor, critics, targets and replay")
    path = Path(directory) / "learning_snapshots" / f"step_{env_step:09d}.pt"
    payload = {"format": FORMAT, "config": deepcopy(config), "config_hash": config_hash(config),
               "environment_profile": deepcopy(provenance["environment_profile"]),
               "network_kind": config["network"]["kind"],
               "transformer_layout": config["network"].get("transformer_layout", "scalar_tokens"),
               "state_features": list(config["state"]["features"]),
               "provenance": deepcopy(provenance), "env_step": int(env_step),
               "update_step": int(agent.update_step),
               "networks": {name: _cpu_state(getattr(agent, name)) for name in NETWORKS},
               "network_training_modes": {name: bool(getattr(agent, name).training) for name in NETWORKS},
               "log_alpha": agent.log_alpha.detach().cpu().clone(),
               "alpha": float(agent.log_alpha.detach().exp().cpu().item()),
               "replay_sample": _sample_replay(agent.replay)}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if not _same(saved, payload):
            raise ValueError(f"Existing learning snapshot differs at step {env_step}: {path}")
        return str(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)
