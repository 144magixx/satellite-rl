"""Small, read-only SAC learning evidence at each complete checkpoint boundary.

These files support offline Critic and terminal TD diagnosis. They are not
recoverable training checkpoints and deliberately omit optimizer/replay state.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from satellite_rl.training.run_monitor import replace_with_retry
from satellite_rl.workbench.checkpoints import config_hash


FORMAT = "satellite_rl_sac_learning_snapshot_v1"
TEMPORAL_FORMAT = "satellite_rl_sac_learning_snapshot_temporal_v1"
SAMPLE_SEED = 51001
SAMPLE_PER_STRATUM = 64
NETWORKS = ("actor", "critic1", "critic2", "target1", "target2")
SOURCE_FIELDS = ("git_commit", "source_root", "source_manifest_sha256")


def _cpu_state(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _cpu_mask(mask):
    if mask is None:
        return None
    return {key: torch.as_tensor(np.asarray(value, dtype=np.bool_)).clone()
            for key, value in mask.items()}


def _sample_replay(replay):
    rows = replay.eligible_rows() if hasattr(replay, "eligible_rows") else replay.rows
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
            if len(row) not in (5, 7, 10):
                raise ValueError("Unsupported SAC replay transition shape")
            sample = {"replay_index": index,
                            "state": torch.as_tensor(np.asarray(row[0], dtype=np.float32)).clone(),
                            "action": torch.as_tensor(row[1], dtype=torch.int64).clone(),
                            "reward": float(row[2]),
                            "next_state": torch.as_tensor(np.asarray(row[3], dtype=np.float32)).clone(),
                            "done": bool(row[4]),
                            "action_mask": _cpu_mask(row[5]) if len(row) >= 7 else None,
                            "next_action_mask": _cpu_mask(row[6]) if len(row) >= 7 else None}
            if len(row) == 10:
                sample.update(episode_id=row[7], episode_step=row[8],
                              actual_action=deepcopy(row[9]))
                logical_index = len(replay.rows) - len(replay) + index
                context = replay._context(logical_index)
                sample["context_states"] = torch.as_tensor(
                    np.stack(context), dtype=torch.float32).clone()
                sample["context_length"] = len(context)
                if row[4]:
                    sample["next_context_states"] = None
                    sample["next_context_length"] = 0
                else:
                    following = (context + [row[3]])[-replay.window:]
                    sample["next_context_states"] = torch.as_tensor(
                        np.stack(following), dtype=torch.float32).clone()
                    sample["next_context_length"] = len(following)
            samples.append(sample)
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


def _source_identity(provenance):
    return tuple(provenance.get(key) for key in SOURCE_FIELDS)


def _verified_source_chain(saved, current, source_history):
    """Accept only reverified I/O repair or approved performance source paths."""
    from satellite_rl.workbench.recovery import REPAIR_ID, verify_io_retry_repair

    prior = _source_identity(saved)
    target = _source_identity(current)
    if not all(prior) or not all(target) or not isinstance(source_history, list):
        raise ValueError("Learning snapshot source differs without a complete I/O repair chain")
    proofs = []
    seen = set()
    while prior != target:
        if prior in seen:
            raise ValueError("Learning snapshot I/O repair source chain has a cycle")
        seen.add(prior)
        matches = [item for item in source_history if isinstance(item, dict) and
                   (item.get("repair_id") == REPAIR_ID or
                    item.get("upgrade_type") == "async_telemetry_v1") and
                   (item.get("old_commit"), item.get("old_source_root"),
                    item.get("old_manifest_sha256")) == prior]
        if len(matches) != 1:
            raise ValueError("Learning snapshot source is absent from the verified I/O repair chain")
        item = matches[0]
        if item.get("upgrade_type") == "async_telemetry_v1":
            from satellite_rl.workbench.performance_upgrade import verify_source_history_entry
            verified = verify_source_history_entry(item)
        else:
            verified = verify_io_retry_repair(item["old_source_root"], item["old_commit"],
                                              item["new_source_root"], item["new_commit"])
        if any(item.get(key) != value for key, value in verified.items()):
            raise ValueError("Learning snapshot I/O repair proof differs from frozen source files")
        prior = (verified["new_commit"], verified["new_source_root"],
                 verified["new_manifest_sha256"])
        proofs.append(verified)
    return proofs


def _record_recovery(path, saved, current, proofs):
    current_commit = current["git_commit"]
    audit = path.with_name(f"{path.stem}_recovery_{current_commit[:12]}.json")
    record = {"format": "satellite_rl_learning_snapshot_source_recovery_v1",
              "snapshot": str(path), "snapshot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "previous_source": {key: saved[key] for key in SOURCE_FIELDS},
              "current_source": {key: current[key] for key in SOURCE_FIELDS},
              "verified_repair_chain": proofs}
    if audit.exists():
        if json.loads(audit.read_text(encoding="utf-8")) != record:
            raise ValueError(f"Existing learning snapshot recovery record differs: {audit}")
        return
    temporary = audit.with_name(audit.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        replace_with_retry(temporary, audit)
    finally:
        temporary.unlink(missing_ok=True)


def save_learning_snapshot(directory, agent, config, provenance, env_step, source_history=None):
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
    if getattr(agent, "temporal_enabled", False):
        payload.update(format=TEMPORAL_FORMAT, temporal_mode=agent.temporal_mode,
                       temporal_window=agent.temporal_window)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if _same(saved, payload):
            return str(path)
        if not isinstance(saved, dict) or not isinstance(saved.get("provenance"), dict):
            raise ValueError(f"Existing learning snapshot differs at step {env_step}: {path}")
        previous_source = saved["provenance"]
        current_source = payload["provenance"]
        if _source_identity(previous_source) == _source_identity(current_source):
            raise ValueError(f"Existing learning snapshot differs at step {env_step}: {path}")
        comparable = dict(saved)
        comparable["provenance"] = dict(previous_source)
        for key in SOURCE_FIELDS:
            comparable["provenance"][key] = current_source.get(key)
        proofs = _verified_source_chain(previous_source, current_source, source_history)
        expected = payload
        if any(proof.get("upgrade_type") == "async_telemetry_v1" for proof in proofs):
            # Only the separately verified operational telemetry choice may
            # differ. No scientific state, tensors, replay sample or RNG changes.
            comparable["config"] = dict(saved["config"])
            comparable["config"].pop("telemetry", None)
            expected = dict(payload)
            expected["config"] = dict(payload["config"])
            expected["config"].pop("telemetry", None)
        if not _same(comparable, expected):
            raise ValueError(f"Existing learning snapshot semantics differ at step {env_step}: {path}")
        _record_recovery(path, previous_source, current_source, proofs)
        return str(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)
