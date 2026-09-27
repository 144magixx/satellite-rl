"""Strict inference packages and recoverable workbench checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch

from satellite_rl.agents.workbench_algorithms import build_agent
from satellite_rl.agents.workbench_networks import model_description
from satellite_rl.training.run_monitor import replace_with_retry


def _canonical(config):
    return json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_hash(config):
    """Hash all semantics, including state order, action mapping and normalization."""
    # Only run budget/log cadence may change without changing an inference policy.
    # Absent and explicitly disabled masking are the same historical semantics.
    # Keep pre-mask policy/checkpoint hashes loadable without relaxing other fields.
    semantic = dict(config)
    semantic["environment"] = dict(config["environment"])
    profile = semantic["environment"].get("profile", "china_legacy")
    if profile == "china_legacy":
        # Old China policies predate regional configuration; the absent default
        # must retain its original hash. Europe uses a distinct physical/input contract.
        semantic["environment"].pop("profile", None)
    else:
        from satellite_rl.envs.regions import profile_metadata
        semantic["environment"]["profile_metadata"] = profile_metadata(profile)
    if semantic["environment"].get("action_mask", "none") == "none":
        semantic["environment"].pop("action_mask", None)
    semantic["network"] = dict(config["network"])
    if semantic["network"].get("transformer_layout", "scalar_tokens") == "scalar_tokens":
        semantic["network"].pop("transformer_layout", None)
    return hashlib.sha256(_canonical(semantic).encode("utf-8")).hexdigest()


def _dimensions(config):
    from satellite_rl.workbench.config import config_summary
    summary = config_summary(config)
    return summary["state_dim"], summary["action_dims"]


def describe_model(config):
    state_dim, action_dims = _dimensions(config)
    return model_description(config, state_dim, action_dims)


def _atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        torch.save(payload, temporary)
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_policy(directory, agent, config, provenance):
    """Write tensor-only policy weights and human-readable, strict metadata."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    policy = agent.policy_network()
    weights = {name: value.detach().cpu() for name, value in policy.state_dict().items()}
    weight_path = directory / "policy.pt"
    _atomic_torch(weight_path, weights)
    from satellite_rl.envs.regions import profile_metadata
    metadata = {"format": "satellite_rl_workbench_policy_v1", "algorithm": config["algorithm"],
                "environment_profile": profile_metadata(config["environment"].get("profile", "china_legacy")),
                "variant": describe_model(config)["variant"], "config": config,
                "config_hash": config_hash(config), "provenance": provenance,
                "weights_sha256": hashlib.sha256(weight_path.read_bytes()).hexdigest(),
                "state_layout": __import__("satellite_rl.workbench.config", fromlist=["config_summary"]).config_summary(config)["state_layout"],
                "action_dims": _dimensions(config)[1]}
    _atomic_json(directory / "model.json", metadata)
    return str(directory)


class Policy:
    def __init__(self, agent, config, metadata):
        self.agent = agent
        self.config = config
        self.metadata = metadata

    @property
    def last_action_info(self):
        return self.agent.last_action_info

    def act(self, state, deterministic=True, action_mask=None):
        return self.agent.act(state, deterministic=deterministic, action_mask=action_mask)


def load_policy(path, config=None, device="cpu"):
    directory = Path(path)
    if directory.is_file() and directory.name == "policy.pt":
        directory = directory.parent
    metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    if metadata.get("format") != "satellite_rl_workbench_policy_v1":
        raise ValueError("Unsupported model package")
    saved = metadata["config"]
    from satellite_rl.envs.regions import profile_metadata
    expected_profile = profile_metadata(saved["environment"].get("profile", "china_legacy"))
    if expected_profile["id"] != "china_legacy" and "environment_profile" not in metadata:
        raise ValueError("Model environment profile metadata missing")
    if metadata.get("environment_profile", expected_profile) != expected_profile:
        raise ValueError("Model environment profile metadata mismatch")
    if config is not None and config_hash(config) != metadata["config_hash"]:
        raise ValueError("Model config differs in state, actions, normalization or other semantics")
    if config_hash(saved) != metadata["config_hash"]:
        raise ValueError("Model metadata config hash mismatch")
    weight_path = directory / "policy.pt"
    if weight_path.stat().st_size > 512 * 1024 ** 2:
        raise ValueError("Inference weight file exceeds the supported 512 MiB limit")
    if hashlib.sha256(weight_path.read_bytes()).hexdigest() != metadata["weights_sha256"]:
        raise ValueError("Model weights checksum mismatch")
    state_dim, action_dims = _dimensions(saved)
    agent = build_agent(saved, state_dim, action_dims, device)
    weights = torch.load(weight_path, map_location=device, weights_only=True)
    if not isinstance(weights, dict) or any(not isinstance(value, torch.Tensor) for value in weights.values()):
        raise ValueError("Model package must contain a tensor-only state dict")
    if any(not torch.isfinite(value).all().item() for value in weights.values()):
        raise ValueError("Model weights contain NaN or Inf; refusing invalid inference package")
    agent.policy_network().load_state_dict(weights, strict=True)
    agent.policy_network().eval()
    return Policy(agent, saved, metadata)


def save_checkpoint(path, agent, progress, config, environment_state):
    """Full trusted local resume point; loaded only from the job's own directory."""
    rng = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch": torch.get_rng_state(),
           "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    from satellite_rl.envs.regions import profile_metadata
    _atomic_torch(path, {"format": "satellite_rl_workbench_checkpoint_v1", "config_hash": config_hash(config),
                         "environment_profile": profile_metadata(config["environment"].get("profile", "china_legacy")),
                         "agent": agent.state_dict(), "progress": progress,
                         "environment_state": environment_state, "rng": rng})
    return str(path)


def load_checkpoint(path, agent, config):
    path = Path(path).resolve()
    if path.name != "checkpoint.pt":
        raise ValueError("Resume only accepts this job's checkpoint.pt")
    # Full optimizer/replay/RNG state requires Python objects. This path must be local and trusted.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "satellite_rl_workbench_checkpoint_v1" or payload.get("config_hash") != config_hash(config):
        raise ValueError("Checkpoint format/config mismatch")
    from satellite_rl.envs.regions import profile_metadata
    expected_profile = profile_metadata(config["environment"].get("profile", "china_legacy"))
    if expected_profile["id"] != "china_legacy" and "environment_profile" not in payload:
        raise ValueError("Checkpoint environment profile metadata missing")
    if payload.get("environment_profile", expected_profile) != expected_profile:
        raise ValueError("Checkpoint environment profile metadata mismatch")
    agent.load_state_dict(payload["agent"])
    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return payload["progress"], payload["environment_state"]
