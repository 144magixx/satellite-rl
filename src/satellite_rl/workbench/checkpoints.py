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

from satellite_rl.agents.workbench_algorithms import build_agent, resolve_gamma
from satellite_rl.agents.workbench_networks import model_description
from satellite_rl.training.run_monitor import replace_with_retry


def _canonical(config):
    return json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sac_structures(config):
    network = config.get("network", {})
    return (network.get("sac_policy_structure", "factorized"),
            network.get("sac_critic_structure", "additive"))


def _experiment_modes(config):
    network = config.get("network", {})
    training = config.get("training", {})
    modes = {"alpha_mode": training.get("alpha_mode", "adaptive"),
            "fixed_alpha": training.get("fixed_alpha", 0.01),
            "temporal_mode": network.get("temporal_mode", "none"),
            "temporal_window": network.get("temporal_window", 16),
            "graph_mode": network.get("graph_mode", "none")}
    if training.get("beam_count_mode", "none") != "none":
        modes.update({key: training[key] for key in
                      ("beam_count_mode", "beam_count_min", "beam_count_max", "held_out_beam_counts")})
    # S2-P18：只有显式启用时才把新学习机制写进模式元数据，旧配置的元数据与
    # experiment_modes 逐位不变，旧断点与旧模型仍按原样加载。
    if int(training.get("progress_strata", 1)) > 1:
        modes["progress_strata"] = int(training["progress_strata"])
    schedule = training.get("gamma_schedule") or {"mode": "none"}
    if schedule.get("mode", "none") != "none":
        modes["gamma_schedule"] = schedule
        modes["effective_gamma_by_beam_count"] = {
            str(count): resolve_gamma(training, count) for count in (50, 100, 120, 160, 200)}
    return modes


def config_hash(config):
    """Hash all semantics, including state order, action mapping and normalization."""
    # Only run budget/log cadence may change without changing an inference policy.
    # Absent and explicitly disabled masking are the same historical semantics.
    # Keep pre-mask policy/checkpoint hashes loadable without relaxing other fields.
    semantic = dict(config)
    # Telemetry changes persistence/display latency only, never learning inputs,
    # losses, RNG, actions or model compatibility. Full configs still retain it.
    semantic.pop("telemetry", None)
    semantic["environment"] = dict(config["environment"])
    if semantic["environment"].get("beam_count_max", 100) == 100:
        semantic["environment"].pop("beam_count_max", None)
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
    if semantic["network"].get("sac_policy_structure", "factorized") == "factorized":
        semantic["network"].pop("sac_policy_structure", None)
    if semantic["network"].get("sac_critic_structure", "additive") == "additive":
        semantic["network"].pop("sac_critic_structure", None)
    temporal_mode = semantic["network"].get("temporal_mode", "none")
    if temporal_mode == "none":
        semantic["network"].pop("temporal_mode", None)
    if temporal_mode in ("none", "previous_action"):
        semantic["network"].pop("temporal_window", None)
    if semantic["network"].get("graph_mode", "none") == "none":
        semantic["network"].pop("graph_mode", None)
    if semantic["network"].get("transformer_layout", "scalar_tokens") == "scalar_tokens":
        semantic["network"].pop("transformer_layout", None)
    # S2-P18 新增键：只有离开默认值时才纳入语义哈希，因此改动前已冻结的配置
    # （V0/V1、T0、欧洲欧洲对照等）保持原来的 config_hash，旧断点与旧模型可原样恢复；
    # 一旦真正启用新奖励/新采样/新逐束编码，哈希随之变化，权重不会被误认为兼容。
    if semantic["network"].get("context_beam_count", "fixed_100") == "fixed_100":
        semantic["network"].pop("context_beam_count", None)
    if semantic["network"].get("readout_mode", "first_token") == "first_token":
        semantic["network"].pop("readout_mode", None)
    if semantic["network"].get("beam_count_conditioning", "none") == "none":
        semantic["network"].pop("beam_count_conditioning", None)
    semantic["reward"] = dict(config["reward"])
    shaping = semantic["reward"].get("shaping")
    if (not any(shaping.get(key, 0) for key in ("w_throughput", "w_spectrum", "w_shape"))
            and shaping.get("throughput_reference", "mean_demand") == "mean_demand"):
        semantic["reward"].pop("shaping", None)
    semantic["evaluation"] = dict(config["evaluation"])
    if semantic["evaluation"].get("extra_steps", []) == []:
        semantic["evaluation"].pop("extra_steps", None)
    semantic["state"] = dict(config["state"])
    if semantic["state"].get("context_level", "none") == "none":
        semantic["state"].pop("context_level", None)
    semantic["training"] = dict(config["training"])
    if semantic["training"].get("beam_count_mode", "none") == "none":
        for key in ("beam_count_mode", "beam_count_min", "beam_count_max", "held_out_beam_counts"):
            semantic["training"].pop(key, None)
    if semantic["training"].get("alpha_mode", "adaptive") == "adaptive":
        semantic["training"].pop("alpha_mode", None)
        semantic["training"].pop("fixed_alpha", None)
    # Old tasks predate scene-order options; the explicit defaults preserve
    # their exact ordering and therefore retain their original semantic hash.
    if semantic["training"].get("scene_order", "sequential") == "sequential":
        semantic["training"].pop("scene_order", None)
        if semantic["training"].get("scene_order_seed", 51001) == 51001:
            semantic["training"].pop("scene_order_seed", None)
    if int(semantic["training"].get("progress_strata", 1)) == 1:
        semantic["training"].pop("progress_strata", None)
    if (semantic["training"].get("gamma_schedule") or {"mode": "none"}).get("mode", "none") == "none":
        semantic["training"].pop("gamma_schedule", None)
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
    from satellite_rl.workbench.config import config_summary
    summary = config_summary(config)
    description = describe_model(config)
    metadata = {"format": "satellite_rl_workbench_policy_v1", "algorithm": config["algorithm"],
                "environment_profile": profile_metadata(config["environment"].get("profile", "china_legacy")),
                "variant": description["variant"], "config": config,
                "config_hash": config_hash(config), "provenance": provenance,
                "weights_sha256": hashlib.sha256(weight_path.read_bytes()).hexdigest(),
                "state_layout": summary["state_layout"], "state_dim": summary["state_dim"],
                "network_token_count": summary["network_token_count"],
                "parameter_count": description["parameter_count"],
                "sac_policy_structure": _sac_structures(config)[0],
                "sac_critic_structure": _sac_structures(config)[1],
                "experiment_modes": _experiment_modes(config),
                "evaluation_extra_steps": config["evaluation"].get("extra_steps", []),
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

    def reset_episode(self):
        self.agent.reset_episode()


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
    if (metadata.get("sac_policy_structure", "factorized"),
            metadata.get("sac_critic_structure", "additive")) != _sac_structures(saved):
        raise ValueError("Model SAC action structures do not match its saved configuration")
    if metadata.get("experiment_modes", _experiment_modes(saved)) != _experiment_modes(saved):
        raise ValueError("Model temperature, temporal or graph mode metadata differs")
    if metadata.get("evaluation_extra_steps", []) != saved.get("evaluation", {}).get("extra_steps", []):
        raise ValueError("Model evaluation steps do not match its saved configuration")
    if metadata.get("variant") != describe_model(saved)["variant"]:
        raise ValueError("Model variant does not match its configured network layout")
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
                         "sac_policy_structure": _sac_structures(config)[0],
                         "sac_critic_structure": _sac_structures(config)[1],
                         "experiment_modes": _experiment_modes(config),
                         "evaluation_extra_steps": config["evaluation"].get("extra_steps", []),
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
    if (payload.get("sac_policy_structure", "factorized"),
            payload.get("sac_critic_structure", "additive")) != _sac_structures(config):
        raise ValueError("Checkpoint SAC action structures mismatch")
    if payload.get("experiment_modes", _experiment_modes(config)) != _experiment_modes(config):
        raise ValueError("Checkpoint temperature, temporal or graph modes mismatch")
    if payload.get("evaluation_extra_steps", []) != config["evaluation"].get("extra_steps", []):
        raise ValueError("Checkpoint evaluation steps mismatch")
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
