"""Frozen, explicit controls for the S2-P19 I0 improvement route."""

from copy import deepcopy
import math


DEFAULT_CONFIG = {
    "schema_version": "s2p19_i0_v1",
    "profile": "europe_v1",
    "power_budget_w": 3000.0,
    "basic_fraction": 0.1,
    "seed": 51001,
    "max_decisions": 32,
    "max_probes": 32,
    "candidate_limit": 16,
    "si_tolerance": 1e-8,
    "sgm_tolerance": 1e-8,
    "gamma": 1.0,
    "alpha_mode": "fixed",
    "fixed_alpha": 0.01,
    "target_entropy_ratio": 0.5,
    "actor_lr": 1e-4,
    "critic_lr": 1e-4,
    "tau": 0.005,
    "batch_size": 128,
    "replay_capacity": 20000,
    "warmup_steps": 2000,
    "update_every": 1,
    "gradient_steps": 1,
    "gradient_clip": 10.0,
    "hidden_width": 128,
    # 网络结构（默认即原 I0 的多层感知机；其余仅在 network 不是 mlp 时生效）
    "network": "mlp",
    "embed_dim": 128,
    "num_layers": 2,
    "num_heads": 8,
    "ffn_mult": 4,
    "pointer": False,
    # 起点：greedy 容量感知贪心（原 I0），spread 分散摆放规则（8 组、组内按干扰挑空位）
    "start": "greedy",
    "total_steps": 400000,
    "pilot_stop_steps": 100000,
    "pilot_max_episodes": 3125,
    "max_episodes": 12500,
    "checkpoint_interval": 10000,
    "device": "cuda",
}


def default_config():
    return deepcopy(DEFAULT_CONFIG)


def validate_config(raw):
    if not isinstance(raw, dict):
        raise ValueError("I0 config must be an object")
    extra = set(raw) - set(DEFAULT_CONFIG)
    if extra:
        raise ValueError(f"Unknown I0 controls: {sorted(extra)}")
    config = default_config()
    config.update(deepcopy(raw))
    s2fix = config["schema_version"] == "s2fix_i0_v1"
    if config["schema_version"] != DEFAULT_CONFIG["schema_version"] and not s2fix:
        raise ValueError("Unknown I0 schema")
    if config["profile"] != "europe_v1" or config["power_budget_w"] != 3000.0 or config["basic_fraction"] != 0.1:
        raise ValueError("I0 requires the frozen European physical domain")
    for key in ("seed", "max_decisions", "max_probes", "candidate_limit", "batch_size",
                "replay_capacity", "warmup_steps", "update_every", "gradient_steps",
                "hidden_width", "total_steps", "pilot_stop_steps", "pilot_max_episodes",
                "max_episodes", "checkpoint_interval"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"Invalid {key}")
    if not s2fix and (config["max_decisions"] != 32 or config["max_probes"] != 32 or config["candidate_limit"] != 16):
        raise ValueError("I0 decision/probe/candidate limits are frozen at 32/32/16")
    if s2fix and (min(config["max_decisions"], config["max_probes"]) < 1
                  or config["candidate_limit"] != 16):
        raise ValueError("S2-FIX requires positive search limits and 16 candidates")
    if config["gamma"] != 1.0 or config["alpha_mode"] != "fixed":
        raise ValueError("I0 reward telescoping requires gamma=1 and frozen fixed-alpha SAC")
    if not s2fix and (config["total_steps"] != 400000 or config["pilot_stop_steps"] != 100000
            or config["pilot_max_episodes"] != 3125 or config["max_episodes"] != 12500):
        raise ValueError("I0 requires a frozen 100k/3125 pilot inside 400k/12500 total")
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("Invalid I0 device")
    for key in ("si_tolerance", "sgm_tolerance", "fixed_alpha", "target_entropy_ratio",
                "actor_lr", "critic_lr", "tau", "gradient_clip"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {key}")
    if config["si_tolerance"] != 1e-8 or config["sgm_tolerance"] != 1e-8:
        raise ValueError("I0 numerical guards are frozen")
    if not 0 < config["fixed_alpha"] or not 0 <= config["target_entropy_ratio"] <= 1:
        raise ValueError("Invalid SAC entropy settings")
    if not 0 < config["tau"] <= 1 or min(config["actor_lr"], config["critic_lr"]) <= 0:
        raise ValueError("Invalid optimizer settings")
    if config["batch_size"] < 1 or config["replay_capacity"] < config["batch_size"]:
        raise ValueError("Invalid replay capacity")
    if config["network"] not in ("mlp", "deepsets", "transformer") or config["start"] not in ("greedy", "spread"):
        raise ValueError("Invalid I0 network or start")
    if config["network"] == "transformer" and config["embed_dim"] % config["num_heads"]:
        raise ValueError("embed_dim must be divisible by num_heads")
    return config
