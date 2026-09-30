"""Explicit first-stage configuration, independent of resource allocation."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math


def default_config():
    return {
        "schema_version": 1, "research_stage": "coverage", "algorithm": "ppo",
        "name": "第一阶段 · 规则基础布局＋PPO微调",
        "network": {"kind": "coverage_mlp", "hidden_sizes": [256, 256], "dropout": 0.0},
        "coverage": {
            "environment_profile": "europe_v1", "initialization": "rule", "method": "ppo",
            "beam_count": 100, "beam_width_deg": 0.5, "sweeps": 2,
            "coordinate_step_deg": 0.1, "max_movement_deg": 0.25,
            "min_coverage_demand": 0.99, "min_coverage_users": 0.99,
            "cluster_iterations": 12,
            "score_coverage_demand_weight": 100.0,
            "score_coverage_users_weight": 1.0, "score_load_cv_weight": 1.0,
            "score_movement_weight": 0.1,
        },
        "training": {
            "device": "cuda", "seed": 51001, "total_steps": 409600,
            "num_envs": 64, "rollout_steps": 64, "minibatch_size": 1024,
            "epochs": 4, "lr": 0.0003, "gamma": 1.0, "gae_lambda": 0.95,
            "clip_ratio": 0.2, "entropy_coef": 0.01, "value_coef": 0.5,
            "max_grad_norm": 0.5, "validation_interval": 20480,
            "checkpoint_interval": 20480, "torch_threads": 2,
        },
        "evaluation": {"action_mode": "deterministic", "repeats": 1},
        "datasets": {},
    }


def _number(value, label, minimum=None, maximum=None, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}须为有限数值")
    if integer and (not isinstance(value, int)):
        raise ValueError(f"{label}须为整数")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label}不得小于{minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label}不得大于{maximum}")


def _merge(default, supplied, path=""):
    if not isinstance(supplied, dict):
        raise ValueError(f"{path or '配置'}须为对象")
    unknown = set(supplied) - set(default)
    if unknown:
        raise ValueError(f"{path}存在未实现的配置项: {sorted(unknown)}")
    result = deepcopy(default)
    for key, value in supplied.items():
        if isinstance(default[key], dict) and key != "datasets":
            result[key] = _merge(default[key], value, f"{path}{key}.")
        else:
            result[key] = deepcopy(value)
    return result


def validate_config(config):
    c = _merge(default_config(), config)
    if c["research_stage"] != "coverage" or c["algorithm"] != "ppo" or c["schema_version"] != 1:
        raise ValueError("当前仅支持第一阶段coverage/ppo/v1配置")
    if not isinstance(c["name"], str) or not 1 <= len(c["name"].strip()) <= 100:
        raise ValueError("方案名称须为1–100字符")
    n, v, t = c["network"], c["coverage"], c["training"]
    if n["kind"] != "coverage_mlp" or n["dropout"] != 0:
        raise ValueError("布局PPO首版使用独立coverage_mlp且dropout=0")
    if not isinstance(n["hidden_sizes"], list) or not 1 <= len(n["hidden_sizes"]) <= 4:
        raise ValueError("hidden_sizes须为1–4层宽度")
    for width in n["hidden_sizes"]:
        _number(width, "隐藏层宽度", 16, 2048, integer=True)
    if v["environment_profile"] != "europe_v1" or v["beam_count"] != 100 or v["beam_width_deg"] != 0.5:
        raise ValueError("首版固定欧洲环境、100束与0.5°全宽")
    if v["initialization"] not in ("rule", "cluster") or v["method"] not in ("none", "local_search", "ppo"):
        raise ValueError("不支持的基础布局或微调方法")
    _number(v["sweeps"], "轮询次数", 1, 20, integer=True)
    _number(v["coordinate_step_deg"], "经纬度格步长", 0.001, 2.0)
    _number(v["max_movement_deg"], "最大累计卫星视角移动", 0.001, 1.0)
    for key in ("min_coverage_demand", "min_coverage_users"):
        _number(v[key], key, 0, 1)
    _number(v["cluster_iterations"], "聚类迭代次数", 1, 200, integer=True)
    for key in ("score_coverage_demand_weight", "score_coverage_users_weight", "score_load_cv_weight", "score_movement_weight"):
        _number(v[key], key, 0, 10000)
    if t["device"] not in ("cuda", "cpu"):
        raise ValueError("设备须为cuda或cpu；正式任务另外强制CUDA")
    for key, lo, hi in (("seed", 0, 2**32-1), ("total_steps", 1, 100000000),
                        ("num_envs", 1, 2048), ("rollout_steps", 1, 1024),
                        ("minibatch_size", 1, 131072), ("epochs", 1, 32),
                        ("validation_interval", 0, 100000000), ("checkpoint_interval", 1, 100000000),
                        ("torch_threads", 1, 32)):
        _number(t[key], key, lo, hi, integer=True)
    if t["total_steps"] % t["num_envs"]:
        raise ValueError("正式交互预算须能被并行环境数整除")
    if t["seed"] != 51001:
        raise ValueError("当前第一阶段已授权范围固定训练种子51001")
    if t["gamma"] != 1.0:
        raise ValueError("首版以有限时域评分差分为奖励，要求gamma=1以对应终态目标")
    for key in ("gae_lambda", "clip_ratio"):
        _number(t[key], key, 0, 1)
    for key in ("lr", "max_grad_norm"):
        _number(t[key], key, 1e-8, 10)
    for key in ("entropy_coef", "value_coef"):
        _number(t[key], key, 0, 100)
    ev = c["evaluation"]
    if ev["action_mode"] not in ("deterministic", "stochastic"):
        raise ValueError("评价动作方式未知")
    _number(ev["repeats"], "评价重复次数", 1, 30, integer=True)
    if not isinstance(c["datasets"], dict) or set(c["datasets"]) - {"train", "validation", "test"}:
        raise ValueError("datasets仅支持train/validation/test三分区")
    for role, record in c["datasets"].items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not record["path"].strip():
            raise ValueError(f"{role}须提供明确的逐机/覆盖场景路径")
        if record.get("partition_role", role) != role:
            raise ValueError(f"{role}数据用途不一致")
    return c


def config_hash(config):
    c = validate_config(config)
    return hashlib.sha256(json.dumps(c, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def summary(config):
    c = validate_config(config)
    return {"research_stage": "coverage", "state_dim": 508, "action_dim": 9,
            "initialization": c["coverage"]["initialization"], "method": c["coverage"]["method"],
            "seed": c["training"]["seed"], "total_steps": c["training"]["total_steps"],
            "reward": "全体需求覆盖、用户覆盖、负载CV与移动代价的评分差分（gamma=1）",
            "config_hash": config_hash(c)}
