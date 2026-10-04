"""S1-03 region reconstruction configuration and strict schema validation."""
from __future__ import annotations

from copy import deepcopy
import math


ROUTES = ("prepare", "R0", "R1", "R2", "R3", "P1")
ROUTE_METHOD = {"prepare": "none", "R0": "none", "R1": "local_search",
                "R2": "local_search", "R3": "local_search", "P1": "ppo"}
ROLES = ("train", "validation", "test")


def reconstruction_default_config():
    from .config import default_config

    config = default_config()
    config.update(schema_version=4, name="S1-03 · 需求区域重构与PPO")
    config["network"] = {"kind": "reconstruction_relation_ppo",
                         "hidden_size": 128, "message_layers": 2}
    config["coverage"].update(initialization="cluster", method="none", sweeps=1,
                              min_coverage_demand=.999, min_coverage_users=.999,
                              reconstruction={
                                  "route": "prepare", "max_macro_steps": 32,
                                  "max_geometry_calls_per_action": 256,
                                  "max_geometry_calls_total": 8192,
                                  "max_initialization_geometry_calls": 8192,
                                  "split_templates": [0, 1, 2, 3],
                                  "search_width": 8,
                                  "neighbor_count": 3, "split_target_count": 12,
                                  "edge_demand_ratio": .5, "max_edge_beams": 10,
                                  "subject_cv_target": .2, "range_tolerance": .2,
                                  "min_subject_beams_in_range": 90,
                              })
    config["training"].update(seed=51001, total_steps=262144, num_envs=32,
                              rollout_steps=32, minibatch_size=256, epochs=4,
                              lr=.0003, gamma=1., gae_lambda=.95, clip_ratio=.2,
                              entropy_coef=.01, value_coef=.5, max_grad_norm=.5,
                              reward_scale=1., torch_threads=2,
                              validation_interval=16384, checkpoint_interval=16384,
                              mechanism_gate_step=65536)
    config["reconstruction_starts"] = {}
    return config


def _integer(value, name, lo, hi):
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f"{name}须为{lo}–{hi}的整数")


def _number(value, name, lo, hi):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{name}须为{lo}–{hi}的有限数值")


def validate_reconstruction_config(supplied):
    from .config import _merge

    config = _merge(reconstruction_default_config(), supplied)
    if config["schema_version"] != 4 or config["research_stage"] != "coverage" or config["algorithm"] != "ppo":
        raise ValueError("S1-03须使用schema4/coverage/ppo")
    if not isinstance(config["name"], str) or not 1 <= len(config["name"].strip()) <= 100:
        raise ValueError("方案名称须为1–100字符")
    if config["network"] != reconstruction_default_config()["network"]:
        raise ValueError("S1-03固定128维、两层关系PPO网络")
    coverage, training = config["coverage"], config["training"]
    default_coverage = reconstruction_default_config()["coverage"]
    for key, expected in default_coverage.items():
        if key not in ("reconstruction", "method") and coverage[key] != expected:
            raise ValueError(f"S1-03覆盖物理/兼容字段{key}须与冻结方案一致")
    reconstruction = coverage["reconstruction"]
    route = reconstruction["route"]
    if route not in ROUTES or coverage["method"] != ROUTE_METHOD[route]:
        raise ValueError("区域重构路线与实际方法不一致")
    for key, lo, hi in (("max_macro_steps", 1, 128),
                        ("max_geometry_calls_per_action", 1, 100000),
                        ("max_geometry_calls_total", 1, 10000000),
                        ("max_initialization_geometry_calls", 1, 10000000),
                        ("search_width", 1, 64),
                        ("neighbor_count", 1, 99), ("split_target_count", 1, 100),
                        ("max_edge_beams", 0, 100),
                        ("min_subject_beams_in_range", 0, 100)):
        _integer(reconstruction[key], key, lo, hi)
    if reconstruction["max_geometry_calls_total"] < reconstruction["max_geometry_calls_per_action"]:
        raise ValueError("回合几何调用额度不得小于单次动作额度")
    if reconstruction["split_templates"] != [0, 1, 2, 3]:
        raise ValueError("本轮固定四种共同需求切分模板0/1/2/3")
    for key, lo, hi in (("edge_demand_ratio", 0, 1),
                        ("subject_cv_target", 0, 2), ("range_tolerance", 0, 1)):
        _number(reconstruction[key], key, lo, hi)
    defaults = reconstruction_default_config()["training"]
    for key, expected in defaults.items():
        if training[key] != expected:
            raise ValueError(f"S1-03训练字段{key}须与授权预算/算法设置一致")
    if config["evaluation"] != {"action_mode": "deterministic", "repeats": 1}:
        raise ValueError("S1-03正式评价每场景仅确定性执行一次")
    datasets = config["datasets"]
    if not isinstance(datasets, dict) or set(datasets) - set(ROLES):
        raise ValueError("数据集仅支持train/validation/test")
    for role, record in datasets.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not record["path"].strip() or record.get("partition_role", role) != role:
            raise ValueError(f"{role}数据集来源与分区不一致")
    starts = config["reconstruction_starts"]
    if not isinstance(starts, dict) or set(starts) - set(ROLES):
        raise ValueError("共同初始化起点仅支持train/validation/test")
    for role, record in starts.items():
        if role not in datasets or not isinstance(record, dict):
            raise ValueError(f"{role}缺少匹配的数据集")
        for field in ("sha256", "dataset_sha256"):
            value = record.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
                raise ValueError(f"{role}共同初始化起点{field}无效")
        if (not isinstance(record.get("path"), str) or not record["path"].strip() or
                not isinstance(record.get("source_run_id"), str) or not record["source_run_id"].strip() or
                record["dataset_sha256"] != datasets[role].get("sha256")):
            raise ValueError(f"{role}共同初始化起点与数据SHA或工作台任务不匹配")
    return config


def reconstruction_summary(config):
    from .config import config_hash

    c = validate_reconstruction_config(config)
    r = c["coverage"]["reconstruction"]
    return {"research_stage": "coverage", "schema_version": 4, "route": r["route"],
            "action_dim": "完整条件宏动作", "max_macro_steps": r["max_macro_steps"],
            "max_geometry_calls_per_action": r["max_geometry_calls_per_action"],
            "max_geometry_calls_total": r["max_geometry_calls_total"],
            "max_initialization_geometry_calls": r["max_initialization_geometry_calls"],
            "terminal_metric": "STOP或额度结束的原终态",
            "seed": c["training"]["seed"], "total_steps": c["training"]["total_steps"],
            "config_hash": config_hash(c)}
