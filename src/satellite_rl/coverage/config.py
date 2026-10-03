"""Explicit first-stage configuration, independent of resource allocation."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math


def default_config():
    return {
        "schema_version": 2, "research_stage": "coverage", "algorithm": "ppo",
        "name": "第一阶段 · 规则基础布局＋PPO微调",
        "network": {"kind": "coverage_candidate_mlp", "hidden_sizes": [256, 256], "dropout": 0.0},
        "coverage": {
            "environment_profile": "europe_v1", "initialization": "rule", "method": "ppo",
            "beam_count": 100, "beam_width_deg": 0.5, "sweeps": 2,
            "coordinate_step_deg": 0.1, "max_movement_deg": 0.25,
            "min_coverage_demand": 0.99, "min_coverage_users": 0.99,
            "cluster_iterations": 12,
            "candidate_capacity": 2048, "heatmap_bins": 16,
            "score_coverage_demand_weight": 100.0,
            "score_coverage_users_weight": 1.0, "score_load_cv_weight": 1.0,
            "score_movement_weight": 0.1,
        },
        "training": {
            "device": "cuda", "seed": 51001, "total_steps": 409600,
            "num_envs": 512, "rollout_steps": 16, "minibatch_size": 1024,
            "epochs": 4, "lr": 0.0003, "gamma": 1.0, "gae_lambda": 0.95,
            "clip_ratio": 0.2, "entropy_coef": 0.01, "value_coef": 0.5,
            "max_grad_norm": 0.5, "validation_interval": 20480,
            "checkpoint_interval": 20480, "torch_threads": 2,
            "reward_scale": 0.01,
        },
        "evaluation": {"action_mode": "deterministic", "repeats": 1},
        "datasets": {},
    }


def balance_default_config():
    """S1-02 settings; schema 2 remains the unchanged historical default."""
    c = default_config()
    c["schema_version"] = 3
    c["name"] = "第一阶段 · 覆盖优先与主体波束均衡"
    c["coverage"].update(initialization="cluster", method="local_search",
                         sweeps=10, min_coverage_demand=1.0,
                         min_coverage_users=1.0,
                         balance={
                             "profile": "coverage_priority_v1",
                             "route": "A0", "action_menu": "X0",
                             "step_angle_deg": 0.025,
                             "coverage_goal": "strict_100",
                             "subject_cv_target": 0.20,
                             "subject_range_tolerance": 0.20,
                             "min_subject_beams_in_range": 90,
                             "max_edge_beams": 10,
                             "edge_demand_ratio": 0.5,
                             "max_rounds": 10,
                             "max_beam_decisions": 1000,
                             "max_repair_decisions": 200,
                             "max_repair_evaluations": 20000,
                             "max_assignment_passes": 20,
                             "max_assignment_attempts": 200000,
                             "post_move_transfers": 1,
                             "max_candidate_evaluations": 20000,
                             "max_refinement_seconds": 60.0,
                         })
    c["training"].update(num_envs=128, rollout_steps=64)
    c["balance_starts"] = {}
    return c


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
    if not isinstance(config, dict):
        raise ValueError("配置须为对象")
    version = config.get("schema_version", 2)
    if version not in (2, 3):
        raise ValueError("仅支持第一阶段配置版本2或3；旧配置不可静默转换")
    c = _merge(default_config() if version == 2 else balance_default_config(), config)
    if c["research_stage"] != "coverage" or c["algorithm"] != "ppo":
        raise ValueError("仅支持第一阶段coverage/ppo配置")
    if not isinstance(c["name"], str) or not 1 <= len(c["name"].strip()) <= 100:
        raise ValueError("方案名称须为1–100字符")
    n, v, t = c["network"], c["coverage"], c["training"]
    if n["kind"] != "coverage_candidate_mlp" or n["dropout"] != 0:
        raise ValueError("布局PPO使用共用候选评分网络且dropout=0")
    _number(n["dropout"], "dropout", 0, 0)
    if not isinstance(n["hidden_sizes"], list) or not 1 <= len(n["hidden_sizes"]) <= 4:
        raise ValueError("hidden_sizes须为1–4层宽度")
    for width in n["hidden_sizes"]:
        _number(width, "隐藏层宽度", 16, 2048, integer=True)
    if v["environment_profile"] != "europe_v1" or v["beam_count"] != 100 or v["beam_width_deg"] != 0.5:
        raise ValueError("首版固定欧洲环境、100束与0.5°全宽")
    if v["initialization"] not in ("rule", "cluster", "empty") or v["method"] not in ("none", "local_search", "ppo"):
        raise ValueError("不支持的基础布局或微调方法")
    if v["initialization"] == "empty" and (v["method"] != "ppo" or v["sweeps"] != 1):
        raise ValueError("无基础布局仅支持纯PPO从空布局构建，轮询次数须为1（100次全局放置）")
    if version == 3:
        _validate_balance(c)
    if v["candidate_capacity"] != 2048 or v["heatmap_bins"] != 16:
        raise ValueError("本轮共用2048候选上限与16×16业务需求栅格，超上限数据不能裁剪")
    _number(v["sweeps"], "轮询次数", 1, 20, integer=True)
    _number(v["coordinate_step_deg"], "经纬度格步长", 0.001, 2.0)
    _number(v["max_movement_deg"], "最大累计卫星视角移动", 0.001, 1.0)
    for key in ("min_coverage_demand", "min_coverage_users"):
        _number(v[key], key, 0, 1)
    _number(v["cluster_iterations"], "聚类迭代次数", 1, 100, integer=True)
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
    _number(t["gamma"], "gamma", 1, 1)
    if t["gamma"] != 1.0:
        raise ValueError("首版以有限时域评分差分为奖励，要求gamma=1以对应终态目标")
    for key in ("gae_lambda", "clip_ratio"):
        _number(t[key], key, 1e-8, 1)
    for key in ("lr", "max_grad_norm"):
        _number(t[key], key, 1e-8, 10)
    for key in ("entropy_coef", "value_coef"):
        _number(t[key], key, 0, 100)
    if t["reward_scale"] != 0.01:
        raise ValueError("本轮三种PPO共用0.01学习奖励缩放，终态物理评分仍单列")
    ev = c["evaluation"]
    if ev["action_mode"] not in ("deterministic", "stochastic"):
        raise ValueError("评价动作方式未知")
    _number(ev["repeats"], "评价重复次数", 1, 30, integer=True)
    if version == 3 and ev["repeats"] != 1:
        raise ValueError("S1-02共同考核每场景只执行一次；重复评价须另列预算")
    if not isinstance(c["datasets"], dict) or set(c["datasets"]) - {"train", "validation", "test"}:
        raise ValueError("datasets仅支持train/validation/test三分区")
    for role, record in c["datasets"].items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not record["path"].strip():
            raise ValueError(f"{role}须提供明确的逐机/覆盖场景路径")
        if record.get("partition_role", role) != role:
            raise ValueError(f"{role}数据用途不一致")
    if version == 3:
        starts = c["balance_starts"]
        if not isinstance(starts, dict) or set(starts) - {"train", "validation", "test"}:
            raise ValueError("A2起点仅支持训练、验证、考核分区")
        for role, record in starts.items():
            if (role not in c["datasets"] or not isinstance(record, dict) or
                    not isinstance(record.get("path"), str) or
                    not isinstance(record.get("sha256"), str) or
                    len(record["sha256"]) != 64 or
                    record.get("dataset_sha256") != c["datasets"][role].get("sha256")):
                raise ValueError(f"{role}的A2起点必须匹配场景来源与SHA")
    return c


def _validate_balance(c):
    v, t = c["coverage"], c["training"]
    b = v["balance"]
    if v["initialization"] != "cluster":
        raise ValueError("S1-02须从同一聚类C1基础布局出发")
    historical = default_config()["coverage"]
    for key in ("coordinate_step_deg", "score_coverage_demand_weight",
                "score_coverage_users_weight", "score_load_cv_weight",
                "score_movement_weight"):
        if v[key] != historical[key]:
            raise ValueError(f"S1-02固定旧C1参照参数{key}，新菜单使用独立卫星视角格宽")
    if v["sweeps"] != b["max_rounds"]:
        raise ValueError("显示轮询次数须与S1-02束处理轮数一致")
    if b["profile"] != "coverage_priority_v1":
        raise ValueError("未知覆盖与主体均衡物理版本")
    if b["route"] not in ("prepare", "A0", "A1", "A2", "A3", "X0", "X1", "X2", "X3", "X4", "ppo"):
        raise ValueError("未知S1-02路线")
    if b["action_menu"] not in ("X0", "X1", "X2", "X3", "X4"):
        raise ValueError("动作菜单须为X0–X4")
    if b["route"] in ("A3", "X0") and b["action_menu"] != "X0":
        raise ValueError("A3/X0使用共同四向5动作菜单")
    if b["route"] in ("X1", "X2", "X3", "X4") and b["action_menu"] != b["route"]:
        raise ValueError("路线与动作菜单必须一致")
    if b["route"] == "ppo" and v["method"] != "ppo":
        raise ValueError("PPO路线须实际使用PPO方法")
    if b["route"] != "ppo" and v["method"] == "ppo":
        raise ValueError("非学习路线不能标为PPO")
    if b["coverage_goal"] not in ("strict_100", "near_99_9"):
        raise ValueError("覆盖目标须明确选100%或双99.9%")
    expected = 1.0 if b["coverage_goal"] == "strict_100" else 0.999
    if v["min_coverage_demand"] != expected or v["min_coverage_users"] != expected:
        raise ValueError("需求与用户覆盖阈值须与所选全局目标一致")
    for key, low, high, integer in (
        ("step_angle_deg", .0001, 1, False),
        ("subject_cv_target", 0, 2, False),
        ("subject_range_tolerance", 0, 1, False),
        ("min_subject_beams_in_range", 0, 100, True),
        ("max_edge_beams", 0, 100, True),
        ("edge_demand_ratio", 0, 1, False),
        ("max_rounds", 1, 10, True),
        ("max_beam_decisions", 1, 1000, True),
        ("max_repair_decisions", 1, 200, True),
        ("max_repair_evaluations", 1, 20000, True),
        ("max_assignment_passes", 1, 20, True),
        ("max_assignment_attempts", 1, 200000, True),
        ("post_move_transfers", 1, 1, True),
        ("max_candidate_evaluations", 1, 20000, True),
        ("max_refinement_seconds", .001, 60, False),
    ):
        _number(b[key], key, low, high, integer=integer)
    if b["max_rounds"] * v["beam_count"] < b["max_beam_decisions"]:
        raise ValueError("束处理上限不能超过轮数×100束")
    if b["route"] == "ppo" and (t["num_envs"] != 128 or t["rollout_steps"] != 64 or
                                      t["minibatch_size"] != 1024 or t["epochs"] != 4 or
                                      t["total_steps"] != 409600 or
                                      t["validation_interval"] != 20480 or
                                      t["checkpoint_interval"] != 20480):
        raise ValueError("S1-02正式PPO固定128环境×64采集、1024小批量×4轮、409600交互及每20480步验证阈值/断点")


def config_hash(config):
    c = validate_config(config)
    return hashlib.sha256(json.dumps(c, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def summary(config):
    c = validate_config(config)
    if c["schema_version"] == 3:
        menu = c["coverage"]["balance"]["action_menu"]
        return {"research_stage": "coverage", "schema_version": 3,
                "route": c["coverage"]["balance"]["route"], "action_menu": menu,
                "action_dim": {"X0":5,"X1":9,"X2":13,"X3":21,"X4":37}[menu],
                "candidate_feature_dim": 8, "state_dim": "由冻结环境实测",
                "coverage_goal": c["coverage"]["balance"]["coverage_goal"],
                "subject_cv_target": c["coverage"]["balance"]["subject_cv_target"],
                "seed": c["training"]["seed"], "total_steps": c["training"]["total_steps"],
                "config_hash": config_hash(c)}
    pure = c["coverage"]["initialization"] == "empty"
    return {"research_stage": "coverage", "state_dim": 964,
            "candidate_feature_dim": 4, "action_dim": 2048 if pure else 9,
            "initialization": c["coverage"]["initialization"], "method": c["coverage"]["method"],
            "seed": c["training"]["seed"], "total_steps": c["training"]["total_steps"],
            "reward": "终态覆盖/负载评分差分×0.01（gamma=1）；纯构建无移动代价",
            "config_hash": config_hash(c)}
