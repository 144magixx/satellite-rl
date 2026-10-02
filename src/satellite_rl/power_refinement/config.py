"""Validated task controls for frozen-policy, fixed-slot power refinement."""

from __future__ import annotations

from copy import deepcopy
import math


METHODS = ("none", "discrete", "slsqp")
TOTAL_POWER_W = 3000.0
DEFAULTS = {
    "methods": list(METHODS),
    "action_mode": "stochastic",
    "action_seed": 53001,
    "repeats": 3,
    "power_budget_mode": "total",
    "total_power_w": TOTAL_POWER_W,
    "si_tolerance": 1e-8,
    "sgm_tolerance": 1e-8,
    "protect_basic_service": True,
    "maxiter": 300,
    "maxeval": 2000,
    "max_time_seconds": 30.0,
    "discrete_step_w": 5.0,
}


def default_config():
    return deepcopy(DEFAULTS)


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name}须为{minimum}–{maximum}的整数")
    return value


def _number(value, name, minimum, maximum):
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name}须为{minimum}–{maximum}的有限数值")
    return float(value)


def validate_config(raw):
    """Reject unknown or misleading bounds; continuous powers have no per-beam cap."""
    if not isinstance(raw, dict):
        raise ValueError("功率细调配置须为对象")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError("未知功率细调参数：" + ", ".join(sorted(unknown)))
    result = default_config()
    result.update(deepcopy(raw))
    if result["methods"] != list(METHODS):
        raise ValueError("首轮须对同一初始动作依次评价原档位、离散细调和连续SLSQP")
    if result["action_mode"] not in ("deterministic", "stochastic"):
        raise ValueError("动作方式须为确定性或随机采样")
    result["action_seed"] = _integer(result["action_seed"], "动作种子", 0, 2**32 - 1)
    result["repeats"] = _integer(result["repeats"], "每场景动作次数", 1, 30)
    if result["power_budget_mode"] not in ("total", "initial"):
        raise ValueError("功率预算口径须为总预算或初始总功率")
    result["total_power_w"] = _number(result["total_power_w"], "总功率预算/W", TOTAL_POWER_W, TOTAL_POWER_W)
    result["si_tolerance"] = _number(result["si_tolerance"], "SI数值容差", 0, 1e-3)
    result["sgm_tolerance"] = _number(result["sgm_tolerance"], "SGM数值容差", 0, 1e-3)
    if result["protect_basic_service"] is not True:
        raise ValueError("首轮必须开启原基本服务保护")
    result["maxiter"] = _integer(result["maxiter"], "最大迭代次数", 1, 10000)
    result["maxeval"] = _integer(result["maxeval"], "最大容量计算次数", 1, 1000000)
    result["max_time_seconds"] = _number(result["max_time_seconds"], "每次优化时限/秒", 0.01, 3600)
    result["discrete_step_w"] = _number(result["discrete_step_w"], "原功率档位步长/W", 5, 5)
    return result
