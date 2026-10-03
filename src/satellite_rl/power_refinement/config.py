"""Validated task controls for frozen-policy, fixed-slot power refinement."""

from __future__ import annotations

from copy import deepcopy
import math


LEGACY_METHODS = ("none", "discrete", "slsqp")
METHODS = (*LEGACY_METHODS, "sca", "de", "pso", "wmmse")
SI_PRIORITY_METHODS = ("sca", "sca_si_lp", "sca_si_lb")
METHOD_LABELS = {
    "none": "T0原分配", "discrete": "原5 W档位调整",
    "slsqp": "SLSQP含需求匹配初始化", "sca": "SCA信赖域凸近似",
    "de": "差分进化DE", "pso": "粒子群PSO",
    "wmmse": "WMMSE速率提议与双指标验收",
    "sca_si_lp": "B：从A继续优先提高SI",
    "sca_si_lb": "C：改进功率估计并优先提高SI",
}
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
    "optimizer_seed": 63001,
    "population_size": 24,
    "de_mutation": 0.7,
    "de_crossover": 0.8,
    "pso_inertia": 0.7,
    "pso_cognitive": 1.4,
    "pso_social": 1.4,
    "si_stall_window": 200,
    "si_stall_tolerance": 1e-6,
    "si_min_evaluations": 200,
    "si_max_fraction": 0.85,
    "service_margin_ratio": 5e-9,
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
    if result["methods"] not in (list(LEGACY_METHODS), list(METHODS), list(SI_PRIORITY_METHODS)):
        raise ValueError("请选择原三方法、完整七路线或SI优先三组，方法顺序须一致")
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
    result["optimizer_seed"] = _integer(result["optimizer_seed"], "优化器基础种子", 0, 2**32 - 1)
    result["population_size"] = _integer(result["population_size"], "种群/粒子数", 4, 256)
    result["de_mutation"] = _number(result["de_mutation"], "DE变异系数", 1e-6, 2)
    result["de_crossover"] = _number(result["de_crossover"], "DE交叉概率", 0, 1)
    result["pso_inertia"] = _number(result["pso_inertia"], "PSO惯性", 0, 0.999999)
    result["pso_cognitive"] = _number(result["pso_cognitive"], "PSO个体系数", 0, 4)
    result["pso_social"] = _number(result["pso_social"], "PSO群体系数", 0, 4)
    result["si_stall_window"] = _integer(result["si_stall_window"], "SI停滞窗口", 1, 10000)
    result["si_stall_tolerance"] = _number(result["si_stall_tolerance"], "SI停滞阈值", 0, 1)
    result["si_min_evaluations"] = _integer(result["si_min_evaluations"], "SI最低容量调用", 1, 1000000)
    result["si_max_fraction"] = _number(result["si_max_fraction"], "SI阶段最大占比", 0.01, 0.99)
    result["service_margin_ratio"] = _number(result["service_margin_ratio"], "基本服务安全裕量", 0, 1e-3)
    if result["methods"] == list(SI_PRIORITY_METHODS):
        fixed = {"si_tolerance": 1e-8, "sgm_tolerance": 1e-8,
                 "maxiter": 300, "maxeval": 2000,
                 "max_time_seconds": 30.0, "si_stall_window": 200,
                 "si_stall_tolerance": 1e-6, "si_min_evaluations": 200,
                 "si_max_fraction": 0.85, "service_margin_ratio": 5e-9}
        if any(result[key] != value for key, value in fixed.items()):
            raise ValueError("SI优先三组须使用方案04冻结的共同容差、阶段规则与追加预算")
    return result
