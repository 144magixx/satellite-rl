"""Shared, dependency-light definitions for the natural T-SAC observation layout."""

from math import prod


NATURAL_CONTEXT_LEVELS = ("none", "compact", "summary", "layout", "full")
BASE_FEATURES = ("beam", "power", "interference", "occupancy")
VARIABLE_BEAM_FEATURES = BASE_FEATURES + (
    "beta", "progress", "remaining_demand", "beam_count_normalized",
    "remaining_beams_normalized", "power_pressure",
)
VARIABLE_BEAM_STATE_DIM = 210
VARIABLE_BEAM_MAX = 200

# S2-P18（方案18 第二版）新增的可选观测扩充。默认关闭，已冻结的 210 维 V0/V1
# 状态、其 state_dim、断点语义与配置哈希保持逐位不变；只有显式选择这些特征的
# 对照臂才会改变维度。
# - ``group``：把当前复用组号 group/7 纳入特征（方案 2.4 第三行）。
# - ``group_remaining_obligation`` / ``remaining_slot_limit``：补入“本组剩余预留
#   义务”与“剩余合法槽上限”（方案 2.4 第四行），掩码实际使用的
#   remaining_group/max_slots 由此可观测。
S2P18_OPTIONAL_FEATURES = ("group", "group_remaining_obligation", "remaining_slot_limit")
S2P18_VARIABLE_FEATURES = VARIABLE_BEAM_FEATURES + S2P18_OPTIONAL_FEATURES
S2P18_VARIABLE_STATE_DIM = VARIABLE_BEAM_STATE_DIM + len(S2P18_OPTIONAL_FEATURES)
S2P18_VARIABLE_TOKEN_COUNT = 14


def natural_shapes(level):
    """Return only fields visible at this level, in observation order.

    ``natural_beam_context`` keeps its historical fixed 100-row layout so the
    frozen natural levels stay comparable. S2-P18 的逐束对照改用
    :func:`natural_shapes_actual_count`，按场景实际束数构造。
    """
    if level not in NATURAL_CONTEXT_LEVELS:
        raise ValueError(f"Unknown natural context level: {level}")
    shapes = {"beam": [3], "power": [1], "interference": [100],
              "occupancy": [100]}
    if level != "none":
        shapes["natural_global_context"] = [3 if level == "compact" else 10]
    if level in ("layout", "full"):
        shapes["natural_beam_context"] = [10, 10, 5 if level == "layout" else 11]
    if level in ("summary", "layout", "full"):
        shapes["natural_resource_context"] = [8, 204 if level == "full" else 4]
    return shapes


def natural_shapes_actual_count(level, count):
    """S2-P18 逐束布局：只保留实际 count 个实体行，不填充到 100。

    方案18 第 2.4 节记录 ``natural_beam_context`` 固定 ``(10,10,11)``，第 338 行
    ``reshape(count, 11)`` 在 N≠100 时直接报错。该函数给出按实际束数构造的形状，
    供 ``context_beam_count="actual"`` 的可选对照使用；默认档位不受影响。
    """
    shapes = natural_shapes(level)
    if "natural_beam_context" not in shapes:
        return shapes
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= VARIABLE_BEAM_MAX:
        raise ValueError(f"实际束数须为1至{VARIABLE_BEAM_MAX}的整数，当前为 {count!r}")
    shapes["natural_beam_context"] = [count, shapes["natural_beam_context"][-1]]
    return shapes


def natural_dim(level):
    return sum(prod(shape) for shape in natural_shapes(level).values())


def natural_dim_actual_count(level, count):
    return sum(prod(shape) for shape in natural_shapes_actual_count(level, count).values())


def natural_token_count(level):
    return 13 + (level != "none") + (10 if level in ("layout", "full") else 0) + (8 if level in ("summary", "layout", "full") else 0)
