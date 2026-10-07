"""可变束数实验的标识集合（S2-P17 与 S2-P18 共用同一套工作台链路）。

S2-P17（第一阶段可变波束数量训练）与 S2-P18（第二轮改进）使用相同的工作台
训练、独立考核与功率细调流程，差别只在配置与冻结场景，不在链路语义。此处集中
定义允许的 `experiment_id`，避免再次把某一轮的编号写死在共享代码里。

**刻意不放宽的范围**：`satellite_rl.workbench.recovery` 中的数值边界修复
（`S2P17_BOUNDARY_REPAIR_ID` / `S2P17_FAILED_RUN`）是 S2-P17 失败任务专属的历史
修复路径，其校验必须继续只认 S2-P17 精确匹配；本模块只是为常规链路提供集合判断，
不改变该修复链的任何语义。
"""

VARIABLE_BEAM_EXPERIMENTS = ("S2-P17", "S2-P18", "S2-P19")
S2P17 = "S2-P17"
S2P18 = "S2-P18"
S2P19 = "S2-P19"


def is_variable_beam_experiment(experiment_id):
    """该标识是否属于可变束数实验集合；None/缺失一律视为否。"""
    return experiment_id in VARIABLE_BEAM_EXPERIMENTS
