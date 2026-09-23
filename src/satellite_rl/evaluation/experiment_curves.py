"""Compare the three backbones using recorded validation rounds only."""

from collections import defaultdict
from pathlib import Path
import statistics

import numpy as np

from satellite_rl.evaluation.experiment_report import (
    FORMULAS, FORMULA_NAMES, MODELS, MODEL_NAMES,
)
from satellite_rl.training.run_monitor import atomic_json


FIELDS = ("si_mean", "sgm_mean", "throughput_bps", "return")


def build_comparison_curves(runs, validation_interval=10000, budget_steps=400000):
    """Average each saved validation round across exactly three training seeds.

    Validation follows an episode boundary, so actual interaction counts may
    differ slightly. Preserve those counts instead of inventing a common point.
    """
    if validation_interval <= 0 or budget_steps % validation_interval:
        raise ValueError("验证间隔必须为正，并整除训练预算")
    expected_variants = {(model, formula) for model in MODELS for formula in FORMULAS}
    grouped_runs = defaultdict(list)
    for run in runs:
        job = run["job"]
        grouped_runs[(job["model"], job["formula"])].append(run)
    if set(grouped_runs) != expected_variants:
        raise RuntimeError("三种网络及两套公式的验证记录不完整")
    expected_rounds = set(range(1, budget_steps // validation_interval + 1))
    curves = {}
    reference_seeds = None
    reference_keys = None
    for (model, formula), variant_runs in grouped_runs.items():
        seeds = {run["job"]["seed"] for run in variant_runs}
        if len(variant_runs) != 3 or len(seeds) != 3:
            raise RuntimeError("每种配置必须有三个不同训练种子")
        if reference_seeds is not None and seeds != reference_seeds:
            raise RuntimeError("各网络的训练种子不一致")
        reference_seeds = seeds
        by_round = defaultdict(list)
        for run in variant_runs:
            config = run["manifest"]["config"]
            expected_keys = {(str(scene), str(action))
                             for scene in config["validation_scene_seeds"]
                             for action in config["action_seeds"]}
            if not expected_keys or (reference_keys is not None and expected_keys != reference_keys):
                raise RuntimeError("各网络的固定验证场景或动作种子不一致")
            reference_keys = expected_keys
            points = defaultdict(list)
            for row in run["evaluations"]:
                phase = row["checkpoint"]
                if phase not in ("validation", "final_validation"):
                    continue
                if row.get("action_mode", "sample") != "sample":
                    continue
                points[(int(row["env_step"]), phase)].append(row)
            rounds = {}
            for (step, phase), rows in sorted(points.items()):
                keys = {(str(row["scene_id"]), str(row["action_seed"])) for row in rows}
                if keys != expected_keys or len(rows) != len(expected_keys):
                    raise RuntimeError(f"{run['job']['run_id']} 在 {step} 步缺少或重复验证场景")
                index = step // validation_interval
                if index not in expected_rounds or step > budget_steps:
                    raise RuntimeError("验证交互步数超出预定范围")
                metrics = {field: statistics.mean(float(row[field]) for row in rows)
                           for field in FIELDS}
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise RuntimeError("验证曲线包含非有限数值")
                point = {"train_seed": run["job"]["seed"], "env_step": step, **metrics}
                # The separate final validation may repeat the last scheduled
                # validation at the identical budget endpoint; it is not a new run.
                if index in rounds:
                    if step != budget_steps or rounds[index]["env_step"] != step:
                        raise RuntimeError("同一验证轮次出现多个不同保存点")
                    if any(not np.isclose(rounds[index][field], point[field],
                                          rtol=1e-8, atol=1e-10) for field in FIELDS):
                        raise RuntimeError("预算终点的重复验证数据不一致")
                rounds[index] = point
            if set(rounds) != expected_rounds:
                raise RuntimeError(f"{run['job']['run_id']} 缺少预定验证轮次")
            for index, point in rounds.items():
                by_round[index].append(point)
        values = []
        for index in sorted(by_round):
            points = sorted(by_round[index], key=lambda point: point["train_seed"])
            steps = [point["env_step"] for point in points]
            values.append({
                "validation_round": index,
                "env_step_mean": statistics.mean(steps),
                "env_step_min": min(steps), "env_step_max": max(steps),
                "by_train_seed": points,
                "metrics": {field: {
                    "mean": statistics.mean(point[field] for point in points),
                    "std": statistics.stdev(point[field] for point in points),
                } for field in FIELDS},
            })
        curves[f"{model}_{formula}"] = values
    return {"validation_interval": validation_interval, "budget_steps": budget_steps,
            "training_repeats": 3, "curves": curves,
            "method": "每次训练先平均固定场景及动作采样，再按三个训练种子计算均值与样本标准差。"
                      "同一验证轮次的横轴为实际交互步数均值；没有插值、外推或额外平滑。"}


def write_comparison_curves(curves, output_directory):
    """Write a standalone figure and its exact numerical data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    data_path = output_directory / "三种网络验证曲线.json"
    figure_path = output_directory / "三种网络验证曲线.png"
    atomic_json(data_path, curves)
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    titles = ("需求满足率 SI（%）", "供需匹配程度 SGM（%）", "总吞吐量（Gbps）", "回合累计奖励")
    scales = (100, 100, 1e-9, 1)
    colors = ("tab:blue", "tab:orange", "tab:green")
    for row_index, formula in enumerate(FORMULAS):
        for model, color in zip(MODELS, colors):
            points = curves["curves"][f"{model}_{formula}"]
            x = [point["env_step_mean"] for point in points]
            for ax, field, scale in zip(axes[row_index], FIELDS, scales):
                means = np.array([point["metrics"][field]["mean"] for point in points]) * scale
                stds = np.array([point["metrics"][field]["std"] for point in points]) * scale
                ax.plot(x, means, color=color, label=MODEL_NAMES[model], linewidth=1.5)
                ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.12)
        for ax, title in zip(axes[row_index], titles):
            ax.set_title(f"{FORMULA_NAMES[formula]}：{title}", fontsize=10)
            ax.set_xlabel("实际环境交互步数（同轮次三次训练的均值）", fontsize=9)
            ax.grid(alpha=0.25)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax.legend(fontsize=7)
    fig.suptitle("三种网络的固定场景验证结果：均值及训练种子间标准差", fontsize=14)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return {"data_path": str(data_path), "figure_path": str(figure_path)}
