"""Reproducible third-stage tables and curves from complete saved case records."""
from pathlib import Path

import numpy as np


def write_report(directory, rows, summaries, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in ("Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"):
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False
    directory = Path(directory)
    methods = list(summaries)
    scenes = sorted({r["source_scene_sha256"] for r in rows})
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    for method in methods:
        selected = [r for r in rows if r["method"] == method]
        for axis, metric, label in zip(axes, ("final_si_mean", "final_sgm_mean"), ("SI", "SGM")):
            values = [np.mean([r["metrics"][metric] for r in selected
                               if r["source_scene_sha256"] == scene]) for scene in scenes]
            axis.plot(range(1, len(scenes) + 1), values, label=method, linewidth=1)
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
    axes[0].set_title("Paired scenes: repeated actions averaged within each scene")
    axes[-1].set_xlabel("Common scene index (sorted SHA256; no smoothing)")
    axes[0].legend()
    fig.savefig(directory / "power_paired.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for method in methods:
        if method == "none":
            continue
        for row in [r for r in rows if r["method"] == method]:
            history = row["solver"]["history"]
            elapsed = [h["elapsed_seconds"] for h in history]
            for axis, metric in zip(axes, ("final_si_mean", "final_sgm_mean")):
                gain = [h[metric] - history[0][metric] for h in history]
                axis.plot(elapsed, gain, alpha=0.12, linewidth=0.7,
                          color="tab:blue" if method == "slsqp" else "tab:orange")
    for axis, label in zip(axes, ("SI gain", "SGM gain")):
        axis.set(xlabel="Solver elapsed seconds", ylabel=label)
        axis.grid(alpha=0.2)
    axes[0].set_title("Accepted feasible records; blue=SLSQP, orange=discrete")
    axes[1].set_title("Every action sample shown; no smoothing")
    fig.savefig(directory / "power_progress.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for method in methods:
        selected = [r for r in rows if r["method"] == method]
        ratios = np.array([b["rate_bps"] / b["required_bps"] for r in selected for b in r["beams"]])
        watts = np.array([b["power_w"] for r in selected for b in r["beams"]])
        axes[0].hist(np.log10(np.maximum(ratios, 1e-6)), bins=60, histtype="step",
                     density=True, label=method)
        axes[1].hist(watts, bins=50, histtype="step", density=True, label=method)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set(xlabel="log10(C / D), zero capacity plotted at -6", ylabel="Density")
    axes[1].set(xlabel="Whole-beam power (W)", ylabel="Density")
    axes[0].legend()
    fig.suptitle("All final beam records; repeated actions are not independent scenes")
    fig.savefig(directory / "power_distributions.png", dpi=150)
    plt.close(fig)

    options = config["power_refinement"]
    lines = ["# 第三阶段固定频率槽功率细调结果", "",
             "冻结M1生成初始分配，各方法共享相同实际频率槽、需求及初始功率。连续方法为SLSQP含需求匹配初始化，功率非负且仅受总预算约束；原档位参照仍为5–50 W、步长5 W。",
             "", f"预算口径：{options['power_budget_mode']}；场景{len(scenes)}个，每场景{options['repeats']}次动作；新增训练交互0。",
             "", "| 方法 | SI | SGM | 基本服务束 | 全束基本服务比例 | 功率W | 求解均时s | 求解95分位s | 回退数 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method, item in summaries.items():
        lines.append(f"| {method} | {item['final_si_mean']:.6f} | {item['final_sgm_mean']:.6f} | "
                     f"{item['basic_met_count']:.3f} | {item['all_basic_met']:.6f} | {item['power_w']:.3f} | "
                     f"{item['mean_solver_seconds']:.4f} | {item['p95_solver_seconds']:.4f} | {item['fallback_count']} |")
    lines += ["", "SI和SGM按全部目标束终态计算，包含后续干扰及未服务束；重复动作先按场景平均。容量为环境计算值，min(C,D)为有效承载估计。",
              "", "![共同场景](power_paired.png)", "", "![优化过程](power_progress.png)", "",
              "过程图是保存的已接受可行候选，不是所有求解器试探点；未平滑，失败和无改善样本保留于原始表。",
              "", "![逐束分布](power_distributions.png)", "",
              "单训练种子及单考核日期限制仍在。连续方案同时放宽了原功率粒度、最低和最高整束功率，差值不能只归因为SQP算法，需求匹配初始化也有贡献；各case的algorithm_components及history.phase分别记录。局部解、停止或回退均不证明全局最优。",
              "", "原始入口：power_cases.jsonl、power_results.csv、power_beams.csv、power_iterations.jsonl；每个case含初始化SHA、来源场景SHA、方法、预算、终止原因和约束误差。",
              "", "本报告由保存的完整case生成；模型/场景冻结证据见job.json、config.json、refinement_manifest.json与results.json。"]
    (directory / "power_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
