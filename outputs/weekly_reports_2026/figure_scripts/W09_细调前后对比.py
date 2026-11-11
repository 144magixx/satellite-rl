# -*- coding: utf-8 -*-
"""W09_细调前后对比.png：按报告表格数字重画（本机没有该实验的逐场景原始记录）。

数字出处：docs/第三阶段/实验报告/01_T0初始解的连续功率细调对照.md 的“全束终态结果”表
（1月25日的100个场景，每个场景3个初始分配取平均），见 _s3_tables.py 的 W09_RESULT。
四个小图：需求满足率、供需匹配程度、基本服务达标束数、100束全部达标的动作占比；
每个小图有两种功率预算（总功率3000 W、总功率不超过初始解）各三种做法。
运行：python W09_细调前后对比.py
"""
import numpy as np

from _s3_style import *
from _s3_tables import W09_METHODS, W09_RESULT

colors = [C_BASE, C_D, C_A]
panels = [
    (0, "需求满足率（%，纵轴从85起）", (85, 100), "{:.1f}"),
    (1, "供需匹配程度", (0, 1.05), "{:.2f}"),
    (2, "基本服务达标束数（每100束，纵轴从98.5起）", (98.5, 100.1), "{:.2f}"),
    (3, "100束全部达标的动作占比（%）", (0, 100), "{:.1f}"),
]
fig, axes = plt.subplots(2, 2, figsize=(6.2, 4.9))
budgets = list(W09_RESULT)
x = np.arange(len(budgets))
w = 0.26
for ax, (idx, title, ylim, fmt) in zip(axes.ravel(), panels):
    for k, m in enumerate(W09_METHODS):
        vals = [W09_RESULT[b][k][idx] for b in budgets]
        ax.bar(x + (k - 1) * w, vals, w, color=colors[k], label=m)
        for xi, v in zip(x + (k - 1) * w, vals):
            ax.text(xi, v + (ylim[1] - ylim[0]) * 0.012, fmt.format(v), ha="center", va="bottom", fontsize=7.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["总功率预算\n3000 W", "总功率不超过\n初始解"])
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=9.2)
    ax.grid(axis="x", visible=False)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, "W09_细调前后对比.png")
