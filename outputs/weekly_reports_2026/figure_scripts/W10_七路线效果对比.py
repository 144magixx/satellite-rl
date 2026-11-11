# -*- coding: utf-8 -*-
"""W10_七路线效果对比.png：按报告表格数字重画（本机没有该实验的逐场景原始记录）。

数字出处：docs/第三阶段/实验报告/02_七路线功率细调与逐场景耗时对照.md 里两个预算的“终态效果与总耗时”表
（1月26日的100个场景，每个场景3次动作取平均），见 _s3_tables.py 的 W10_RESULT。
上图是需求满足率（纵轴从85%起），下图是供需匹配程度；每种做法画两根柱：总功率预算3000 W、总功率不超过初始解。
运行：python W10_七路线效果对比.py
"""
import numpy as np

from _s3_style import *
from _s3_tables import W10_METHODS, W10_RESULT

fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.5), sharex=True)
budgets = list(W10_RESULT)
x = np.arange(len(W10_METHODS))
w = 0.38
for ax, idx, title, ylim, fmt in [(axes[0], 0, "需求满足率（%，纵轴从85起）", (85, 96), "{:.1f}"),
                                   (axes[1], 1, "供需匹配程度", (0, 1.1), "{:.2f}")]:
    for j, b in enumerate(budgets):
        vals = [r[idx] for r in W10_RESULT[b]]
        ax.bar(x + (j - 0.5) * w, vals, w, color=BUDGET_COLORS[j], label=b)
        for xi, v in zip(x + (j - 0.5) * w, vals):
            ax.text(xi, v + (ylim[1] - ylim[0]) * 0.012, fmt.format(v), ha="center", va="bottom", fontsize=7)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=9.5)
    ax.grid(axis="x", visible=False)
axes[1].set_xticks(x)
axes[1].set_xticklabels(W10_METHODS, fontsize=9)
axes[0].legend(loc="upper left", ncol=2, fontsize=8.5, frameon=False)
fig.tight_layout()
save(fig, "W10_七路线效果对比.png")
