# -*- coding: utf-8 -*-
"""W11_达标情况.png：按报告表格数字重画（本机没有该实验的逐场景原始记录）。

数字出处：docs/第三阶段/实验报告/03_SI优先功率细调与容量近似对照.md 两个预算下的结果表
（1月27日的100个场景，每景3次动作取平均），见 _s3_tables.py 的 W11_RESULT。
左图是每100束里基本服务达标的束数（纵轴从99.3起），右图是100束全部达标的动作占比。
运行：python W11_达标情况.py
"""
import numpy as np

from _s3_style import *
from _s3_tables import W11_METHODS, W11_RESULT

budgets = list(W11_RESULT)
x = np.arange(3)
w = 0.38
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4))
for ax, idx, title, ylim, fmt in [(axes[0], 2, "基本服务达标束数（每100束，纵轴从99.3起）", (99.3, 99.8), "{:.2f}"),
                                  (axes[1], 3, "100束全部达标的动作占比（%）", (0, 100), "{:.1f}")]:
    for j, b in enumerate(budgets):
        vals = [r[idx] for r in W11_RESULT[b]]
        ax.bar(x + (j - 0.5) * w, vals, w, color=BUDGET_COLORS[j], label=b)
        for xi, v in zip(x + (j - 0.5) * w, vals):
            ax.text(xi, v + (ylim[1] - ylim[0]) * 0.012, fmt.format(v), ha="center", va="bottom", fontsize=7.8)
    ax.set_xticks(x)
    ax.set_xticklabels(W11_METHODS, fontsize=7.8)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="x", visible=False)
axes[1].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=8, frameon=False, ncol=1)
fig.tight_layout()
save(fig, "W11_达标情况.png")
