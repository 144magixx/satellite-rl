# -*- coding: utf-8 -*-
"""W11_配对差.png：按报告表格数字重画（本机没有该实验的逐场景原始记录）。

数字出处：docs/第三阶段/实验报告/03_SI优先功率细调与容量近似对照.md 两个预算下的“配对差中的SI……”表
（B−A、C−A、C−B；1月27日的100个场景，每景3次动作取平均再配对），见 _s3_tables.py 的 W11_PAIRED。
这里把三组叫：原SCA（没改规则的起点）、原近似（先提满足率＋原来的容量近似）、新近似（先提满足率＋新的容量近似）。
左图：需求满足率的平均变化，单位0.0001个百分点；右图：供需匹配程度的平均变化，单位0.0001。
柱外的大数字是平均变化，小数字是100个场景中“提高/持平/降低”的场景数。
运行：python W11_配对差.py
"""
import numpy as np

from _s3_style import *
from _s3_tables import W11_PAIRED

pairs = ["原近似\n相对原SCA", "新近似\n相对原SCA", "新近似\n相对原近似"]
budgets = list(W11_PAIRED)
x = np.arange(3)
w = 0.38
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.7))
for ax, vi, ci, title, step in [(axes[0], 1, 2, "需求满足率的平均变化（0.0001个百分点）", 1.9),
                                (axes[1], 3, 4, "供需匹配程度的平均变化（0.0001）", 0.3)]:
    for j, b in enumerate(budgets):
        vals = [W11_PAIRED[b][k][vi] * 1e4 for k in range(3)]
        ax.bar(x + (j - 0.5) * w, vals, w, color=BUDGET_COLORS[j], label=b)
        for k, xi in enumerate(x + (j - 0.5) * w):
            c = W11_PAIRED[b][k][ci]
            v = vals[k]
            sign = 1 if v >= 0 else -1
            va = "bottom" if v >= 0 else "top"
            ax.text(xi, v + sign * step * 0.15, f"{v:.1f}", ha="center", va=va, fontsize=8)
            ax.text(xi, v + sign * step * 1.15, f"{c[0]}/{c[1]}/{c[2]}", ha="center", va=va, fontsize=6.2, color="#444")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=8.3)
    ax.set_title(title, fontsize=8.8)
    ax.grid(axis="x", visible=False)
axes[0].set_ylim(0, 31)
axes[1].set_ylim(-4.0, 0.5)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=2, fontsize=8.3, frameon=False)
fig.text(0.5, 0.012, "柱外小数字：100个场景中 提高/持平/降低 的场景数；原近似、新近似都是先提满足率之后用的容量近似", ha="center", fontsize=7.4)
fig.tight_layout(rect=(0, 0.1, 1, 1))
save(fig, "W11_配对差.png")
