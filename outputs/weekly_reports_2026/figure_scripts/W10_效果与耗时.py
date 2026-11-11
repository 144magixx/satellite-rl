# -*- coding: utf-8 -*-
"""W10_效果与耗时.png：按报告表格数字重画（本机没有该实验的逐场景原始记录）。

数字出处：docs/第三阶段/实验报告/02_七路线功率细调与逐场景耗时对照.md 的“终态效果与总耗时”表，
见 _s3_tables.py 的 W10_RESULT。横轴是单次总耗时（秒，包含约0.7秒的初始分配），纵轴左图是需求满足率，
右图是供需匹配程度；实心点是总功率预算3000 W，空心点是总功率不超过初始解。越靠左上越好。
运行：python W10_效果与耗时.py
"""
import numpy as np

from _s3_style import *
from _s3_tables import W10_METHODS, W10_RESULT

cols = [C_BASE, C_D, C_C, C_F, C_E, "#8d6e63", "#e377c2"]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4))
budgets = list(W10_RESULT)
# 标签位置（数据坐标）：拥挤的几个点统一排到 x=2.8 处，用细线连回去；SLSQP、SCA 直接放在点旁边
lab = {
    0: {"不细调": (2.8, 89.55), "原档位调整": (2.8, 90.0), "差分进化": (2.8, 90.5), "WMMSE": (0.0, 91.05),
        "粒子群": (2.8, 91.6), "SCA": (1.6, 94.05), "SLSQP": (9.4, 94.05)},
    1: {"不细调": (2.8, 0.695), "原档位调整": (2.8, 0.715), "差分进化": (2.8, 0.738), "粒子群": (2.8, 0.76),
        "WMMSE": (2.8, 0.785), "SCA": (1.6, 0.955), "SLSQP": (9.4, 0.955)},
}
for ax, idx, ylab, ylim in [(axes[0], 0, "需求满足率（%）", (89, 94.5)), (axes[1], 1, "供需匹配程度", (0.68, 0.97))]:
    for j, b in enumerate(budgets):
        for k, m in enumerate(W10_METHODS):
            t, v = W10_RESULT[b][k][6], W10_RESULT[b][k][idx]
            ax.scatter(t, v, s=45, color=cols[k], facecolor=cols[k] if j == 0 else "white", linewidth=1.6, zorder=3)
            if j == 0:
                ax.annotate(m, (t, v), xytext=lab[idx][m], textcoords="data", fontsize=8, va="center",
                            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6, shrinkA=1, shrinkB=3))
    ax.set_xlabel("单次总耗时（秒）")
    ax.set_ylabel(ylab)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.3, 15)
axes[0].scatter([], [], s=45, color="#555", label="总功率预算3000 W")
axes[0].scatter([], [], s=45, color="#555", facecolor="white", linewidth=1.6, label="总功率不超过初始解")
axes[0].legend(loc="lower right", fontsize=8, frameon=False)
fig.tight_layout()
save(fig, "W10_效果与耗时.png")
