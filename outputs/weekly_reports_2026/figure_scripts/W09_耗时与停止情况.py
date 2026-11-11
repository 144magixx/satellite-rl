# -*- coding: utf-8 -*-
"""W09_耗时与停止情况.png：按报告表格数字重画（本机没有该实验的逐场景原始记录）。

数字出处：docs/第三阶段/实验报告/01_T0初始解的连续功率细调对照.md 的“求解开销、停止与失败记录”表，
见 _s3_tables.py 的 W09_COST。左图：每种做法单次求解的平均耗时（柱）和P95耗时（黑点），纵轴为对数坐标；
右图：每种做法300次求解各自怎么结束的（未改动、用完2000次容量计算上限、求解器未收敛、正常结束）。
原分配的求解耗时报告里只给了约0.00008秒，图上按这个数画。
运行：python W09_耗时与停止情况.py
"""
import numpy as np

from _s3_style import *
from _s3_tables import W09_COST, W09_METHODS

budgets = list(W09_COST)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 3.6), gridspec_kw={"width_ratios": [1, 1.1]})

# 左：耗时
x = np.arange(3)
w = 0.36
for j, b in enumerate(budgets):
    mean = [c[0] for c in W09_COST[b]]
    p95 = [c[1] for c in W09_COST[b]]
    xs = x + (j - 0.5) * w
    ax1.bar(xs, mean, w, color=BUDGET_COLORS[j], label=b)
    ax1.plot(xs, p95, "o", color="black", ms=3.5, label="P95（95%的求解不超过）" if j == 0 else None)
fmt = lambda v: "<0.001" if v < 0.001 else f"{v:.2f}" if v < 1 else f"{v:.1f}"
for k in range(3):
    m0, m1 = W09_COST[budgets[0]][k][0], W09_COST[budgets[1]][k][0]
    top = max(W09_COST[budgets[0]][k][1], W09_COST[budgets[1]][k][1])
    t = fmt(m0) if k == 0 else f"{fmt(m0)} / {fmt(m1)}"
    ax1.text(k, top * 1.7, t, ha="center", va="bottom", fontsize=7.8)
ax1.set_yscale("log")
ax1.set_ylim(3e-5, 400)
ax1.set_xticks(x)
ax1.set_xticklabels(W09_METHODS, fontsize=8.5)
ax1.set_ylabel("单次求解耗时（秒）")
ax1.set_title("平均耗时（柱，数字）和P95（黑点）", fontsize=9.2)
ax1.grid(axis="x", visible=False)
h, l = ax1.get_legend_handles_labels()
ax1.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=7.8, frameon=False)

# 右：停止原因（每种300次）
names = ["未改动", "用完2000次容量计算上限", "求解器未收敛", "正常结束"]
cols = ["#bdbdbd", C_B, C_F, C_A]
rows = []
for b in budgets:
    for k, m in enumerate(W09_METHODS):
        mean, p95, nonconv, evalcap = W09_COST[b][k]
        if m == "原分配":
            seg = [300, 0, 0, 0]
        elif m == "原档位调整":
            seg = [0, evalcap, 0, 300 - evalcap]
        else:
            seg = [0, 0, nonconv, 300 - nonconv]
        rows.append(((("3000 W预算｜") if b.endswith("3000 W") else "初始功率上限｜") + m, seg))
y = np.arange(len(rows))[::-1]
left = np.zeros(len(rows))
for i, (nm, c) in enumerate(zip(names, cols)):
    v = np.array([r[1][i] for r in rows], dtype=float)
    ax2.barh(y, v, left=left, color=c, label=nm, height=0.7)
    for yi, l_, vv in zip(y, left, v):
        if vv >= 25:
            ax2.text(l_ + vv / 2, yi, f"{int(vv)}", ha="center", va="center", fontsize=7.8,
                     color="white" if c != "#bdbdbd" else "black")
    left += v
ax2.set_yticks(y)
ax2.set_yticklabels([r[0] for r in rows], fontsize=8)
ax2.set_xlim(0, 300)
ax2.set_xlabel("动作数（每种做法300次）")
ax2.set_title("求解是怎么结束的", fontsize=9.2)
ax2.grid(axis="y", visible=False)
ax2.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2, fontsize=7.8, frameon=False)
fig.tight_layout()
save(fig, "W09_耗时与停止情况.png")
