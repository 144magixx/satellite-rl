# -*- coding: utf-8 -*-
"""W18 图：第三种方案（贪心起点＋SAC调整）相对贪心、规则搜索、随机搜索的需求满足率提升及95%区间。

数据来源：第二阶段实验报告15 第4节“I0相对参照”表（直接Δ及母级95%区间、最终B Δ及区间，单位百分点）。
区间是以40个母场景为单位的配对bootstrap 95%区间（2000次）。逐场景原始数据不在本机，取报告表格数字。
"""
from _c1821 import *

rows = [
    # 标签, 提升, 下界, 上界, 阶段
    ("比贪心高", 3.2609, 2.6322, 3.8168, "before"),
    ("比贪心高", 3.0658, 2.4483, 3.6347, "after"),
    ("比规则搜索高", 1.9471, 1.5156, 2.3590, "before"),
    ("比规则搜索高", 1.7490, 1.3344, 2.1367, "after"),
    ("比随机搜索高", 0.8434, 0.6217, 1.0565, "before"),
    ("比随机搜索高", 0.6847, 0.4951, 0.8697, "after"),
]
fig, ax = plt.subplots(figsize=(7.2, 3.3))
ypos = [5.2, 4.55, 3.2, 2.55, 1.2, 0.55]
for (lab, m, lo, hi, st), y in zip(rows, ypos):
    c = C_BLUE if st == "before" else C_RED
    ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", color=c, capsize=4, lw=2, ms=7)
    ax.text(hi + 0.08, y, f"+{m:.2f}", va="center", fontsize=10.5, color=c)
ax.axvline(0, color=C_GREY, ls="--", lw=1.2)
ax.set_yticks([4.875, 2.875, 0.875])
ax.set_yticklabels(["比贪心高", "比规则搜索高", "比随机搜索高"])
ax.set_xlim(-0.2, 4.4)
ax.set_ylim(-0.1, 5.9)
ax.set_xlabel("需求满足率提升（百分点），线段为95%区间")
ax.grid(axis="y", visible=False)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", color=C_BLUE, lw=2, label="细调前（直接分配）"),
                   Line2D([], [], marker="o", color=C_RED, lw=2, label="细调后（功率优化后）")],
          loc="lower right", frameon=False)
ax.set_title("第三种方案（贪心起点＋SAC调整）的提升", loc="left")
save_png(fig, "W18_配对提升区间.png")
