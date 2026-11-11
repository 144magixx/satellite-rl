# -*- coding: utf-8 -*-
"""W18 图：七种方法在880个测试场景上细调前后的需求满足率。

数据来源：第二阶段实验报告15 第4节“独立考核结果”表（直接SI、最终B SI 两列，单位 %）。
该表数字来自 sac_routes_s2p19_20261007/analysis/final_report_material/summary.json，
逐场景原始 csv 不在本机，所以这里直接取报告表格里的数字。
方法名对应：C0=从空构造（只改奖励）、C1=从空构造＋贪心样本、I0=贪心起点＋SAC调整、V2=上一轮旧模型。
"""
from _c1821 import *

names = ["贪心", "规则搜索", "随机搜索", "上一轮\n旧模型",
         "从空构造\n（只改奖励）", "从空构造\n＋贪心样本", "贪心起点\n＋SAC调整"]
before = [90.7907, 92.1045, 93.2082, 89.0485, 86.6018, 88.4616, 94.0516]
after = [94.2675, 95.5844, 96.6486, 93.1220, 91.8427, 92.5744, 97.3334]

fig, ax = plt.subplots(figsize=(6.3, 4.5))
y = list(range(len(names)))[::-1]
h = 0.38
for yi, b, a, n in zip(y, before, after, names):
    key = n.startswith("贪心起点")
    ax.barh(yi + h / 2, b, h, color="#9db8d6" if not key else "#e8a79f", edgecolor="none")
    ax.barh(yi - h / 2, a, h, color=C_BLUE if not key else C_RED, edgecolor="none")
    ax.text(b + 0.12, yi + h / 2, f"{b:.1f}", va="center", fontsize=10)
    ax.text(a + 0.12, yi - h / 2, f"{a:.1f}", va="center", fontsize=10, fontweight="bold" if key else None)
ax.axvline(before[0], color=C_GREY, lw=0.9, ls=":")
ax.axvline(after[0], color=C_GREY, lw=0.9, ls=":")
ax.set_yticks(y)
ax.set_yticklabels(names)
ax.set_xlim(84, 99.5)
ax.set_xlabel("需求满足率（%）（横轴从84%起）")
ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax.legend(handles=[Patch(color="#9db8d6", label="细调前（直接分配）"),
                   Patch(color=C_BLUE, label="细调后（功率优化后）"),
                   Line2D([], [], color=C_GREY, ls=":", label="贪心的水平")],
          loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False,
          columnspacing=1.2, handlelength=1.4)
save_png(fig, "W18_七种方法需求满足率.png")
