# -*- coding: utf-8 -*-
"""W25 图：最初手写的环境里，八种第二阶段方法的需求满足率、最差10%波束和单景耗时。

数据来源：docs/Claude维护/原版框架复现/20_原版框架上八种方法不重训对比报告.md 第2节“结果”表
（100个随机场景，每景100束）。该实验的逐景原始记录
outputs/experiments/s2_orig_compare_20261022/main/results.jsonl 本机没有，只剩报告里的图，
所以这里按报告表格里的数字重画，没有误差线。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *
from matplotlib.patches import Patch

# (名称, 满足率%, 最差10%波束%, 单景耗时, 颜色)
rows = [
    ("T-SAC直接输出", 96.1, 60.2, "0.3 秒", C_LEARN),
    ("不用Transformer的MLP-SAC", 87.0, 5.8, "0.2 秒", C_LEARN),
    ("改进式SAC（贪心起点+局部改进）", 91.7, 40.0, "6.0 秒", C_LEARN),
    ("容量感知贪心", 88.2, 19.8, "0.03 秒", C_RULE),
    ("纯禁忌搜索32次", 61.5, 7.4, "5.1 秒", C_SEARCH),
    ("贪心+禁忌搜索32次", 90.9, 36.7, "4.9 秒", C_SEARCH),
    ("贪心+禁忌搜索96次", 92.2, 46.9, "14.4 秒", C_SEARCH),
    ("贪心+禁忌搜索320次", 92.5, 48.7, "46.6 秒", C_SEARCH),
    ("原版自带贪心（参照）", 72.8, 13.6, "0.15 秒", "#b9bfcb"),
]
labels = [r[0] for r in rows]
y = np.arange(len(rows))[::-1]
fig = plt.figure(figsize=(5.8, 3.75))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.34], wspace=0.06, left=0.395, right=0.985, top=0.90, bottom=0.17)
ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharey=ax1); ax3 = fig.add_subplot(gs[2], sharey=ax1)
for ax, idx, title, xl in zip((ax1, ax2), (1, 2), ("需求满足率（%）", "最差10%波束满足率（%）"), ((40, 118), (0, 118))):
    vals = [r[idx] for r in rows]
    ax.barh(y, [v - xl[0] for v in vals], left=xl[0], color=[r[4] for r in rows], height=0.68)
    for yy, v in zip(y, vals):
        ax.text(v + 1.5, yy, f"{v:.1f}", va="center", fontsize=8.5)
    ax.set_xlim(*xl)
    ax.set_title(title, fontsize=9)
    ax.xaxis.grid(True, color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
ax1.set_xticks([40, 60, 80, 100])
ax2.set_xticks([0, 25, 50, 75, 100])
ax1.set_yticks(y)
ax1.set_yticklabels(labels, fontsize=8.5)
ax2.tick_params(axis="y", length=0, labelleft=False)
ax3.set_xlim(0, 1)
for yy, r in zip(y, rows):
    ax3.text(0.5, yy, r[3], ha="center", va="center", fontsize=8.5)
ax3.set_title("单景耗时", fontsize=9)
ax3.axis("off")
fig.legend(handles=[Patch(color=C_LEARN, label="学习方法"), Patch(color=C_SEARCH, label="搜索方法"),
                    Patch(color=C_RULE, label="规则方法")],
           loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.62, 0.0))
save_png(fig, "W25_原版环境八种方法.png", dpi=220)
