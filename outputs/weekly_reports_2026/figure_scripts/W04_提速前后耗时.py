# -*- coding: utf-8 -*-
"""W04 图：训练提速前后，每500步环境交互加500次参数更新的耗时。

数字取自 docs/第二阶段/实验报告/05_异步采集与训练提速验证.md “同工作量稳态训练”一节的表
（三轮测试，每轮原版、优化后各两次取平均；柱高是三轮中位数，点是各轮的值）：
  204维（13个输入令牌）：原版 94.56 / 93.32 / 94.72，中位 94.56；优化后 36.24 / 34.28 / 35.53，中位 35.53；倍数 2.66
  2946维（32个输入令牌）：原版 113.67 / 110.37 / 115.63，中位 113.67；优化后 46.29 / 46.18 / 46.39，中位 46.29；倍数 2.46
（每次计时的原始记录本机没有，按报告表重画。）
运行：python outputs/weekly_reports_2026/figure_scripts/W04_提速前后耗时.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w01_04_tools import *
import numpy as np

plt = setup_mpl()

labels = ["204维状态\n（13个输入令牌）", "2946维状态\n（32个输入令牌）"]
orig_rounds = [[94.56, 93.32, 94.72], [113.67, 110.37, 115.63]]
new_rounds = [[36.24, 34.28, 35.53], [46.29, 46.18, 46.39]]
orig_med = [np.median(r) for r in orig_rounds]
new_med = [np.median(r) for r in new_rounds]
ratio = [o / n for o, n in zip(orig_med, new_med)]

fig, ax = plt.subplots(figsize=(7.2, 3.8))
x = np.arange(2)
w = 0.34
b1 = ax.bar(x - w / 2 - 0.01, orig_med, w, color=C_ORANGE, label="原版（同步）")
b2 = ax.bar(x + w / 2 + 0.01, new_med, w, color=C_TEAL, label="优化后（批量检查＋异步记录）")
for i in range(2):
    for xs, rs in ((x[i] - w / 2 - 0.01, orig_rounds[i]), (x[i] + w / 2 + 0.01, new_rounds[i])):
        ax.scatter(xs + np.array([-0.07, 0, 0.07]), rs, s=16, color="white", edgecolor="#333", linewidth=0.8, zorder=4)
    ax.text(x[i] - w / 2 - 0.01, orig_med[i] + 9, f"{orig_med[i]:.1f}", ha="center", fontsize=10)
    ax.text(x[i] + w / 2 + 0.01, new_med[i] + 9, f"{new_med[i]:.1f}", ha="center", fontsize=10)
    ax.text(x[i], max(orig_med[i], new_med[i]) + 22, f"快了约{ratio[i]:.1f}倍", ha="center", fontsize=11,
            color=C_RED, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 150)
ax.set_ylabel("500步交互＋500次更新的耗时（秒）")
ax.grid(axis="x", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=2)
ax.text(1.0, -0.2, "柱高为三轮中位数，圆点是各轮的值", transform=ax.transAxes, ha="right", fontsize=8.5, color="#555")
save_fig(fig, "W04_提速前后耗时.png")
