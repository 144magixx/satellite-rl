# -*- coding: utf-8 -*-
"""W04 图：把一次验证和保存也算进去的100步训练窗口，原版和优化后的耗时。

数字取自 docs/第二阶段/实验报告/05_异步采集与训练提速验证.md “验证/保存边界与纯算法参考”一节正文：
  204维（13个输入令牌）：原版 95.95 秒、优化后 84.43 秒，窗口减少 12.0%
  2946维（32个输入令牌）：原版 93.27 秒、优化后 90.40 秒，减少 3.1%
  窗口里包含60行固定验证、保存与关闭，单次验证本身约 70～78 秒；每1万步才验证一次，这里100步就验一次。
（每个窗口只测了一对，原始记录本机没有，按报告正文数字重画。）
运行：python outputs/weekly_reports_2026/figure_scripts/W04_含验证保存的100步窗口.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w01_04_tools import *
import numpy as np

plt = setup_mpl()

labels = ["204维状态\n（13个输入令牌）", "2946维状态\n（32个输入令牌）"]
orig = [95.95, 93.27]
new = [84.43, 90.40]
drop = ["少了12.0%", "少了3.1%"]

fig, ax = plt.subplots(figsize=(7.2, 3.8))
x = np.array([0.0, 1.3])
w = 0.34
ax.bar(x - w / 2 - 0.01, orig, w, color=C_ORANGE, label="原版（同步）")
ax.bar(x + w / 2 + 0.01, new, w, color=C_TEAL, label="优化后")
for i in range(2):
    ax.text(x[i] - w / 2 - 0.01, orig[i] + 2, f"{orig[i]:.1f}", ha="center", fontsize=10)
    ax.text(x[i] + w / 2 + 0.01, new[i] + 2, f"{new[i]:.1f}", ha="center", fontsize=10)
    ax.text(x[i], 112, drop[i], ha="center", fontsize=10.5, color=C_RED, fontweight="bold")
# 验证耗时的参考带：70～78 秒
ax.axhspan(70, 78, color=C_GOLD, alpha=0.35, zorder=3)
ax.text(0.65, 74, "单次验证约70～78秒", ha="center", va="center", fontsize=9.5, color="#6b4e0c", zorder=5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlim(-0.6, 1.9)
ax.set_ylim(0, 128)
ax.set_ylabel("100步窗口的耗时（秒）")
ax.grid(axis="x", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
ax.text(1.0, -0.2, "窗口内含一次验证（60行评价）、保存和关闭，各测一对", transform=ax.transAxes, ha="right",
        fontsize=8.5, color="#555")
save_fig(fig, "W04_含验证保存的100步窗口.png")
