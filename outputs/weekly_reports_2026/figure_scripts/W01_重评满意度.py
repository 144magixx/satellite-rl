# -*- coding: utf-8 -*-
"""W01 图1：五套旧权重和贪心在同一批100个场景上的总满意度。
数字取自 Q&A/评估与指标/五套预训练模型的策略偏好与泛化评估.md 的“默认分布：同一批100个场景”表
（原始 episodes.csv 所在目录本机已清理，按该表重画）。误差线为表里的场景间标准差。
运行：python outputs/weekly_reports_2026/figure_scripts/W01_重评满意度.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w01_04_tools import *
import numpy as np

plt = setup_mpl()

names = ["T-SAC", "MLP版SAC", "贪心", "DQN", "CNN版SAC", "PPO"]
mean = [95.36, 80.75, 77.23, 73.23, 67.94, 67.11]
std = [1.91, 2.53, 3.49, 3.51, 3.21, 3.54]
greedy = 77.23
cols = [C_GREY if n == "贪心" else (C_BLUE if m > greedy else C_ORANGE) for n, m in zip(names, mean)]

fig, ax = plt.subplots(figsize=(6.3, 3.4))
x = np.arange(len(names))
ax.bar(x, mean, 0.6, color=cols, yerr=std, error_kw=dict(ecolor="#444", capsize=3, lw=1))
ax.axhline(greedy, color="#555", ls="--", lw=1)
for i, m in enumerate(mean):
    ax.text(i, m + std[i] + 1.5, f"{m:.1f}", ha="center", fontsize=9.5)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 108)
ax.set_ylabel("总满意度（满分100）")
ax.text(len(names) - 0.45, greedy + 1.5, "贪心 77.2", ha="right", fontsize=9, color="#555")
ax.grid(axis="x", visible=False)
save_fig(fig, "W01_重评满意度.png")
