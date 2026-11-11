# -*- coding: utf-8 -*-
"""W01 图4：五套旧权重和同结构、没训练过的网络的总满意度（同样的10个场景）。
数字取自 Q&A/评估与指标/五套预训练模型的策略偏好与泛化评估.md 的“是否学到了东西：同结构未训练对照”表。
未训练网络每种用3个固定随机初始化，先在场景内平均再在场景间平均。
运行：python outputs/weekly_reports_2026/figure_scripts/W01_旧权重对比未训练网络.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w01_04_tools import *
import numpy as np

plt = setup_mpl()

names = ["T-SAC", "CNN版SAC", "MLP版SAC", "PPO", "DQN"]
old = [95.35, 68.25, 81.51, 68.05, 72.50]
new = [51.85, 51.85, 52.37, 50.08, 6.50]

fig, ax = plt.subplots(figsize=(6.3, 3.4))
x = np.arange(len(names))
w = 0.36
ax.bar(x - w / 2, old, w, color=C_BLUE, label="旧权重")
ax.bar(x + w / 2, new, w, color=C_GREY, label="同结构、未训练的网络")
for i in range(len(names)):
    ax.text(x[i] - w / 2, old[i] + 1.5, f"{old[i]:.1f}", ha="center", fontsize=9)
    ax.text(x[i] + w / 2, new[i] + 1.5, f"{new[i]:.1f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 118)
ax.set_ylabel("总满意度（满分100）")
ax.legend(loc="upper right", ncol=2, frameon=False)
ax.grid(axis="x", visible=False)
fig.tight_layout()
save_fig(fig, "W01_旧权重对比未训练网络.png")
