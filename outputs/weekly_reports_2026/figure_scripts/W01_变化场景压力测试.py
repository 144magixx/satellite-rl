# -*- coding: utf-8 -*-
"""W01 图3：五套旧权重和贪心在四种变化场景下的总满意度（同样的20个场景）。
数字取自 Q&A/评估与指标/五套预训练模型的策略偏好与泛化评估.md 的“泛化与过拟合”表。
运行：python outputs/weekly_reports_2026/figure_scripts/W01_变化场景压力测试.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w01_04_tools import *
import numpy as np

plt = setup_mpl()

cats = ["默认\n20个场景", "需求均值\n300 Mbps", "需求均值\n800 Mbps", "组内顺序\n重排", "均匀合成\n地理分布"]
data = {
    "T-SAC": [95.62, 97.05, 71.84, 83.93, 94.45],
    "CNN版SAC": [68.75, 77.65, 57.24, 68.89, 85.54],
    "MLP版SAC": [81.18, 83.10, 69.17, 78.77, 91.43],
    "PPO": [67.64, 77.21, 55.60, 67.89, 83.98],
    "DQN": [73.60, 78.90, 61.81, 71.01, 84.58],
    "贪心": [76.75, 86.97, 66.19, 76.98, 93.61],
}
sty = {"T-SAC": (C_BLUE, "-", 2.4, "o"), "CNN版SAC": (C_ORANGE, "-", 1.4, "s"), "MLP版SAC": (C_TEAL, "-", 1.4, "^"),
       "PPO": (C_PURPLE, "-", 1.4, "D"), "DQN": (C_GOLD, "-", 1.4, "v"), "贪心": ("#333", "--", 1.8, "o")}

fig, ax = plt.subplots(figsize=(6.6, 3.7))
for k, v in data.items():
    c, ls, lw, mk = sty[k]
    ax.plot(range(5), v, ls, color=c, lw=lw, marker=mk, ms=4.5, label=k)
ax.set_xticks(range(5))
ax.set_xticklabels(cats, fontsize=9)
ax.set_ylim(50, 100)
ax.set_ylabel("总满意度（满分100）")
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9)
ax.grid(axis="x", visible=False)
fig.tight_layout()
save_fig(fig, "W01_变化场景压力测试.png")
