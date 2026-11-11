# -*- coding: utf-8 -*-
"""W01 图2：同一批100个场景上的平均总功率和平均处理波束数。
数字取自 Q&A/评估与指标/五套预训练模型的策略偏好与泛化评估.md 的“默认分布：同一批100个场景”表
（平均总功率、平均处理波束数、超功率回合）。红色柱表示100个回合全部超过功率预算。
运行：python outputs/weekly_reports_2026/figure_scripts/W01_重评功率与处理波束.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w01_04_tools import *
import numpy as np

plt = setup_mpl()

names = ["T-SAC", "CNN版SAC", "MLP版SAC", "PPO", "DQN", "贪心"]
power = [2002.25, 3028.30, 3024.15, 2116.44, 2217.85, 2452.50]
beams = [100.00, 87.30, 95.68, 100.00, 100.00, 100.00]
over = [0, 100, 100, 0, 0, 0]
cols = [C_RED if o else (C_GREY if n == "贪心" else C_BLUE) for n, o in zip(names, over)]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.3))
x = np.arange(len(names))
a1.bar(x, power, 0.6, color=cols)
a1.axhline(3000, color=C_RED, ls="--", lw=1)
a1.text(5.45, 3040, "功率预算 3000 W", fontsize=9, color=C_RED, va="bottom", ha="right")
for i, p in enumerate(power):
    a1.text(i, p + 40, f"{p:.0f}", ha="center", va="bottom", fontsize=8.5)
a1.set_xticks(x)
a1.set_xticklabels(names, rotation=35, ha="right")
a1.set_ylim(0, 3400)
a1.set_ylabel("平均总功率（W）")
a1.grid(axis="x", visible=False)

a2.bar(x, beams, 0.6, color=cols)
for i, b in enumerate(beams):
    a2.text(i, b + 1, f"{b:.1f}" if b < 100 else "100", ha="center", fontsize=8.5)
a2.set_xticks(x)
a2.set_xticklabels(names, rotation=35, ha="right")
a2.set_ylim(0, 112)
a2.set_ylabel("平均处理波束数（共100束）")
a2.grid(axis="x", visible=False)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=C_RED, label="红色：这个模型100个回合全部超过了功率预算")],
           loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=9)
fig.tight_layout(w_pad=1.5, rect=(0, 0.06, 1, 1))
save_fig(fig, "W01_重评功率与处理波束.png")
