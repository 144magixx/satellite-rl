# -*- coding: utf-8 -*-
"""W07_供需差额：数字取自 docs/第二阶段/实验报告/11_联合动作四组独立场景对照.md 第6节表
（每次分配内先求和，再对300次平均）：平均欠供波束数、平均欠供总量、平均过供总量（Gbit/s）。"""
import numpy as np
import matplotlib.pyplot as plt
from _plot_common import *
labels = ["原策略\n＋加性Q", "原策略\n＋联合Q", "条件策略\n＋加性Q", "条件策略\n＋联合Q"]
nb = [19.28, 33.62, 19.18, 25.48]
under = [2.95, 7.68, 2.75, 6.41]
over = [10.73, 13.18, 10.39, 20.19]
colors = [C1, C2, C3, C4]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.28})
ax = axs[0]
x = np.arange(4)
bars = ax.bar(x, nb, color=colors, width=0.62)
for b, v in zip(bars, nb):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", fontsize=10.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 40); ax.set_ylabel("平均欠供波束数（个/100束）")
ax.grid(axis="y", color="#e6e6e6"); ax.set_axisbelow(True)
ax = axs[1]
w = 0.36
b1 = ax.bar(x - w / 2, under, w, color="#4c78a8", label="欠供总量")
b2 = ax.bar(x + w / 2, over, w, color="#e08a3c", label="过供总量")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f"{b.get_height():.1f}", ha="center", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 24); ax.set_ylabel("平均每次分配的差额（Gbps）")
ax.legend(frameon=False, loc="upper left", fontsize=10.5)
ax.grid(axis="y", color="#e6e6e6"); ax.set_axisbelow(True)
fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.17)
save(fig, "W07_供需差额")
note("W07_供需差额", """
按报告表格数字重画。数字取自 docs/第二阶段/实验报告/11_联合动作四组独立场景对照.md 第6节表：平均欠供波束数、平均欠供总量、平均过供总量（Gbit/s，
每次分配内先求和再对300次平均，原表单位写 Gbit/s，这里按 Gbps 标）。脚本：W07_供需差额.py。
""")
