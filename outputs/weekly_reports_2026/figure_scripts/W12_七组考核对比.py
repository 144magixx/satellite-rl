# -*- coding: utf-8 -*-
"""W12_七组考核对比.png：按报告表格数字重画（本机没有该实验的逐场景原始记录）。

数字出处：docs/第一阶段/实验报告/02_七组覆盖规划独立场景对照.md 的“七组共同考核成绩”表
（1月22日预先选好的100个时刻，每组100个场景的平均），见 _s3_tables.py 的 W12_RESULT。
三个小图：需求覆盖（横轴从99.3%起）、负载不均程度（负载标准差÷平均，越小越均匀）、100个场景里达标（可行）的场景数。
运行：python W12_七组考核对比.py
"""
import numpy as np

from _s3_style import *
from _s3_tables import W12_RESULT

names = ["规则布局，不调整", "规则布局＋局部搜索", "规则布局＋PPO微调", "聚类布局，不调整", "聚类布局＋局部搜索",
         "聚类布局＋PPO微调", "纯PPO，从空布局构建"]
colors = ["#9db4d1", "#6b92c0", "#3a6ea5", "#a8d5cf", "#6fbfb4", "#2a9d8f", "#c0392b"]
y = np.arange(7)[::-1]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.3), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1]})
for ax, idx, title, xlim, fmt in [(axes[0], 0, "需求覆盖（%，横轴从99.3起）", (99.3, 100.0), "{:.2f}"),
                                  (axes[1], 2, "负载不均程度（越小越均匀）", (0, 1.15), "{:.2f}"),
                                  (axes[2], 3, "达标场景数（共100个）", (0, 118), "{:.0f}")]:
    vals = [r[idx] for r in W12_RESULT]
    ax.barh(y, vals, color=colors, height=0.68)
    for yi, v in zip(y, vals):
        ax.text(v + (xlim[1] - xlim[0]) * 0.012, yi, fmt.format(v), va="center", fontsize=8)
    ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=8.6)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(y)
axes[0].set_yticklabels(names, fontsize=8)
fig.tight_layout()
save(fig, "W12_七组考核对比.png")
