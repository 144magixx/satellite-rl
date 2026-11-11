# -*- coding: utf-8 -*-
"""W05_过供欠供来源：数字取自 docs/第二阶段/实验报告/08_SGM低分的过供欠供诊断.md 第1节表（过供欠供占SGM缺口）和第3节表（容量需求分档占比）。"""
import numpy as np
import matplotlib.pyplot as plt
from _plot_common import *

labels = ["原模型", "A 关dropout", "B 打乱场景", "C1 少量全局信息", "C2 资源概况", "C3 布局与需求", "C4 布局、分配与干扰", "固定分配", "贪心分配"]
over  = [85.20, 83.75, 84.18, 78.51, 77.65, 73.74, 83.12, 85.09, 86.71]
under = [14.80, 16.25, 15.82, 21.49, 22.35, 26.26, 16.88, 14.91, 13.29]

fig, ax = plt.subplots(figsize=(7.2, 3.6))
ys = np.arange(len(labels))
ax.barh(ys, over, color="#e08a3c", height=0.66, label="过供（容量超过需求）")
ax.barh(ys, under, left=over, color="#4c78a8", height=0.66, label="欠供（容量不足）")
for y, o, u in zip(ys, over, under):
    ax.text(o / 2, y, f"{o:.0f}%", ha="center", va="center", color="white", fontsize=10.5)
    ax.text(o + u / 2, y, f"{u:.0f}%", ha="center", va="center", color="white", fontsize=10.5)
ax.set_yticks(ys); ax.set_yticklabels(labels); ax.invert_yaxis()
ax.set_xlim(0, 100); ax.set_xlabel("占供需匹配程度总缺口的比例（%）")
ax.tick_params(axis="y", length=0)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False)
fig.subplots_adjust(left=0.31, right=0.96, top=0.9, bottom=0.14)
save(fig, "W05_过供欠供来源")

note("W05_过供欠供来源", """
按报告表格数字重画。数字取自 docs/第二阶段/实验报告/08_SGM低分的过供欠供诊断.md 第1节表（过供占SGM缺口/欠供占SGM缺口，9行）。
脚本：W05_过供欠供来源.py。
""")

