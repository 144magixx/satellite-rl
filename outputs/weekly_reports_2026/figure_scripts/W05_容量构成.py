# -*- coding: utf-8 -*-
"""W05_容量构成：数字取自 docs/第二阶段/实验报告/07_欧洲六组独立场景对照.md 第3节第一张表（容量单位 Gbps）。"""
import numpy as np
import matplotlib.pyplot as plt
from _plot_common import *

labels = ["原模型", "A 关dropout", "B 打乱场景", "C1 少量全局信息", "C2 资源概况", "C3 布局与需求", "C4 布局、分配与干扰", "固定分配", "贪心分配"]
eff   = [15.371, 16.269, 16.245, 13.862, 13.254, 12.137, 14.365, 13.571, 15.767]
over  = [23.038, 9.074, 11.172, 16.319, 17.117, 13.941, 20.028, 36.180, 25.347]
short = [5.948, 5.051, 5.075, 7.458, 8.066, 9.183, 6.955, 7.748, 5.553]
demand = 21.319

fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.9), gridspec_kw={"width_ratios": [2.0, 1.0], "wspace": 0.1})
ax = axs[0]
ys = np.arange(len(labels))
ax.barh(ys, eff, color="#4c78a8", height=0.66, label="有效承载")
ax.barh(ys, over, left=eff, color="#e08a3c", height=0.66, label="超过需求的部分")
ax.axvline(demand, color="#333333", linestyle="--", linewidth=1.2, label="平均总需求21.3")

for y, e, o in zip(ys, eff, over):
    ax.text(e + o + 0.6, y, f"{e + o:.1f}", va="center", fontsize=10.5)
ax.set_yticks(ys); ax.set_yticklabels(labels); ax.invert_yaxis()
ax.set_xlim(0, 56); ax.set_xlabel("总容量（Gbps）")
ax.tick_params(axis="y", length=0)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=10.5, columnspacing=1.0, handlelength=1.6)
ax = axs[1]
hbar(ax, labels, short, ["#d95f5f"] * len(labels), fmt="{:.1f}", xlim=(0, 12), show_ylabels=False)
ax.set_xlabel("未满足需求（Gbps）")
ax.set_title("")
fig.subplots_adjust(left=0.27, right=0.985, top=0.9, bottom=0.14)
save(fig, "W05_容量构成")
note("W05_容量构成", """
按报告表格数字重画。数字取自 docs/第二阶段/实验报告/07_欧洲六组独立场景对照.md 第3节第一张表（总容量、有效承载、超需求容量、未满足需求，Gbps，9行）。
左图是总容量拆成“有效承载”和“超过需求的部分”，虚线是本批场景平均总需求约21.3 Gbps；右图是未满足需求。脚本：W05_容量构成.py。
""")
