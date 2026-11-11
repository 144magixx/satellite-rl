# -*- coding: utf-8 -*-
"""W20 图：统一单机请求（每架飞机21.6 Mbps）下的教师标签：多少个训练母场景有合格标签，以及教师布局相对共同起点降了多少峰值。

数据：第一阶段实验报告19（逐景统计表 case576_statistics.csv 不在本机可读位置，数字取自报告19）：
  左：576个训练母场景 → 574个有合格且非平凡的教师标签 → 其中564个同时改变了束数并使用了不同且不全取下界的束宽
      （2个因基础设施中断没有标签）；
  右：574个正常母场景的平均峰值请求负载（Mbps）和平均束数：共同起点 2196.99、教师 1040.53、
      教师分组再经修复器一次 791.34；束数 80.57、88.34、94.57。
"""
from _c1821 import *

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1.1]})
ax = axes[0]
labs = ["训练\n母场景", "有合格的\n教师标签", "同时变束数\n和束宽"]
vals = [576, 574, 564]
cols = [C_LIGHT, C_BLUE, C_RED]
for i, (v, c) in enumerate(zip(vals, cols)):
    ax.bar(i, v, 0.6, color=c)
    ax.text(i, v + 8, str(v), ha="center", fontsize=11)
ax.set_xticks(range(3))
ax.set_xticklabels(labs, fontsize=9.5)
ax.set_ylim(0, 680)
ax.set_title("教师标签数量", fontsize=10.5)
ax.grid(axis="x", visible=False)

ax = axes[1]
names = ["共同起点", "教师布局", "教师分组再\n修复一次"]
peak = [2196.99, 1040.53, 791.34]
nb = [80.57, 88.34, 94.57]
for i, (v, n, c) in enumerate(zip(peak, nb, (C_GREY, C_BLUE, C_RED))):
    ax.bar(i, v, 0.6, color=c)
    ax.text(i, v + 40, f"{v:.0f}", ha="center", fontsize=11)
    ax.text(i, 70, f"{n:.0f}束", ha="center", fontsize=10.5, color="white")
ax.set_xticks(range(3))
ax.set_xticklabels(names, fontsize=9.5)
ax.set_ylim(0, 2500)
ax.set_title("平均峰值负载/Mbps\n（柱内为平均束数）", fontsize=10.5)
ax.grid(axis="x", visible=False)
save_png(fig, "W20_教师标签.png")
