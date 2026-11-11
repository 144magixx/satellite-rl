# -*- coding: utf-8 -*-
"""W21 图：固定模型给局部转移排序（模型排序），对比按几何位置排序和随机排序，20个开发场景。

数据：第一阶段实验报告21 两张表（逐景原始表 analysis20x3.csv 不在本机可读位置，取报告表格数字）：
  左：平均峰值请求负载（Mbps）：模型排序 1069.20、几何排序 1063.80、随机排序 1138.32；
      虚线是同一批场景上原先“规则分组＋修复器”的 892.08。
  右：模型排序减对照的逐景胜/平/负次数（负载指标更低为胜）：
      对几何排序 峰值 4/10/6、上尾 8/2/10、CV 10/0/10；对随机排序 峰值 7/12/1、上尾 18/1/1、CV 19/0/1。
“质量达标”（比同景较好的对照好1%以上且不变差）：20个场景里只有2个，预定门槛是5个。
"""
from _c1821 import *

fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.3), gridspec_kw={"wspace": 0.35, "width_ratios": [0.95, 1.25]})
ax = axes[0]
names = ["模型\n排序", "几何\n排序", "随机\n排序"]
peak = [1069.20, 1063.80, 1138.32]
cols = [C_RED, C_BLUE, C_GREY]
for i, (v, c) in enumerate(zip(peak, cols)):
    ax.bar(i, v, 0.62, color=c)
    ax.text(i, v + 15, f"{v:.0f}", ha="center", fontsize=10.5)
ax.axhline(892.08, color=C_GREEN, ls="--", lw=1.6, label="原规则分组：892")
ax.set_xticks(range(3))
ax.set_xticklabels(names, fontsize=10)
ax.set_ylim(0, 1650)
ax.set_title("平均峰值负载/Mbps", fontsize=10.5)
ax.legend(loc="upper left", frameon=False, fontsize=9.5)
ax.grid(axis="x", visible=False)

ax = axes[1]
rows = [("峰值", (4, 10, 6)), ("上尾", (8, 2, 10)), ("CV", (10, 0, 10)),
        ("峰值", (7, 12, 1)), ("上尾", (18, 1, 1)), ("CV", (19, 0, 1))]
ypos = [5.6, 4.6, 3.6, 1.8, 0.8, -0.2]
for (met, (w, t, l)), y in zip(rows, ypos):
    ax.barh(y, w, color=C_GREEN, height=0.72)
    ax.barh(y, t, left=w, color=C_LIGHT, height=0.72)
    ax.barh(y, l, left=w + t, color=C_RED, height=0.72)
    for x0, n, c in ((0, w, "white"), (w, t, "black"), (w + t, l, "white")):
        if n >= 2:
            ax.text(x0 + n / 2, y, str(n), ha="center", va="center", fontsize=10, color=c)
ax.set_yticks(ypos)
ax.set_yticklabels([r[0] for r in rows], fontsize=10)
ax.text(-0.02, 6.35, "对几何排序", transform=ax.get_yaxis_transform(), ha="right", fontsize=10.5, fontweight="bold")
ax.text(-0.02, 2.55, "对随机排序", transform=ax.get_yaxis_transform(), ha="right", fontsize=10.5, fontweight="bold")
ax.set_ylim(-0.8, 6.8)
ax.set_xlim(0, 20)
ax.set_xlabel("20个场景里的个数")
ax.set_title("模型排序逐景对比", fontsize=10.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_GREEN, label="模型排序更低"), Patch(color=C_LIGHT, label="持平"),
                   Patch(color=C_RED, label="模型排序更高")], loc="upper center", bbox_to_anchor=(0.45, -0.2),
          ncol=3, frameon=False, fontsize=9.5, handlelength=1.2, columnspacing=1.0)
ax.grid(axis="y", visible=False)
save_png(fig, "W21_局部引导排序.png")
