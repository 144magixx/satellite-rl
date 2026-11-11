# -*- coding: utf-8 -*-
"""W21 图：共同初始布局，以及三种全景匹配、两种顺序重构的峰值负载和负载CV（12个训练场景的平均）。

数据：第一阶段实验报告23 完整结果表（逐母原始表不在本机可读位置，取报告表格数字）：
  峰值请求负载 Mbps：共同初始布局 271.800；全景匹配 按距离 268.200、少改归属 255.600、最小扩宽 268.200；
                   顺序重构 束数不变 234.000、可变束数 230.400；
  负载CV：0.5177、0.5142、0.5051、0.5139、0.4936、0.4698；
  括号里是各方法达到“最重束人数降到比起点少1人”目标的场景数：—、2、9、2、5、6（共12个）。
"""
from _c1821 import *

rows = [("共同初始布局", 271.800, 0.517710, None, C_GREY),
        ("全景匹配\n（按距离）", 268.200, 0.514180, 2, C_LIGHT),
        ("全景匹配\n（少改归属）", 255.600, 0.505110, 9, C_BLUE),
        ("全景匹配\n（少扩宽）", 268.200, 0.513901, 2, C_LIGHT),
        ("顺序重构\n（束数不变）", 234.000, 0.493586, 5, C_TEAL),
        ("顺序重构\n（可变束数）", 230.400, 0.469750, 6, C_RED)]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.6), sharey=True, gridspec_kw={"wspace": 0.08})
y = list(range(len(rows)))[::-1]
for ax, idx, title, fmt, xmax in ((axes[0], 1, "最重波束的请求负载/Mbps", "{:.0f}", 345),
                                   (axes[1], 2, "负载CV", "{:.2f}", 0.68)):
    for yi, r in zip(y, rows):
        ax.barh(yi, r[idx], 0.62, color=r[4])
        ax.text(r[idx] + xmax * 0.012, yi, fmt.format(r[idx]), va="center", fontsize=10)
    ax.set_xlim(0, xmax)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(y)
axes[0].set_yticklabels([r[0] for r in rows], fontsize=10)
axes[1].set_xticks([0, 0.2, 0.4, 0.6])
save_png(fig, "W21_四种重构方式.png")
