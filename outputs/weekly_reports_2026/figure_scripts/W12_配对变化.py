# -*- coding: utf-8 -*-
"""W12_配对变化.png：按报告表格数字重画（本机没有该实验的逐场景原始记录）。

数字出处：docs/第一阶段/实验报告/02_七组覆盖规划独立场景对照.md 的“同场景配对变化”表
（同一个场景里“新组减旧组”，1月22日的100个场景），见 _s3_tables.py 的 W12_PAIRED。
左图：需求覆盖的平均变化（百分点，向右是覆盖变多）；右图：负载不均程度的平均变化（向左是更均匀）。
括号里的数字：左图是100个场景中 覆盖变好/变差/不变 的场景数，右图是 变均匀/变不均 的场景数。
运行：python W12_配对变化.py
"""
import numpy as np

from _s3_style import *
from _s3_tables import W12_PAIRED

labels = [f"{a} 相对 {b}" for a, b, *_ in W12_PAIRED]
y = np.arange(len(labels))[::-1]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=True)
cov = [p[2] for p in W12_PAIRED]
cv = [p[4] for p in W12_PAIRED]
ax1.barh(y, cov, color=[C_A if v >= 0 else C_B for v in cov], height=0.66)
ax2.barh(y, cv, color=[C_C if v <= 0 else C_B for v in cv], height=0.66)
for yi, p in zip(y, W12_PAIRED):
    c = p[3]
    ax1.text(max(p[2], 0) + 0.015, yi, f"{p[2]:+.3f}（{c[0]}/{c[1]}/{c[2]}）", va="center", ha="left", fontsize=7.4)
    k = p[5]
    ax2.text(max(p[4], 0) + 0.004, yi, f"{p[4]:+.3f}（{k[0]}/{k[1]}）", va="center", ha="left", fontsize=7.4)
for ax in (ax1, ax2):
    ax.axvline(0, color="black", lw=0.8)
    ax.grid(axis="y", visible=False)
ax1.set_xlim(-0.5, 0.5)
ax2.set_xlim(-0.14, 0.2)
ax1.set_yticks(y)
ax1.set_yticklabels(labels, fontsize=7.8)
ax1.set_title("需求覆盖的平均变化（百分点）", fontsize=8.8)
ax2.set_title("负载不均程度的平均变化", fontsize=8.8)
fig.text(0.5, 0.014, "括号里：左图 覆盖变好/变差/不变 的场景数，右图 变均匀/变不均 的场景数（共100个场景）", ha="center", fontsize=7.6)
fig.tight_layout(rect=(0, 0.07, 1, 1))
save(fig, "W12_配对变化.png")
