# -*- coding: utf-8 -*-
"""W10_耗时分解.png：按报告表格数字重画（本机没有该实验的逐场景原始记录）。

数字出处：docs/第三阶段/实验报告/02_七路线功率细调与逐场景耗时对照.md 的“时间分段”表中总功率预算3000 W的七行
（单次均值，秒），见 _s3_tables.py 的 W10_TIMING。每根条分成：初始分配（T0生成）、
物理准备与核验（两段都只有约0.003秒和0.001秒，合并）、算法求解（扣掉写日志的回调时间）、写日志的回调时间。
运行：python W10_耗时分解.py
"""
import numpy as np

from _s3_style import *
from _s3_tables import W10_METHODS, W10_TIMING

names = W10_METHODS
t0 = np.array([W10_TIMING[m][0] for m in names])
prep = np.array([W10_TIMING[m][1] + W10_TIMING[m][5] for m in names])
algo = np.array([W10_TIMING[m][4] for m in names])
cb = np.array([W10_TIMING[m][3] for m in names])
total = np.array([W10_TIMING[m][6] for m in names])

fig, ax = plt.subplots(figsize=(6.4, 3.4))
y = np.arange(len(names))[::-1]
left = np.zeros(len(names))
for v, c, lab in [(t0, "#bdbdbd", "初始分配"), (prep, "#7f7f7f", "物理准备与核验"),
                  (algo, C_A, "算法求解"), (cb, C_D, "写日志的回调")]:
    ax.barh(y, v, left=left, color=c, label=lab, height=0.68)
    left += v
for yi, t in zip(y, total):
    ax.text(t + 0.15, yi, f"{t:.1f} 秒", va="center", fontsize=8.5)
ax.set_yticks(y)
ax.set_yticklabels(names)
ax.set_xlabel("单次总耗时（秒，3000 W总功率预算，单次均值）")
ax.set_xlim(0, 14.5)
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", fontsize=8.5, frameon=False)
fig.tight_layout()
save(fig, "W10_耗时分解.png")
