# -*- coding: utf-8 -*-
"""W11_耗时分解.png：按报告表格数字重画（本机没有该实验的逐场景原始记录）。

数字出处：docs/第三阶段/实验报告/03_SI优先功率细调与容量近似对照.md 的“两预算与时间分段”表
（单次均值，秒），见 _s3_tables.py 的 W11_TIMING。每根条分三段：初始分配与物理准备（T0初始+物理准备）、
原SCA求解与核验、追加求解与核验（原SCA不做追加）。
运行：python W11_耗时分解.py
"""
import numpy as np

from _s3_style import *
from _s3_tables import W11_TIMING

rows = []
for b, d in W11_TIMING.items():
    for m, (t0, prep, a, add, addv, tot) in d.items():
        rows.append((("3000 W预算｜" if b.endswith("3000 W") else "初始功率上限｜") + m, t0 + prep, a, add + addv, tot))
y = np.arange(len(rows))[::-1]
fig, ax = plt.subplots(figsize=(6.6, 3.4))
left = np.zeros(len(rows))
for idx, c, lab in [(1, "#bdbdbd", "初始分配与物理准备"), (2, C_A, "原SCA求解与核验"), (3, C_D, "追加求解与核验")]:
    v = np.array([r[idx] for r in rows])
    ax.barh(y, v, left=left, color=c, label=lab, height=0.66)
    left += v
for yi, r in zip(y, rows):
    ax.text(r[4] + 0.15, yi, f"{r[4]:.1f} 秒", va="center", fontsize=8.5)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=8.3)
ax.set_xlim(0, 13.5)
ax.set_xlabel("单次完整耗时（秒，单次均值）")
ax.grid(axis="y", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.17), ncol=3, fontsize=8, frameon=False)
fig.tight_layout()
save(fig, "W11_耗时分解.png")
