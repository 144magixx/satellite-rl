# -*- coding: utf-8 -*-
"""W11_追加阶段.png：按报告表格数字重画（本机没有该实验的逐场景原始记录）。

数字出处：docs/第三阶段/实验报告/03_SI优先功率细调与容量近似对照.md 两个预算下的“基本服务、SGM取舍和停止情况”表
（每种做法300次动作），见 _s3_tables.py 的 W11_STOP 和 W11_CALLS。
左图：每种做法的追加求解是怎么停下来的；右图：这一段平均用了多少次物理容量计算（每次都要真算一遍容量）。
运行：python W11_追加阶段.py
"""
import numpy as np

from _s3_style import *
from _s3_tables import W11_CALLS, W11_STOP

names = ["原SCA", "先提满足率＋原近似", "先提满足率＋新近似"]
colors = {"用完2000次容量调用": C_B, "调整步长过小": C_A, "候选方向静止": C_D, "迭代上限": C_E, "30秒时间上限": C_F}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.5), gridspec_kw={"width_ratios": [1.25, 1]})
rows = []
for b in W11_STOP:
    for k, m in enumerate(names):
        rows.append(("3000 W预算｜" if b.endswith("3000 W") else "初始功率上限｜", m, k, b))
y = np.arange(len(rows))[::-1]
left = np.zeros(len(rows))
for reason, col in colors.items():
    v = np.array([W11_STOP[r[3]][reason][r[2]] for r in rows], dtype=float)
    ax1.barh(y, v, left=left, color=col, label=reason, height=0.7)
    for yi, l_, vv in zip(y, left, v):
        if vv >= 25:
            ax1.text(l_ + vv / 2, yi, f"{int(vv)}", ha="center", va="center", fontsize=7.8, color="white")
    left += v
ax1.set_yticks(y)
ax1.set_yticklabels([r[0] + r[1].replace("先提满足率＋", "") if r[2] else r[0] + r[1] for r in rows], fontsize=8)
ax1.set_xlim(0, 300)
ax1.set_xlabel("动作数（每种做法300次）")
ax1.set_title("追加求解怎么停下来的", fontsize=9.5)
ax1.grid(axis="y", visible=False)
ax1.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=7.8, frameon=False)

x = np.arange(3)
w = 0.38
for j, (b, vals) in enumerate(W11_CALLS.items()):
    ax2.bar(x + (j - 0.5) * w, vals, w, color=BUDGET_COLORS[j], label=b)
    for xi, v in zip(x + (j - 0.5) * w, vals):
        ax2.text(xi, v + 30, f"{v:.0f}", ha="center", va="bottom", rotation=90, fontsize=7.6)
ax2.set_xticks(x)
ax2.set_xticklabels(["原SCA", "先提满足率\n＋原近似", "先提满足率\n＋新近似"], fontsize=8)
ax2.set_ylabel("平均容量计算次数")
ax2.set_title("平均用了多少次容量计算", fontsize=9.5)
ax2.set_ylim(0, 3000)
ax2.grid(axis="x", visible=False)
ax2.legend(loc="upper right", fontsize=7.8, frameon=False)
fig.tight_layout()
save(fig, "W11_追加阶段.png")
