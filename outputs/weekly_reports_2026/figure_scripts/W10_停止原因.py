# -*- coding: utf-8 -*-
"""W10_停止原因.png：按报告表格数字重画（本机没有该实验的逐场景原始记录）。

数字出处：docs/第三阶段/实验报告/02_七路线功率细调与逐场景耗时对照.md 末尾的“停止原因/300”表中总功率预算3000 W的七行
（每种做法300次动作），见 _s3_tables.py 的 W10_STOP。
运行：python W10_停止原因.py
"""
import numpy as np

from _s3_style import *
from _s3_tables import W10_METHODS, W10_STOP

colors = {"未改动": "#bdbdbd", "用完2000次容量计算上限": C_B, "求解器未收敛": C_F, "正常结束": C_A,
          "信赖域太小": C_D, "没有可接受的下一步": C_E}
fig, ax = plt.subplots(figsize=(6.4, 3.4))
y = np.arange(len(W10_METHODS))[::-1]
left = np.zeros(len(W10_METHODS))
for name, vals in W10_STOP.items():
    v = np.array(vals, dtype=float)
    ax.barh(y, v, left=left, color=colors[name], label=name, height=0.68)
    for yi, l_, vv in zip(y, left, v):
        if vv >= 20:
            ax.text(l_ + vv / 2, yi, f"{int(vv)}", ha="center", va="center", fontsize=8,
                    color="black" if name == "未改动" else "white")
    left += v
ax.set_yticks(y)
ax.set_yticklabels(W10_METHODS)
ax.set_xlim(0, 300)
ax.set_xlabel("动作数（3000 W总功率预算，每种做法300次）")
ax.grid(axis="y", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3, fontsize=8, frameon=False)
fig.tight_layout()
save(fig, "W10_停止原因.png")
