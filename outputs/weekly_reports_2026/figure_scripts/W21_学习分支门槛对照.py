# -*- coding: utf-8 -*-
"""W21 图：几条“学习帮忙”的分支，事先定的门槛和实际结果（实际 ÷ 门槛）。

数据：第一阶段实验报告20、21、22、23、25、26 各自的门槛表（逐项取报告数字，没有再算别的）：
  直接预测布局（20景）  修复后能用上新提议的景数 1，门槛15；质量达标的景数 0，门槛5
  局部引导排序（20景）  质量达标的景数 2，门槛5
  候选排序（12个场景）  超过强规则前8名的场景数 0，门槛3
  全景匹配（12个场景）  比更强的顺序重构还好的场景数 1，门槛3
  两步重构（12个场景）  对强参照达标的场景数 0，门槛3
  提前停止（12个场景）  比逐母事后最佳固定规则再少的尝试次数占比 35/211=16.6%，门槛20%
条的长度是 实际÷门槛，竖线是门槛（100%）。
“质量达标”：峰值、上尾、CV 都要比对照好到事先规定的程度。
"""
from _c1821 import *

rows = [("直接预测布局：\n修复后能用上新提议", 1, 15, "{}个（门槛{}个）"),
        ("直接预测布局：\n质量达标", 0, 5, "{}个（门槛{}个）"),
        ("局部引导排序：\n质量达标", 2, 5, "{}个（门槛{}个）"),
        ("候选排序：\n超过强规则前8名", 0, 3, "{}个（门槛{}个）"),
        ("全景匹配：\n超过顺序重构", 1, 3, "{}个（门槛{}个）"),
        ("两步重构：\n对强参照达标", 0, 3, "{}个（门槛{}个）"),
        ("提前停止：\n比强规则包络再省", 16.6, 20, "{}%（门槛{}%）")]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
y = list(range(len(rows)))[::-1]
for yi, (lab, act, thr, fmt) in zip(y, rows):
    r = act / thr * 100
    ax.barh(yi, r, 0.58, color=C_RED if r < 100 else C_GREEN)
    ax.text(max(r, 0) + 3, yi, fmt.format(act if not isinstance(act, float) else f"{act:.1f}", thr), va="center", fontsize=9.8,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
ax.axvline(100, color=C_GREY, ls="--", lw=1.4)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=10)
ax.set_xlim(0, 210)
ax.set_xlabel("实际 ÷ 门槛（%），100%表示刚好达到")
ax.grid(axis="y", visible=False)
save_png(fig, "W21_学习分支门槛对照.png")
