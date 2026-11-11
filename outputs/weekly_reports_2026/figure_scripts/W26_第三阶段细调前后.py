# -*- coding: utf-8 -*-
"""W26 图：SCA功率细调前后，块指针头T-SAC和容量感知贪心的三项指标（上排旧确认集，下排新确认集）。

数据来源（只读）：D:/paper/outputs/experiments/s3lab_20261029/runs/main/rows.jsonl
  method=sca 的行：细调前 W0 / worst10_0 / F0（供需匹配程度），细调后 a_W / a_worst10 / a_SGM（保SGM接入后）。
  块指针头 tier=ptr（3套方案平均），贪心 tier=greedy。对全部输入取平均。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *
from _s3 import load

df = load("main")
sca = df[df.method == "sca"]
metrics = [("W0", "a_W", "需求满足率"), ("worst10_0", "a_worst10", "最差10%飞机"), ("F0", "a_SGM", "供需匹配程度")]
plans = [("ptr", "块指针头T-SAC", C_KEY), ("greedy", "容量感知贪心", "#555b66")]
fig, axes = plt.subplots(2, 3, figsize=(5.8, 4.1), sharex=True)
for r, (st, stname) in enumerate((("old", "旧确认集"), ("new", "新确认集"))):
    for c, (b, a, title) in enumerate(metrics):
        ax = axes[r, c]
        vals = {}
        for tier, name, col in plans:
            sub = sca[(sca.set == st) & (sca.tier == tier)]
            vals[tier] = (float(sub[b].mean()), float(sub[a].mean()))
            ax.plot([0, 1], vals[tier], color=col, marker="o", ms=5, lw=1.8)
        # 数字标注：两条线靠得近时上下错开
        lo = min(v[0] for v in vals.values()); hi = max(v[0] for v in vals.values())
        span = max(hi - lo, 1e-9)
        for side, x, ha, dx in ((0, 0, "right", -0.08), (1, 1, "left", 0.08)):
            items = sorted(((vals[t][side], t) for t, _, _ in plans), reverse=True)
            ys = [it[0] for it in items]
            rng = ax_range = 0.0
            allv = [v for tv in vals.values() for v in tv]
            gap = (max(allv) - min(allv)) * 0.16 + 1e-9
            if ys[0] - ys[1] < gap:
                mid = (ys[0] + ys[1]) / 2
                ys = [mid + gap / 2, mid - gap / 2]
            for (v, t), yy in zip(items, ys):
                colr = dict((p[0], p[2]) for p in plans)[t]
                ax.text(x + dx, yy, f"{v:.3f}" if title == "供需匹配程度" else f"{v:.3f}", ha=ha, va="center", fontsize=8.3, color=colr)
        ax.set_xlim(-0.95, 1.95)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["细调前", "SCA后"], fontsize=8.3)
        allv = [v for tv in vals.values() for v in tv]
        pad = (max(allv) - min(allv)) * 0.25
        ax.set_ylim(min(allv) - pad, max(allv) + pad)
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(True, axis="x", color="#eeeeee", lw=0.6)
        if r == 0:
            ax.set_title(title, fontsize=9)
        if c == 0:
            ax.text(-0.02, 1.0, stname, transform=ax.transAxes, ha="right", va="center", fontsize=8.5, rotation=90)
        print(st, title, {t: np.round(v, 3) for t, v in vals.items()})
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([0], [0], color=p[2], marker="o", lw=1.8, label=p[1]) for p in plans], loc="lower center", ncol=2,
           frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.14, wspace=0.12, hspace=0.25)
save_png(fig, "W26_第三阶段细调前后.png", dpi=220)
