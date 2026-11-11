# -*- coding: utf-8 -*-
"""W20 图：同一个被打乱的输入分别交给旧修复器和新修复器，最重波束的请求负载对比（每点一个案例）。

数据：E:/paper_archives/outputs/experiments/first_stage_ai_search_20261013/round5/analysis/p0_attempt1/paired48.csv
（48行 = 24个案例 × 2种打乱方式；D4_maxload、D5_maxload 是旧、新修复器发布布局的最大束请求负载，bit/s，这里除以10^6）。
对角线以下表示新修复器的最重波束负载更小。脚本里核对报告17的胜/平/负：遮蔽归属 13/7/4，加锚点替换 15/6/3。
"""
import pandas as pd
from _c1821 import *

p = pd.read_csv(E_EXP / "first_stage_ai_search_20261013/round5/analysis/p0_attempt1/paired48.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), sharex=True, sharey=True, gridspec_kw={"wspace": 0.08})
exp = {"OWNER10": (13, 7, 4), "JOINT10": (15, 6, 3)}
for ax, (meth, title) in zip(axes, (("OWNER10", "遮蔽10%飞机的归属"), ("JOINT10", "遮蔽归属＋换锚点"))):
    g = p[p.method == meth]
    x, y = g.D4_maxload / 1e6, g.D5_maxload / 1e6
    d = g.delta_maxload / 1e6
    lo, eq, hi = d < -1e-6, d.abs() <= 1e-6, d > 1e-6
    assert (int(lo.sum()), int(eq.sum()), int(hi.sum())) == exp[meth]
    ax.scatter(x[lo], y[lo], s=28, color=C_RED, label=f"新修复器更小 {lo.sum()}个")
    ax.scatter(x[eq], y[eq], s=28, color=C_GREY, label=f"持平 {eq.sum()}个")
    ax.scatter(x[hi], y[hi], s=28, color=C_BLUE, label=f"新修复器更大 {hi.sum()}个")
    lim = max(x.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=C_GREY, ls="--", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("旧修复器的最重波束负载/Mbps")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, handletextpad=0.2)
axes[0].set_ylabel("新修复器的最重波束负载/Mbps")
save_png(fig, "W20_修复器同输入配对.png")
