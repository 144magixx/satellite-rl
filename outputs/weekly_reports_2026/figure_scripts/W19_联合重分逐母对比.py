# -*- coding: utf-8 -*-
"""W19 图：多束联合重分与邻接图单机转移、足迹约束聚类在40个母场景上的最重波束负载倍数对比。

数据：first_stage_geometry_load_20261012 的 report/per_mother.csv（role=test，每点是一个母场景，
先对4种请求分布取平均）。位置：
E:/paper_archives/outputs/experiments/first_stage_geometry_load_20261012/analysis/final_attempt3/report/
点在对角线下方表示联合重分的最重波束负载更小。
"""
import pandas as pd
from _c1821 import *

B = E_EXP / "first_stage_geometry_load_20261012/analysis/final_attempt3/report"
pm = pd.read_csv(B / "per_mother.csv")
t = pm[pm.role == "test"]
p = t.pivot(index="mother_id", columns="method", values="P")
assert len(p) == 40

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), gridspec_kw={"wspace": 0.28})
for ax, other, name in ((axes[0], "09-LEX", "邻接图单机转移"), (axes[1], "07", "足迹约束聚类")):
    x, yv = p[other].values, p["JOINT"].values
    better = yv < x - 1e-9
    worse = yv > x + 1e-9
    ax.scatter(x[better], yv[better], s=26, color=C_RED, label=f"联合重分更小 {better.sum()}个")
    ax.scatter(x[worse], yv[worse], s=26, color=C_BLUE, label=f"联合重分更大 {worse.sum()}个")
    ax.scatter(x[~better & ~worse], yv[~better & ~worse], s=26, color=C_GREY, label="持平")
    lim = max(x.max(), yv.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=C_GREY, lw=1, ls="--")
    ax.set_xlim(2, lim)
    ax.set_ylim(2, lim)
    ax.set_xlabel(f"{name}\n最重波束负载（平均的倍数）")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, handletextpad=0.2)
axes[0].set_ylabel("多束联合重分\n最重波束负载（平均的倍数）")
save_png(fig, "W19_联合重分逐母对比.png")
