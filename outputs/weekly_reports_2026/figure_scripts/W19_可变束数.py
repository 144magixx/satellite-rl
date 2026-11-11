# -*- coding: utf-8 -*-
"""W19 图：可变束数试验：目标束数与实际束数，以及原束数/多20%束数两个配额分支的负载。

数据：first_stage_geometry_load_20261012 的 figures_v2/03_variable_N.csv
  E:/paper_archives/outputs/experiments/first_stage_geometry_load_20261012/analysis/final_attempt3/figures_v2/
  每行是一个“母场景×请求分布”的案例（160个）在某个阶段的结果。
  stage=construct：按目标束数重新构造（目标为原束数的约0.8倍或1.2倍；outcome=completed 为构造成功，
  not_found 为在本次预算内没找到，此时 N 是保留下来的旧布局的束数）。
  stage=same_N_quota / target_quota：构造后的热点配额调整；maxload、top10_mean 是原单位（bit/s）的
  最重束负载和最重10%波束平均负载。
左图：每个点是一个案例（目标束数，实际束数）；右图：两个配额分支的平均负载（Gbit/s）。
"""
import pandas as pd
from _c1821 import *

B = E_EXP / "first_stage_geometry_load_20261012/analysis/final_attempt3/figures_v2"
v = pd.read_csv(B / "03_variable_N.csv")
c = v[v.stage == "construct"]
ok, nf = c[c.outcome == "completed"], c[c.outcome == "not_found"]
assert len(ok) == 160 and len(nf) == 160
q0 = v[v.stage == "same_N_quota"]
q1 = v[v.stage == "target_quota"].dropna(subset=["maxload"])
assert len(q0) == 160 and len(q1) == 160

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.4, "width_ratios": [1.15, 1]})
ax = axes[0]
ax.plot([25, 100], [25, 100], color=C_GREY, ls="--", lw=1)
ax.scatter(nf.target_N, nf.N, s=14, color=C_BLUE, alpha=0.65, label="少20%束：没找到")
ax.scatter(ok.target_N, ok.N, s=14, color=C_RED, alpha=0.65, label="多20%束：构造成功")
ax.set_xlabel("目标束数")
ax.set_ylabel("实际束数")
ax.set_xlim(25, 100)
ax.set_ylim(25, 100)
ax.legend(loc="upper left", frameon=False, fontsize=9.5, handletextpad=0.2)
ax.text(0.97, 0.04, "对角线：达到目标", transform=ax.transAxes, ha="right", fontsize=9.5, color=C_GREY)

ax = axes[1]
vals = {"原束数": (q0.maxload.mean() / 1e9, q0.top10_mean.mean() / 1e9, q0.N.mean()),
        "多20%束": (q1.maxload.mean() / 1e9, q1.top10_mean.mean() / 1e9, q1.N.mean())}
w = 0.36
for i, (k, (pk, tp, n)) in enumerate(vals.items()):
    ax.bar(i - w / 2, pk, w, color=C_BLUE if i == 0 else C_RED, alpha=0.95)
    ax.bar(i + w / 2, tp, w, color=C_BLUE if i == 0 else C_RED, alpha=0.5)
    ax.text(i - w / 2, pk + 0.05, f"{pk:.2f}", ha="center", fontsize=10)
    ax.text(i + w / 2, tp + 0.05, f"{tp:.2f}", ha="center", fontsize=10)
ax.set_xticks([0, 1])
ax.set_xticklabels([f"原束数\n（平均{vals['原束数'][2]:.0f}束）", f"多20%束\n（平均{vals['多20%束'][2]:.0f}束）"])
ax.set_ylabel("请求负载/Gbit/s")
ax.set_ylim(0, 5.4)
ax.grid(axis="x", visible=False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_GREY, label="最重波束"), Patch(color=C_GREY, alpha=0.5, label="最重10%波束平均")],
          loc="upper right", frameon=False, fontsize=9.5)
save_png(fig, "W19_可变束数.png")
