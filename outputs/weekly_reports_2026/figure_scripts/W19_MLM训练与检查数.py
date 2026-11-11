# -*- coding: utf-8 -*-
"""W19 图：MLM 排序模块的训练损失、4个验证节点的最重波束负载倍数，以及在线每个案例检查的候选数。

数据：first_stage_geometry_load_20261012（E:/paper_archives/outputs/experiments/.../analysis/final_attempt3/report/）
  learningcurve.csv      10000次更新的原始 loss（浅线是原值，深线是100次滑动平均）
  validation_curves.csv  4个验证节点（2500/5000/7500/10000次更新）的 P（母场景均值）
  summary.csv            共同验证上 联合重分/规则/随机 的 P，作为水平参照
  timing.csv             task 含 common_test、stage=online 的 attempts：每个案例在线检查的候选数均值
"""
import pandas as pd
from _c1821 import *

B = E_EXP / "first_stage_geometry_load_20261012/analysis/final_attempt3/report"
lc = pd.read_csv(B / "learningcurve.csv")
vc = pd.read_csv(B / "validation_curves.csv")
sm = pd.read_csv(B / "summary.csv")
sm = sm[sm.role == "validation"].set_index("method")
tm = pd.read_csv(B / "timing.csv")
tm = tm[(tm.stage == "online") & tm.task.str.contains("common_test")]
att = tm.groupby("method").attempts.mean()
assert len(lc) == 10000 and len(vc) == 4

fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), gridspec_kw={"wspace": 0.6, "width_ratios": [1.1, 1, 1]})
ax = axes[0]
ax.plot(lc["update"], lc["loss"], color=C_LIGHT, lw=0.5)
ax.plot(lc["update"], lc["loss"].rolling(100, min_periods=1).mean(), color=C_PURPLE, lw=1.8)
ax.set_yscale("log")
ax.set_xlabel("训练更新次数")
ax.set_ylabel("训练损失")
ax.set_xticks([0, 5000, 10000])
ax.set_xticklabels(["0", "5千", "1万"])
ax.set_title("训练损失", fontsize=11)

ax = axes[1]
ax.plot(vc.node, vc.P, color=C_PURPLE, marker="o", lw=1.8, label="MLM各节点")
for k, name, c, ls in (("JOINT", "联合重分", C_RED, "--"), ("rule", "规则顺序", C_ORANGE, ":")):
    ax.axhline(sm.loc[k, "P"], color=c, ls=ls, lw=1.4, label=name)
ax.set_ylim(5.5, 6.3)
ax.set_xticks([2500, 5000, 7500, 10000])
ax.set_xticklabels(["2.5千", "5千", "7.5千", "1万"], fontsize=9.5)
ax.set_xlabel("训练更新次数")
ax.set_ylabel("最重波束负载倍数")
ax.set_title("验证结果", fontsize=11)
ax.legend(loc="lower center", frameon=False, fontsize=9.5)

ax = axes[2]
names = [("JOINT", "联合重分", C_RED), ("rule", "规则顺序", C_ORANGE), ("random", "随机顺序", C_GREY), ("MLM", "MLM排序", C_PURPLE)]
for i, (k, n, c) in enumerate(names):
    ax.bar(i, att[k], color=c, width=0.65)
    ax.text(i, att[k] - 40, f"{att[k]:.0f}", ha="center", va="top", fontsize=10.5, color="white", fontweight="bold",
            rotation=90)
ax.set_xticks(range(4))
ax.set_xticklabels([n for _, n, _ in names], rotation=35, ha="right", fontsize=10)
ax.set_ylim(0, 1450)
ax.set_ylabel("每个案例检查的候选数")
ax.set_title("在线检查量", fontsize=11)
ax.grid(axis="x", visible=False)
save_png(fig, "W19_MLM训练与检查数.png")
