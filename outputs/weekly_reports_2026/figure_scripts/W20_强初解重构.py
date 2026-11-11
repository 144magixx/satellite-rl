# -*- coding: utf-8 -*-
"""W20 图：共同强初始布局上的两种有限重构：较大邻域（D0）和只改最多64架飞机归属（D1）。

数据：
  左图：E:/paper_archives/outputs/experiments/first_stage_ai_search_20261013/round2/analysis/extension/
        full80_attempt1/gates/case_summary.csv（160行 = 80个案例 × 2种方法，target_hit 是“峰值降低至少1%、
        最重10%波束的平均负载和CV不变差”）；脚本里核对为 30 对 36 个案例（13 对 16 个母场景）。
  中图、右图：第一阶段实验报告14 的两张表（20个母场景的平均峰值请求负载；80个案例的射线对比较次数）。
"""
import pandas as pd
from _c1821 import *

SRC = E_EXP / "first_stage_ai_search_20261013/round2/analysis/extension/full80_attempt1/gates/case_summary.csv"
cs = pd.read_csv(SRC)
hit = cs.groupby("variant").target_hit.sum()
mothers = cs[cs.target_hit].groupby("variant").mother_id.nunique()
assert int(hit["D0"]) == 30 and int(hit["D1"]) == 36 and mothers["D0"] == 13 and mothers["D1"] == 16

fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.7), gridspec_kw={"wspace": 0.55})
ax = axes[0]
for i, (k, lab, c) in enumerate((("D0", "较大邻域", C_ORANGE), ("D1", "只改最多\n64架飞机", C_RED))):
    ax.bar(i, hit[k], 0.6, color=c)
    ax.text(i, hit[k] + 0.8, f"{int(hit[k])}", ha="center", fontsize=11)
ax.set_xticks([0, 1])
ax.set_xticklabels(["较大邻域", "只改最多\n64架飞机"], fontsize=9.5)
ax.set_ylim(0, 48)
ax.set_title("达到目标的案例数\n（共80个）", fontsize=10.5)
ax.grid(axis="x", visible=False)

ax = axes[1]
vals = [("共同起点", 1776.87, C_GREY), ("较大邻域", 1711.86, C_ORANGE), ("只改64架", 1665.17, C_RED)]
for i, (lab, v, c) in enumerate(vals):
    ax.bar(i, v, 0.62, color=c)
    ax.text(i, v + 12, f"{v:.0f}", ha="center", fontsize=10.5)
ax.set_xticks(range(3))
ax.set_xticklabels([v[0] for v in vals], rotation=30, ha="right", fontsize=9.5)
ax.set_ylim(1400, 1850)
ax.set_title("平均峰值负载/Mbps\n（纵轴从1400起）", fontsize=10.5)
ax.grid(axis="x", visible=False)

ax = axes[2]
vals = [("较大邻域", 11.952, C_ORANGE), ("只改64架", 4.023, C_RED)]
for i, (lab, v, c) in enumerate(vals):
    ax.bar(i, v, 0.6, color=c)
    ax.text(i, v + 0.25, f"{v:.1f}", ha="center", fontsize=10.5)
ax.set_xticks(range(2))
ax.set_xticklabels([v[0] for v in vals], rotation=30, ha="right", fontsize=9.5)
ax.set_ylim(0, 14.5)
ax.set_title("射线对比较次数\n（百万次）", fontsize=10.5)
ax.grid(axis="x", visible=False)
save_png(fig, "W20_强初解重构.png")
