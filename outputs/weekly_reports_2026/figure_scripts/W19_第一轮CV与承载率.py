# -*- coding: utf-8 -*-
"""W19 图：第一轮十二条非强化学习路线里入选方法的负载CV与承载率（100个母场景中有飞机的68个）。

数据：first_stage_nondrl12_20261012 最终交付包
  E:/paper_archives/outputs/experiments/first_stage_nondrl12_20261012/delivery/final_b7c869c028d3b114fadb/
  nd12_final_b7c869c028d3b114fadb.zip 里的 publication/figures_v2/summary.csv
  （group=三档母平均，metric=load_cv_all 和 aircraft_carry，mean 列；与报告10第一张表一致）。
CV越低越好，承载率越高越好；虚线是预先定的“CV比共同受限搜索低5%”门槛。
"""
import zipfile

import pandas as pd
from _c1821 import *

ZIP = E_EXP / "first_stage_nondrl12_20261012/delivery/final_b7c869c028d3b114fadb/nd12_final_b7c869c028d3b114fadb.zip"
s = pd.read_csv(zipfile.ZipFile(ZIP).open("publication/figures_v2/summary.csv"), encoding="utf-8-sig")
s = s[(s.group == "三档母平均")]
s["route"] = s["route"].astype(str)


def pick(route, method, metric):
    r = s[(s.route == route) & (s.method == method) & (s.metric == metric)]
    assert len(r) == 1 and int(r.defined.iloc[0]) == 68
    return float(r["mean"].iloc[0])


items = [  # (route, method, 名称, 颜色, 标签偏移(x,y), 对齐)
    ("6", "BH", "共同受限搜索（对照）", C_GREY, (8, 8), "left"),
    ("1", "M", "MLM直接补全", C_PINK, (-9, -4), "right"),
    ("1", "MO", "MLM缩小候选＋优化", C_PURPLE, (9, -16), "left"),
    ("6", "main", "需求约束聚类", C_TEAL, (9, 10), "left"),
    ("7", "main", "足迹约束聚类", C_BLUE, (9, -2), "left"),
    ("9", "main", "邻接图划分", C_RED, (9, -4), "left"),
    ("11", "main", "大邻域重建", C_ORANGE, (-9, -8), "right"),
    ("12", "main", "多目标进化", C_GREEN, (-9, 10), "right"),
]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
pts = {}
for r, m, name, c, off, ha in items:
    cv = pick(r, m, "load_cv_all")
    carry = pick(r, m, "aircraft_carry") * 100
    pts[name] = (cv, carry)
    ax.scatter(cv, carry, s=90 if name.startswith("邻接") or name.startswith("共同") else 60, color=c, zorder=3)
    ax.annotate(name, (cv, carry), textcoords="offset points", xytext=off, ha=ha, fontsize=10.5, color=c)
bh_cv, bh_carry = pts["共同受限搜索（对照）"]
ax.axvline(bh_cv * 0.95, color=C_GREY, ls="--", lw=1)
ax.text(bh_cv * 0.95 + 0.0004, 83.9, "预定门槛：\nCV比对照低5%", fontsize=10, color=C_GREY, va="bottom", ha="right")
ax.axvline(bh_cv, color=C_LIGHT, ls=":", lw=1)
ax.axhline(bh_carry, color=C_LIGHT, ls=":", lw=1)
ax.set_xlim(0.452, 0.493)
ax.set_ylim(83.8, 86.2)
ax.set_xlabel("负载CV（越低越均衡）")
ax.set_ylabel("需求承载率/%（越高越好）")
ax.invert_xaxis()
save_png(fig, "W19_第一轮CV与承载率.png")
