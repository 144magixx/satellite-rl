# -*- coding: utf-8 -*-
"""W19 图：第一轮里（a）邻接图划分相对共同受限搜索、（b）MLM三组配对对照，在68个有效母场景上的CV变化。

数据：first_stage_nondrl12_20261012 最终交付包
  E:/paper_archives/outputs/experiments/first_stage_nondrl12_20261012/delivery/final_b7c869c028d3b114fadb/
  nd12_final_b7c869c028d3b114fadb.zip 里
    publication/figures_v2/paired_mothers.csv（route=9, method=main, baseline=BH, group=三档母平均：cv_relative_gain）
    publication/figures_v2/attribution_pairs.csv（M−H、MO−HO、MO−O 逐母 cv_delta）
（a）每根柱是一个母场景的CV降低比例，正值表示比共同受限搜索更均衡；
（b）每个点是一个母场景的CV差（前者减后者），负值表示前者更均衡。
"""
import zipfile

import numpy as np
import pandas as pd
from _c1821 import *

ZIP = E_EXP / "first_stage_nondrl12_20261012/delivery/final_b7c869c028d3b114fadb/nd12_final_b7c869c028d3b114fadb.zip"
z = zipfile.ZipFile(ZIP)
pm = pd.read_csv(z.open("publication/figures_v2/paired_mothers.csv"), encoding="utf-8-sig")
g = pm[(pm.route == 9) & (pm.method == "main") & (pm.baseline == "BH") & (pm.group == "三档母平均")]
g = g.dropna(subset=["cv_relative_gain"])
assert len(g) == 68
gain = np.sort(g.cv_relative_gain.values * 100)[::-1]
ap = pd.read_csv(z.open("publication/figures_v2/attribution_pairs.csv"), encoding="utf-8-sig")
ap = ap[ap.group == "三档母平均"].dropna(subset=["cv_delta"])

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.38})
ax = axes[0]
cols = [C_GREEN if v > 1e-9 else (C_RED if v < -1e-9 else C_LIGHT) for v in gain]
ax.bar(range(len(gain)), gain, color=cols, width=0.85)
ax.axhline(0, color=C_GREY, lw=0.8)
n_imp, n_reg = int((gain > 1e-9).sum()), int((gain < -1e-9).sum())
ax.text(0.97, 0.95, f"变均衡 {n_imp} 个\n变差 {n_reg} 个\n其余持平", transform=ax.transAxes, ha="right", va="top", fontsize=10.5)
ax.set_xlabel("68个母场景（按改善排序）")
ax.set_ylabel("CV降低比例/%")
ax.set_xticks([])
ax.set_title("邻接图划分 对 共同受限搜索", fontsize=11)

ax = axes[1]
pairs = [("M−H", "A"), ("MO−HO", "B"), ("MO−O", "C")]
rng = np.random.default_rng(0)
for i, (k, lab) in enumerate(pairs):
    d = ap[ap.comparison == k].cv_delta.values
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d, s=10, color=C_BLUE, alpha=0.6)
    ax.hlines(d.mean(), i - 0.3, i + 0.3, color=C_RED, lw=2)
    ni, nr = int((d < -1e-9).sum()), int((d > 1e-9).sum())
    ax.text(i, 0.0305, f"更均衡{ni}\n更差{nr}", ha="center", va="center", fontsize=10)
ax.axhline(0, color=C_GREY, lw=0.8)
ax.set_xticks(range(3))
ax.set_xticklabels([p[1] for p in pairs])
ax.set_xlabel("A：MLM直接补全\n　　对 规则补全\nB：MLM缩域＋优化\n　　对 规则缩域＋优化\nC：MLM缩域＋优化\n　　对 全候选池优化",
              fontsize=10.2, loc="left", linespacing=1.3)
ax.set_ylim(-0.03, 0.04)
ax.set_ylabel("CV差（前减后，负为更均衡）")
ax.set_title("MLM的三组配对", fontsize=11)
save_png(fig, "W19_第一轮逐母变化.png")
