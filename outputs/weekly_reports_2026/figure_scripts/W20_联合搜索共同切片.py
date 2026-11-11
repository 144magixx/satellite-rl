# -*- coding: utf-8 -*-
"""W20 图：可变束数与束宽联合搜索，六种配置在“束数和角域都不超过共同初始布局”限制内的峰值负载和CV。

数据：first_stage_ai_search_20261013 首轮开发的
  E:/paper_archives/outputs/experiments/first_stage_ai_search_20261013/analysis/initial_g1/post_attempt2/gates/resource_slices.csv
  取 slice=common、view=lex_maxload_tail_CV、status=completed 的案例（80个案例 = 20个母场景 × 4种请求分布）。
  先对同一母场景的4种分布取平均（只用4种分布都有解的母场景），再对母场景等权平均；
  与报告12第二张表逐项一致（INIT 63个案例/14母/3001.17 Mbps/1.8255 等）。
各方法有解的案例数和母场景数不同，数值不能直接相除。
"""
import numpy as np
import pandas as pd
from _c1821 import *

SRC = E_EXP / "first_stage_ai_search_20261013/analysis/initial_g1/post_attempt2/gates/resource_slices.csv"
rs = pd.read_csv(SRC)
c = rs[(rs.slice == "common") & (rs.view == "lex_maxload_tail_CV")]

names = {"INIT": "三档宽度\n几何初解", "RULE": "规则顺序搜索", "RANDOM": "随机顺序搜索",
         "N_ONLY": "只改束数", "W_ONLY": "只改束宽", "POST_SHRINK": "固定宽度搜索\n＋末端缩宽"}
rows = []
for m in names:
    g = c[c.method == m]
    ok = g[g.status == "completed"]
    per = ok.groupby("mother_id").agg(n=("case_id", "count"), p=("maxload", "mean"), cv=("CV", "mean"))
    full = per[per.n == 4]
    rows.append((m, len(ok), len(full), full.p.mean() / 1e6, full.cv.mean()))
exp = {"INIT": (63, 14, 3001.17, 1.8255), "POST_SHRINK": (80, 20, 1884.25, 1.4066)}
for m, n, nm, p, cv in rows:
    if m in exp:
        assert (n, nm) == exp[m][:2] and abs(p - exp[m][2]) < 0.01, (m, n, nm, p)

fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.6), sharey=True, gridspec_kw={"wspace": 0.08})
y = list(range(len(rows)))[::-1]
cols = {"INIT": C_GREY, "RULE": C_BLUE, "RANDOM": C_LIGHT, "N_ONLY": C_TEAL, "W_ONLY": C_PURPLE, "POST_SHRINK": C_RED}
for ax, idx, title, fmt, xmax in ((axes[0], 3, "最重波束的请求负载/Mbps", "{:.0f}", 4400),
                                   (axes[1], 4, "负载CV", "{:.2f}", 2.2)):
    for yi, r in zip(y, rows):
        ax.barh(yi, r[idx], 0.62, color=cols[r[0]])
        ax.text(r[idx] + xmax * 0.012, yi, fmt.format(r[idx]), va="center", fontsize=10)
    ax.set_xlim(0, xmax)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(y)
axes[0].set_yticklabels([f"{names[r[0]]}\n（{r[1]}/80个案例有解）" for r in rows], fontsize=9.8)
save_png(fig, "W20_联合搜索共同切片.png")
