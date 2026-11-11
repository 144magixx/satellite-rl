# -*- coding: utf-8 -*-
"""W19 图：第二轮（几何约束负载重分组）九种配置的最重波束负载倍数和负载CV。

数据：first_stage_geometry_load_20261012 的 report/per_mother.csv（role=test，40个母场景，
每个母场景先对4种请求分布取平均，再对40母取平均；与报告11第一张表一致）。
文件位置：E:/paper_archives/outputs/experiments/first_stage_geometry_load_20261012/analysis/final_attempt3/report/
P=最重一束的负载是平均负载的几倍，越小越均匀；CV越小越均匀。
"""
import pandas as pd
from _c1821 import *

B = E_EXP / "first_stage_geometry_load_20261012/analysis/final_attempt3/report"
pm = pd.read_csv(B / "per_mother.csv")
t = pm[pm.role == "test"]
assert t.mother_id.nunique() == 40
m = t.groupby("method")[["P", "CV"]].mean()

order = [("BH", "共同受限搜索\n（对照）", C_GREY), ("09-CV", "邻接图单机转移\n（按CV）", C_LIGHT),
         ("09-LEX", "邻接图单机转移\n（按负载倍数）", C_LIGHT), ("09+SWAP", "邻接图转移＋交换", C_LIGHT),
         ("07", "足迹约束聚类", C_BLUE), ("JOINT", "多束联合重分", C_RED),
         ("rule", "规则顺序\n＋联合重分", C_PINK), ("random", "随机顺序\n＋联合重分", C_PINK),
         ("MLM", "MLM排序\n＋联合重分", C_PURPLE)]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.7), sharey=True, gridspec_kw={"wspace": 0.06})
y = list(range(len(order)))[::-1]
for ax, col, title, fmt, xmax in ((axes[0], "P", "最重波束负载（平均的倍数）", "{:.2f}", 11.8),
                                   (axes[1], "CV", "负载CV", "{:.2f}", 2.2)):
    for yi, (k, name, c) in zip(y, order):
        v = m.loc[k, col]
        ax.barh(yi, v, 0.62, color=c)
        ax.text(v + xmax * 0.012, yi, fmt.format(v), va="center", fontsize=10)
    ax.set_xlim(0, xmax)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(y)
axes[0].set_yticklabels([n for _, n, _ in order], fontsize=10)
axes[1].set_xticks([0, 0.5, 1.0, 1.5, 2.0])
save_png(fig, "W19_第二轮九种配置.png")
