"""W22 图8：可变束数、可变束宽的新布局上，四个SAC模型相对随机、固定、贪心三种规则的提升（40个确认场景平均）。

数据来源：报告“第二阶段/实验报告/18_T0束宽与干扰状态的直出基线对照.md”第1节第二张表
（每格是SI百分点差、SGM的0到1原值差）。
本机没有这批实验的逐场景原始记录，所以按报告表格数字重画。
"""
from _w22_24_common import *
import numpy as np

models = ["原状态", "加本束宽度", "加干扰关系摘要", "宽度和干扰关系都加"]
rules = [("随机", PINK), ("固定", GRAY), ("贪心", YELLOW)]
si = {"随机": [5.9020, 6.5381, 6.3082, 6.5330], "固定": [2.3042, 2.9402, 2.7103, 2.9351], "贪心": [2.7880, 3.4241, 3.1942, 3.4190]}
sgm = {"随机": [0.310570, 0.323455, 0.321925, 0.319588], "固定": [0.404658, 0.417544, 0.416013, 0.413677],
       "贪心": [0.328570, 0.341456, 0.339925, 0.337589]}

fig, axes = plt.subplots(1, 2, figsize=(W, 3.3), sharey=True)
y = np.arange(len(models))
h = 0.26
for ax, data, title, fmt, xmax in zip(axes, [si, sgm], ["需求满足率比规则高（百分点）", "供需匹配程度比规则高（0到1）"],
                                      ["{:.1f}", "{:.2f}"], [8.2, 0.52]):
    for j, (name, col) in enumerate(rules):
        pos = y + (j - 1) * h
        ax.barh(pos, data[name], height=h * 0.92, color=col, label="相对" + name, edgecolor="white")
        for p, v in zip(pos, data[name]):
            ax.text(v, p, " " + fmt.format(v), va="center", ha="left", fontsize=7.5)
    ax.set_xlim(0, xmax)
    ax.set_title(title, fontsize=9.5)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(y)
axes[0].set_yticklabels(models, fontsize=8.5)
axes[0].invert_yaxis()
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=8.5, handlelength=1.4)
fig.text(0.5, -0.17, "每个SAC模型各训练15万步；40个确认场景的平均差", ha="center", fontsize=8.5)
fig.tight_layout(w_pad=1.0)
save_png(fig, "W22_四个SAC相对三种规则的提升.png")
