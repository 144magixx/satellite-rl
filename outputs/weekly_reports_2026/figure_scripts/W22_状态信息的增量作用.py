"""W22 图6：给SAC的状态里加入本束宽度、干扰关系摘要后，需求满足率和供需匹配程度的变化（同一场景配对）。

数据来源：报告“第二阶段/实验报告/18_T0束宽与干扰状态的直出基线对照.md”第3节“宽度与关系信息的增量作用”表
（均值和95%描述性区间；SI差为百分点，SGM差为0到1的原值差；括号里是40个场景中改善/变差的场景数）。
本机没有这批实验的逐场景原始记录，所以按报告表格数字重画。
"""
from _w22_24_common import *
import numpy as np

rows = [  # 名称, SI均差, SI下界, SI上界, SGM均差, SGM下界, SGM上界, SI改善数, SI变差数, SGM改善数, SGM变差数
    ("加本束宽度（对原状态）",     0.636036, 0.233048, 1.062428, 0.012886, 0.004637, 0.020564, 26, 14, 30, 10),
    ("加干扰关系摘要（对原状态）", 0.406148, 0.001667, 0.831673, 0.011355, 0.004116, 0.018965, 22, 18, 27, 13),
    ("已有干扰关系再加宽度",       0.224819, -0.116643, 0.574622, -0.002337, -0.009902, 0.004554, 20, 20, 20, 20),
    ("已有宽度再加干扰关系",       -0.005069, -0.382484, 0.349627, -0.003867, -0.008398, 0.000678, 21, 19, 14, 26),
]
fig, axes = plt.subplots(1, 2, figsize=(W, 2.8), sharey=True)
y = np.arange(len(rows))
for ax, (m, lo, hi, a, b), title, fmt in zip(axes, [(1, 2, 3, 7, 8), (4, 5, 6, 9, 10)],
                                              ["需求满足率变化（百分点）", "供需匹配程度变化（0到1）"],
                                              ["{:+.2f}", "{:+.3f}"]):
    for yi, r in zip(y, rows):
        col = BLUE if r[lo] > 0 else GRAY
        ax.plot([r[lo], r[hi]], [yi, yi], color=col, lw=3.2, solid_capstyle="butt")
        ax.plot(r[m], yi, marker="o", ms=5.5, mfc="white", mec="black", mew=1.2, zorder=5)
        ax.text(r[m], yi - 0.27, fmt.format(r[m]) + f"（{r[a]}/{r[b]}）", ha="center", va="bottom", fontsize=8.5,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8), zorder=6)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_title(title, fontsize=9.5)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(y)
axes[0].set_yticklabels([r[0] for r in rows], fontsize=8.5)
axes[0].set_ylim(3.45, -0.95)
axes[0].set_xlim(-0.65, 1.45)
axes[0].set_xticks([-0.5, 0, 0.5, 1.0])
axes[1].set_xlim(-0.016, 0.03)
fig.text(0.5, -0.02, "线段是95%区间，空心点是平均变化；蓝色表示区间整体在0右侧，灰色表示区间跨过0\n括号里是40个确认场景中变好/变差的场景数",
         ha="center", va="top", fontsize=8.5)
fig.tight_layout(w_pad=1.0)
save_png(fig, "W22_状态信息的增量作用.png")
