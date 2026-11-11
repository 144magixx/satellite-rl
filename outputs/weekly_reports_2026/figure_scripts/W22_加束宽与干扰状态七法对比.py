"""W22 图5：可变束数、可变束宽的新布局上，四个SAC（状态不同）和三种规则在40个确认场景上的需求满足率与供需匹配程度。

数据来源：报告“第二阶段/实验报告/18_T0束宽与干扰状态的直出基线对照.md”第1节“七法共同确认”表
（SI为百分数，SGM为0到1的原值，40个场景×动作重复先平均再等权）。
本机没有这批实验的逐场景原始记录（D:\\paper\\outputs\\experiments\\s2_width26_20261014 下只剩报告引用的PNG），
所以这张图按报告里的表格数字重画，没有用原图。
"""
from _w22_24_common import *
import numpy as np

names = ["原状态", "加本束宽度", "加干扰关系摘要", "宽度和干扰关系都加", "随机", "固定", "贪心"]
si = [94.9591, 95.5952, 95.3653, 95.5901, 89.0571, 92.6550, 92.1711]
sgm = [0.608633, 0.621518, 0.619988, 0.617651, 0.298063, 0.203974, 0.280062]
cols = [BLUE, BLUE, BLUE, BLUE, PINK, GRAY, YELLOW]
hatch = ["", "", "", "", "", "", ""]

fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), sharey=True)
y = np.arange(len(names))
for ax, vals, title, fmt, xlim in zip(axes, [si, sgm], ["需求满足率（%）", "供需匹配程度（0到1）"],
                                       ["{:.1f}", "{:.2f}"], [(85, 97.5), (0, 0.78)]):
    bars = ax.barh(y, vals, color=cols, height=0.62, edgecolor="white")
    bars[1].set_edgecolor("black")
    bars[1].set_linewidth(1.2)
    for yi, v in zip(y, vals):
        ax.text(v, yi, " " + fmt.format(v), va="center", ha="left", fontsize=8.5)
    ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=9.5)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(y)
axes[0].set_yticklabels(names)
axes[0].invert_yaxis()
fig.text(0.5, -0.02, "蓝色是四个SAC模型（各训练15万步），黑框是加入本束宽度的模型；40个确认场景的平均", ha="center", fontsize=8.5)
fig.tight_layout(w_pad=1.0)
save_png(fig, "W22_加束宽与干扰状态七法对比.png")
