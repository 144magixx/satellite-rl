"""W22 图7：可变束数、可变束宽的新布局上，七种方法在40个确认场景上的平均总功率、可提供总容量和每景完整分配耗时。

数据来源：报告“第二阶段/实验报告/18_T0束宽与干扰状态的直出基线对照.md”第2节“资源使用与可提供容量”表
（总功率W、可提供总容量Gbps、完整算法时间秒/回合；“回合”就是给一个场景做完整分配）。
本机没有这批实验的逐场景原始记录，所以按报告表格数字重画。
"""
from _w22_24_common import *
import numpy as np

names = ["原状态", "加本束宽度", "加干扰关系摘要", "宽度和干扰关系都加", "随机", "固定", "贪心"]
power = [1858.46, 2019.83, 1648.46, 1693.88, 2190.62, 2406.75, 1739.38]
cap = [26.812, 27.295, 27.666, 27.040, 40.297, 63.596, 44.308]
sec = [1.7257, 1.7143, 1.7350, 1.7023, 1.4212, 1.4922, 1.4766]
cols = [BLUE, BLUE, BLUE, BLUE, PINK, GRAY, YELLOW]

fig, axes = plt.subplots(1, 3, figsize=(W, 2.9), sharey=True)
y = np.arange(len(names))
for ax, vals, title, fmt, xmax in zip(axes, [power, cap, sec],
                                      ["平均总功率（W）", "可提供总容量（Gbps）", "每景完整分配耗时（秒）"],
                                      ["{:.0f}", "{:.1f}", "{:.2f}"], [3100, 82, 2.3]):
    ax.barh(y, vals, color=cols, height=0.62, edgecolor="white")
    for yi, v in zip(y, vals):
        ax.text(v, yi, " " + fmt.format(v), va="center", ha="left", fontsize=8)
    ax.set_xlim(0, xmax)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="x", labelsize=8)
axes[0].set_yticks(y)
axes[0].set_yticklabels(names, fontsize=8.5)
axes[0].invert_yaxis()
fig.text(0.5, -0.02, "蓝色是四个SAC模型；40个确认场景的平均。容量没有按需求截断，过供的波束也会算进去", ha="center", fontsize=8.5)
fig.tight_layout(w_pad=0.8)
save_png(fig, "W22_新布局资源使用与耗时.png")
