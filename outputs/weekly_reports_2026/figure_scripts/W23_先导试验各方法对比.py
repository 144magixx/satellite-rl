"""W23 图：先导试验三个模型与三种规则的共同对照（40个场景）。
数字来源：实验报告19 第1节表“方法（模型取50k节点）”（SI、SGM、平均总功率）与
第3节耗时表“完整算法均值（s）”。旧版同节点历史模型与原模型逐值相同，这里不重复画。
方法名：B=原模型（加了本束宽度的SAC，对照），E=温度系数降到0.003，LP=槽数与功率联合建模。
"""
from _w22_24_common import *  # noqa: F401,F403

plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10.5,
                     "ytick.labelsize": 11, "legend.fontsize": 10, "axes.titlesize": 11.5})

names = ["原模型（对照）", "温度系数降到0.003", "槽数与功率联合建模", "随机选择", "固定偏好", "贪心规则"]
sgm = [0.614078, 0.604908, 0.584768, 0.298063, 0.203974, 0.280062]
si = [95.5509, 95.1809, 94.4192, 89.0571, 92.6550, 92.1711]
power = [1816.71, 1785.29, 2202.62, 2190.62, 2406.75, 1739.38]
sec = [1.7906, 1.8175, 1.7825, 1.5516, 1.6143, 1.5974]
cols = [BLUE, ORANGE, GREEN, "#9a9a9a", "#9a9a9a", "#9a9a9a"]

fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), sharey="row")
panels = [
    (axs[0, 0], sgm, "供需匹配程度（越高越好）", "{:.3f}", (0, 0.78)),
    (axs[0, 1], si, "需求满足率（%，横轴从80起）", "{:.1f}", (80, 99.5)),
    (axs[1, 0], power, "平均总功率（W）", "{:.0f}", (0, 2900)),
    (axs[1, 1], sec, "每个场景完整分配耗时（秒）", "{:.2f}", (0, 2.3)),
]
y = list(range(len(names)))[::-1]
for ax, vals, title, fmt, xlim in panels:
    ax.barh(y, vals, color=cols, height=0.62)
    for yy, v in zip(y, vals):
        ax.text(v + (xlim[1] - xlim[0]) * 0.012, yy, fmt.format(v), va="center", fontsize=10)
    ax.set_xlim(*xlim)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
axs[0, 1].tick_params(labelleft=False)
axs[1, 1].tick_params(labelleft=False)
fig.tight_layout(h_pad=1.2, w_pad=0.8)
save_png(fig, "W23_先导试验各方法对比.png", dpi=210)
