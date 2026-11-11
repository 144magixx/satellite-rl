"""W23 图：四种功率档位，不带/带余量摘要，15万步终点的开发评价（20个开发场景，每场景3次）。
数字来源：实验报告23 第2节“主要结果（15万步终点）”表：SGM、SI、总功率三列。
档位：原档位（5～50瓦十档）、只细化5瓦以上、原档位＋0.5瓦、0.5～50瓦全部细化。
“余量摘要”是额外给模型看的一组信息：已经分配好的波束在某个槽上还有多少富余。
SI取表中四位小数（如0.9625）写成百分数两位小数（96.25%）。
"""
from _w22_24_common import *  # noqa: F401,F403

plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
                     "ytick.labelsize": 10.5, "legend.fontsize": 10, "axes.titlesize": 11})

groups = ["原档位", "只细化5瓦以上", "原档位＋0.5瓦", "0.5～50瓦全部细化"]
sgm = {"无": [0.608, 0.590, 0.748, 0.746], "有": [0.615, 0.603, 0.747, 0.749]}
si = {"无": [96.34, 96.25, 96.10, 96.19], "有": [96.61, 96.77, 96.67, 96.52]}
pw = {"无": [1983, 2054, 1493, 1354], "有": [1866, 1775, 1351, 1841]}

fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.7), sharey=True)
y = [3, 2, 1, 0]
h = 0.36
for ax, d, title, fmt, xlim in (
    (axs[0], sgm, "供需匹配程度", "{:.3f}", (0, 1.06)),
    (axs[1], pw, "平均总功率（W）", "{:.0f}", (0, 3300)),
    (axs[2], si, "需求满足率（%，横轴从94起）", "{:.2f}", (94, 98.6)),
):
    span = xlim[1] - xlim[0]
    base = xlim[0]
    ax.barh([i + h / 2 for i in y], [v - base for v in d["无"]], h, color=BLUE, label="不带余量摘要", left=base)
    ax.barh([i - h / 2 for i in y], [v - base for v in d["有"]], h, color=ORANGE, label="带余量摘要", left=base)
    for i, v1, v2 in zip(y, d["无"], d["有"]):
        ax.text(v1 + span * 0.015, i + h / 2, fmt.format(v1), va="center", fontsize=9)
        ax.text(v2 + span * 0.015, i - h / 2, fmt.format(v2), va="center", fontsize=9)
    ax.set_xlim(*xlim)
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.grid(axis="y", visible=False)
    ax.set_yticks(y)
    ax.set_yticklabels(groups)
    ax.tick_params(axis="x", labelsize=9.5)
axs[0].set_xticks([0, 0.2, 0.4, 0.6, 0.8])
axs[1].set_xticks([0, 1000, 2000, 3000])
axs[2].set_xticks([94, 95, 96, 97])
hd, lb = axs[0].get_legend_handles_labels()
fig.legend(hd, lb, loc="lower center", ncol=2, frameon=False, fontsize=10.5)
fig.tight_layout(w_pad=0.8, rect=[0, 0.07, 1, 1])
save_png(fig, "W23_八组功率档位对比.png", dpi=210)
