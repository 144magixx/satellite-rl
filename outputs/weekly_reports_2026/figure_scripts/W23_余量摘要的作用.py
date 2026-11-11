"""W23 图：给模型加上余量摘要以后，各功率档位下的变化。
数字来源：实验报告23 第3.4节表“加摘要后SGM变化 / SI变化 / 功率变化”：
  原档位 +0.007、+0.0027、-117瓦；细化5瓦以上 +0.013、+0.0053、-279瓦；
  原档位＋0.5瓦 -0.0004、+0.0057、-143瓦；全区间细化 +0.003、+0.0033、+486瓦。
SI变化乘100画成百分点。
"""
from _w22_24_common import *  # noqa: F401,F403

plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10.5,
                     "ytick.labelsize": 10.5, "legend.fontsize": 10, "axes.titlesize": 11})

groups = ["原档位", "只细化\n5瓦以上", "原档位\n＋0.5瓦", "0.5～50瓦\n全部细化"]
dsgm = [0.007, 0.013, -0.0004, 0.003]
dsi = [0.27, 0.53, 0.57, 0.33]
dpw = [-117, -279, -143, 486]

fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.7))
x = list(range(4))
for ax, vals, title, fmt, lim in (
    (axs[0], dsgm, "供需匹配程度变化", "{:+.4f}", (-0.004, 0.02)),
    (axs[1], dsi, "需求满足率变化（百分点）", "{:+.2f}", (-0.1, 0.8)),
    (axs[2], dpw, "总功率变化（W）", "{:+.0f}", (-380, 600)),
):
    cs = [BLUE if v >= 0 else ORANGE for v in vals]
    ax.bar(x, vals, 0.6, color=cs)
    span = lim[1] - lim[0]
    for xx, v in zip(x, vals):
        ax.text(xx, v + (span * 0.02 if v >= 0 else -span * 0.02), fmt.format(v), ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_ylim(*lim)
    ax.set_xticks(x)
    ax.set_xticklabels(["原档位", "细化\n5瓦以上", "加\n0.5瓦", "全部\n细化"], fontsize=9)
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="y", labelsize=9.5)
fig.tight_layout(w_pad=0.8)
save_png(fig, "W23_余量摘要的作用.png", dpi=210)
