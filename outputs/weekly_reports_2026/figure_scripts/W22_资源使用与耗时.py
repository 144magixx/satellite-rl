"""W22 图4：六种方法的资源使用（功率档位、每景总功率、每景总槽数）和每景耗时。
数据（都在 E:\\satellite_rl_archives\\s2_t0_fair24_20261013\\publication_t24_v4\\）：
  05_resource_mass.csv   逐束名义功率分布（先场景内平均，再对40个场景平均）
  05_scene_resources.csv 每个场景的总功率（W）和总槽数
  06_direct_time.csv     每个场景直接输出的耗时（秒/100束）
"""
from _w22_24_common import *
import numpy as np
import pandas as pd

P = ARCHIVE_T24
mass = pd.read_csv(P / "05_resource_mass.csv")
scene = pd.read_csv(P / "05_scene_resources.csv")
tm = pd.read_csv(P / "06_direct_time.csv")
methods = [("sac", "SAC", BLUE), ("ppo", "PPO", ORANGE), ("dqn", "DQN", GREEN),
           ("random", "随机", PINK), ("greedy", "贪心", YELLOW), ("fixed", "固定", GRAY)]
labels = [m[1] for m in methods]

fig, axes = plt.subplots(2, 2, figsize=(W, 4.6))
# (a) 功率档位分布
ax = axes[0, 0]
pw = mass[mass.field == "nominal_power_w"]
for key, name, col in methods:
    t = pw[pw.method == key].sort_values("bin")
    ax.plot(t.bin, t["mass"] * 100, marker="o", ms=3, lw=1.2, color=col, label=name)
ax.set_xlabel("每束功率（W）")
ax.set_ylabel("该档位的波束占比（%）")
ax.set_title("(a) 功率档位的使用", fontsize=9.5)
ax.set_xlim(3, 63)
ax.set_xticks(range(10, 55, 10))
ax.legend(frameon=False, fontsize=8, ncol=1, loc="upper right", handlelength=1.4, labelspacing=0.3)
# (b) 总功率
def boxes(ax, df, col, ylabel, title, scale=1.0):
    data = [df[df.method == k][col].to_numpy() * scale for k, _, _ in methods]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showmeans=True,
                    meanprops=dict(marker="D", mfc="white", mec="black", ms=4.5),
                    medianprops=dict(color="black", lw=1.0), flierprops=dict(ms=2.5, mfc="none"))
    for patch, (_, _, c) in zip(bp["boxes"], methods):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.5)
    ax.grid(axis="x", visible=False)
boxes(axes[0, 1], scene, "power_w", "每景总功率（W）", "(b) 每景总功率（上限3000 W）")
boxes(axes[1, 0], scene, "slots", "每景总槽数", "(c) 每景分配的频率槽数")
boxes(axes[1, 1], tm, "direct_mother_mean_seconds", "每景耗时（秒）", "(d) 每景直接输出耗时")
fig.text(0.5, -0.01, "(b)(c)(d) 每个方框是40个考核场景的分布，菱形是平均", ha="center", fontsize=8.5)
fig.tight_layout(h_pad=1.0, w_pad=1.0)
save_png(fig, "W22_资源使用与耗时.png")
