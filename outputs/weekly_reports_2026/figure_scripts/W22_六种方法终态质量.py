"""W22 图2：六种方法在40个考核场景上的终态需求满足率和供需匹配程度。
数据：E:\\satellite_rl_archives\\s2_t0_fair24_20261013\\publication_t24_v4\\03_quality.csv
（每个点是一个场景，SAC/PPO/DQN/随机的3次动作先取平均；考核场景是2月21日、22日各20个）。
"""
from _w22_24_common import *
import numpy as np
import pandas as pd

q = pd.read_csv(ARCHIVE_T24 / "03_quality.csv")
methods = [("sac", "SAC", BLUE), ("ppo", "PPO", ORANGE), ("dqn", "DQN", GREEN),
           ("random", "随机", PINK), ("greedy", "贪心", YELLOW), ("fixed", "固定", GRAY)]
rng = np.random.default_rng(1)

fig, axes = plt.subplots(1, 2, figsize=(W, 3.0), sharey=True)
for ax, metric, title, xlim in zip(axes, ["SI", "SGM"], ["需求满足率（%）", "供需匹配程度（%）"], [(70, 102), (0, 90)]):
    for i, (key, name, col) in enumerate(methods):
        v = q[(q.method == key) & (q.metric == metric)].value.to_numpy() * 100
        y = i + rng.uniform(-0.22, 0.22, len(v))
        ax.scatter(v, y, s=11, color=col, alpha=0.6, linewidths=0)
        ax.plot(v.mean(), i, marker="D", ms=6.5, mfc="white", mec="black", mew=1.4, zorder=5)
        ax.text(v.mean(), i - 0.36, f"{v.mean():.1f}%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.6), zorder=6)
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m[1] for m in methods])
    ax.grid(axis="y", visible=False)
axes[0].set_ylim(len(methods) - 0.4, -1.0)
fig.supxlabel("每个小圆点是一个考核场景（共40个），空心菱形是40个场景的平均", fontsize=9)
fig.tight_layout(w_pad=1.0)
save_png(fig, "W22_六种方法终态质量.png")

