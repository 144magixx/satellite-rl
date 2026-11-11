"""W22 图3：SAC 相对其余五种方法的需求满足率差和供需匹配程度差（同一场景配对，单位百分点）。
数据：E:\\satellite_rl_archives\\s2_t0_fair24_20261013\\publication_t24_v4\\04_paired_CI.csv
粗线是按日期分层重采样的99.5%区间，细线是按时间块重采样的99.5%区间，0是没有差别。
"""
from _w22_24_common import *
import numpy as np
import pandas as pd

c = pd.read_csv(ARCHIVE_T24 / "04_paired_CI.csv")
names = {"ppo": "PPO", "dqn": "DQN", "random": "随机", "greedy": "贪心", "fixed": "固定"}
order = ["ppo", "dqn", "random", "greedy", "fixed"]

fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), sharey=True)
for ax, metric, title in zip(axes, ["SI", "SGM"], ["需求满足率差（百分点）", "供需匹配程度差（百分点）"]):
    for i, k in enumerate(order):
        m = c[(c.right == k) & (c.metric == metric) & (c.interval == "main")].iloc[0]
        b = c[(c.right == k) & (c.metric == metric) & (c.interval == "block")].iloc[0]
        ax.plot([b.low_pp, b.high_pp], [i, i], color=BLUE, lw=1.4, alpha=0.45, solid_capstyle="butt")
        ax.plot([m.low_pp, m.high_pp], [i, i], color=BLUE, lw=4.0, solid_capstyle="butt")
        ax.plot(m.mean_delta_pp, i, marker="o", ms=5.5, mfc="white", mec="black", mew=1.2, zorder=5)
        ax.text(m.mean_delta_pp, i + 0.24, f"{m.mean_delta_pp:+.1f}", ha="center", va="top", fontsize=8.5,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6), zorder=6)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_title(title, fontsize=9.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"SAC − {names[k]}" for k in order])
    ax.grid(axis="y", visible=False)
axes[0].set_ylim(len(order) - 0.35, -0.5)
axes[0].set_xlim(-6, 10)
axes[1].set_xlim(0, 50)
fig.text(0.5, -0.02, "粗线：按日期分层的99.5%区间；细线：按时间块的99.5%区间；空心点是平均差；0表示没有差别",
         ha="center", fontsize=8.5)
fig.tight_layout(w_pad=1.0)
save_png(fig, "W22_SAC配对差.png")

