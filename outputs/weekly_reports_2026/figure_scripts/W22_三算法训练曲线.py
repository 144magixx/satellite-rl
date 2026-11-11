"""W22 图1：SAC、PPO、DQN 三种算法从头训练15万步的训练曲线。
数据：E:\\satellite_rl_archives\\s2_t0_fair24_20261013\\publication_t24_v4\\01_learning.csv
（每个算法1500个训练回合的回报、终态需求满足率、终态供需匹配程度，以及18个验证节点，
验证节点是20个固定场景×3次动作的平均）。
"""
from _w22_24_common import *
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

d = pd.read_csv(ARCHIVE_T24 / "01_learning.csv")
d["is_train"] = d["phase"].eq("train")
algos = [("sac", "SAC", BLUE), ("ppo", "PPO", ORANGE), ("dqn", "DQN", GREEN)]
rows = [("return", None, "每回合累计回报", 1),
        ("final_si_mean", "final_si_mean", "终态需求满足率（%）", 100),
        ("final_sgm_mean", "final_sgm_mean", "终态供需匹配程度（%）", 100)]

fig, axes = plt.subplots(3, 1, figsize=(W, 4.9), sharex=True)
for ax, (metric, vmetric, ylabel, k) in zip(axes, rows):
    for key, name, col in algos:
        t = d[(d.algorithm == key) & d.is_train & (d.metric == metric)].sort_values("env_step")
        ax.plot(t.env_step / 1e4, moving_average(t.value * k, 50), color=col, lw=1.3)
        if vmetric:
            v = d[(d.algorithm == key) & ~d.is_train & (d.metric == vmetric)].sort_values("env_step")
            ax.plot(v.env_step / 1e4, v.value * k, color=col, lw=0.8, marker="o", ms=3.4, mfc="white", mew=1.0, ls="--")
    ax.set_ylabel(ylabel, fontsize=9)
    if metric == "final_si_mean":
        ax.set_ylim(72, 94)
    if metric == "final_sgm_mean":
        ax.set_ylim(30, 76)
axes[2].set_xlabel("训练步数（万步）")
axes[2].set_xlim(0, 15.2)
handles = [Line2D([], [], color=c, lw=2, label=n) for _, n, c in algos]
handles += [Line2D([], [], color="k", lw=1.5, label="训练回合（50回合滑动平均）"),
            Line2D([], [], color="k", lw=0.9, ls="--", marker="o", ms=3.4, mfc="white", label="固定验证场景")]
fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=8.5,
           bbox_to_anchor=(0.5, 1.03), columnspacing=1.0, handlelength=1.6)
fig.align_ylabels(axes)
fig.tight_layout(h_pad=0.5, rect=(0, 0, 1, 0.97))
save_png(fig, "W22_三算法训练曲线.png")
