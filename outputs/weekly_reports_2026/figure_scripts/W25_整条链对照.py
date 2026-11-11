# -*- coding: utf-8 -*-
"""W25 图：三个阶段连起来的整条链方案对比（确认88景、每机43.2 Mbps）。

数据来源（只读）：
  D:/paper/outputs/experiments/litcmp_20261018/analysis/stage23_table.csv 里 tier=d432 的行
  E1_320   蜂窝布局 + 模拟退火320次 + 注水功率
  E1_3200  同上，退火加到3200次
  G1@A2    聚类布局 + 贪心 + SCA功率细调
  G1       精确指派(FLOW-AO)布局 + 贪心 + SCA功率细调
  S2       精确指派 + 贪心+禁忌搜索320次 + SCA功率细调
  L1       精确指派 + 改进式SAC（贪心起点+局部改进，学习方法） + SCA功率细调
72.6%对应搜索320次，67.5%对应3200次，两个口径分开画。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *

df = pd.read_csv(os.path.join(EXP, "litcmp_20261018", "analysis", "stage23_table.csv"))
df = df[df.tier == "d432"].set_index("arm")
arms = ["E1_320", "E1_3200", "G1@A2", "G1", "S2", "L1"]
labels = [
    "传统三段式\n蜂窝+退火320次+注水",
    "传统三段式\n退火加到3200次",
    "聚类布局+贪心\n+功率细调",
    "精确指派+贪心\n+功率细调",
    "精确指派+搜索320次\n+功率细调",
    "精确指派+改进式SAC\n+功率细调",
]
colors = [C_RULE, "#aab2c0", C_BLUE, "#4a90a4", "#5bb3a5", C_LEARN]
metrics = [("W", "需求满足率（%）", 100, "{:.1f}", (0, 128), [0, 50, 100]),
           ("w10", "最差10%飞机（%）", 100, "{:.1f}", (0, 128), [0, 50, 100]),
           ("SGM", "供需匹配程度", 1, "{:.3f}", (0, 1.36), [0, 0.5, 1.0])]

fig, axes = plt.subplots(1, 3, figsize=(5.8, 3.85), sharey=True,
                         gridspec_kw={"wspace": 0.08, "left": 0.30, "right": 0.985, "top": 0.88, "bottom": 0.12})
y = np.arange(len(arms))[::-1]
for ax, (col, title, k, fmt, xl, ticks) in zip(axes, metrics):
    vals = [df.loc[a, col] * k for a in arms]
    ax.barh(y, vals, color=colors, height=0.66)
    for yy, v in zip(y, vals):
        ax.text(v + xl[1] * 0.02, yy, fmt.format(v), va="center", fontsize=8.5)
    ax.set_xlim(*xl)
    ax.set_title(title, fontsize=8.8)
    ax.xaxis.grid(True, color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks(ticks)
axes[0].set_yticks(y)
axes[0].set_yticklabels(labels, fontsize=8.3)
for ax in axes[1:]:
    ax.tick_params(axis="y", length=0)
save_png(fig, "W25_整条链对照.png", dpi=220)
