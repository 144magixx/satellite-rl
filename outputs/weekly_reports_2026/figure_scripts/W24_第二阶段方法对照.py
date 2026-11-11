"""W24 图：确认 88 景、每机 43.2 Mbps 下，第二阶段几种方法的需求满足率和最差 10% 飞机的满足率（按报告表格重画）。
数字来源：
  贪心 / 贪心＋禁忌搜索320次 / 学习改进 / 直出T-SAC：docs/Claude维护/S2-FIX第二阶段需求压力/12_S2-FIX第二阶段需求压力实验报告.md
      第 3.1 节表格“确认88景 满足率、确认 最差10%”（G1 92.5/67.2，S2 94.9/83.9，L1 94.9/84.4，L2 88.3/50.0；
      贪心和贪心＋禁忌搜索的两个数也与 s2_pure_tabu 原始汇总一致）。
  传统三段式（蜂窝布局＋退火320次＋注水功率）：docs/Claude维护/文献对照/11_文献对照实验报告.md 表 6-1，
      43.2 W 列 0.726、43.2 最差10% 列 0.327；它是整条链（布局、频率、功率都按传统做法），其余四个是新骨架布局上功率细调后的第二阶段结果。
  “学习改进”是从贪心解出发、每步从候选修改里选一个的离散改进 SAC，只用 32 次试算；“直出T-SAC”是一次直接给出整套频率和功率的 Transformer SAC。
"""
from _w24_common import *

rows = [
    ("传统三段式\n（蜂窝布局＋退火＋注水）", 72.6, 32.7, "#A67C52"),
    ("直出T-SAC", 88.3, 50.0, C_TSAC),
    ("按需求配槽的贪心", 92.5, 67.2, C_GREEDY),
    ("贪心＋禁忌搜索320次", 94.9, 83.9, C_TABU),
    ("学习改进（32次试算）", 94.9, 84.4, C_LEARN),
]
ypos = np.arange(len(rows))[::-1]
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True, gridspec_kw={"wspace": 0.06})
for ax, (i, xl) in zip(axes, [(1, "需求满足率（%）"), (2, "最差10%飞机的满足率（%）")]):
    for y, r in zip(ypos, rows):
        ax.barh(y, r[i], color=r[3], height=0.66)
        ax.text(r[i] + 1.2, y, f"{r[i]:.1f}", va="center", fontsize=10)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel(xl)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(ypos)
axes[0].set_yticklabels([r[0] for r in rows], fontsize=10)
fig.text(0.5, -0.05, "确认88景，每机43.2 Mbps，场景等权平均。传统三段式是整条链，其余四个是新布局上功率细调后的结果。",
         ha="center", va="top", fontsize=8.6, color="#444444")
save_png(fig, "W24_第二阶段方法对照.png", **SAVE_KW)
