# -*- coding: utf-8 -*-
"""W26 图：第三阶段九种功率方法，在四种第二阶段方案上把需求满足率提高了多少（供需匹配程度不降）。

数据来源（只读）：D:/paper/outputs/experiments/s3lab_20261029/runs/main/rows.jsonl
  每行一份第二阶段方案×一种功率方法×一个求解器种子；这里取“保SGM接入”后的 a_W 减细调前 W0，
  在全部输入上取平均（旧确认集287景、新确认集300景；随机和块指针头各3套方案，两段式3个求解器种子一并平均）。
  最上面一行是细调前的需求满足率（method=none 的 W0 平均）。
  左侧小字：各方法单景完整求解耗时（t_native）的中位数，在8种（确认集×方案）组合里取最小到最大。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *
from _s3 import load, METHODS, TIERS

df = load("main")
df0 = df[df.method == "none"]
fig, axes = plt.subplots(1, 2, figsize=(5.8, 4.2), sharey=True)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("wt", ["#ffffff", "#2a9d8f"])
tm = df.groupby(["method", "set", "tier"]).t_native.median().unstack(["set", "tier"])  # 单景完整求解耗时


def tlabel(m):
    lo, hi = float(tm.loc[m].min()), float(tm.loc[m].max())
    if hi < 0.01:
        return "<0.01 秒"
    return f"{lo:.2f}–{hi:.2f} 秒" if lo >= 0.005 else f"<0.01–{hi:.2f} 秒"


rows_lbl = ["细调前满足率"] + [f"{m[1]}\n{tlabel(m[0])}" for m in METHODS]
for ax, st, title in zip(axes, ("old", "new"), ("旧确认集（287景）", "新确认集（300景）")):
    mat = np.zeros((len(METHODS), len(TIERS)))
    base = []
    for j, (tier, _) in enumerate(TIERS):
        base.append(float(df0[(df0.set == st) & (df0.tier == tier)].W0.mean()))
        for i, (m, _) in enumerate(METHODS):
            sub = df[(df.set == st) & (df.tier == tier) & (df.method == m)]
            mat[i, j] = float(sub.dW.mean())
    for j in range(len(TIERS)):
        col = mat[:, j]
        norm = col / col.max()
        for i in range(len(METHODS)):
            ax.add_patch(plt.Rectangle((j - 0.5, i + 1 - 0.5), 1, 1, facecolor=cmap(0.08 + 0.85 * norm[i]), edgecolor="white", lw=1.2))
            best = abs(col[i] - col.max()) < 5e-4
            ax.text(j, i + 1, f"+{col[i]:.3f}", ha="center", va="center", fontsize=8.3, fontweight="bold" if best else "normal")
        ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, 1, facecolor="#eceff3", edgecolor="white", lw=1.2))
        ax.text(j, 0, f"{base[j]:.3f}", ha="center", va="center", fontsize=8.3, color="#333333")
    ax.set_xlim(-0.5, len(TIERS) - 0.5)
    ax.set_ylim(len(METHODS) + 0.5, -0.5)
    ax.set_xticks(range(len(TIERS)))
    ax.set_xticklabels([t[1] for t in TIERS], fontsize=8.3)
    ax.xaxis.tick_top()
    ax.set_title(title, fontsize=9, pad=34)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    print(st, "SCA", np.round(mat[0], 3), "细调前", np.round(base, 3))
axes[0].set_yticks(range(len(rows_lbl)))
axes[0].set_yticklabels(rows_lbl, fontsize=8.3, linespacing=1.0)
fig.text(0.5, 0.012, "方格：需求满足率比细调前提高多少（加粗为每列最高）；左侧小字是单景中位耗时范围", ha="center", fontsize=8.3)
fig.subplots_adjust(left=0.235, right=0.995, top=0.815, bottom=0.07, wspace=0.05)
save_png(fig, "W26_第三阶段九种功率方法.png", dpi=220)
