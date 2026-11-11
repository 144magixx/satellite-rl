# -*- coding: utf-8 -*-
"""W26 图：第二阶段新口径四轮实验，每轮最好的一组在287个旧确认场景上的三项指标，对比容量感知贪心。

数据来源（只读）：D:/paper/outputs/experiments/
  第1轮 s2lab_20261024/eval/confirm/extra-legacydelta-s{1,2,3}.csv   T-SAC（组由贪心定，新奖励）
  第2轮 s2lab_round2_20261025 的 t14-in-delta-s{1,2,3}/eval_confirm_best_sgm.csv  本组10块各占一个输入+新奖励
  第3轮 s2lab_round3_20261027 的 r3-xcombo-s{1,2,3}/eval_confirm_best_sgm.csv     补七项状态+大回放池
  第4轮 s2lab_round4_20261028 的 r4-ptr-s{1,2,3}/eval_confirm_best_sgm.csv        频谱块指针T-SAC
  贪心 s2lab_20261024/eval/confirm/rule-greedy.csv
每个运行先对287景取平均，再对3个种子取平均。直接输出，没接第三阶段细调。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *

sets = [
    ("第1轮", [os.path.join(EXP, R1, "eval", "confirm", f"extra-legacydelta-s{i}.csv") for i in (1, 2, 3)]),
    ("第2轮", run_paths(R2, "", "t14-in-delta")),
    ("第3轮", run_paths(R3, "r3-", "xcombo")),
    ("第4轮", run_paths(R4, "r4-", "ptr")),
]
g, _, _ = seed_mean([greedy_confirm()])
res = []
for name, ps in sets:
    m, s, _ = seed_mean(ps)
    res.append(m)
    print(name, {k: round(m[k], 3) for k in ("SGM", "W", "worst10")})
print("贪心", {k: round(g[k], 3) for k in ("SGM", "W", "worst10")})

metrics = [("SGM", "供需匹配程度", C_KEY, "o"), ("W", "需求满足率", C_BLUE, "s"), ("worst10", "最差10%飞机满足率", C_LEARN, "^")]
fig, ax = plt.subplots(figsize=(5.8, 3.5))
x = np.arange(4)
for col, name, c, mk in metrics:
    v = [r[col] for r in res]
    ax.plot(x, v, color=c, marker=mk, ms=6, lw=1.8, label=name)
    ax.axhline(g[col], color=c, ls="--", lw=1.0, alpha=0.8)
    for xi, vi in zip(x, v):
        dy = 0.033 if col != "worst10" else -0.045
        ax.text(xi, vi + dy, f"{vi:.2f}" if col != "SGM" else f"{vi:.3f}", ha="center", va="center", fontsize=8.5, color=c,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.85), zorder=2)
    ax.text(3.3, g[col], f"贪心 {g[col]:.2f}", va="center", ha="left", fontsize=8.3, color=c,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=5)
ax.set_xticks(x)
ax.set_xticklabels(["第1轮\nT-SAC（组由贪心定）\n+新奖励", "第2轮\n本组10块各占\n一个输入", "第3轮\n补七项状态\n+大回放池", "第4轮\n频谱块指针\nT-SAC"], fontsize=8.3)
ax.set_xlim(-0.35, 4.15)
ax.set_ylim(0.3, 1.02)
ax.set_ylabel("287个确认场景的平均")
ax.grid(True, color="#e6e6e6", lw=0.6)
ax.set_axisbelow(True)
ax.legend(loc="lower right", frameon=False, fontsize=8.5, bbox_to_anchor=(1.0, 0.0))
fig.subplots_adjust(left=0.12, right=0.985, top=0.975, bottom=0.22)
save_png(fig, "W26_四轮最好组对比.png", dpi=220)
