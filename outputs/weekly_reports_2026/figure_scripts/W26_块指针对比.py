# -*- coding: utf-8 -*-
"""W26 图1（文件名沿用，月报M10也引用它）：频谱块指针T-SAC和几种规则方法在287个旧确认场景上的三项指标。

数据来源（只读）：D:/paper/outputs/experiments/
  频谱块指针T-SAC  s2lab_round4_20261028/account_B/runs4/r4-ptr-s{1,2,3}/eval_confirm_best_sgm.csv（3个种子平均）
  容量感知贪心、分散摆放规则、固定分配、随机分配  s2lab_round2_20261025/account_A/eval2/confirm/ 或 s2lab_20261024/eval/confirm/ 的 rule-*.csv
  遗传算法320次（实际评价321次）、模拟退火320次  s2lab_20261024/eval/confirm/meta-ga320.csv、meta-sa320.csv
和原来的图数据相同，只是把“SGM”“W”改成中文名称重画。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *

R1d = os.path.join(EXP, R1, "eval", "confirm")
methods = [  # 名称, 路径列表, 颜色
    ("频谱块指针T-SAC", run_paths(R4, "r4-", "ptr"), "#1f6fb2"),
    ("容量感知贪心", [os.path.join(R1d, "rule-greedy.csv")], "#d9531e"),
    ("遗传算法（321次评价）", [os.path.join(R1d, "meta-ga320.csv")], "#6aa84f"),
    ("模拟退火（320次评价）", [os.path.join(R1d, "meta-sa320.csv")], "#7b3f98"),
    ("分散摆放规则", [os.path.join(EXP, R2, "account_A", "eval2", "confirm", "rule-dispersed.csv")], "#e6a817"),
    ("固定分配", [os.path.join(R1d, "rule-fixed.csv")], "#4fb8e8"),
    ("随机分配", [os.path.join(R1d, "rule-random.csv")], "#555555"),
]
cols = [("SGM", "供需匹配程度"), ("W", "需求满足率"), ("worst10", "最差10%飞机满足率")]
res = [seed_mean(ps)[0] for _, ps, _ in methods]
for (n, _, _), m in zip(methods, res):
    print(n, {k: round(m[k], 3) for k, _ in cols})

fig, ax = plt.subplots(figsize=(5.8, 3.3))
nb = len(methods)
bw = 0.115
for j, (name, _, c) in enumerate(methods):
    xs = np.arange(3) + (j - (nb - 1) / 2) * bw
    vals = [res[j][k] for k, _ in cols]
    ax.bar(xs, vals, width=bw * 0.92, color=c, label=name)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.012, f"{v:.2f}", rotation=90, ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(3))
ax.set_xticklabels([n for _, n in cols], fontsize=8.8)
ax.set_ylim(0, 1.32)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylabel("287个确认场景的平均")
ax.yaxis.grid(True, color="#e6e6e6", lw=0.6)
ax.set_axisbelow(True)
ax.legend(ncol=4, frameon=False, fontsize=8.3, loc="upper center", bbox_to_anchor=(0.5, 1.03), columnspacing=0.9,
          handlelength=1.1, handletextpad=0.4)
fig.subplots_adjust(left=0.105, right=0.99, top=0.98, bottom=0.085)
save_png(fig, "W26_块指针对比.png", dpi=220)
