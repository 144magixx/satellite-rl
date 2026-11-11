# -*- coding: utf-8 -*-
"""W25 图：第二阶段新口径第一轮，各方法在287个确认场景上的三项指标（直接输出，没接功率细调）。

数据来源（只读）：D:/paper/outputs/experiments/s2lab_20261024/eval/confirm/
  rule-greedy.csv / rule-fixed.csv / rule-random.csv / meta-ga320.csv   规则与搜索（每景一行）
  extra-legacydelta-s{1,2,3}.csv  组由贪心定+新奖励的T-SAC
  main-tsaclegacy-s*.csv          组由贪心定的T-SAC（原论文奖励）
  reward-delta-s*.csv             自己选组+新奖励的T-SAC
  main-tsac-s*.csv / main-ppo-s* / main-dqn-s* / main-cnnsac-s* / main-mlpsac-s*
每个运行先对287景取平均，学习方法再对3个种子取平均，误差线是种子间标准差。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *

R = os.path.join(EXP, "s2lab_20261024", "eval", "confirm")


def seeds(n):
    return [os.path.join(R, f"{n}-s{i}.csv") for i in (1, 2, 3)]


def one(n):
    return [os.path.join(R, f"{n}.csv")]


methods = [  # 名称, 路径, 颜色
    ("容量感知贪心", one("rule-greedy"), C_RULE),
    ("遗传算法320次（从贪心出发）", one("meta-ga320"), C_RULE),
    ("T-SAC（组由贪心定，新奖励）", seeds("extra-legacydelta"), C_KEY),
    ("T-SAC（组由贪心定）", seeds("main-tsaclegacy"), C_LEARN),
    ("T-SAC（自己选组，新奖励）", seeds("reward-delta"), C_LEARN),
    ("T-SAC（自己选组）", seeds("main-tsac"), C_LEARN),
    ("PPO", seeds("main-ppo"), C_BLUE),
    ("DQN", seeds("main-dqn"), C_BLUE),
    ("CNN-SAC", seeds("main-cnnsac"), C_PURPLE),
    ("MLP-SAC", seeds("main-mlpsac"), C_PURPLE),
    ("固定分配", one("rule-fixed"), "#b9bfcb"),
    ("随机分配", one("rule-random"), "#b9bfcb"),
]
res = []
for name, ps, c in methods:
    m, s, _ = seed_mean(ps)
    res.append((name, m, s, c, len(ps) > 1))
    print(f"{name}: SGM {m['SGM']:.3f} W {m['W']:.3f} 最差10% {m['worst10']:.3f}")

y = np.arange(len(res))[::-1]
fig = plt.figure(figsize=(5.8, 4.0))
gs = fig.add_gridspec(1, 3, wspace=0.07, left=0.355, right=0.985, top=0.93, bottom=0.075)
axes = [fig.add_subplot(gs[0])]
axes += [fig.add_subplot(gs[1], sharey=axes[0]), fig.add_subplot(gs[2], sharey=axes[0])]
for ax, (col, title) in zip(axes, [("SGM", "供需匹配程度"), ("W", "需求满足率"), ("worst10", "最差10%飞机")]):
    vals = [r[1][col] for r in res]
    errs = [r[2][col] if r[4] else 0 for r in res]
    ax.barh(y, vals, xerr=errs, color=[r[3] for r in res], height=0.7, error_kw=dict(lw=0.8, capsize=1.5))
    for yy, v, e in zip(y, vals, errs):
        ax.text(v + e + 0.02, yy, f"{v:.2f}", va="center", fontsize=8.3,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9), zorder=5)
    ax.set_xlim(0, 1.2)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_title(title, fontsize=9)
    ax.xaxis.grid(True, color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    # 贪心参考线
    ax.axvline(res[0][1][col], color="#555555", lw=0.8, ls="--", zorder=0)
axes[0].set_yticks(y)
axes[0].set_yticklabels([r[0] for r in res], fontsize=8.3)
for ax in axes[1:]:
    ax.tick_params(axis="y", length=0, labelleft=False)
save_png(fig, "W25_第一轮主结果.png", dpi=220)
