# -*- coding: utf-8 -*-
"""W26 图：第二阶段新口径第2轮，各学习方法在60个开发场景上的得分随训练步数的变化。

数据来源（只读）：D:/paper/outputs/experiments/s2lab_round2_20261025/account_{A,B}/runs2/<运行名>/dev.jsonl
  每1万步在开发集评一次；线是3个种子的平均，阴影是种子间最小到最大。
  贪心参考线：s2lab_20261024/eval/dev/rule-greedy.csv 对60景取平均。
运行名：t14-in-delta（本组10块各占一个输入+新奖励）、extra-legacydelta（沿用第1轮输入+新奖励）、
        reward-delta（自己选组+新奖励）、t14-dbw-paper（本组10块各占一个输入+原论文奖励）、
        main-ppo、main-dqn、main-cnnsac
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *

items = [
    ("T-SAC 本组10块各占一个输入，新奖励", "t14-in-delta", C_KEY),
    ("T-SAC 沿用第1轮输入，新奖励", "extra-legacydelta", C_LEARN),
    ("T-SAC 自己选组，新奖励", "reward-delta", "#7fc8bd"),
    ("T-SAC 本组10块各占一个输入，原奖励", "t14-dbw-paper", C_GOLD),
    ("PPO", "main-ppo", C_BLUE),
    ("DQN", "main-dqn", C_PURPLE),
    ("CNN-SAC", "main-cnnsac", "#8b8b8b"),
]
g = read_eval(os.path.join(EXP, R1, "eval", "dev", "rule-greedy.csv"))
fig, axes = plt.subplots(1, 2, figsize=(5.8, 3.65), sharex=True)
for ax, field, title in zip(axes, ("SGM", "W"), ("供需匹配程度", "需求满足率")):
    for name, run, c in items:
        ps = [find_run(R2, f"{run}-s{i}", "dev.jsonl") for i in (1, 2, 3)]
        st, mu, lo, hi = dev_curve(ps, field)
        ax.plot(st / 1e4, mu, color=c, lw=1.5, label=name)
        ax.fill_between(st / 1e4, lo, hi, color=c, alpha=0.13, lw=0)
    ax.axhline(g[field].mean(), color="#333333", lw=1.0, ls="--", label="容量感知贪心")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("训练步数（万步）")
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel("60个开发场景的平均")
axes[0].set_ylim(0.25, 0.92)
axes[1].set_ylim(0.45, 0.88)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.3, bbox_to_anchor=(0.5, -0.005), handlelength=1.4,
           columnspacing=0.8)
fig.subplots_adjust(left=0.105, right=0.985, top=0.93, bottom=0.40, wspace=0.2)
save_png(fig, "W26_第2轮训练曲线.png", dpi=220)
