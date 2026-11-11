# -*- coding: utf-8 -*-
"""W26 图：第3轮、第4轮几组主要方法在60个开发场景上的供需匹配程度随训练步数的变化。

数据来源（只读）：D:/paper/outputs/experiments/
  左图 s2lab_round3_20261027/account_{A,B}/runs3/r3-<组>-s{1,2,3}/dev.jsonl
       r0 基线、f-all 补七项状态、xcombo 补七项状态+大回放池、x-noglobal 去掉全局信息、p2 连续功率、dqn
  右图 s2lab_round4_20261028/account_B/runs4/r4-<组>-s{1,2,3}/dev.jsonl
       base 第3轮最好组重跑、ptr 频谱块指针T-SAC、AB 先定槽数+低干扰前8位、A 先定槽数、g1 折扣因子1.0
每1万步评一次（前几轮都是20万步）；线是3个种子的平均，阴影是种子间最小到最大。
贪心参考线：s2lab_20261024/eval/dev/rule-greedy.csv 对60景取平均。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *

g = read_eval(os.path.join(EXP, R1, "eval", "dev", "rule-greedy.csv"))["SGM"].mean()
left = [
    ("补七项状态+大回放池", R3, "r3-xcombo", C_KEY),
    ("补七项状态", R3, "r3-f-all", C_LEARN),
    ("基线（第2轮最好组）", R3, "r3-r0", C_BLUE),
    ("连续功率", R3, "r3-p2", C_GOLD),
    ("去掉全局信息", R3, "r3-x-noglobal", C_PURPLE),
    ("DQN", R3, "r3-dqn", "#8b8b8b"),
]
right = [
    ("频谱块指针T-SAC", R4, "r4-ptr", C_KEY),
    ("先定槽数+低干扰前8位", R4, "r4-AB", C_LEARN),
    ("先定槽数", R4, "r4-A", "#7fc8bd"),
    ("补七项状态+大回放池（上一轮最好）", R4, "r4-base", C_BLUE),
    ("折扣因子1.0", R4, "r4-g1", C_GOLD),
]
fig, axes = plt.subplots(1, 2, figsize=(5.8, 3.7), sharey=True)
for ax, items, title in zip(axes, (left, right), ("第3轮", "第4轮")):
    for name, rnd, run, c in items:
        ps = [find_run(rnd, f"{run}-s{i}", "dev.jsonl") for i in (1, 2, 3)]
        st, mu, lo, hi = dev_curve(ps, "SGM")
        ax.plot(st / 1e4, mu, color=c, lw=1.5, label=name)
        ax.fill_between(st / 1e4, lo, hi, color=c, alpha=0.13, lw=0)
    ax.axhline(g, color="#333333", lw=1.0, ls="--", label="容量感知贪心")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("训练步数（万步）")
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8.3, handlelength=1.3, ncol=1)
axes[0].set_ylabel("供需匹配程度（60个开发场景）")
axes[0].set_ylim(0.35, 0.95)
fig.subplots_adjust(left=0.115, right=0.985, top=0.93, bottom=0.40, wspace=0.08)
save_png(fig, "W26_第3轮第4轮训练曲线.png", dpi=220)
