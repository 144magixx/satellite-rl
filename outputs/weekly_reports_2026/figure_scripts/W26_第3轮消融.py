# -*- coding: utf-8 -*-
"""W26 图：第3轮每改一样东西，供需匹配程度比基线高多少或低多少（287个旧确认场景，3个种子平均）。

数据来源（只读）：D:/paper/outputs/experiments/s2lab_round3_20261027/account_{A,B}/runs3/r3-<组>-s{1,2,3}/
  eval_confirm_best_sgm.csv（逐景评价）。基线是第2轮最好的一组在新代码里重跑（r3-r0）。
  贪心 s2lab_20261024/eval/confirm/rule-greedy.csv。
没画的两项：补信噪比（r3-f-snr）+0.002、补邻束敏感度（r3-f-sens）+0.004，都小于0.005。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *
from matplotlib.patches import Patch

CAT = {"state": ("补充状态信息", "#2a9d8f"), "drop": ("去掉信息", "#c0392b"), "act": ("改动作方式", "#3a6ea5"),
       "pow": ("改功率动作", "#7b5ea7"), "train": ("训练设置", "#e09f3e"), "algo": ("换成其他算法", "#8c96a8")}
groups = [  # 标签, 运行名, 类别
    ("补七项状态（全部）", "f-all", "state"),
    ("补七项状态+大回放池", "xcombo", "state"),
    ("补全局频谱", "f-gspec", "state"),
    ("补本束所需功率", "f-pneed", "state"),
    ("补功率份额", "f-share", "state"),
    ("补出向干扰", "f-outint", "state"),
    ("补后续邻居", "f-future", "state"),
    ("去掉干扰输入", "x-nointf", "drop"),
    ("去掉全局信息", "x-noglobal", "drop"),
    ("只能选放得下的起点", "legal", "act"),
    ("先定槽数再选起点", "cond", "act"),
    ("先定槽数+只留低干扰前8位", "condtop8", "act"),
    ("评论家功率项看槽数", "p1", "pow"),
    ("连续功率", "p2", "pow"),
    ("连续功率+功率看起点", "p3", "pow"),
    ("回放池加大到10万", "replay", "train"),
    ("自动温度", "auto", "train"),
    ("历史取法的双Q", "hist", "train"),
    ("网络加宽", "wide", "train"),
    ("演员学习率调大", "lr", "train"),
    ("DQN", "dqn", "algo"),
    ("MLP-SAC", "mlpsac", "algo"),
    ("PPO", "ppo", "algo"),
]
base, _, _ = seed_mean(run_paths(R3, "r3-", "r0"))
gm, _, _ = seed_mean([greedy_confirm()])
print("基线", round(base["SGM"], 3), "贪心", round(gm["SGM"], 3))
vals = []
for label, run, cat in groups:
    m, _, _ = seed_mean(run_paths(R3, "r3-", run))
    d = m["SGM"] - base["SGM"]
    vals.append(d)
    print(f"{label}: {m['SGM']:.3f} ({d:+.3f})")

n = len(groups)
y = np.arange(n)[::-1]
fig, ax = plt.subplots(figsize=(5.8, 4.55))
ax.barh(y, vals, color=[CAT[g[2]][1] for g in groups], height=0.72)
bb = dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.9)
for yy, v in zip(y, vals):
    if v >= 0:
        ax.text(v + 0.004, yy, f"{v:+.3f}", va="center", ha="left", fontsize=8.3, bbox=bb, zorder=5)
    else:
        ax.text(v - 0.004, yy, f"{v:+.3f}", va="center", ha="right", fontsize=8.3, bbox=bb, zorder=5)
ax.axvline(0, color="#333333", lw=0.9)
gd = gm["SGM"] - base["SGM"]
ax.axvline(gd, color="#555555", lw=1.0, ls="--")
ax.text(gd - 0.004, n - 0.35, f"容量感知贪心（{gm['SGM']:.3f}）", fontsize=8.3, ha="right", va="bottom", color="#555555")
ax.set_yticks(y)
ax.set_yticklabels([g[0] for g in groups], fontsize=8.3)
ax.set_xlim(-0.31, 0.085)
ax.set_xticks([-0.3, -0.2, -0.1, 0, 0.05])
ax.set_xticklabels(["-0.3", "-0.2", "-0.1", "0", "+0.05"])
ax.set_ylim(-0.7, n + 0.2)
ax.set_xlabel(f"供需匹配程度比基线（{base['SGM']:.3f}）高多少")
ax.xaxis.grid(True, color="#e6e6e6", lw=0.6)
ax.set_axisbelow(True)
fig.legend(handles=[Patch(color=v[1], label=v[0]) for v in CAT.values()], loc="lower center", ncol=3, frameon=False,
           fontsize=8.3, bbox_to_anchor=(0.55, -0.005), columnspacing=1.0, handlelength=1.2)
fig.subplots_adjust(left=0.335, right=0.975, top=0.985, bottom=0.19)
save_png(fig, "W26_第3轮消融.png", dpi=220)
