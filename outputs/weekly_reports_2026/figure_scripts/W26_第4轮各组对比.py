# -*- coding: utf-8 -*-
"""W26 图：第4轮各组在旧确认场景（1、2月，287景）和新确认场景（2024年6、8、10月的300景）上的供需匹配程度。

数据来源（只读）：D:/paper/outputs/experiments/s2lab_round4_20261028/
  旧确认集  account_B/runs4/r4-<组>-s{1,2,3}/eval_confirm_best_sgm.csv
  新确认集  eval_c2/confirm/r4-<组>-s{1,2,3}.csv 和 eval_c2/confirm/rule-greedy.csv
  旧集贪心  s2lab_20261024/eval/confirm/rule-greedy.csv
每个运行先对场景取平均，再对3个种子取平均。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *

groups = [  # 标签, 运行名前缀
    ("频谱块指针T-SAC", "ptr"),
    ("先定槽数+块指针", "ptrA"),
    ("先定槽数+低干扰前8位", "AB"),
    ("先定槽数+评论家起点看槽数", "Aqx"),
    ("先定槽数", "A"),
    ("每步更新2次", "utd2"),
    ("补七项状态+大回放池（上一轮最好）", "base"),
    ("先模仿贪心再训练", "bc"),
    ("回放池加到20万", "rb200k"),
    ("只留三项状态", "f3"),
    ("贪心建议作状态", "hint"),
    ("回放池减到5万", "rb50k"),
    ("折扣因子0.995", "g995"),
    ("折扣因子1.0", "g1"),
]
old, new = [], []
for label, run in groups:
    mo, _, _ = seed_mean([find_run(R4, f"r4-{run}-s{i}", "eval_confirm_best_sgm.csv") for i in (1, 2, 3)])
    mn, _, _ = seed_mean([os.path.join(EXP, R4, "eval_c2", "confirm", f"r4-{run}-s{i}.csv") for i in (1, 2, 3)])
    old.append(mo["SGM"])
    new.append(mn["SGM"])
    print(f"{label}: 旧 {mo['SGM']:.3f} 新 {mn['SGM']:.3f}")
go, _, _ = seed_mean([greedy_confirm()])
gn, _, _ = seed_mean([os.path.join(EXP, R4, "eval_c2", "confirm", "rule-greedy.csv")])
print("贪心 旧", round(go["SGM"], 3), "新", round(gn["SGM"], 3))

n = len(groups)
y = np.arange(n)[::-1]
fig, ax = plt.subplots(figsize=(5.8, 4.2))
for yy, a, b in zip(y, old, new):
    ax.plot([b, a], [yy, yy], color="#c9ced6", lw=2.0, zorder=1)
ax.scatter(old, y, s=36, color=C_BLUE, zorder=3, label="旧确认集（287景）")
ax.scatter(new, y, s=36, color=C_SEARCH, marker="D", zorder=3, label="新确认集（300景）")
for yy, a, b in zip(y, old, new):
    ax.text(a + 0.008, yy, f"{a:.3f}", va="center", ha="left", fontsize=8.3, color=C_BLUE)
    ax.text(b - 0.008, yy, f"{b:.3f}", va="center", ha="right", fontsize=8.3, color="#b65a3f")
ax.axvline(go["SGM"], color=C_BLUE, ls="--", lw=1.0, alpha=0.8)
ax.axvline(gn["SGM"], color=C_SEARCH, ls="--", lw=1.0, alpha=0.9)
ax.text(go["SGM"] + 0.004, n - 0.35, f"旧集贪心 {go['SGM']:.3f}", fontsize=8.3, color=C_BLUE, ha="left", va="bottom")
ax.text(gn["SGM"] - 0.004, n - 0.35, f"新集贪心 {gn['SGM']:.3f}", fontsize=8.3, color="#b65a3f", ha="right", va="bottom")
ax.set_yticks(y)
ax.set_yticklabels([g[0] for g in groups], fontsize=8.3)
ax.set_xlim(0.72, 0.99)
ax.set_ylim(-0.7, n + 0.5)
ax.set_xlabel("供需匹配程度（3个种子平均）")
ax.grid(True, color="#e6e6e6", lw=0.6)
ax.set_axisbelow(True)
ax.legend(loc="lower center", ncol=2, frameon=False, fontsize=8.3, bbox_to_anchor=(0.45, 1.0), handletextpad=0.2)
fig.subplots_adjust(left=0.37, right=0.985, top=0.935, bottom=0.115)
save_png(fig, "W26_第4轮各组对比.png", dpi=220)
