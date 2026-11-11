# -*- coding: utf-8 -*-
"""W26 图：负载不够的场景里，飞机最多的6个波束（平均约26架）分到多少槽、多少功率、满足率多少。

数据来源（只读）：D:/paper/outputs/experiments/s2lab_round4_20261028/beamdiag/b{1,2,3}_beams.csv
  逐束记录（每个波束一行：飞机数 members、槽数 slots、功率 power、满足率 si）。
  取 stratum=short（不够档）的4个事先选好的场景，每个场景按 members 取最大的6个波束，
  对场景和3个种子取平均。贪心没有种子，只有一份。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *

D = os.path.join(EXP, R4, "beamdiag")
df = pd.concat([pd.read_csv(os.path.join(D, f"b{i}_beams.csv")) for i in (1, 2, 3)])
df = df[df.stratum == "short"].copy()
df["grp"] = df.policy.str.replace(r"-s\d$", "", regex=True)
pols = [("rule-greedy", "容量感知贪心", C_RULE), ("r4-base", "补七项状态\n+大回放池", C_BLUE),
        ("r4-ptr", "频谱块指针\nT-SAC", C_KEY), ("r4-AB", "先定槽数+低干扰\n前8位", C_LEARN)]
res = {}
for p, _, _ in pols:
    out = []
    for (pl, k), g in df[df.grp == p].groupby(["policy", "key"]):
        top = g.sort_values(["members", "beam"], ascending=[False, True]).head(6)
        out.append((top.slots.mean(), top.power.mean(), top.si.mean(), top.members.mean()))
    res[p] = np.mean(out, axis=0)
    print(p, np.round(res[p], 2))

fig, axes = plt.subplots(1, 3, figsize=(5.8, 3.0), gridspec_kw={"wspace": 0.32, "left": 0.07, "right": 0.99, "top": 0.88, "bottom": 0.27})
titles = ["每束槽数（最多10）", "每束功率（W）", "每束满足率"]
fmts = ["{:.1f}", "{:.0f}", "{:.2f}"]
lims = [(0, 12.5), (0, 42), (0, 0.6)]
for k, (ax, t, fm, lm) in enumerate(zip(axes, titles, fmts, lims)):
    vals = [res[p][k] for p, _, _ in pols]
    ax.bar(range(4), vals, color=[c for _, _, c in pols], width=0.7)
    for i, v in enumerate(vals):
        ax.text(i, v + lm[1] * 0.02, fm.format(v), ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(*lm)
    ax.set_title(t, fontsize=9)
    ax.set_xticks([])
    ax.yaxis.grid(True, color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=c, label=n.replace("\n", "")) for _, n, c in pols], loc="lower center", ncol=2, frameon=False,
           fontsize=8.3, bbox_to_anchor=(0.5, -0.005), columnspacing=1.2, handlelength=1.2)
save_png(fig, "W26_大波束分到的资源.png", dpi=220)
