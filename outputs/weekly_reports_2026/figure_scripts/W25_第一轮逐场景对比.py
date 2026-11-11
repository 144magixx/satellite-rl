# -*- coding: utf-8 -*-
"""W25 图：第二阶段新口径第一轮，学习方法和贪心逐个场景比（287个确认场景）。

数据来源（只读）：D:/paper/outputs/experiments/s2lab_20261024/eval/confirm/
  左图：extra-legacydelta-s{1,2,3}.csv（组由贪心定+新奖励的T-SAC，3个种子逐景平均）减 rule-greedy.csv，
        横轴是该场景的负载比（总需求/110 Gbps），点的颜色是负载档（stratum列：ample宽裕/marginal勉强/short不够）。
  右图：几种方法分档的平均差（供需匹配程度，方法减贪心）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *

R = os.path.join(EXP, "s2lab_20261024", "eval", "confirm")
g = read_eval(os.path.join(R, "rule-greedy.csv")).set_index("key")


def seeds(n):
    return [os.path.join(R, f"{n}-s{i}.csv") for i in (1, 2, 3)]


best = scene_mean(seeds("extra-legacydelta"), "SGM")
diff = (best - g["SGM"]).reindex(g.index)
print("比贪心高的场景比例", float((diff > 0).mean()), "平均差", float(diff.mean()))

tiers = [("ample", "宽裕档", C_LEARN), ("marginal", "勉强档", C_GOLD), ("short", "不够档", C_SEARCH)]
fig, axes = plt.subplots(1, 2, figsize=(5.8, 3.4), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.3,
                                                                "left": 0.115, "right": 0.985, "top": 0.88, "bottom": 0.28})
ax = axes[0]
for key, name, c in tiers:
    m = g.stratum == key
    ax.scatter(g.load_ratio[m], diff[m], s=7, color=c, alpha=0.7, label=name, lw=0)
ax.axhline(0, color="#333333", lw=0.9)
ax.set_xlabel("场景负载比（总需求/110 Gbps）")
ax.set_ylabel("供需匹配程度比贪心高多少")
ax.set_title("T-SAC（组由贪心定，新奖励）\n逐场景减贪心", fontsize=8.8)
ax.grid(True, color="#e6e6e6", lw=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8.3, loc="lower right", handletextpad=0.2, markerscale=1.8)

ax = axes[1]
cmp = [("T-SAC（组由贪心定，新奖励）", seeds("extra-legacydelta"), C_KEY),
       ("T-SAC（自己选组）", seeds("main-tsac"), C_LEARN),
       ("PPO", seeds("main-ppo"), C_BLUE),
       ("遗传算法320次", [os.path.join(R, "meta-ga320.csv")], C_RULE)]
x = np.arange(3)
w = 0.2
for i, (name, ps, c) in enumerate(cmp):
    s = scene_mean(ps, "SGM").reindex(g.index) - g["SGM"]
    vals = [float(s[g.stratum == k].mean()) for k, _, _ in tiers]
    ax.bar(x + (i - 1.5) * w, vals, width=w, color=c, label=name)
ax.axhline(0, color="#333333", lw=0.9)
ax.set_xticks(x)
ax.set_xticklabels([t[1] for t in tiers])
ax.set_title("分档平均差", fontsize=8.8)
ax.grid(True, axis="y", color="#e6e6e6", lw=0.6)
ax.set_axisbelow(True)
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.3, bbox_to_anchor=(0.62, -0.005), columnspacing=1.0, handlelength=1.4)
save_png(fig, "W25_第一轮逐场景对比.png", dpi=220)
