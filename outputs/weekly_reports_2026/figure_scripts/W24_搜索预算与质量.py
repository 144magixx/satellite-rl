"""W24 图：禁忌搜索的探测次数（0、32、96、320）与结果质量，两种起点各一条线（从原始记录重画）。
数据（只读）：D:/paper/outputs/experiments/s2_pure_tabu_*/eval/summary.csv
  场景集 conf_d432_n150（确认 88 景，每机 43.2 Mbps，最多 150 束），主初解种子 51001；全部场景等权平均。
  方法：Random0＝随机合法初解（0 次），PTS32/96/320＝纯禁忌搜索；G1＝容量感知贪心（0 次），GTS32/96/320＝贪心＋禁忌搜索。
  需求满足率取 before_W / after_W（按业务量加权，功率细调前／后），供需匹配程度取 before_SGM / after_SGM。
"""
from _w24_common import *

s = pd.read_csv(PURE / "summary.csv")
c = s[(s.set == "conf_d432_n150") & (s.seed == 51001)].set_index("method")
pure = ["Random0", "PTS32", "PTS96", "PTS320"]
comb = ["G1", "GTS32", "GTS96", "GTS320"]
x = np.arange(4)

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"wspace": 0.5})
for ax, (b, a, yl, scale, fmt) in zip(axes, [("before_W", "after_W", "需求满足率（%）", 100, "{:.1f}"),
                                             ("before_SGM", "after_SGM", "供需匹配程度", 1, "{:.3f}")]):
    for ms, col, lab in [(comb, C_TABU, "贪心＋禁忌搜索"), (pure, C_PURE, "纯禁忌搜索（随机起点）")]:
        yb = [c.loc[m, b] * scale for m in ms]
        ya = [c.loc[m, a] * scale for m in ms]
        ax.plot(x, ya, "-o", color=col, lw=2, ms=6, label=f"{lab}，功率细调后")
        ax.plot(x, yb, "--s", color=col, lw=1.4, ms=5, alpha=0.75, label=f"{lab}，功率细调前")
        ax.text(3.14, ya[-1], fmt.format(ya[-1] + 1e-9), va="center", fontsize=9.5, color=col, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["0\n（只有初解）", "32", "96", "320"])
    ax.set_xlim(-0.2, 3.65)
    ax.set_xlabel("搜索探测次数（上限）")
    ax.set_ylabel(yl)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.24), columnspacing=2)
save_png(fig, "W24_搜索预算与质量.png", **SAVE_KW)

for m in pure + comb:
    print(m, "SCA前 W %.1f SGM %.3f | SCA后 W %.1f SGM %.3f" % (c.loc[m, "before_W"] * 100, c.loc[m, "before_SGM"], c.loc[m, "after_W"] * 100, c.loc[m, "after_SGM"]))
