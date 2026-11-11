"""W24 图：完整计算时间（初解＋搜索＋核验＋功率细调）与结果质量，两种起点各一条线（从原始记录重画）。
数据（只读）：D:/paper/outputs/experiments/s2_pure_tabu_*/eval/summary.csv
  场景集 conf_d432_n150（确认 88 景，43.2 Mbps），seed 51001，列 total_seconds（初解、搜索、核验、功率细调四段实测之和，
  远端同一软件环境、4 个固定 CPU 核并行；不含第一阶段布局生成和文件读写）、after_W、after_SGM（功率细调后）。
"""
from _w24_common import *

s = pd.read_csv(PURE / "summary.csv")
c = s[(s.set == "conf_d432_n150") & (s.seed == 51001)].set_index("method")
pure = ["Random0", "PTS32", "PTS96", "PTS320"]
comb = ["G1", "GTS32", "GTS96", "GTS320"]
bud = ["0", "32", "96", "320"]

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharex=True, gridspec_kw={"wspace": 0.5})
for ax, (col, yl, scale, fmt) in zip(axes, [("after_W", "需求满足率（%）", 100, "{:.1f}"), ("after_SGM", "供需匹配程度", 1, "{:.3f}")]):
    for ms, cc, lab, dy in [(comb, C_TABU, "贪心＋禁忌搜索", 1), (pure, C_PURE, "纯禁忌搜索（随机起点）", -1)]:
        xs = [c.loc[m, "total_seconds"] for m in ms]
        ys = [c.loc[m, col] * scale for m in ms]
        ax.plot(xs, ys, "-o", color=cc, lw=2, ms=6, label=lab)
        for xx, yy, bb in zip(xs, ys, bud):
            ax.annotate(bb, (xx, yy), textcoords="offset points", xytext=((0, 8) if dy > 0 else (13, -13)), ha="center", fontsize=9, color=cc)
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 5, 10])
    ax.set_xticklabels(["2", "3", "5", "10"])
    ax.minorticks_off()
    ax.set_xlabel("总计算时间（秒，对数轴）")
    ax.set_ylabel(yl)
axes[0].set_ylim(84.8, 95.8)
axes[1].set_ylim(0.922, 0.996)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.12), columnspacing=2)
fig.text(0.5, -0.19, "点旁数字是搜索探测次数上限，0是只有初解。两种起点都包含功率细调的时间。", ha="center", va="top", fontsize=9, color="#444444")
save_png(fig, "W24_质量与总耗时.png", **SAVE_KW)

for m in pure + comb:
    print(m, "总耗时 %.2f 秒 W %.1f SGM %.3f" % (c.loc[m, "total_seconds"], c.loc[m, "after_W"] * 100, c.loc[m, "after_SGM"]))
