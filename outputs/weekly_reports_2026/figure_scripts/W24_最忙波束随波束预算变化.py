"""W24 图：波束预算从 60 加到 200 时，各布局方法的最忙波束飞机数（从原始记录重画，数据同 W24_与文献布局方法的最忙波束对比.py）。
24 个开发场景，白天（飞机数>=1200，12 景）和夜间（12 景）各一幅，纵轴取对数。
"""
from _w24_common import *

rows, wide = stage1_peaks()
d = wide[wide.scene_set == "DEV24"]
budgets = [60, 100, 150, 200]
series = [("LB", "理论下界", "#444444", ":", None, 1.4),
          ("FLOW-AO-no-spacing", "FLOW-AO（不加间隔规则）", C_FLOW, "-", "o", 2.0),
          ("FLOW-AO", "FLOW-AO（加间隔规则）", C_FLOW2, "--", "s", 1.8),
          ("A3", "Voronoi均衡", C_VOR, "-", "^", 1.6),
          ("A2", "聚类", C_CLU, "-", "v", 1.6),
          ("A1", "蜂窝网格", C_HEX, "-", "D", 1.6)]

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True, gridspec_kw={"wspace": 0.08})
for ax, per, title in zip(axes, ["day", "night"], ["白天场景（12景）", "夜间场景（12景）"]):
    s = d[d.period == per]
    for col, lab, c, ls, mk, lw in series:
        y = [s[s.N_budget == b][col].mean() for b in budgets]
        ax.plot(budgets, y, color=c, ls=ls, marker=mk, ms=5, lw=lw, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(budgets)
    ax.set_xlabel("波束预算（最多用几个波束）")
    ax.set_title(title, fontsize=10.5)
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(False, which="minor")
axes[0].set_ylabel("最忙波束的飞机数（架，对数轴）")
axes[0].set_yticks([3, 5, 10, 20, 50, 100, 200])
axes[0].set_yticklabels(["3", "5", "10", "20", "50", "100", "200"])
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.2), columnspacing=1.6, handlelength=2.2)
save_png(fig, "W24_最忙波束随波束预算变化.png", **SAVE_KW)

print(d.groupby(["period", "N_budget"])[["LB", "FLOW-AO-no-spacing", "FLOW-AO", "A1", "A2", "A3"]].mean().round(1).to_string())
