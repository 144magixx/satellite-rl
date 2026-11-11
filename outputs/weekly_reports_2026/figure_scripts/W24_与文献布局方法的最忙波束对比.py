"""W24 图：100 个波束时，FLOW-AO 与三种文献布局方法的最忙波束飞机数（从原始记录重画）。
数据：D:/paper/outputs/experiments/litcmp_*/stage1/static.csv（文献方法：蜂窝网格、聚类、Voronoi 均衡）
      D:/paper/outputs/experiments/s1_fix_*/A/static_results.csv（FLOW-AO，加/不加间隔规则）
场景：24 个开发场景中，飞机数>=1200 的 12 个白天场景、其余 12 个夜间场景，波束预算 100。
倍数＝文献方法÷FLOW-AO（不加间隔规则），逐场景相除后再平均。
"""
from _w24_common import *

rows, wide = stage1_peaks()
d = wide[(wide.scene_set == "DEV24") & (wide.N_budget == 100)]

order = [("FLOW-AO-no-spacing", "FLOW-AO（不加间隔规则）", C_FLOW),
         ("FLOW-AO", "FLOW-AO（加间隔规则）", C_FLOW2),
         ("A3", "Voronoi均衡", C_VOR),
         ("A2", "聚类", C_CLU),
         ("A1", "蜂窝网格", C_HEX)]

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True, gridspec_kw={"wspace": 0.06, "width_ratios": [1.15, 1]})
for ax, per, title in zip(axes, ["day", "night"], ["白天场景", "夜间场景"]):
    s = d[d.period == per]
    lb = s.LB.mean()
    ref = s["FLOW-AO-no-spacing"]
    vals = [s[a].mean() for a, _, _ in order]
    ypos = np.arange(len(order))[::-1]
    ax.barh(ypos, vals, color=[c for _, _, c in order], height=0.66)
    ax.axvline(lb, color="#444444", ls=":", lw=1.3)
    for y, (a, _, _), v in zip(ypos, order, vals):
        txt = f"{v:.1f}"
        if a in ("A1", "A2", "A3"):
            txt += f"（{(s[a] / ref).mean():.1f}倍）"
        ax.text(v + max(vals) * 0.015, y, txt, va="center", ha="left", fontsize=10)
    ax.set_xlim(0, max(vals) * 1.42)
    ax.set_yticks(ypos)
    ax.set_yticklabels([l for _, l, _ in order])
    ax.set_title(f"{title}（{len(s)}景，平均约{round(s.n.mean(), -1):.0f}架）", fontsize=10.5)
    ax.set_xlabel(f"飞机数（架），虚线为理论下界{lb:.1f}", fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
fig.suptitle("100个波束时，最忙波束里的飞机数（越低越均衡）", fontsize=11.5, y=1.03)
save_png(fig, "W24_与文献布局方法的最忙波束对比.png", **SAVE_KW)

# 核对：写出给报告核数用的数字
print(d.groupby("period")[["LB", "FLOW-AO-no-spacing", "FLOW-AO", "A1", "A2", "A3"]].mean().round(2).to_string())
print(d.groupby("period").n.mean().round(0).to_string())
