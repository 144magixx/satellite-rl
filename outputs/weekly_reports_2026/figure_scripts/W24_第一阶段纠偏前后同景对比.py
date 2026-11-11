"""W24 图：第一阶段新旧方法在历史方法原来的场景上、用不超过它们的波束数的最忙波束飞机数（从原始记录重画）。
数据（只读）：D:/paper/outputs/experiments/s1_fix_*/A/hist_compare.csv   历史方法与 FLOW-AO（不加间隔规则）逐场景结果
                                          A/hist_spacing.csv   同样场景、FLOW-AO 加间隔规则
上面一幅：20 个场景（飞机数平均约 1089，波束数平均约 92）。历史方法：几何排序＋修补（列 G_peak）、
         学习辅助（用图网络打分，列 A_peak）。
下面一幅：12 个场景（飞机数平均约 393，波束数平均约 79）。历史方法：强顺序结构重构（列 SH-C_peak）。
"""
from _w24_common import *

h = pd.read_csv(S1FIX / "A" / "hist_compare.csv")
sp = pd.read_csv(S1FIX / "A" / "hist_spacing.csv")
h20, h12 = h[h.scene_set == "HIST20"], h[h.scene_set == "HIST12"]
s20, s12 = sp[sp.scene_set == "HIST20"], sp[sp.scene_set == "HIST12"]

panels = [
    (f"{len(h20)}个场景（飞机平均约{h20.n.mean():.0f}架，波束约{h20.G_N.mean():.0f}个）",
     [("历史方法：几何排序＋修补", h20.G_peak.mean(), C_HEX),
      ("历史方法：加学习打分辅助", h20.A_peak.mean(), C_HEX),
      ("FLOW-AO（加间隔规则）", s20.peak.mean(), C_FLOW2),
      ("FLOW-AO（不加间隔规则）", h20.FLOW_actual_peak.mean(), C_FLOW)]),
    (f"{len(h12)}个场景（飞机平均约{h12.n.mean():.0f}架，波束约{h12['SH-C_N'].mean():.0f}个）",
     [("历史方法：强顺序结构重构", h12["SH-C_peak"].mean(), C_HEX),
      ("FLOW-AO（加间隔规则）", s12.peak.mean(), C_FLOW2),
      ("FLOW-AO（不加间隔规则）", h12.FLOW_actual_peak.mean(), C_FLOW)]),
]
fig, axes = plt.subplots(2, 1, figsize=FIGSIZE, gridspec_kw={"height_ratios": [4, 3], "hspace": 0.62})
for ax, (title, items) in zip(axes, panels):
    ypos = np.arange(len(items))[::-1]
    vals = [v for _, v, _ in items]
    ax.barh(ypos, vals, color=[c for _, _, c in items], height=0.66)
    for y, v in zip(ypos, vals):
        ax.text(v + max(vals) * 0.012, y, f"{v + 1e-9:.1f}", va="center", fontsize=10)
    ax.set_yticks(ypos)
    ax.set_yticklabels([n for n, _, _ in items], fontsize=10)
    ax.set_xlim(0, max(vals) * 1.12)
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
axes[1].set_xlabel("最忙波束里的飞机数（架），每个场景里新旧方法用的波束数相同")
save_png(fig, "W24_第一阶段纠偏前后同景对比.png", **SAVE_KW)

print("HIST20: G %.2f A %.2f FLOW-AO加间隔 %.2f 不加 %.2f" % (h20.G_peak.mean(), h20.A_peak.mean(), s20.peak.mean(), h20.FLOW_actual_peak.mean()))
print("HIST12: SH-C %.2f 加间隔 %.2f 不加 %.2f" % (h12["SH-C_peak"].mean(), s12.peak.mean(), h12.FLOW_actual_peak.mean()))
print("CV: HIST20 G %.2f -> FLOW %.2f | HIST12 SH-C %.2f -> FLOW %.2f" % (h20.G_CV.mean(), h20.FLOW_actual_CV.mean(), h12["SH-C_CV"].mean(), h12.FLOW_actual_CV.mean()))
