"""W24 图：单机请求从 21.6 加到 86.4 Mbps（以及主档加 200 束预算）时，四种做法的结果（从原始记录重画）。
数据（只读）：D:/paper/outputs/experiments/s2_pure_tabu_*/eval/summary.csv
  场景集（开发 20 景，seed 51001，功率细调后的 after_W、after_SGM）：
    dev_d216_n150 ＝ 21.6 Mbps；dev_d432_n150 ＝ 43.2 Mbps（主档）；dev_d864_n150 ＝ 86.4 Mbps；dev_d432_n200 ＝ 43.2 Mbps、波束预算放宽到 200。
  方法：Random0 随机合法初解（不搜索）、PTS320 纯禁忌搜索 320 次、G1 容量感知贪心（不搜索）、GTS320 贪心＋禁忌搜索 320 次。
  数字与报告 25 第 4 节表格一致；G1 和 GTS320 在 21.6 / 43.2 / 86.4 档与 S2-FIX 报告第 5 节表格一致。
  图上只标纯禁忌和贪心＋禁忌两条线的满足率数值，其余数值见脚本末尾打印。
"""
from _w24_common import *

s = pd.read_csv(PURE / "summary.csv")
sets = [("dev_d216_n150", "21.6\nMbps"), ("dev_d432_n150", "43.2\nMbps\n（主档）"),
        ("dev_d864_n150", "86.4\nMbps"), ("dev_d432_n200", "43.2\nMbps\n200束")]
methods = [("Random0", "随机初解（不搜索）", C_RAND, "^"), ("G1", "贪心（不搜索）", C_GREEDY, "s"),
           ("PTS320", "纯禁忌搜索320次", C_PURE, "o"), ("GTS320", "贪心＋禁忌搜索320次", C_TABU, "o")]
x = np.array([0, 1, 2, 3.45])

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"wspace": 0.3})
for ax, (col, yl, scale, fmt, ylim) in zip(axes, [("after_W", "需求满足率（%）", 100, "{:.1f}", (58, 104)),
                                                  ("after_SGM", "供需匹配程度", 1, "{:.2f}", (0.70, 1.02))]):
    for m, lab, cc, mk in methods:
        vals = [s[(s.set == st) & (s.method == m) & (s.seed == 51001)].iloc[0][col] * scale for st, _ in sets]
        ax.plot(x[:3], vals[:3], "-", color=cc, lw=2)
        ax.plot(x[3:], vals[3:], mk, color=cc, ms=7.5)
        ax.plot(x[:3], vals[:3], mk, color=cc, ms=7.5, label=lab)
        if col == "after_W" and m in ("PTS320", "GTS320"):
            up = m == "GTS320"
            for i, (xx, v) in enumerate(zip(x, vals)):
                if i == 3:
                    off, ha = (0, 7) if up else (0, -15), "center"
                else:
                    off, ha = ((9, 4), "left") if up else ((-8, -14), "right")
                ax.annotate(fmt.format(v + 1e-9), (xx, v), textcoords="offset points", xytext=off, ha=ha, fontsize=8.8, color=cc)
    ax.axvline(2.72, color="#BBBBBB", lw=0.9, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in sets], fontsize=9)
    ax.set_xlim(-0.4, 3.9)
    ax.set_ylim(*ylim)
    ax.set_ylabel(yl)
    ax.grid(axis="x", visible=False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=9.3, frameon=False, bbox_to_anchor=(0.5, -0.2), columnspacing=1.2, handletextpad=0.3, handlelength=1.6)
fig.text(0.5, -0.29, "每一档里各方法用同一批20个开发场景，纵轴不从0开始。", ha="center", va="top", fontsize=9, color="#444444")
save_png(fig, "W24_需求压力下两种初始化对照.png", **SAVE_KW)

for st, _ in sets:
    xx = s[(s.set == st) & (s.seed == 51001)].set_index("method")
    print(st, {m: (round(xx.loc[m, "after_W"] * 100, 1), round(xx.loc[m, "after_SGM"], 3)) for m, _, _, _ in methods})
