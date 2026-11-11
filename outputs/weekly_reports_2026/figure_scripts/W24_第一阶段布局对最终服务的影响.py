"""W24 图：资源紧张（总功率 750 W）时，四种第一阶段布局接到第二阶段贪心和第三阶段功率优化后的最终服务（从原始记录重画）。
数据（只读）：D:/paper/outputs/experiments/s1_fix_*/C/service_stress.csv（power_budget_w==750）、C/layout_n.csv（给出每个布局文件对应的飞机数 n）。
  取 100 束预算、飞机数>=1200 的白天场景 36 景（开发 12 景＋确认 24 景）；文件名里的布局类型：
    INIT ＝ 自适应网＋就近分配；FLOW-AO-peak ＝ FLOW-AO 只压峰值；FLOW-AO ＝ FLOW-AO 加间隔规则（完整版）；FLOW-AO-no-spacing ＝ FLOW-AO 不加间隔规则。
  列 peak＝最忙波束飞机数，sca_aircraft_mean＝飞机平均需求满足率，sca_aircraft_worst10＝最差 10% 飞机的平均满足率（第三阶段功率优化后）。
  数字与 S1-FIX 报告第 4 节表格一致（94.9/88.0/98.3/98.4，65.1/44.6/86.9/86.4）。
"""
from _w24_common import *

C_DIR = S1FIX / "C"
lay = pd.read_csv(C_DIR / "layout_n.csv")
d = pd.read_csv(C_DIR / "service_stress.csv").merge(lay[["file", "n"]], on="file")
d["variant"] = d.file.str.extract(r"_N\d+_(.*)\.npz")[0]
d["N"] = d.file.str.extract(r"_N(\d+)_")[0].astype(int)
x = d[(d.n >= 1200) & (d.N == 100) & (d.power_budget_w == 750)]
g = x.groupby("variant")[["peak", "sca_aircraft_mean", "sca_aircraft_worst10", "width_mean"]].mean()

order = [("INIT", "自适应网＋就近分配", C_HEX),
         ("FLOW-AO-peak", "FLOW-AO只压峰值", "#E69F00"),
         ("FLOW-AO", "FLOW-AO（加间隔规则）", C_FLOW2),
         ("FLOW-AO-no-spacing", "FLOW-AO（不加间隔规则）", C_FLOW)]
ypos = np.arange(len(order))[::-1]
panels = [("peak", "最忙波束飞机数（架）", 1, "{:.0f}", 100),
          ("sca_aircraft_mean", "飞机平均满足率（%）", 100, "{:.1f}", 135),
          ("sca_aircraft_worst10", "最差10%飞机满足率（%）", 100, "{:.1f}", 135)]
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=True, gridspec_kw={"wspace": 0.08})
for ax, (col, xl, sc, fmt, xmax) in zip(axes, panels):
    for y, (k, _, c) in zip(ypos, order):
        v = g.loc[k, col] * sc
        ax.barh(y, v, color=c, height=0.66)
        ax.text(v + xmax * 0.015, y, fmt.format(v + 1e-9), va="center", fontsize=9.5)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xl, fontsize=9.5)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    if sc == 100:
        ax.set_xticks([0, 50, 100])
axes[0].set_xticks([0, 25, 50, 75])
axes[0].set_yticks(ypos)
axes[0].set_yticklabels([n for _, n, _ in order], fontsize=9.5)
fig.suptitle("总功率压到750 W、100个波束、36个白天场景", fontsize=10.5, y=1.0)
save_png(fig, "W24_第一阶段布局对最终服务的影响.png", **SAVE_KW)
print(g.round(3).to_string())
