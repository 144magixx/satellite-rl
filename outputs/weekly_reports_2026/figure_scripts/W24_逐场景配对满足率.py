"""W24 图：88 个确认场景里，每个场景的纯禁忌搜索与贪心＋禁忌搜索的需求满足率对比（从原始记录重画）。
数据（只读）：D:/paper/outputs/experiments/s2_pure_tabu_*/eval/conf_d432_n150/results.csv
  seed==51001，列 after_W（功率细调后、按业务量加权的需求满足率）；每个点一个场景。
  “更高/持平/更低”：纯禁忌减贪心＋禁忌的差，大于 1e-9 记更高，绝对值不超过 1e-9 记持平，小于 -1e-9 记更低；
  与报告 25 的 2/0/86、3/2/83、3/2/83 一致。
"""
from _w24_common import *

r = pd.read_csv(PURE / "conf_d432_n150" / "results.csv")
r = r[r.seed == 51001]
p = r.pivot(index="scene_id", columns="method", values="after_W") * 100

fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharex=True, sharey=True, gridspec_kw={"wspace": 0.12})
lo, hi = 62, 101
for ax, b in zip(axes, (32, 96, 320)):
    xs, ys = p[f"GTS{b}"], p[f"PTS{b}"]
    d = (ys - xs) / 100
    up, eq, dn = int((d > 1e-9).sum()), int((d.abs() <= 1e-9).sum()), int((d < -1e-9).sum())
    ax.plot([lo, hi], [lo, hi], "--", color="#555555", lw=1.2)
    ax.scatter(xs, ys, s=16, color=C_PURE, alpha=0.75, edgecolor="none")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title(f"搜索{b}次", fontsize=10.5)
    ax.text(0.05, 0.95, f"纯禁忌更高 {up}\n持平 {eq}\n更低 {dn}", transform=ax.transAxes, va="top", ha="left", fontsize=9.2, linespacing=1.3)
    ax.set_xticks([70, 80, 90, 100])
    ax.set_yticks([70, 80, 90, 100])
axes[1].set_xlabel("贪心＋禁忌搜索的需求满足率（%）")
axes[0].set_ylabel("纯禁忌搜索的需求满足率（%）")
save_png(fig, "W24_逐场景配对满足率.png", **SAVE_KW)
