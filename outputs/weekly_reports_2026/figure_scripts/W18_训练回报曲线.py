# -*- coding: utf-8 -*-
"""W18 图：三条SAC路线训练过程中每个回合回报的滑动平均（最近50个回合）。

数据说明：本机没有逐回合原始记录，只有报告15引用的三张训练成品图
  D:/paper/outputs/experiments/sac_routes_s2p19_20261007/analysis/final_report_material/figures/
      c0_training.png、c1_training.png、i0_training.png 的右半（“训练”面板）里的橙色线。
这里按坐标轴刻度线把橙色线每个像素列的位置换算成数值，再用中文标签重画（只画滑动平均，
原图里浅色的逐回合原值不重画）。横轴是训练步数，纵轴是回报。
前两条路线的回报是“从空构造到最终分配的满足率改善×100”，第三条的回报是“相对贪心起点的改善”，
含义不同，所以分成两个面板，不放在同一条坐标上比较。
读出的曲线存成 figure_scripts/data/W18_训练回报曲线_读点.csv（每个像素列一个点）。
"""
import numpy as np
from _c1821 import *

BASE = D_EXP / "sac_routes_s2p19_20261007/analysis/final_report_material/figures"
# y轴：两点 (像素, 数值)；x轴：0步和400000步的像素
CAL = {
    "c0": dict(y=((81.5, 100.0), (494.5, 65.0)), x=(1022.5, 1691.5), skip=lambda x, y: x >= 1485 and y >= 470),
    "c1": dict(y=((81.5, 100.0), (473.5, 60.0)), x=(1022.5, 1691.5), skip=lambda x, y: 1240 <= x <= 1475 and y >= 470),
    "i0": dict(y=((110.5, 14.0), (518.5, 0.0)), x=(1016.5, 1691.5), skip=lambda x, y: x <= 1222 and y <= 130),
}


def curve(name):
    im = np.array(Image.open(BASE / f"{name}_training.png").convert("RGB")).astype(int)
    m = np.abs(im - np.array((255, 127, 14))).sum(axis=2) < 40
    cal = CAL[name]
    m[:, :992] = False
    ys, xs = np.where(m)
    keep = np.array([not cal["skip"](x, y) for x, y in zip(xs, ys)])
    ys, xs = ys[keep], xs[keep]
    (ya, va), (yb, vb) = cal["y"]
    x0, x1 = cal["x"]
    steps, vals = [], []
    for x in np.unique(xs):
        yy = ys[xs == x]
        steps.append((x - x0) / (x1 - x0) * 400000)
        vals.append(va + (yy.mean() - ya) * (vb - va) / (yb - ya))
    return np.array(steps), np.array(vals)


cur = {k: curve(k) for k in CAL}
with open(DATA_DIR / "W18_训练回报曲线_读点.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["路线", "训练步数", "最近50回合平均回报"])
    names = {"c0": "从空构造（只改奖励）", "c1": "从空构造＋贪心样本", "i0": "贪心起点＋SAC调整"}
    for k, (s, v) in cur.items():
        for a, b in zip(s, v):
            w.writerow([names[k], int(a), round(float(b), 3)])

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"wspace": 0.32})
ax = axes[0]
for k, c, lab in (("c0", C_PINK, "从空构造（只改奖励）"), ("c1", C_BLUE, "从空构造＋贪心样本")):
    s, v = cur[k]
    ax.plot(s / 1e3, v, color=c, lw=1.5, label=lab)
ax.set_ylim(80, 96)
ax.set_title("前两种：从空构造", fontsize=11.5)
ax.set_xlabel("训练步数（千步）")
ax.set_ylabel("回报（最近50回合平均）")
ax.set_xticks([0, 100, 200, 300, 400])
ax.legend(loc="upper right", frameon=False, fontsize=9.5)
ax = axes[1]
s, v = cur["i0"]
ax.plot(s / 1e3, v, color=C_RED, lw=1.5)
ax.set_ylim(0, 6)
ax.set_title("第三种：贪心起点＋SAC调整", fontsize=11.5)
ax.set_xlabel("训练步数（千步）")
ax.set_ylabel("相对贪心起点的改善")
ax.set_xticks([0, 100, 200, 300, 400])
save_png(fig, "W18_训练回报曲线.png")
