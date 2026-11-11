# -*- coding: utf-8 -*-
"""W18 图：三条SAC路线在固定验证场景（1月12日的72个场景）上的需求满足率随训练步数变化。

数据说明：本机没有逐验证点的原始记录，只有报告15引用的三张训练成品图
  D:/paper/outputs/experiments/sac_routes_s2p19_20261007/analysis/final_report_material/figures/
      c0_training.png、c1_training.png、i0_training.png 的左半（“验证”面板）。
这里把面板里每个验证点（蓝色实心点）的位置读出来，按坐标轴刻度线换算成数值，再用中文标签重画。
校核：报告15第6节文字给出了几个点的精确值（C0 87.3414%→86.2510%，C1 87.3414%→87.7095%，
I0 90.8953%→93.5902%、200k步93.8341%），本脚本运行时逐个比对，差超过0.03个百分点就报错。
读出的数值存成 figure_scripts/data/W18_训练验证曲线_读点.csv。
路线对应：C0=从空构造（只改奖励）、C1=从空构造＋贪心样本、I0=贪心起点＋SAC调整。
"""
import numpy as np
from scipy import ndimage as ndi
from _c1821 import *

BASE = D_EXP / "sac_routes_s2p19_20261007/analysis/final_report_material/figures"
# 各图验证面板的坐标：y轴 (像素, 数值) 两点；x轴 0步和400000步的像素位置（由刻度线读出）
CAL = {
    "c0": dict(y=((80.5, 0.880), (534.5, 0.845)), x=(164.5, 832.5)),
    "c1": dict(y=((113.5, 0.875), (505.5, 0.850)), x=(164.5, 832.5)),
    "i0": dict(y=((130.5, 0.935), (502.5, 0.910)), x=(164.5, 839.5)),
}
KNOWN = {  # 报告15 第6节文字里的精确值（%）：(步数, 值)
    "c0": [(0, 87.3414), (400000, 86.2510)],
    "c1": [(0, 87.3414), (400000, 87.7095)],
    "i0": [(0, 90.8953), (200000, 93.8341), (397112, 93.5902)],
}


def markers(name):
    im = np.array(Image.open(BASE / f"{name}_training.png").convert("RGB")).astype(int)
    m = np.abs(im - np.array((31, 119, 180))).sum(axis=2) < 60
    m[:, 866:] = False
    er = ndi.binary_erosion(m, structure=np.ones((3, 3)))
    lab, n = ndi.label(er)
    cs = ndi.center_of_mass(er, lab, range(1, n + 1))
    sizes = ndi.sum(er, lab, range(1, n + 1))
    cal = CAL[name]
    (ya, va), (yb, vb) = cal["y"]
    x0, x1 = cal["x"]
    pts = []
    for (cy, cx), s in zip(cs, sizes):
        if s < 6:        # 过滤折线本身
            continue
        step = (cx - x0) / (x1 - x0) * 400000
        val = (va + (cy - ya) * (vb - va) / (yb - ya)) * 100
        pts.append((step, val))
    return sorted(pts)


raw = {k: markers(k) for k in CAL}
# 取整到实际验证节点
nodes = {"c0": [0, 50000, 100000, 200000, 300000, 400000],
         "c1": [0, 50000, 100000, 200000, 300000, 400000],
         "i0": [0, 50000, 97112, 100000, 200000, 397112]}
series = {}
for k, pts in raw.items():
    arr = np.array(pts)
    if k == "i0":
        # 97112 与 100000 两个点挨得很近，读成一个标记，取同一数值
        vals = []
        for nd in nodes[k]:
            j = np.argmin(np.abs(arr[:, 0] - nd))
            vals.append(arr[j, 1])
    else:
        vals = [arr[np.argmin(np.abs(arr[:, 0] - nd)), 1] for nd in nodes[k]]
    series[k] = (np.array(nodes[k], dtype=float), np.array(vals))
    for st, v in KNOWN[k]:
        got = vals[nodes[k].index(st)]
        assert abs(got - v) < 0.03, (k, st, got, v)

with open(DATA_DIR / "W18_训练验证曲线_读点.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["路线", "训练步数", "验证需求满足率(%)"])
    names = {"c0": "从空构造（只改奖励）", "c1": "从空构造＋贪心样本", "i0": "贪心起点＋SAC调整"}
    for k, (st, v) in series.items():
        for a, b in zip(st, v):
            w.writerow([names[k], int(a), round(float(b), 3)])

fig, ax = plt.subplots(figsize=(7.2, 3.5))
style = {"c0": (C_PINK, "从空构造（只改奖励）"), "c1": (C_BLUE, "从空构造＋贪心样本"),
         "i0": (C_RED, "贪心起点＋SAC调整")}
for k, (c, lab) in style.items():
    st, v = series[k]
    ax.plot(st / 1e3, v, color=c, lw=2.6 if k == "i0" else 2.0, marker="o", ms=5, label=lab)
    ax.annotate(f"{v[-1]:.1f}", (st[-1] / 1e3, v[-1]), textcoords="offset points", xytext=(6, -3),
                color=c, fontsize=10.5)
ax.set_xlim(-12, 440)
ax.set_xlabel("训练步数（千步）")
ax.set_ylabel("固定验证场景的需求满足率（%）")
ax.legend(loc="center right", frameon=False)
save_png(fig, "W18_训练验证曲线.png")
