"""W17 画图（按报告表格和实验目录里的 json 重画）。

数字来源：
  * 第一轮（可变束数泛化，报告 第二阶段/实验报告/13）：4.1、4.2 两张逐束数表（固定100束训练和混合训练的需求满足率差），
    第3节总表（直接分配与功率细调后的需求满足率、供需匹配程度）。
  * 第二轮（报告 14）：3.1 逐束数表（改奖励、改折扣和整形在 50/100/200 束的直接分配满足率），
    3.2 每束槽数表，3.3 总表；
    variable_beams_s2p18_20261006/analysis/ 下 sca_per_method.json、g0_sca_per_method.json（功率细调前后）、
    g0_assessment.json（改网络结构在11个束数的满足率和每束槽数）、beam_tails_{V2,V3,G0}.json（逐束容量需求比），
    analysis_inputs/p17_anchor_by_n.json 和 per_n_snapshots.json（第一轮混合训练、贪心的每束槽数参照）。
本机没有第一轮和第二轮的逐场景原始记录，所以只能画到这些汇总数字的粒度。
"""
import json
import os

import numpy as np

import _style as S
from _style import BLUE, GREEN, GREY, LIGHT, ORANGE, PURPLE, RED, TEAL
import matplotlib.pyplot as plt

S.setup()
J = r"D:\paper\outputs\experiments\variable_beams_s2p18_20261006\analysis"


def load(name):
    return json.load(open(os.path.join(J, name), encoding="utf-8"))


# ---------------------------------------------------------------- 图 第一轮留出束数上的泛化
N11 = [50, 60, 75, 90, 100, 125, 140, 150, 175, 180, 200]
held = {60, 90, 140, 180}
# 报告13 表4.1、4.2：混合训练减固定100束训练的需求满足率差（百分点）
diff_direct = [-1.088, -0.705, -0.488, -0.092, 0.022, 0.418, 1.535, 1.947, 2.059, 2.149, 2.673]
diff_final = [0.040, -0.459, -0.236, -0.186, 0.192, 0.007, 0.732, 1.298, 0.921, 1.028, 1.380]
fig, ax = plt.subplots(figsize=(6.6, 3.7))
x = np.arange(len(N11))
w = 0.38
b1 = ax.bar(x - w / 2, diff_direct, w, color=BLUE, label="直接分配")
b2 = ax.bar(x + w / 2, diff_final, w, color=ORANGE, label="功率细调后")
for i, n in enumerate(N11):
    if n in held:
        b1[i].set_edgecolor("k")
        b1[i].set_hatch("//")
        b2[i].set_edgecolor("k")
        b2[i].set_hatch("//")
ax.axhline(0, color="k", lw=1)
ax.set_xticks(x)
ax.set_xticklabels([("★" if n in held else "") + str(n) for n in N11])
ax.set_xlabel("目标波束数（★斜线柱：训练里完全没出现过的束数）")
ax.set_ylabel("需求满足率差（百分点）")
ax.set_title("混合训练比固定100束训练高多少")
ax.legend(loc="upper left", frameon=True)
ax.grid(axis="x", visible=False)
fig.subplots_adjust(left=0.11, right=0.99, top=0.91, bottom=0.20)
S.save(fig, "W17_第一轮留出束数上的泛化.png")

# ---------------------------------------------------------------- 图 第二轮整体对比
sca = load("sca_per_method.json")["arms"]
g0s = load("g0_sca_per_method.json")["methods"]
# (名称, 直接SI, 细调后SI, 直接SGM, 细调后SGM)；V0/V1/贪心/固定规则取自报告13第3节总表，V2/V3/G0 取自 json
rows = [
    ("固定规则", 83.680, 86.960, 0.379148, 0.863818),
    ("贪心", 89.332, 93.335, 0.371259, 0.930858),
    ("固定100束训练", 86.951, 91.421, 0.694022, 0.914359),
    ("混合波束数训练", 87.718, 91.850, 0.686065, 0.917824),
    ("第二轮：改奖励", sca["V2"]["none"]["si"] * 100, sca["V2"]["sca"]["si"] * 100,
     sca["V2"]["none"]["sgm"], sca["V2"]["sca"]["sgm"]),
    ("第二轮：改折扣和整形", sca["V3"]["none"]["si"] * 100, sca["V3"]["sca"]["si"] * 100,
     sca["V3"]["none"]["sgm"], sca["V3"]["sca"]["sgm"]),
    ("第二轮：改网络结构", g0s["none"]["si"] * 100, g0s["sca"]["si"] * 100, g0s["none"]["sgm"], g0s["sca"]["sgm"]),
]
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.9), sharey=True, gridspec_kw={"wspace": 0.06})
y = np.arange(len(rows))[::-1]
for ax, (ia, ib), title, fmt, xlim in zip(axs, [(1, 2), (3, 4)], ["需求满足率 %", "供需匹配程度"],
                                          ["{:.1f}", "{:.2f}"], [(75, 96), (0.2, 1.02)]):
    a = [r[ia] for r in rows]
    b = [r[ib] for r in rows]
    ax.barh(y + 0.19, a, height=0.36, color=LIGHT, edgecolor=GREY, label="直接分配")
    ax.barh(y - 0.19, b, height=0.36, color=ORANGE, label="功率细调后")
    for yy, va, vb in zip(y, a, b):
        ax.text(va + (xlim[1] - xlim[0]) * 0.008, yy + 0.19, fmt.format(va), va="center", fontsize=8.5)
        ax.text(vb + (xlim[1] - xlim[0]) * 0.008, yy - 0.19, fmt.format(vb), va="center", fontsize=8.5)
    ax.set_xlim(*xlim)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
axs[0].set_yticks(y)
axs[0].set_yticklabels([r[0] for r in rows])
axs[1].tick_params(axis="y", length=0)
axs[1].set_xlim(0.2, 1.12)
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.6, -0.01))
fig.subplots_adjust(left=0.30, right=0.99, top=0.92, bottom=0.14)
S.save(fig, "W17_直接分配与功率细调后对比.png")
for r in rows:
    print(r[0], "直接/细调 SI", round(r[1], 2), round(r[2], 2), "SGM", round(r[3], 3), round(r[4], 3))

# ---------------------------------------------------------------- 图 第二轮逐束数
g0 = load("g0_assessment.json")["per_n"]
g0_n = [int(k) for k in g0]
g0_si = [g0[k]["si"] * 100 for k in g0]
g0_sl = [g0[k]["slots_per_beam"] for k in g0]
# 报告13 表4.1：直接分配逐束数需求满足率（百分点）
v0_si = [88.227, 89.338, 90.157, 90.341, 89.844, 87.443, 86.174, 85.454, 83.627, 83.597, 82.260]
v1_si = [87.139, 88.633, 89.670, 90.249, 89.867, 87.861, 87.709, 87.400, 85.686, 85.746, 84.934]
# 报告14 表3.1、3.2：改奖励、改折扣和整形在 50/100/200 束
v2_si = {50: 86.66, 100: 89.72, 200: 86.41}
v3_si = {50: 89.73, 100: 88.57, 200: 82.05}
v2_sl = {100: 2.85, 200: 1.96}
v3_sl = {100: 2.72, 200: 1.68}
# 第一轮混合训练、贪心的每束槽数（报告14 3.3 表和 p17_anchor / per_n_snapshots 的参照）
snap = load("..\\analysis_inputs\\per_n_snapshots.json")["p17_reference_slots_per_beam"]
v1_sl = {int(k): v for k, v in snap["V1"].items()}
gr_sl = {int(k): v for k, v in snap["greedy"].items()}

fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.9), gridspec_kw={"wspace": 0.28})
ax = axs[0]
ax.axhline(89.332, color=GREY, ls=":", lw=1.4)
ax.text(201, 89.45, "贪心（整体均值）", fontsize=8.5, color=GREY, ha="right", va="bottom")
ax.plot(N11, v0_si, "-o", color=LIGHT, ms=3.5, lw=1.6, label="固定100束训练")
ax.plot(N11, v1_si, "-o", color=BLUE, ms=4, lw=1.8, label="混合波束数训练")
ax.plot(g0_n, g0_si, "-o", color=GREEN, ms=4, lw=1.8, label="第二轮：改网络结构")
ax.plot(list(v2_si), list(v2_si.values()), "-s", color=ORANGE, ms=5, lw=1.8, label="第二轮：改奖励")
ax.plot(list(v3_si), list(v3_si.values()), "-^", color=PURPLE, ms=5, lw=1.8, label="第二轮：改折扣和整形")
ax.set_title("需求满足率 %（直接分配）")
ax.set_xlabel("目标波束数")
ax = axs[1]
ax.plot(list(gr_sl), list(gr_sl.values()), "-o", color=GREY, ms=3.5, lw=1.6, label="贪心")
ax.plot(list(v1_sl), list(v1_sl.values()), "-o", color=BLUE, ms=4, lw=1.8)
ax.plot(g0_n, g0_sl, "-o", color=GREEN, ms=4, lw=1.8)
ax.plot(list(v2_sl), list(v2_sl.values()), "-s", color=ORANGE, ms=5, lw=1.8)
ax.plot(list(v3_sl), list(v3_sl.values()), "-^", color=PURPLE, ms=5, lw=1.8)
ax.set_title("每束占用的频率槽数")
ax.set_xlabel("目标波束数")
h, l = axs[0].get_legend_handles_labels()
h2, l2 = axs[1].get_legend_handles_labels()
fig.legend(h + h2, l + l2, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=9)
fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.30)
S.save(fig, "W17_第二轮逐束数对照.png")

# ---------------------------------------------------------------- 图 细调前后逐束容量需求比
arms = [("改奖励", "V2"), ("改折扣和整形", "V3"), ("改网络结构", "G0")]
tails = {k: load(f"beam_tails_{k}.json")["methods"] for _, k in arms}
meths = [("none", "直接分配", LIGHT), ("sca", "功率优化细调", ORANGE), ("sca_si_lp", "先保需求满足率的细调", BLUE)]
fig, axs = plt.subplots(1, 3, figsize=(6.8, 3.7), gridspec_kw={"wspace": 0.42})
x = np.arange(3)
wid = 0.26
for ax, key, title, as_pct in zip(axs, ["share_below_1", "share_below_0_8", "ratio_max"],
                                  ["容量不够需求的\n波束占比 %", "严重不够（不足80%）\n的波束占比 %", "容量是需求的\n最多几倍"],
                                  [True, True, False]):
    for j, (mk, lab, colr) in enumerate(meths):
        vals = [tails[k][mk][key] * (100 if as_pct else 1) for _, k in arms]
        bars = ax.bar(x + (j - 1) * wid, vals, wid, color=colr, edgecolor=GREY if colr == LIGHT else None, label=lab)
        for xi, v in zip(x + (j - 1) * wid, vals):
            ax.text(xi, v * 1.02, f"{v:.0f}" if as_pct else (f"{v:.0f}" if v > 5 else f"{v:.1f}"), ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["改奖励", "改折扣\n和整形", "改网络\n结构"], fontsize=8.5)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", visible=False)
axs[2].set_yscale("log")
axs[2].set_ylim(0.8, 60)
axs[0].set_ylim(0, 62)
axs[1].set_ylim(0, 40)
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=9)
fig.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.27)
S.save(fig, "W17_细调前后逐束容量需求比.png")
for _, k in arms:
    for mk, lab, _c in meths:
        t = tails[k][mk]
        print(k, mk, "<1", round(t["share_below_1"] * 100, 1), "<0.8", round(t["share_below_0_8"] * 100, 1), "max", round(t["ratio_max"], 2))

