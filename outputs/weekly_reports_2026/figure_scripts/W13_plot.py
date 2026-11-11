"""W13 画图：按报告里的表格数字重画（逐场景原始记录不在本机，只有报告里的汇总表和图）。

数字来源（docs/第一阶段/实验报告）：
  04_覆盖优先与主体负载均衡A阶段对照.md
     第2节“覆盖与负载结果”表、配对表；第3节“已记录分段合计”表和“实测阶段”表；
  05_动作空间与PPO负载均衡B阶段对照.md
     第2节五菜单主体CV；第3节六法结果表；第4节PPO对同菜单搜索的配对表；第5节原终态到保留结果表；第6节耗时表。
"""
import numpy as np

import _style as S
from _style import BLUE, GREEN, GREY, LIGHT, ORANGE, PURPLE, RED, TEAL
import matplotlib.pyplot as plt

S.setup()

# ------------------------------------------------------------------ 图1 A阶段四步对照（04 第2节表）
A = ["聚类\n基础布局", "＋覆盖\n修补", "＋负载感知\n用户归属", "＋四向\n调位"]
strict = [35, 91, 91, 92]
cv = [0.7926, 0.8050, 0.5521, 0.5491]
band = [19.71, 19.54, 23.46, 25.99]
peak = [4.2590, 4.3684, 2.0752, 2.0852]
cols = [LIGHT, BLUE, ORANGE, GREEN]
fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.2), gridspec_kw={"wspace": 0.30, "hspace": 0.62})
x = np.arange(4)


def bars(ax, vals, title, fmt, ylim, target=None, tlabel=None):
    ax.bar(x, vals, color=cols, width=0.66, edgecolor=GREY)
    for xi, v in zip(x, vals):
        ax.text(xi, v + (ylim[1] - ylim[0]) * 0.015, fmt.format(v), ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(A, fontsize=8)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)
    if target is not None:
        ax.axhline(target, color=RED, ls="--", lw=1.3)
        ax.text(-0.45, target + (ylim[1] - ylim[0]) * 0.02, tlabel, color=RED, fontsize=8.5, ha="left", bbox=dict(facecolor="white", edgecolor="none", pad=1.0))


bars(axs[0, 0], strict, "严格全覆盖的场景数（共100）", "{:.0f}", (0, 118))
bars(axs[0, 1], cv, "主体负载CV（越小越均匀）", "{:.2f}", (0, 1.0), 0.20, "目标 0.20")
bars(axs[1, 0], band, "落在均值±20%内的波束数", "{:.1f}", (0, 100), 90, "目标 90")
bars(axs[1, 1], peak, "最重波束负载÷均值", "{:.1f}", (0, 5.2))
S.save(fig, "W13_覆盖优先与负载均衡四步对照.png")

# ------------------------------------------------------------------ 图2 配对结果与耗时（04 配对表、分段表）
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.9), gridspec_kw={"wspace": 0.55, "width_ratios": [1, 1]})
ax = axs[0]
pairs = ["覆盖修补 对\n基础布局", "＋用户归属 对\n基础布局", "＋四向调位 对\n基础布局", "＋四向调位 对\n＋用户归属"]
imp = np.array([7, 100, 100, 78])
eq = np.array([36, 0, 0, 1])
wo = np.array([57, 0, 0, 21])
yy = np.arange(4)[::-1]
left = np.zeros(4)
for v, c, lab in [(imp, GREEN, "改善"), (eq, LIGHT, "持平"), (wo, RED, "退步")]:
    ax.barh(yy, v, left=left, color=c, height=0.62, label=lab)
    for yi, l, w in zip(yy, left, v):
        if w >= 6:
            ax.text(l + w / 2, yi, str(w), ha="center", va="center", fontsize=8.5, color="k" if c == LIGHT else "white")
    left += v
ax.set_yticks(yy)
ax.set_yticklabels(pairs, fontsize=8.5)
ax.set_xlim(0, 100)
ax.set_xticks([0, 50, 100])
ax.set_xlabel("场景数（主体CV）")
ax.set_title("同一场景前后对比")
ax.grid(axis="y", visible=False)
fig.legend(*ax.get_legend_handles_labels(), loc="lower left", ncol=3, frameon=False, bbox_to_anchor=(0.03, -0.01), fontsize=8.5)
ax = axs[1]
tot = [9.9896, 10.2158, 10.2522, 11.6929]
c1 = [9.7980] * 4
edge = [0.1866] * 4
fix = [0, 0.2262, 0.2262, 0.2262]
asg = [0, 0, 0.0367, 0.0367]
mv = [0, 0, 0, 1.4413]
rest = [t - (a + b + c + d + e) for t, a, b, c, d, e in zip(tot, c1, edge, fix, asg, mv)]
stages = [(c1, BLUE, "基础布局局部搜索"), (edge, TEAL, "边缘识别"), (fix, ORANGE, "覆盖修补"),
          (asg, PURPLE, "用户归属"), (mv, GREEN, "四向调位"), (rest, GREY, "其他")]
yy = np.arange(4)[::-1]
left = np.zeros(4)
for v, c, lab in stages:
    ax.barh(yy, v, left=left, color=c, height=0.62, label=lab)
    left += np.array(v)
for yi, t in zip(yy, tot):
    ax.text(t + 0.12, yi, f"{t:.1f}", va="center", fontsize=8.5)
ax.set_yticks(yy)
ax.set_yticklabels(["基础布局", "＋覆盖修补", "＋用户归属", "＋四向调位"], fontsize=8.5)
ax.set_xlim(0, 13.5)
ax.set_xlabel("每景已记录耗时（秒）")
ax.set_title("耗时分段")
ax.grid(axis="y", visible=False)
fig.legend(*ax.get_legend_handles_labels(), loc="lower right", ncol=3, frameon=False, bbox_to_anchor=(0.995, -0.01), fontsize=7.5, columnspacing=0.8, handlelength=1.1)
fig.subplots_adjust(left=0.17, right=0.985, top=0.90, bottom=0.27)
S.save(fig, "W13_配对结果与耗时分段.png")

# ------------------------------------------------------------------ 图3 B阶段六法对比（05 第3、6节表）
menus = ["四向", "八方向", "相邻双束"]
search = {"cover": [93, 93, 92], "cv": [0.532496, 0.534820, 0.535377], "band": [26.60, 25.16, 24.95], "sec": [11.538, 11.428, 11.737]}
ppo = {"cover": [92, 92, 92], "cv": [0.535129, 0.537551, 0.534191], "band": [25.43, 26.06, 25.87], "sec": [28.797, 27.223, 64.096]}
fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.2), gridspec_kw={"wspace": 0.32, "hspace": 0.58})
x = np.arange(3)
spec = [("cover", "严格全覆盖的场景数（共100）", "{:.0f}", (90.5, 94.0), 92),
        ("cv", "主体负载CV（越小越均匀）", "{:.3f}", (0.5290, 0.5400), 0.536974),
        ("band", "落在均值±20%内的波束数", "{:.1f}", (22.8, 27.6), 23.75),
        ("sec", "每景已记录耗时（秒）", "{:.1f}", (0, 74), None)]
for ax, (k, title, fmt, yl, ref) in zip(axs.ravel(), spec):
    if k == "sec":
        ax.bar(x - 0.18, search[k], 0.34, color=TEAL, label="同动作下的搜索")
        ax.bar(x + 0.18, ppo[k], 0.34, color=PURPLE, label="PPO")
        for xi, v in zip(x - 0.18, search[k]):
            ax.text(xi, v + 1, fmt.format(v), ha="center", fontsize=8)
        for xi, v in zip(x + 0.18, ppo[k]):
            ax.text(xi, v + 1, fmt.format(v), ha="center", fontsize=8)
    else:
        for xi, a, b in zip(x, search[k], ppo[k]):
            ax.plot([xi - 0.15, xi + 0.15], [a, b], color=GREY, lw=1)
        ax.plot(x - 0.15, search[k], "o", color=TEAL, ms=7, label="同动作下的搜索")
        ax.plot(x + 0.15, ppo[k], "s", color=PURPLE, ms=7, label="PPO")
        rng = yl[1] - yl[0]
        for xi, a, b in zip(x, search[k], ppo[k]):
            for xx, v, other in [(xi - 0.15, a, b), (xi + 0.15, b, a)]:
                up = v >= other
                ax.text(xx, v + (0.04 if up else -0.04) * rng, fmt.format(v), ha="center",
                        va="bottom" if up else "top", fontsize=8)
        ax.axhline(ref, color=GREY, ls="--", lw=1.2, label="共同起点")
    ax.set_xlim(-0.7, 2.7)
    ax.set_xticks(x)
    ax.set_xticklabels(menus)
    ax.set_title(title, fontsize=10.5)
    ax.set_ylim(*yl)
    ax.grid(axis="x", visible=False)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=9)
fig.subplots_adjust(left=0.10, right=0.99, top=0.93, bottom=0.13)
S.save(fig, "W13_三种动作下PPO与搜索对照.png")

# ------------------------------------------------------------------ 图4 配对与原终态（05 第4、5节表）
fig, axs = plt.subplots(1, 3, figsize=(6.9, 3.6), gridspec_kw={"wspace": 0.55, "width_ratios": [1.15, 1, 1]})
ax = axs[0]
imp = np.array([32, 40, 54])
eq = np.array([0, 1, 1])
wo = np.array([68, 59, 45])
yy = np.arange(3)[::-1]
left = np.zeros(3)
for v, c, lab in [(imp, GREEN, "PPO更好"), (eq, LIGHT, "相同"), (wo, RED, "PPO更差")]:
    ax.barh(yy, v, left=left, color=c, height=0.6, label=lab)
    for yi, l, ww in zip(yy, left, v):
        if ww >= 8:
            ax.text(l + ww / 2, yi, str(ww), ha="center", va="center", fontsize=8.5, color="k" if c == LIGHT else "white")
    left += v
ax.set_yticks(yy)
ax.set_yticklabels(menus)
ax.set_xlim(0, 100)
ax.set_xticks([0, 50, 100])
ax.set_xlabel("场景数")
ax.set_title("PPO与同动作搜索\n逐场景比主体CV", fontsize=10)
ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=3, frameon=False, fontsize=7.5, columnspacing=0.8, handlelength=1.0)
raw = [0.539326, 0.546101, 0.538838]
kept = [0.535129, 0.537551, 0.534191]
srch = [0.532496, 0.534820, 0.535377]
for ax, key, title, vals in [(axs[1], "cv", "主体CV\n（越小越均匀）", (srch, raw, kept))]:
    xx = np.arange(3)
    ww = 0.26
    ax.bar(xx - ww, vals[0], ww, color=TEAL, label="搜索")
    ax.bar(xx, vals[1], ww, color=LIGHT, edgecolor=GREY, label="PPO策略停止时")
    ax.bar(xx + ww, vals[2], ww, color=PURPLE, label="PPO保留的最好布局")
    ax.set_ylim(0.52, 0.552)
    ax.set_xticks(xx)
    ax.set_xticklabels(menus)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", visible=False)
ax = axs[2]
braw = [22.93, 22.74, 23.39]
bkept = [25.43, 26.06, 25.87]
bsrch = [26.60, 25.16, 24.95]
ax.bar(xx - ww, bsrch, ww, color=TEAL)
ax.bar(xx, braw, ww, color=LIGHT, edgecolor=GREY)
ax.bar(xx + ww, bkept, ww, color=PURPLE)
ax.set_ylim(20, 28)
ax.set_xticks(xx)
ax.set_xticklabels(menus)
ax.set_title("落在均值±20%内\n的波束数", fontsize=10)
ax.grid(axis="x", visible=False)
h, l = axs[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.66, -0.01), fontsize=8.5)
fig.subplots_adjust(left=0.10, right=0.99, top=0.82, bottom=0.30)
S.save(fig, "W13_PPO逐场景配对与保留布局.png")

# ------------------------------------------------------------------ 图5 动作菜单筛选与逐景耗时（05 第2、6节）
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.6), gridspec_kw={"wspace": 0.45})
ax = axs[0]
menu5 = ["四向", "八方向", "四向\n多步长", "按业务\n引导位置", "相邻\n双束"]
v5 = [0.546744, 0.547784, 0.552997, 0.550700, 0.548798]
col5 = [PURPLE, PURPLE, LIGHT, LIGHT, PURPLE]
ax.bar(range(5), v5, color=col5, edgecolor=GREY)
for i, v in enumerate(v5):
    ax.text(i, v + 0.0004, f"{v:.3f}", ha="center", fontsize=8)
ax.set_ylim(0.540, 0.558)
ax.set_xticks(range(5))
ax.set_xticklabels(menu5, fontsize=8)
ax.set_title("五种动作菜单筛选\n（20个验证场景的主体CV）", fontsize=10)
ax.grid(axis="x", visible=False)
ax.set_xlabel("紫色：选入训练的三种", color=PURPLE, fontsize=8.5)
ax = axs[1]
names6 = ["四向搜索", "四向PPO", "八方向搜索", "八方向PPO", "双束搜索", "双束PPO"]
pub = [9.915456 + 0.120074 + 0.102116 + 0.027490] * 6
adj = [1.369153, 18.565363, 1.258974, 16.959639, 1.567981, 53.367780]
env = [0, 0.062048, 0, 0.093967, 0, 0.557281]
fin = [0.003383, 0.004133, 0.003492, 0.004309, 0.003609, 0.005519]
yy = np.arange(6)[::-1]
ax.barh(yy, pub, color=BLUE, height=0.62, label="共同起点")
ax.barh(yy, np.array(adj) + np.array(env) + np.array(fin), left=pub, color=ORANGE, height=0.62, label="搜索或PPO调整")
for yi, a, b, c, d in zip(yy, pub, adj, env, fin):
    ax.text(a + b + c + d + 1, yi, f"{a + b + c + d:.1f}", va="center", fontsize=8)
ax.set_yticks(yy)
ax.set_yticklabels(names6, fontsize=8.5)
ax.set_xlim(0, 85)
ax.set_xlabel("每景已记录耗时（秒）")
ax.set_title("六种做法的耗时分段", fontsize=10)
ax.grid(axis="y", visible=False)
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.40), fontsize=8, frameon=True)
fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.20)
S.save(fig, "W13_动作菜单筛选与耗时.png")


