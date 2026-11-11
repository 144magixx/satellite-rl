"""W14 画图：按报告里的表格数字重画（逐回合、逐场景原始记录不在本机，只有报告里的汇总表和图）。

数字来源（docs/第一阶段/实验报告）：
  06_区域重构PPO机制检查与停止诊断.md   第2节闸门表、第3节拒绝类别表和分层表、第4节验证节点表和五方法表、第5节耗时表；
  07_区域重构PPO修复版共同考核.md       第1节结论、第3节考核表、第4节配对表、第5节接受率、第6节分段耗时表。
"""
import numpy as np

import _style as S
from _style import BLUE, GREEN, GREY, LIGHT, ORANGE, PURPLE, RED, TEAL
import matplotlib.pyplot as plt

S.setup()

# ------------------------------------------------------------------ 图1 动作为什么被拒绝（06 第3节）
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.8), gridspec_kw={"wspace": 0.75, "width_ratios": [1.1, 1]})
ax = axs[0]
labs = ["接受并改变布局", "归属超出波束足迹", "找不到合法中心", "覆盖与归属不一致", "布局没有变化"]
cnt = [578, 9959, 5686, 42, 17]
cols = [GREEN, RED, ORANGE, LIGHT, LIGHT]
yy = np.arange(5)[::-1]
ax.barh(yy, cnt, color=cols, height=0.62, edgecolor=GREY)
for yi, v in zip(yy, cnt):
    ax.text(v + 150, yi, f"{v}", va="center", fontsize=8.5)
ax.set_yticks(yy)
ax.set_yticklabels(labs, fontsize=8.5)
ax.set_xlim(0, 12500)
ax.set_xticks([0, 5000, 10000])
ax.set_xlabel("次数")
ax.set_title("修复前最后一段训练里\n16282次非停止动作的结果", fontsize=10)
ax.grid(axis="y", visible=False)
ax = axs[1]
names = ["修复前\n全部动作", "修复前\n本回合还\n没接受过", "修复前\n本回合已\n接受过", "修复后\n同一检查点"]
rate = [578 / 16282 * 100, 559 / 912 * 100, 19 / 15370 * 100, 9787 / 16380 * 100]
cc = [RED, LIGHT, RED, GREEN]
ax.bar(range(4), rate, color=cc, edgecolor=GREY, width=0.66)
for i, v in enumerate(rate):
    ax.text(i, v + 1.5, f"{v:.1f}%" if v >= 1 else f"{v:.2f}%", ha="center", fontsize=8.5)
ax.set_xticks(range(4))
ax.set_xticklabels(names, fontsize=7)
ax.set_ylim(0, 78)
ax.axhline(50, color=GREY, ls="--", lw=1.1)
ax.text(-0.45, 52, "事先定的门槛 50%", fontsize=8, color=GREY, ha="left")
ax.set_title("动作接受率", fontsize=10)
ax.grid(axis="x", visible=False)
fig.subplots_adjust(left=0.19, right=0.99, top=0.84, bottom=0.25)
S.save(fig, "W14_动作被拒绝的原因与修复前后接受率.png")

# ------------------------------------------------------------------ 图2 65k 检查点的验证（06 第4节）
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.5), gridspec_kw={"wspace": 0.30})
step = np.array([16384, 32768, 49152, 65536]) / 1e3
cvp = [0.796810, 0.781906, 0.785709, 0.794613]
qp = [1.493087, 1.501316, 1.499602, 1.494009]
ref_cv = {"共同初始布局": 0.835246, "规则重构": 0.778707, "随机重构": 0.827480, "搜索重构": 0.775644}
ref_q = {"共同初始布局": 1.465297, "规则重构": 1.502878, "随机重构": 1.471864, "搜索重构": 1.505491}
refc = {"共同初始布局": LIGHT, "规则重构": BLUE, "随机重构": ORANGE, "搜索重构": TEAL}
for ax, vals, ref, title in [(axs[0], cvp, ref_cv, "主体负载CV（越小越均匀）"), (axs[1], qp, ref_q, "覆盖等级得分Q（越大越好）")]:
    for k, v in ref.items():
        ax.axhline(v, color=refc[k] if k != "共同初始布局" else GREY, ls="--", lw=1.2, label=k)
    ax.plot(step, vals, "-o", color=PURPLE, lw=2, ms=6, label="区域重构PPO（训练到6.5万步）")
    ax.set_title(title)
    ax.set_xlabel("训练步数（千）")
    ax.set_xticks(step)
    ax.set_xticklabels(["16", "33", "49", "66"])
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=8.5)
fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.30)
S.save(fig, "W14_修复前训练到六万五千步的验证结果.png")

# ------------------------------------------------------------------ 图3 修复后五种做法共同考核（07 第3、6节）
M = ["初始\n布局", "规则\n重构", "随机\n重构", "搜索\n重构", "PPO\n重构"]
cv = [0.855564, 0.717996, 0.789587, 0.663655, 0.602222]
band = [18.03, 19.16, 19.57, 20.54, 23.68]
sec = [0.493029, 4.784578, 2.654028, 3.548662, 1.875337]
cols = [LIGHT, BLUE, ORANGE, TEAL, PURPLE]
fig, axs = plt.subplots(1, 3, figsize=(6.9, 3.6), gridspec_kw={"wspace": 0.42})
x = np.arange(5)
for ax, vals, title, fmt, yl, target, tl in [
        (axs[0], cv, "主体负载CV\n（越小越均匀）", "{:.2f}", (0, 1.0), 0.20, "目标0.20"),
        (axs[1], band, "落在均值±20%内\n的波束数", "{:.1f}", (0, 100), 90, "目标90"),
        (axs[2], sec, "每景已记录耗时\n（秒）", "{:.1f}", (0, 5.6), None, None)]:
    ax.bar(x, vals, color=cols, edgecolor=GREY, width=0.68)
    for xi, v in zip(x, vals):
        ax.text(xi, v + (yl[1] - yl[0]) * 0.012, fmt.format(v), ha="center", fontsize=8)
    if target is not None:
        ax.axhline(target, color=RED, ls="--", lw=1.2)
        ax.set_xlim(-0.6, 6.2)
        ax.text(6.15, target + (yl[1] - yl[0]) * 0.02, tl, color=RED, fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(M, fontsize=7.5)
    ax.set_ylim(*yl)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", visible=False)
fig.subplots_adjust(left=0.06, right=0.995, top=0.82, bottom=0.17)
S.save(fig, "W14_修复后五种做法共同考核.png")

# ------------------------------------------------------------------ 图4 PPO 对搜索的逐景配对（07 第4节）
fig, ax = plt.subplots(figsize=(6.6, 3.3))
rows = [("覆盖等级得分Q", (70, 0, 30)), ("主体负载CV", (71, 0, 29)), ("落在均值±20%内的波束数", (61, 13, 26))]
yy = np.arange(3)[::-1]
left = np.zeros(3)
for k, (c, lab) in enumerate([(GREEN, "PPO更好"), (LIGHT, "持平"), (RED, "PPO更差")]):
    v = np.array([r[1][k] for r in rows])
    ax.barh(yy, v, left=left, color=c, height=0.58, label=lab)
    for yi, l, w in zip(yy, left, v):
        if w >= 6:
            ax.text(l + w / 2, yi, str(w), ha="center", va="center", fontsize=9, color="k" if c == LIGHT else "white")
    left += v
ax.set_yticks(yy)
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0, 100)
ax.set_xlabel("场景数（共100个新场景，PPO 对 搜索重构）")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
ax.grid(axis="y", visible=False)
fig.subplots_adjust(left=0.31, right=0.99, top=0.88, bottom=0.2)
S.save(fig, "W14_PPO与搜索重构的逐景配对.png")

# ------------------------------------------------------------------ 图5 耗时分段（07 第6节表）
seg = {
    "共同初始化": [0.492741] * 5,
    "策略选择": [0, 4.154766, 2.041139, 1.052656, 1.265860],
    "实际重构": [0, 0.136780, 0.119864, 2.002975, 0.116458],
    "最终核验": [0.000287, 0.000290, 0.000283, 0.000290, 0.000277],
}
names = ["共同初始布局", "规则重构", "随机重构", "搜索重构", "PPO重构"]
sc = [("共同初始化", GREY), ("策略选择", BLUE), ("实际重构", ORANGE), ("最终核验", GREEN)]
fig, ax = plt.subplots(figsize=(6.6, 3.5))
yy = np.arange(5)[::-1]
left = np.zeros(5)
for k, c in sc:
    ax.barh(yy, seg[k], left=left, color=c, height=0.6, label=k)
    left += np.array(seg[k])
for yi, t in zip(yy, left):
    ax.text(t + 0.08, yi, f"{t:.1f}", va="center", fontsize=9)
ax.set_yticks(yy)
ax.set_yticklabels(names)
ax.set_xlim(0, 5.6)
ax.set_xlabel("每个场景的已记录耗时（秒）")
ax.legend(loc="lower right", frameon=True, fontsize=9)
ax.grid(axis="y", visible=False)
fig.subplots_adjust(left=0.2, right=0.99, top=0.98, bottom=0.17)
S.save(fig, "W14_每景耗时分段.png")
