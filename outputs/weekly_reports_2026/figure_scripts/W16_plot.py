"""W16 画图：六种做法的独立考核对比、两个PPO的训练过程。
先运行 W16_extract.py 生成 data/W16/*.csv，再运行本脚本，图保存到 ../figures/。
"""
import os

import numpy as np
import pandas as pd

import _style as S
from _style import BLUE, GREEN, GREY, LIGHT, ORANGE, PURPLE, RED, TEAL
import matplotlib.pyplot as plt

S.setup()
D = os.path.join(S.DATA_DIR, "W16")

ORDER = ["I0", "H1", "H2", "BC", "P0", "P1"]
FULL = {"I0": "共同初解（不调整）", "H1": "即时规则", "H2": "两步搜索", "BC": "行为克隆",
        "P0": "PPO（从零训练）", "P1": "PPO（行为克隆后再训练）"}
SHORT = {"I0": "共同\n初解", "H1": "即时\n规则", "H2": "两步\n搜索", "BC": "行为\n克隆", "P0": "PPO\n从零", "P1": "克隆后\nPPO"}
COL = {"I0": LIGHT, "H1": BLUE, "H2": TEAL, "BC": GREEN, "P0": ORANGE, "P1": PURPLE}

st = pd.read_csv(os.path.join(D, "static_per_scene.csv"))
st["pub_s"] = st.algo_s + st.published_phys_s
# 先把同一母场景的三档需求平均，再对 100 个母场景求平均/做配对
mo = st.groupby(["method", "mother"]).mean(numeric_only=True).reset_index()
g = mo.groupby("method").mean(numeric_only=True).loc[ORDER]
med = mo.groupby("method")[["total_s", "pub_s"]].median().loc[ORDER]
p95 = mo.groupby("method")[["total_s", "pub_s"]].quantile(0.95).loc[ORDER]

# ---------------------------------------------------------------- 图1 六种做法总体对比
fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.2), gridspec_kw={"wspace": 0.30, "hspace": 0.55})
x = np.arange(6)
cols = [COL[m] for m in ORDER]


def bars(ax, vals, title, fmt, ylim=None, log=False):
    ax.bar(x, vals, color=cols, width=0.68)
    for xi, v in zip(x, vals):
        ax.text(xi, v * (1.08 if log else 1.0) + (0 if log else (ax.get_ylim()[1] * 0.01)), fmt.format(v),
                ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in ORDER], fontsize=8.5)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    if log:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)


bars(axs[0, 0], g.cv.values, "负载CV（越小越均匀）", "{:.2f}", (0, 1.0))
bars(axs[0, 1], g.carried_aircraft.values * 100, "逐机承载 %（越大越好）", "{:.1f}", (60, 90))
bars(axs[1, 0], g.best_j.values, "参考目标J（越小越好）", "{:.3f}", (0.003, 0.6), log=True)
bars(axs[1, 1], g.pub_s.values, "每景耗时（秒）", "{:.1f}", (0, 3.0))
S.save(fig, "W16_六种做法总体对比.png")
print(g[["cv", "carried_aircraft", "best_j", "pub_s"]].round(4).to_string())

# ---------------------------------------------------------------- 图2 三档需求
tier = st.groupby(["method", "mult"]).mean(numeric_only=True).reset_index()
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.7), gridspec_kw={"wspace": 0.28})
for m in ORDER:
    t = tier[tier.method == m].sort_values("mult")
    axs[0].plot(t.mult, t.cv, "-o", color=COL[m] if m != "I0" else GREY, lw=1.8, ms=4.5, label=FULL[m])
    axs[1].plot(t.mult, t.carried_aircraft * 100, "-o", color=COL[m] if m != "I0" else GREY, lw=1.8, ms=4.5)
for a in axs:
    a.set_xlim(0.68, 1.32)
    a.set_xticks([0.75, 1.0, 1.25])
    a.set_xticklabels(["低需求\n×0.75", "中需求\n×1", "高需求\n×1.25"])
axs[0].set_title("负载CV（越小越均匀）")
axs[1].set_title("逐机承载 %（越大越好）")
fig.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=9)
fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.30)
S.save(fig, "W16_三档需求下的取舍.png")

# ---------------------------------------------------------------- 图3 同母场景配对
pairs = [("BC", "I0"), ("P0", "I0"), ("P1", "I0"), ("BC", "H1"), ("P0", "H1"), ("P1", "H1"), ("P0", "BC"), ("P1", "BC")]
piv = {m: mo[mo.method == m].set_index("mother") for m in ORDER}
rows = []
for a, b in pairs:
    dj = piv[a].best_j - piv[b].best_j
    dc = piv[a].carried_aircraft - piv[b].carried_aircraft
    dv = piv[a].cv - piv[b].cv
    rows.append(dict(pair=f"{FULL[a]}\n对 {FULL[b]}",
                     j=((dj < -1e-8).sum(), (dj.abs() <= 1e-8).sum(), (dj > 1e-8).sum()),
                     c=((dc > 1e-8).sum(), (dc.abs() <= 1e-8).sum(), (dc < -1e-8).sum()),
                     cv=((dv < -1e-8).sum(), (dv.abs() <= 1e-8).sum(), (dv > 1e-8).sum()),
                     dj=dj.mean(), dc=dc.mean() * 100, dv=dv.mean()))
fig, axs = plt.subplots(1, 3, figsize=(6.8, 4.4), sharey=True, gridspec_kw={"wspace": 0.18})
yy = np.arange(len(rows))[::-1]
for ax, key, title in zip(axs, ["j", "cv", "c"], ["参考目标J", "负载CV", "逐机承载"]):
    left = np.zeros(len(rows))
    for k, (lab, colr) in enumerate([("改善", GREEN), ("持平", LIGHT), ("退步", RED)]):
        v = np.array([r[key][k] for r in rows])
        ax.barh(yy, v, left=left, color=colr, height=0.66, label=lab if ax is axs[0] else None)
        for yi, l, w in zip(yy, left, v):
            if w >= 8:
                ax.text(l + w / 2, yi, str(w), ha="center", va="center", fontsize=8.5, color="k" if colr == LIGHT else "white")
        left += v
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("母场景数")
    ax.grid(axis="y", visible=False)
axs[0].set_yticks(yy)
axs[0].set_yticklabels([r["pair"] for r in rows], fontsize=8.5)
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.6, -0.01))
fig.subplots_adjust(left=0.28, right=0.975, top=0.93, bottom=0.17)
S.save(fig, "W16_同母场景配对结果.png")
for r in rows:
    print(r["pair"].replace("\n", " "), r["j"], r["c"], r["cv"], f"dJ={r['dj']:+.5f} d承载={r['dc']:+.2f}pp dCV={r['dv']:+.4f}")

# ---------------------------------------------------------------- 图4 耗时
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.6), sharey=True, gridspec_kw={"wspace": 0.10})
for ax, key, title in zip(axs, ["total_s", "pub_s"], ["完整记录（含多个布局的物理核验）", "算法加一次发布布局核验"]):
    ax.bar(x, med[key], color=cols, width=0.66)
    ax.errorbar(x, med[key], yerr=[np.zeros(6), p95[key] - med[key]], fmt="none", ecolor="k", capsize=3, lw=1.2)
    for xi, m, p in zip(x, med[key], p95[key]):
        ax.text(xi, p + 0.08, f"{m:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in ORDER], fontsize=8.5)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 4.8)
axs[0].set_ylabel("每景耗时（秒）：柱为中位数，须到95%分位")
S.save(fig, "W16_耗时对比.png")
print(med.round(3).to_string())
print(p95.round(3).to_string())

# ---------------------------------------------------------------- 图5 PPO 训练回报
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.5), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, tag, name, c in zip(axs, ["P0", "P1"], ["PPO（从零训练）", "PPO（行为克隆后再训练）"], [ORANGE, PURPLE]):
    ep = pd.read_csv(os.path.join(D, f"train_{tag}_episodes.csv"))
    ep = ep[ep.completed == 1].sort_values("terminal_env_step")
    ax.scatter(ep.terminal_env_step / 1e3, ep.episode_return, s=1.5, color=LIGHT, rasterized=True)
    ax.plot(ep.terminal_env_step / 1e3, ep.episode_return.rolling(100, min_periods=20).mean(), color=c, lw=1.8,
            label="最近100回合平均")
    ax.set_title(f"{name}\n（{len(ep)}个完整回合）", fontsize=10)
    ax.set_xlabel("训练步数（千）")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    print(tag, "完整回合", len(ep), "回报末段均值", ep.episode_return.tail(500).mean())
axs[0].set_ylabel("回合回报（最好J改善量×100）")
fig.subplots_adjust(left=0.12, right=0.99, top=0.86, bottom=0.16)
S.save(fig, "W16_PPO训练回报.png")

# ---------------------------------------------------------------- 图6 验证与行为克隆拟合
v0 = pd.read_csv(os.path.join(D, "train_P0_validation.csv"))
v1 = pd.read_csv(os.path.join(D, "train_P1_validation.csv"))
bc = pd.read_csv(os.path.join(D, "bc_updates.csv"))
u0 = pd.read_csv(os.path.join(D, "train_P0_updates.csv"))
u1 = pd.read_csv(os.path.join(D, "train_P1_updates.csv"))
fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.6), gridspec_kw={"wspace": 0.34, "hspace": 0.85})
a = axs[0, 0]
a.plot(v0.env_step / 1e3, v0.mean_best_j, "-o", color=ORANGE, ms=3.5, label="从零训练")
a.plot(v1.env_step / 1e3, v1.mean_best_j, "-s", color=PURPLE, ms=3.5, label="克隆后再训练")
a.set_title("固定验证的平均目标J\n（越小越好）")
fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
a = axs[0, 1]
a.plot(v0.env_step / 1e3, v0.first_step_stop_count, "-o", color=ORANGE, ms=3.5)
a.plot(v1.env_step / 1e3, v1.first_step_stop_count, "-s", color=PURPLE, ms=3.5)
a.set_title("验证里第一步就停止\n的场景数（共60条）")
a = axs[1, 0]
a.plot(bc.epoch, bc.mean_cross_entropy, "-o", color=GREEN, ms=3.5)
a.set_title("行为克隆的拟合误差\n（交叉熵，20轮）")
a.set_xlabel("训练轮数")
a = axs[1, 1]
a.plot(u0.env_step / 1e3, u0.entropy, color=ORANGE, lw=1.6)
a.plot(u1.env_step / 1e3, u1.entropy, color=PURPLE, lw=1.6)
a.set_title("策略熵\n（越小越确定）")
for a in (axs[0, 0], axs[0, 1], axs[1, 1]):
    a.set_xlabel("训练步数（千）")
fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.17)
S.save(fig, "W16_验证曲线与行为克隆.png")
print("P0 验证J首末", v0.mean_best_j.iloc[0], v0.mean_best_j.iloc[-1], "P1", v1.mean_best_j.iloc[0], v1.mean_best_j.iloc[-1])
print("P1 首步停止", v1.first_step_stop_count.tolist())
print("BC 交叉熵首末", bc.mean_cross_entropy.iloc[0], bc.mean_cross_entropy.iloc[-1])

# ---------------------------------------------------------------- 图7 逐机承载分布
fig, ax = plt.subplots(figsize=(6.6, 3.7))
for m in ORDER:
    v = np.sort(st[st.method == m].carried_aircraft.values) * 100
    ax.plot(np.arange(1, len(v) + 1), v, color=COL[m] if m != "I0" else GREY, lw=1.8, label=FULL[m])
ax.set_xlabel("把各做法的300条结果按承载从低到高排序（横轴为名次）")
ax.set_ylabel("逐机承载 %")
ax.legend(loc="upper left", frameon=True, fontsize=9)
S.save(fig, "W16_逐机承载分布.png")




