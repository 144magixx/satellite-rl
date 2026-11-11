"""W15 画图：十二条覆盖路线的对比、两条学习路线的训练过程、连续时刻试验。
先运行 W15_extract.py 生成 data/W15/*.csv，再运行本脚本，图保存到 ../figures/。
"""
import os

import numpy as np
import pandas as pd

import _style as S
from _style import BLUE, GREEN, GREY, LIGHT, ORANGE, PURPLE, RED, TEAL
import matplotlib.pyplot as plt

S.setup()
D = os.path.join(S.DATA_DIR, "W15")

NAMES = {
    "R01": "参考压力增束", "R02": "候选集整数覆盖", "R03": "固定中心需求再归属",
    "R04": "热点拆分、轻载合并", "R05": "可变波束宽度", "R06": "运输松弛加整数修复",
    "R07": "区域拆除重建", "R08": "局部重构PPO", "R09": "从空布局构造PPO",
    "R10": "下游反馈重构", "R11": "需求扰动搜索", "R12": "时序重配置",
}
ORDER = list(NAMES)
LEARN = {"R08", "R09"}


def color_of(r):
    if r == "R04":
        return ORANGE
    if r in LEARN:
        return PURPLE
    return BLUE


st = pd.read_csv(os.path.join(D, "static_per_scene.csv"))
st["method_s"] = st.total_s - st.final_s - st.down_s

# ---------------------------------------------------------------- 图1 十二路线均衡与承载
g = st.groupby("route").agg(cv=("load_cv", "mean"), car=("carried_aircraft", "mean")).loc[ORDER]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 4.5), sharey=True, gridspec_kw={"wspace": 0.06})
y = np.arange(len(ORDER))[::-1]
cols = [color_of(r) for r in ORDER]
axs[0].barh(y, g.cv, color=cols, height=0.68)
axs[1].barh(y, g.car * 100, color=cols, height=0.68)
for yy, v in zip(y, g.cv):
    axs[0].text(v + 0.012, yy, f"{v:.2f}", va="center", fontsize=9)
for yy, v in zip(y, g.car * 100):
    axs[1].text(v + 0.6, yy, f"{v:.1f}", va="center", fontsize=9)
axs[0].set_yticks(y)
axs[0].set_yticklabels([NAMES[r] for r in ORDER])
axs[0].set_xlim(0, 1.15)
axs[1].set_xlim(60, 92)
axs[0].set_title("需求CV（越小越均匀）")
axs[1].set_title("逐机承载 %（越大越好）")
axs[0].grid(axis="y", visible=False)
axs[1].grid(axis="y", visible=False)
axs[1].tick_params(axis="y", length=0)
fig.text(0.5, 0.005, "橙色：热点拆分、轻载合并；紫色：两条学习路线；蓝色：其余非学习路线", ha="center", fontsize=9, color=GREY)
fig.subplots_adjust(left=0.30, right=0.985, top=0.93, bottom=0.09)
S.save(fig, "W15_十二路线均衡与承载.png")

# ---------------------------------------------------------------- 图2 低中高需求
tier = st.groupby(["route", "mult"]).agg(cv=("load_cv", "mean"), car=("carried_aircraft", "mean")).reset_index()
show = ["R01", "R04", "R05", "R11", "R08", "R09"]
colr = {"R01": GREY, "R04": ORANGE, "R05": TEAL, "R11": GREEN, "R08": PURPLE, "R09": RED}
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.7), gridspec_kw={"wspace": 0.28})
for r in show:
    t = tier[tier.route == r].sort_values("mult")
    axs[0].plot(t.mult, t.cv, "-o", color=colr[r], lw=1.8, ms=4.5, label=NAMES[r])
    axs[1].plot(t.mult, t.car * 100, "-o", color=colr[r], lw=1.8, ms=4.5)
for a in axs:
    a.set_xlim(0.68, 1.32)
    a.set_xticks([0.75, 1.0, 1.25])
    a.set_xticklabels(["低需求\n×0.75", "中需求\n×1", "高需求\n×1.25"])
axs[0].set_title("需求CV（越小越均匀）")
axs[1].set_title("逐机承载 %（越大越好）")
fig.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.30)
S.save(fig, "W15_低中高需求下的取舍.png")

# ---------------------------------------------------------------- 图3 每景耗时分段
tm = st.groupby("route").agg(m=("method_s", "mean"), d=("down_s", "mean"), f=("final_s", "mean"),
                             tot=("total_s", "mean")).loc[ORDER]
p95 = st.groupby("route").total_s.quantile(0.95).loc[ORDER]
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.barh(y, tm.m, color=BLUE, height=0.66, label="方法本身（生成布局、搜索或学习策略）")
ax.barh(y, tm.d + tm.f, left=tm.m, color=ORANGE, height=0.66, label="共同下游（分配频率槽和功率、物理核验）")
ax.plot(p95, y, "D", color="k", ms=3.8, label="95%分位")
for yy, v, q in zip(y, tm.tot, p95):
    ax.text(41.5, yy, f"{v:.1f} / {q:.1f}", va="center", ha="right", fontsize=9)
ax.text(41.5, 12.0, "均值 / 95%分位", va="center", ha="right", fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels([NAMES[r] for r in ORDER])
ax.set_xlabel("每个场景的耗时（秒，均值）")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, 42)
ax.set_ylim(-0.6, 12.6)
ax.legend(loc="center right", bbox_to_anchor=(0.84, 0.40), frameon=True, fontsize=9)
fig.subplots_adjust(left=0.30, right=0.985, top=0.98, bottom=0.11)
S.save(fig, "W15_每景耗时分段.png")

# ---------------------------------------------------------------- 图4 波束数分布
fig, ax = plt.subplots(figsize=(6.6, 4.4))
data = [st[st.route == r].N.values for r in ORDER]
bp = ax.boxplot(data[::-1], vert=False, whis=(0, 100), widths=0.6, patch_artist=True,
                medianprops=dict(color="k", lw=1.4), flierprops=dict(markersize=3))
for patch, r in zip(bp["boxes"], ORDER[::-1]):
    patch.set_facecolor(color_of(r))
    patch.set_alpha(0.75)
ax.axvline(50, color=RED, ls="--", lw=1.3)
ax.text(51, 14.0, "第二阶段模型训练的最小束数50", color=RED, fontsize=9, va="center")
ax.set_yticks(range(1, 13))
ax.set_yticklabels([NAMES[r] for r in ORDER[::-1]])
ax.set_xlabel("每个场景实际用到的波束数（须线为最小到最大值）")
ax.set_xlim(30, 125)
for i, r in enumerate(ORDER[::-1]):
    n_low = int((st[st.route == r].N < 50).sum())
    ax.text(116, i + 1, f"{n_low}", va="center", ha="center", fontsize=9)
ax.text(116, 13.2, "不足50束\n的场景数", ha="center", va="center", fontsize=9)
ax.set_ylim(0.4, 14.5)
ax.grid(axis="y", visible=False)
fig.subplots_adjust(left=0.30, right=0.985, top=0.985, bottom=0.11)
S.save(fig, "W15_波束数分布.png")

# ---------------------------------------------------------------- 图5 相对参考增束路线逐景改善
base = st[st.route == "R01"].set_index(["scene_id", "mult"])
rows = []
for r in ORDER[1:]:
    t = st[st.route == r].set_index(["scene_id", "mult"]).loc[base.index]
    rows.append((r, int((t.load_cv < base.load_cv - 1e-9).sum()), int((t.carried_aircraft > base.carried_aircraft + 1e-9).sum())))
rr = pd.DataFrame(rows, columns=["route", "cv", "car"])
fig, ax = plt.subplots(figsize=(6.6, 4.3))
yy = np.arange(len(rr))[::-1]
ax.barh(yy + 0.19, rr.cv, height=0.36, color=BLUE, label="需求CV比参考增束更小的场景数")
ax.barh(yy - 0.19, rr.car, height=0.36, color=ORANGE, label="逐机承载比参考增束更高的场景数")
for a, c, b in zip(yy, rr.cv, rr.car):
    ax.text(c + 3, a + 0.19, str(c), va="center", fontsize=9)
    ax.text(b + 3, a - 0.19, str(b), va="center", fontsize=9)
ax.axvline(150, color=GREY, ls=":", lw=1.2)
ax.set_yticks(yy)
ax.set_yticklabels([NAMES[r] for r in rr.route])
ax.set_xlim(0, 330)
ax.set_xlabel("共300个场景（100个母场景×三档需求）里的场景数；虚线为一半")
ax.grid(axis="y", visible=False)
ax.legend(loc="upper right", fontsize=9, frameon=True)
fig.subplots_adjust(left=0.30, right=0.985, top=0.985, bottom=0.12)
S.save(fig, "W15_相对参考增束的逐景改善.png")
print(rr.to_string())

# ---------------------------------------------------------------- 图6 局部重构PPO训练
up = pd.read_csv(os.path.join(D, "train_R08_updates.csv"))
ab = pd.read_csv(os.path.join(D, "train_R08_action_blocks.csv"))
va = pd.read_csv(os.path.join(D, "train_R08_validation.csv"))
fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.0), gridspec_kw={"wspace": 0.32, "hspace": 0.52})
a = axs[0, 0]
a.plot(ab.env_step_end / 1e3, ab.stop / ab.n * 100, "-o", color=PURPLE, ms=4)
a.set_title("动作里“立即停止”的占比 %")
a.set_ylim(0, 105)
a = axs[0, 1]
a.plot(ab.env_step_end / 1e3, ab.reward_sum / ab.n, "-o", color=ORANGE, ms=4)
a.axhline(1.995, color=GREY, ls="--", lw=1)
a.text(100, 1.7, "一上来就停止的回报 1.995", fontsize=9, color=GREY)
a.set_title("每一步的平均回报")
a = axs[1, 0]
a.plot(up.env_step / 1e3, up.entropy, color=BLUE, lw=1.8)
a.set_yscale("log")
a.set_title("策略熵（越小越确定）")
a = axs[1, 1]
a.plot(va.env_step / 1e3, va.mean_load_cv, "-o", color=GREEN, ms=4)
a.set_ylim(0.7, 0.9)
a.set_title("固定验证60景的平均负载CV")
a.text(60, 0.735, "16个验证点数值完全相同", fontsize=9, color=GREY)
for a in axs.ravel():
    a.set_xlabel("训练步数（千）")
S.save(fig, "W15_局部重构PPO训练过程.png")

# ---------------------------------------------------------------- 图7 构造PPO训练
ep = pd.read_csv(os.path.join(D, "train_R09_episodes.csv"))
ep = ep[ep.completed == 1].sort_values("terminal_env_step")
up9 = pd.read_csv(os.path.join(D, "train_R09_updates.csv"))
va9 = pd.read_csv(os.path.join(D, "train_R09_validation.csv"))
roll = ep.episode_return.rolling(100, min_periods=20).mean()
rollcv = ep.load_cv.rolling(100, min_periods=20).mean()
fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.0), gridspec_kw={"wspace": 0.32, "hspace": 0.52})
a = axs[0, 0]
a.scatter(ep.terminal_env_step / 1e3, ep.episode_return, s=2, color=LIGHT, rasterized=True)
a.plot(ep.terminal_env_step / 1e3, roll, color=RED, lw=1.8, label="最近100回合平均")
a.set_title("每个完整回合的回报")
a.legend(loc="lower right", frameon=True, fontsize=9)
a = axs[0, 1]
a.plot(ep.terminal_env_step / 1e3, rollcv, color=PURPLE, lw=1.8)
a.set_title("训练回合终态的负载CV\n（最近100回合平均）")
a = axs[1, 0]
a.plot(up9.env_step / 1e3, up9.entropy, color=BLUE, lw=1.8)
a.set_title("策略熵（越小越确定）")
a = axs[1, 1]
a.plot(va9.env_step / 1e3, va9.mean_load_cv, "-o", color=GREEN, ms=4)
a.set_title("固定验证60景的平均负载CV")
for a in axs.ravel():
    a.set_xlabel("训练步数（千）")
S.save(fig, "W15_构造PPO训练过程.png")
print("R09 验证 CV 首/末", va9.mean_load_cv.iloc[0], va9.mean_load_cv.iloc[-1])

# ---------------------------------------------------------------- 图8 连续时刻试验
sq = pd.read_csv(os.path.join(D, "sequence_per_frame.csv"))
a4 = sq[sq.route == "R04"].set_index(["sequence_id", "time_utc", "mult"])
a12 = sq[sq.route == "R12"].set_index(["sequence_id", "time_utc", "mult"]).loc[a4.index]
dcar = (a12.carried_aircraft - a4.carried_aircraft) * 100
dcv = a12.load_cv - a4.load_cv
dn = a12.N - a4.N
fig, axs = plt.subplots(1, 3, figsize=(6.6, 3.0), gridspec_kw={"wspace": 0.38})
axs[0].hist(dcar, bins=30, color=BLUE)
axs[0].axvline(0, color="k", lw=1)
axs[0].set_title(f"逐机承载差\n（百分点，均值{dcar.mean():.1f}）")
axs[1].hist(dcv, bins=30, color=ORANGE)
axs[1].axvline(0, color="k", lw=1)
axs[1].set_title(f"需求CV差\n（均值{dcv.mean():+.2f}）")
axs[2].hist(dn, bins=30, color=GREEN)
axs[2].axvline(0, color="k", lw=1)
axs[2].set_title(f"波束数差\n（均值{dn.mean():.1f}）")
for a in axs:
    a.set_xlabel("连续调用 减 逐帧重新规划")
axs[0].set_ylabel("帧数（共360帧）")
fig.subplots_adjust(left=0.09, right=0.99, top=0.80, bottom=0.20)
S.save(fig, "W15_连续时刻试验.png")
r12 = sq[sq.route == "R12"]
print("时序：后续帧", int((r12.temporal_mode == "sequence_warm_start").sum()),
      "其中回退重建", int(r12[r12.temporal_mode == "sequence_warm_start"].warm_start_fallback.sum()))
print("承载差均值", dcar.mean(), "CV差均值", dcv.mean(), "N差均值", dn.mean(),
      "承载提高/下降/持平", int((dcar > 1e-6).sum()), int((dcar < -1e-6).sum()), int((dcar.abs() <= 1e-6).sum()))




