"""W24 图：第一阶段动态重配置，各策略的最忙波束超出下界多少架、每分钟有多大比例的飞机被换了波束（从原始记录重画）。
数据（只读）：D:/paper/outputs/experiments/s1_fix_*/B/eval_summary.csv、eval_summary_fixed.csv
  开发日 16 段、确认日 32 段，每段 30 分钟，每分钟一帧，波束预算 100。
  列 mean_gap＝最忙波束比理论下界 ceil(n/N) 多几架，mean_churn＝两帧都在场的飞机里换了波束的比例。
策略对应：B0 每帧全部重算；B1 只修补；B3(param=5) 每 5 帧全重算其余修补；B2(param=1) 峰值比上次修复后高出 1 架以上才局部修复；
  B2+(param=1) 同上再加 60 秒位置预测；F1 每分钟轻量局部修复（固定动作 1）；
  PPO（不带预测）：csv 里标 B4-forecast 的 3 个种子（λ=0.2）取平均——这个标签在 csv 里与含义相反，
  它的平均值 4.99 / 7.8%（开发）、7.41 / 7.6%（确认）与报告 04 表格里“PPO，不带预测（3 个种子平均）”一致。
"""
from _w24_common import *

B = S1FIX / "B"
d = pd.read_csv(B / "eval_summary.csv")
f = pd.read_csv(B / "eval_summary_fixed.csv")


def pick(role):
    g = lambda arm, p: d[(d.role == role) & (d.arm == arm) & (np.isclose(d.param, p))].iloc[0]
    ppo = d[(d.role == role) & (d.arm == "B4-forecast") & np.isclose(d.param, 0.2)]
    fx = f[(f.role == role) & (f.arm == "F1")].iloc[0]
    return {
        "每帧全部重算": g("B0", 0),
        "只修补，不重算": g("B1", 0),
        "每5帧重算一次": g("B3", 5),
        "超阈值才修复": g("B2", 1),
        "超阈值才修复＋位置预测": g("B2+", 1),
        "每分钟轻量局部修复": fx,
        "PPO（不带预测）": ppo[["mean_gap", "mean_churn"]].mean(),
    }


dev, conf = pick("dev"), pick("conf")
names = list(dev.keys())
ypos = np.arange(len(names))[::-1]
hl = names.index("每分钟轻量局部修复")

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True, gridspec_kw={"wspace": 0.08})
for ax, col, scale, xl, fmt in [(axes[0], "mean_gap", 1, "最忙波束比下界多几架（越低越好）", "{:.2f}"),
                                (axes[1], "mean_churn", 100, "换了波束的飞机比例（%，越低越好）", "{:.1f}")]:
    for off, data, c, lab in [(0.19, dev, BLUE, "开发日"), (-0.19, conf, "#9DC3E6", "确认日")]:
        vals = [data[n][col] * scale for n in names]
        bars = ax.barh(ypos + off, vals, height=0.36, color=c, label=lab)
        for y, v in zip(ypos + off, vals):
            ax.text(v + 0.012 * max(max(dev[n][col] * scale for n in names), 1), y, fmt.format(v + 1e-9), va="center", fontsize=8.6)
    ax.set_xlabel(xl, fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
axes[0].set_xlim(0, 17.5)
axes[1].set_xlim(0, 58)
axes[0].set_yticks(ypos)
axes[0].set_yticklabels(names, fontsize=10)
axes[0].get_yticklabels()[hl].set_fontweight("bold")
axes[0].legend(loc="lower right", fontsize=9.5, frameon=False)
save_png(fig, "W24_动态重配置各策略对比.png", **SAVE_KW)

for n in names:
    print(f"{n}: 开发 {dev[n]['mean_gap']:.2f}架 {dev[n]['mean_churn']*100:.1f}% | 确认 {conf[n]['mean_gap']:.2f}架 {conf[n]['mean_churn']*100:.1f}%")
