# -*- coding: utf-8 -*-
"""W21 图：强顺序重构能不能更早停下来省计算（12个训练场景）。

数据：first_stage_ai_search_20261013 第十三轮（报告26）分析结果，在归档目录
  E:/paper_archives/outputs/experiments/first_stage_ai_search_20261013/R13_archive_closure_01/objects/ 里按内容哈希存放的两个 csv：
  9881ab0d…（historical_attempt_proxies_by_mother.csv）每个场景的完整尝试次数、逐母事后最佳固定规则（强包络）的尝试次数、
        最早保持教师质量的尝试次数；
  64ffbd8e…（policy_summary.csv）八条固定停止规则各自能保住教师质量的场景数、12个场景合计要付出的尝试次数。
脚本里核对：合计 384 / 211 / 176 次，额外少 35 次，占 211 的 16.59%（原定门槛 20%）。
这里的“尝试次数”是历史轨迹里已经保存的核对次数，不是秒数，也不是实测加速。
"""
import pandas as pd
from _c1821 import *

O = E_EXP / "first_stage_ai_search_20261013/R13_archive_closure_01/objects"
px = pd.read_csv(O / "9881ab0d5d36b0548687f82e9a64b0bfb415e7ea20f111dbe5aa17c4acc1e219")
ps = pd.read_csv(O / "64ffbd8e06e5839feab327e029f2ba2e5fa76a1c6e9db8e837d4ad1efd08cd4a")
assert px.full_SHC_historical_kernel_records.sum() == 384
assert px.retrospective_strong_fixed_envelope.sum() == 211 and px.earliest_safe_oracle.sum() == 176
assert abs(35 / 211 - 0.165877) < 1e-5

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"wspace": 0.55, "width_ratios": [1.15, 1]})
ax = axes[0]
x = range(len(px))
ax.plot(x, px.full_SHC_historical_kernel_records, "o", color=C_LIGHT, ms=7, label="跑完整条轨迹")
ax.plot(x, px.retrospective_strong_fixed_envelope, "s", color=C_BLUE, ms=6, label="每个场景事后挑最好的固定规则")
ax.plot(x, px.earliest_safe_oracle, "^", color=C_RED, ms=7, label="最早能停的位置")
ax.set_xticks(list(x))
ax.set_xticklabels(px.original_index.astype(int), fontsize=9, rotation=0)
ax.set_xlabel("12个训练场景（编号）")
ax.set_ylabel("需要的尝试次数")
ax.set_ylim(-2, 50)
ax.legend(loc="upper center", frameon=False, fontsize=9.3, handletextpad=0.1, bbox_to_anchor=(0.5, 1.0))
ax.text(0.02, 0.45, "合计 384 / 211 / 176", transform=ax.transAxes, fontsize=9.5, color=C_GREY)

ax = axes[1]
names = {"cap_0": "不调整", "cap_4": "最多4次", "cap_8": "最多8次", "cap_16": "最多16次", "cap_32": "最多32次",
         "window_2": "连续2次没改进", "window_4": "连续4次没改进", "window_8": "连续8次没改进"}
d = ps[ps.policy.isin(names)].set_index("policy").loc[list(names)]
y = list(range(len(d)))[::-1]
for yi, (k, r) in zip(y, d.iterrows()):
    ax.barh(yi, r.quality_resource_safe_mothers, 0.6, color=C_GREEN if r.quality_resource_safe_mothers == 12 else C_TEAL)
    ax.text(r.quality_resource_safe_mothers + 0.2, yi, f"{int(r.quality_resource_safe_mothers)}个（共{int(r.total_paid_attempts_proxy)}次）",
            va="center", fontsize=9.3)
ax.set_yticks(y)
ax.set_yticklabels([names[k] for k in d.index], fontsize=9.5)
ax.set_xlim(0, 21)
ax.set_xticks([0, 4, 8, 12])
ax.set_xlabel("能保住教师质量的场景数（共12个）")
ax.grid(axis="y", visible=False)
ax.set_title("八条固定停止规则", fontsize=10.5)
save_png(fig, "W21_提前停止机会.png")
