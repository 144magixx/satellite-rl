# -*- coding: utf-8 -*-
"""W20 图：把教师分组打乱以后，旧修复器和新修复器恢复到教师质量的案例数（24个案例）。

数据：first_stage_ai_search_20261013
  E:/paper_archives/outputs/experiments/first_stage_ai_search_20261013/round5/analysis/p0_attempt1/method168.csv（168行方法结果）
  E:/paper_archives/outputs/experiments/first_stage_ai_search_20261013/round4/analysis/b0_attempt1/case_method_metrics.csv（教师布局）
“达到教师质量”按报告的定义重算：不是回退到起点，且峰值 ≤ 教师×1.02、最重10%波束的平均负载 ≤ 教师×1.05、CV ≤ 教师×1.05。
脚本里核对：旧修复器 2/24、3/24，新修复器 12/24、14/24，教师原样输入 24/24，与报告17一致。
“教师质量”的预定门槛是 18/24。
"""
import pandas as pd
from _c1821 import *

D = E_EXP / "first_stage_ai_search_20261013"
m = pd.read_csv(D / "round5/analysis/p0_attempt1/method168.csv")
t = pd.read_csv(D / "round4/analysis/b0_attempt1/case_method_metrics.csv")
T = t[t.method == "TEACHER"].set_index("case_id")


def n_good(method):
    g = m[m.method == method]
    k = 0
    for _, r in g.iterrows():
        tt = T.loc[r.case_id]
        k += int(r.status == "success" and r.maxload <= 1.02 * tt.maxload
                 and r.top10_mean <= 1.05 * tt.top10_mean and r.CV <= 1.05 * tt.CV)
    return k, len(g)


cnt = {k: n_good(k) for k in ("D4_OWNER10", "D4_JOINT10", "D5_OWNER10", "D5_JOINT10", "D5_EXACT")}
assert [cnt[k][0] for k in ("D4_OWNER10", "D4_JOINT10", "D5_OWNER10", "D5_JOINT10", "D5_EXACT")] == [2, 3, 12, 14, 24]

fig, ax = plt.subplots(figsize=(7.2, 3.9))
groups = [("遮蔽10%飞机的归属", "D4_OWNER10", "D5_OWNER10"), ("遮蔽归属并把部分锚点\n换成相邻飞机", "D4_JOINT10", "D5_JOINT10")]
w = 0.34
for i, (lab, old, new) in enumerate(groups):
    ax.bar(i - w / 2, cnt[old][0], w, color=C_LIGHT, label="旧修复器" if i == 0 else None)
    ax.bar(i + w / 2, cnt[new][0], w, color=C_RED, label="新修复器" if i == 0 else None)
    ax.text(i - w / 2, cnt[old][0] + 0.4, f"{cnt[old][0]}/24", ha="center", fontsize=11)
    ax.text(i + w / 2, cnt[new][0] + 0.4, f"{cnt[new][0]}/24", ha="center", fontsize=11, fontweight="bold")
ax.bar(2, cnt["D5_EXACT"][0], 0.5, color=C_GREEN, label="教师分组原样输入")
ax.text(2, cnt["D5_EXACT"][0] + 0.4, "24/24", ha="center", fontsize=11)
ax.axhline(18, color=C_GREY, ls="--", lw=1.4)
ax.text(-0.45, 18.5, "预定门槛 18个", ha="left", fontsize=10.5, color=C_GREY)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels([g[0] for g in groups] + ["不打乱（对照）"], fontsize=10)
ax.set_ylim(0, 28)
ax.set_ylabel("恢复到教师质量的案例数")
ax.legend(loc="upper left", frameon=False, ncol=1, fontsize=10)
ax.grid(axis="x", visible=False)
save_png(fig, "W20_修复器达标数.png")
