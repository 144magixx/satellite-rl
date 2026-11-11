"""W23 图：只换一步动作、后面让固定模型走完后，最终结果的平均变化。
数字来源：实验报告22 “主要结果”开头两条：
  - 按当前奖励挑最大的动作：最终SI变化 -0.137900个百分点，SGM变化 +0.002103415，SI、SGM分别在6/20、11/20个场景变好；
  - 事后按最终SGM挑最好的动作：SI变化 -0.003033个百分点，SGM变化 +0.005160007，SGM在20/20个场景变好。
事先定的门槛：平均SGM至少提高0.01。20个开发场景×3个位置，相对原动作的参照。
"""
from _w22_24_common import *  # noqa: F401,F403

plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10.5,
                     "ytick.labelsize": 10.5, "legend.fontsize": 10, "axes.titlesize": 11.5})

labels = ["按当前奖励\n挑最大的动作", "事后按最终匹配\n程度挑最好的动作\n（只作参照）"]
sgm = [0.002103415, 0.005160007]
si = [-0.137900, -0.003033]
cols = [BLUE, ORANGE]

fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.6))
a.bar([0, 1], sgm, color=cols, width=0.5)
a.axhline(0.01, color=RED, ls="--", lw=1.4)
a.text(1.42, 0.0103, "事先定的门槛 +0.01", color=RED, ha="right", fontsize=10)
for xx, v in zip([0, 1], sgm):
    a.text(xx, v + 0.0003, f"+{v:.4f}", ha="center", fontsize=10.5)
a.text(0, -0.0017, "11/20个场景变好", ha="center", fontsize=9.5, color="#444")
a.text(1, -0.0017, "20/20个场景变好", ha="center", fontsize=9.5, color="#444")
a.set_xticks([0, 1])
a.set_xticklabels(labels, fontsize=9.5)
a.set_xlim(-0.55, 1.55)
a.set_ylim(-0.0025, 0.012)
a.axhline(0, color="#555", lw=0.8)
a.set_title("供需匹配程度的平均变化", loc="left")
b.bar([0, 1], si, color=cols, width=0.5)
for xx, v in zip([0, 1], si):
    b.text(xx, v - 0.012 if v < -0.05 else v + 0.005, f"{v:.3f}", ha="center", va="top" if v < -0.05 else "bottom", fontsize=10.5)
b.text(0, 0.014, "6/20个场景变好", ha="center", fontsize=9.5, color="#444")
b.set_xticks([0, 1])
b.set_xticklabels(labels, fontsize=9.5)
b.set_xlim(-0.55, 1.55)
b.set_ylim(-0.19, 0.04)
b.axhline(0, color="#555", lw=0.8)
b.set_title("需求满足率的平均变化（百分点）", loc="left")
for ax in (a, b):
    ax.grid(axis="x", visible=False)
fig.tight_layout(w_pad=1.5)
save_png(fig, "W23_单步干预平均效果.png", dpi=210)
