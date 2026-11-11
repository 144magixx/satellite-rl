"""W23 图：固定前面已分配的波束，只改当前这一束的功率时，前缀SGM的平均增量。
数字来源：实验报告21 第4节两张表
  - 嵌套处理表：D1-D0 +0.000005226（14/60状态）、D2-D1 +0.002602346（30/60状态）；
  - 正文：D0相对原执行功率的平均前缀SGM增量0.000948832，35/60状态超过千分之一的百万分之一（1e-6）。
D0=原十档（5、10、…、50瓦），D1=5～50瓦细化到0.5瓦间隔（仍保持5瓦下限），D2=0.5～50瓦。
20个开发场景×3个位置共60个状态，前缀结果不是整景终态成绩。
"""
from _w22_24_common import *  # noqa: F401,F403

plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10.5,
                     "ytick.labelsize": 10.5, "legend.fontsize": 10, "axes.titlesize": 11.5})

labels = ["只在原来十档\n里换功率", "细化到0.5瓦间隔\n（下限仍是5瓦）", "再放宽到\n5瓦以下"]
inc = [0.000948832, 0.000005226, 0.002602346]
cnt = [35, 14, 30]
cols = [GRAY, BLUE, ORANGE]
x = [0, 1, 2]

fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.7))
a.bar(x, inc, color=cols, width=0.6)
for xx, v in zip(x, inc):
    a.text(xx, v + 0.00007, f"+{v:.4f}" if v > 1e-4 else f"+{v:.6f}", ha="center", fontsize=10)
a.set_xticks(x)
a.set_xticklabels(labels, fontsize=9.5)
a.set_ylim(0, 0.0032)
a.set_title("供需匹配程度的平均增量", loc="left")
b.bar(x, cnt, color=cols, width=0.6)
for xx, v in zip(x, cnt):
    b.text(xx, v + 0.8, f"{v}/60", ha="center", fontsize=10)
b.set_xticks(x)
b.set_xticklabels(labels, fontsize=9.5)
b.set_ylim(0, 44)
b.set_title("有提高的状态数（共60个）", loc="left")
for ax in (a, b):
    ax.grid(axis="x", visible=False)
fig.tight_layout(w_pad=1.5)
save_png(fig, "W23_功率档位细化与下限放松.png", dpi=210)
