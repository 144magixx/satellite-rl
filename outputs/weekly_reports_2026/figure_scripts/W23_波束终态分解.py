"""W23 图：15万步评价里每个波束的终态，三组（均不带余量摘要）对比。
数字来源：实验报告23 第3.1节表：容量超过需求两倍的波束、容量不足需求90%的波束、选了5瓦以下的波束（占比%）、每束平均功率（瓦）。
三组：原档位（对照）、原档位＋0.5瓦、0.5～50瓦全部细化。
"""
from _w22_24_common import *  # noqa: F401,F403

plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10.5,
                     "ytick.labelsize": 10.5, "legend.fontsize": 10, "axes.titlesize": 11})

metrics = ["容量超过需求\n两倍的波束", "容量不足需求\n九成的波束", "选了5瓦以下\n的波束"]
data = {
    "原档位（对照）": ([34.6, 8.3, 0.0], 24.7, GRAY),
    "原档位＋0.5瓦": ([23.6, 8.8, 30.6], 18.6, GREEN),
    "0.5～50瓦全部细化": ([24.2, 9.0, 51.6], 16.9, ORANGE),
}
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.7), gridspec_kw={"width_ratios": [2.3, 1]})
w = 0.26
for k, (name, (vals, avg, c)) in enumerate(data.items()):
    xs = [i + (k - 1) * w for i in range(3)]
    a.bar(xs, vals, w, color=c, label=name)
    for xx, v in zip(xs, vals):
        a.text(xx, v + 0.9, f"{v:.1f}", ha="center", fontsize=9)
    b.bar(k, avg, 0.6, color=c)
    b.text(k, avg + 0.5, f"{avg:.1f}", ha="center", fontsize=10)
a.set_xticks(range(3))
a.set_xticklabels(metrics, fontsize=10)
a.set_ylim(0, 62)
a.set_title("占全部波束的比例（%）", loc="left", fontsize=10.5)
a.legend(loc="upper left", frameon=False, fontsize=9.5)
b.set_xticks(range(3))
b.set_xticklabels(["原档位", "加0.5瓦", "全部细化"], fontsize=9.5)
b.set_ylim(0, 30)
b.set_title("每束平均功率（W）", loc="left", fontsize=10.5)
for ax in (a, b):
    ax.grid(axis="x", visible=False)
fig.tight_layout(w_pad=1.2)
save_png(fig, "W23_波束终态分解.png", dpi=210)
