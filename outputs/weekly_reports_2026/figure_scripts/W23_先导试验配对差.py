"""W23 图：先导试验同一场景配对差和95%区间。
数字来源：实验报告19 第5节“完整母配对差附表”（40个场景，B-random、B-fixed、B-greedy、E-B、LP-B）。
SI差为百分点，SGM差为原值；区间是报告里的95%描述性bootstrap区间。
方法名：B=原模型，E=温度系数降到0.003，LP=槽数与功率联合建模。
"""
from _w22_24_common import *  # noqa: F401,F403

plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10.5,
                     "ytick.labelsize": 11, "legend.fontsize": 10, "axes.titlesize": 11.5})

# (名称, SI均值, SI下限, SI上限, SGM均值, SGM下限, SGM上限, SGM变好/变差场景数)
rows = [
    ("降温度系数 减 原模型", -0.370084, -0.820324, 0.068546, -0.009170, -0.019220, 0.002013, "12好/28差"),
    ("联合建模 减 原模型", -1.131737, -1.565189, -0.730668, -0.029310, -0.038424, -0.021162, "5好/35差"),
    ("原模型 减 随机选择", 6.493805, 5.279669, 7.650059, 0.316015, 0.301010, 0.329043, "40好/0差"),
    ("原模型 减 固定偏好", 2.895957, 2.164318, 3.630969, 0.410104, 0.386517, 0.431251, "40好/0差"),
    ("原模型 减 贪心规则", 3.379825, 2.521410, 4.214563, 0.334016, 0.319819, 0.347021, "40好/0差"),
]
cols = [ORANGE, GREEN, BLUE, BLUE, BLUE]
y = list(range(len(rows)))[::-1]

fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
for ax, k, xl, title in ((a, 1, "需求满足率之差（百分点）", "需求满足率"), (b, 4, "供需匹配程度之差", "供需匹配程度")):
    for yy, r, c in zip(y, rows, cols):
        m, lo, hi = r[k], r[k + 1], r[k + 2]
        ax.errorbar(m, yy, xerr=[[m - lo], [hi - m]], fmt="o", color=c, capsize=3.5, lw=1.8, ms=6)
    ax.axvline(0, color="#555", lw=1)
    ax.set_xlabel(xl)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
a.set_yticks(y)
a.set_yticklabels([r[0] for r in rows])
b.set_xlim(-0.07, 0.62)
for yy, r in zip(y, rows):
    b.text(0.62, yy, r[7], va="center", ha="right", fontsize=9.5, color="#444")
fig.tight_layout(w_pad=1.0)
save_png(fig, "W23_先导试验配对差.png", dpi=210)
