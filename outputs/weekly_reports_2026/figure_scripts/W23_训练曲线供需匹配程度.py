# -*- coding: utf-8 -*-
"""W23 图：八组训练过程中，训练回合终态供需匹配程度（200回合滑动平均）。报告现成图改标签。
原图：D:\\paper\\outputs\\experiments\\s2_w31_20261017\\analysis\\fig1_training_curves.png
（实验报告23 第4节 图2，三幅并排，这里只裁出第一幅“终态SGM”）。
只改文字，曲线、网格、坐标轴一个像素不动、不缩放：
  - 子图标题“训练回合终态SGM（供需匹配，越高越好）”改成“训练回合终态供需匹配程度（越高越好）”；
  - 图内带编号P00-M0…P11-M1的图例整块盖白（被盖住的网格线按原样补回），
    在图下方另加一条中文图例：颜色表示档位，实线不带余量摘要、虚线带摘要；
  - 总图标题（带W31和种子编号）和另外两幅（需求满足率、总功率）未用。
配色：蓝=原档位，橙=只细化5瓦以上，绿=原档位＋0.5瓦，红=全区间细化（与原图图例的颜色对应）。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _img_edit import *  # noqa: E402,F401,F403

SRC = r"D:\paper\outputs\experiments\s2_w31_20261017\analysis\fig1_training_curves.png"
OUT = Path(__file__).resolve().parents[1] / "figures" / "W23_训练曲线供需匹配程度.png"

full = load(SRC)
arr = np.asarray(full).astype(int).min(axis=2)
im = full.crop((0, 60, 840, 706))   # 裁出第一幅：标题+绘图区+横轴标题；Y偏移60
OY = 60


def yy(y):
    return y - OY


# 1) 子图标题（原 x 227–652, y 71–109）
cover(im, (215, yy(66), 665, yy(99)))
put_text(im, "训练回合终态供需匹配程度（越高越好）", (65 + 829) / 2, yy(82), 22, anchor="mm")

# 2) 图内图例盖白并补回网格
x0, y0, x1, y1 = 550, 503, 826, 622
hrows = [r for r in range(y0, y1) if arr[r, 450] < 250]
vcols = [c for c in range(x0, x1) if arr[500, c] < 250]
print("grid rows", hrows, "grid cols", vcols)
patch = np.full((y1 - y0, x1 - x0, 3), 255, dtype=np.uint8)
rgb = np.asarray(full)
for r in hrows:
    patch[r - y0, :, :] = rgb[r, 450]
for c in vcols:
    patch[:, c - x0, :] = np.minimum(patch[:, c - x0, :], rgb[500, c])
im.paste(Image.fromarray(patch), (x0, yy(y0)))

# 3) 下方中文图例
entries = [("#4C72B0", "原档位"), ("#DD8452", "只细化5瓦以上"), ("#55A868", "原档位＋0.5瓦"),
           ("#C44E52", "全区间细化"), ("#444444", "实线：不带余量摘要"), ("#444444", "虚线：带余量摘要", "dash")]
entries[4] = ("#444444", "实线：不带摘要")
entries[5] = ("#444444", "虚线：带摘要", "dash")
im = legend_strip(im, entries, 22, rows=2, handle_w=44, gap=30, marker=False)
finish(im, OUT)
