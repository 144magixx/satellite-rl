# -*- coding: utf-8 -*-
"""W23 图：60个决策点上，原动作的当前容量与“5瓦下容量最小的合法组合”的容量（都除以当前束需求）。
报告现成图改标签。
原图：D:\\paper\\outputs\\experiments\\s2_bottleneck29_20261016\\report_analysis\\A0_01\\01_original_floor.png
（实验报告21 第3节 图1）。
只改文字，点、对角虚线、两条黑色虚线、网格一个像素不动、不缩放：
  - 标题、横轴标题、纵轴标题（两行）、图内图例的四行英文换成中文；
  - 图例里的点形（圆/方/三角）和点线样例原样保留；图例文字盖白后，被盖住的网格线和竖直虚线按原样补回。
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _img_edit import *  # noqa: E402,F401,F403

SRC = r"D:\paper\outputs\experiments\s2_bottleneck29_20261016\report_analysis\A0_01\01_original_floor.png"
OUT = Path(__file__).resolve().parents[1] / "figures" / "W23_原动作下限诊断.png"

im = load(SRC)
arr = np.asarray(im).copy()          # 只读参照，用来补网格


def restore(box):
    """盖白后把网格线（水平3条、竖直2条）和竖直虚线补回，等于没有这段文字时的样子。"""
    x0, y0, x1, y1 = box
    out = np.asarray(im).copy()
    out[y0:y1, x0:x1] = 255
    for yy in (74, 75, 76, 196, 197, 198, 317, 318, 319):
        if y0 <= yy < y1:
            out[yy, x0:x1] = arr[yy, 1000]
    for xx in (705, 706, 707, 983, 984, 985):
        if x0 <= xx < x1:
            col = arr[336, xx]
            out[y0:y1, xx] = np.minimum(out[y0:y1, xx], col)
    # 竖直虚线（x=428..430，周期约17.67像素）：从下方干净处按整周期搬上来
    for yy in range(y0, y1):
        k = math.ceil((y1 + 2 - yy) / 17.667)
        src = yy + round(k * 17.667)
        out[yy, 428:431] = np.minimum(out[yy, 428:431], arr[src, 428:431])
    im.paste(Image.fromarray(out[y0:y1, x0:x1]), (x0, y0))


# 1) 图内图例：四行文字（原来都在 x>=262）
rows = [(92, 119, 585), (134, 161, 485), (177, 201, 510), (220, 244, 496)]
new = ["两者相等的线", "前段：已处理约1/8处", "中段：已处理约1/2处", "后段：已处理约7/8处"]
centers = [105, 147, 189, 232]
for (y0, y1, x1), s, cy in zip(rows, new, centers):
    restore((256, y0 - 4, x1 + 6, y1 + 5))
    put_text(im, s, 262, cy, 29, anchor="lm")

# 2) 标题、横轴标题
cover(im, (500, 8, 1770, 56))
put_text(im, "原动作下限诊断：60个决策点，来自20个开发场景", 1133, 31, 36, anchor="mm")
cover(im, (826, 998, 1442, 1044))
put_text(im, "原动作的当前容量 / 需求", 1134, 1020, 33, anchor="mm")

# 3) 纵轴标题（两行，旋转）
cover(im, (0, 150, 96, 850))
put_text_rot(im, "最低容量 / 需求", 28, 496, 33, angle=90)
put_text_rot(im, "（5瓦下所有合法组合中取最小）", 68, 496, 33, angle=90)

finish(im, OUT)
