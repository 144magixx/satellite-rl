# -*- coding: utf-8 -*-
"""W23 图：15万步终点，各组相对原档位的逐场景供需匹配程度差。报告现成图改标签。
原图：D:\\paper\\outputs\\experiments\\s2_w31_20261017\\analysis\\fig3_paired_sgm_150k.png
（实验报告23 第2节 图1）。
只改文字，绘图区（点、黑线、网格、门槛线和图例框）一个像素不动、不缩放：
  - 标题：“每个点是一个开发母场景”改成“开发场景”；
  - 纵轴标题：把带编号P00-M0和SGM的文字改成“相对原档位的供需匹配程度差（同场景配对）”；
  - 横轴7个带编号的刻度（P00-M1…P11-M1）改成两行中文（档位名＋是否带余量摘要），画布向下加高40像素放第二行；
  - 图例“改善门槛 +0.01”本来就没有代号，保持原样。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _img_edit import *  # noqa: E402,F401,F403

SRC = r"D:\paper\outputs\experiments\s2_w31_20261017\analysis\fig3_paired_sgm_150k.png"
OUT = Path(__file__).resolve().parents[1] / "figures" / "W23_逐场景供需匹配差.png"

im = extend_canvas(load(SRC), bottom=42)

# 标题
replace_text(im, (90, 10, 1640, 52), "15万步终点：每个点是一个开发场景，黑线为平均", px=23)
# 纵轴标题
replace_text(im, (0, 100, 52, 620), "相对原档位的供需匹配程度差（同场景配对）", px=21, rotate=90)
# 横轴刻度
labels = [("原档位", "带摘要"), ("只细化5瓦以上", "不带摘要"), ("只细化5瓦以上", "带摘要"),
          ("原档位＋0.5瓦", "不带摘要"), ("原档位＋0.5瓦", "带摘要"), ("全区间细化", "不带摘要"),
          ("全区间细化", "带摘要")]
segs = text_segments(im, (100, 672, 1650, 696), gap=40)
assert len(segs) == 7, segs
px = 21
for bb, (l1, l2) in zip(segs, labels):
    cx = (bb[0] + bb[2]) / 2
    cover(im, (int(cx) - 100, 672, int(cx) + 100, 718))
    put_text(im, l1, cx, 683, px, anchor="mm")
    put_text(im, l2, cx, 683 + px * 1.3, px, anchor="mm")
finish(im, OUT)
