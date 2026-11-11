# -*- coding: utf-8 -*-
"""W11_逐场景差值.png：报告现成图改文字，不是重画。

原图：D:\\paper\\outputs\\experiments\\stage3_si_priority_20261003\\report_20261003\\total_paired_delta.png
（报告 第三阶段/实验报告/03 里的“3000 W预算配对差值”，webp 格式，转存成真 PNG）。
文字改动（绘图区的点一个像素没动，也没缩放）：
  1. 纵轴“SI差（10⁻⁴百分点）”→“需求满足率差（单位0.0001个百分点）”，“SGM差（原量纲）”→“供需匹配程度差”；
  2. 横轴“场景序号（SHA排序）”→“场景序号”；
  3. 图例“B−A、C−A、C−B”换成人话：原近似−原SCA、新近似−原SCA、新近似−原近似
     （原近似、新近似都是“先提满足率”之后的两种容量近似，原SCA是没改规则的起点），图例框向右加宽以放下文字。
两个子图标题原本就是中文，没动。
运行：python W11_逐场景差值.py
"""
from _patch_tools import *

SRC = EXP + r"\stage3_si_priority_20261003\report_20261003\total_paired_delta.png"
im = load_any(SRC)

for (x0, y0, x1, y1), s in [((26, 262, 44, 436), "需求满足率差（单位0.0001个百分点）"), ((965, 269, 986, 429), "供需匹配程度差")]:
    erase(im, (x0 - 3, y0 - 4, x1 + 3, y1 + 4))
    write(im, s, ((x0 + x1) / 2, (y0 + y1) / 2), 21, rot=90)

for cx in (519, 1511):
    erase(im, (cx - 110, 664, cx + 110, 694))
    write(im, "场景序号（共100个）", (cx, 679), 22)

# 图例：加宽框，重写文字（标记点保持原样）
erase(im, (1217, 510, 1350, 616))        # 擦掉原图例框的右边线和右侧空白
d = ImageDraw.Draw(im)
GRAY = (214, 214, 214)                    # 原图例框线的颜色，粗2像素
d.rectangle((1217, 514, 1343, 515), fill=GRAY)   # 上边线接长
d.rectangle((1217, 611, 1343, 612), fill=GRAY)   # 下边线接长
d.rectangle((1342, 514, 1343, 612), fill=GRAY)   # 新的右边线
for cy, s in [(532, "原近似−原SCA"), (562, "新近似−原SCA"), (592, "新近似−原近似")]:
    erase(im, (1166, cy - 12, 1222, cy + 12))
    write(im, s, (1174, cy), 18, anchor="lm")

save_png(im, "W11_逐场景差值.png")
