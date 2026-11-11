# -*- coding: utf-8 -*-
"""W23 图：单步干预后，20个开发场景各自最终供需匹配程度的变化。报告现成图改标签。
原图：D:\\paper\\outputs\\experiments\\s2_bottleneck29_20261016\\report_analysis\\A1_01\\figures\\01_mother_terminal_effects.png
（实验报告22 单步干预结果图的右半幅，即SGM那一幅；左半幅需求满足率变化未用，平均值见W23_单步干预平均效果）。
只改文字，点、水平参考线、虚线/点线（两种做法的平均值）和网格一个像素不动、不缩放：
  - 只裁出右半幅，上方原来的图例条和小字说明去掉，用原图图例里的圆点/方点样例拼一条中文图例；
  - 子图标题“(b) SGM变化”改成“供需匹配程度的变化”；
  - 纵轴标题“终态ΔSGM（绝对差）”改成“最终供需匹配程度的变化（差值）”；
  - 横轴标题“原manifest母场景顺序”改成“开发场景序号”。
蓝色虚线是“按当前奖励挑最大的动作”的20个场景平均（约+0.002），橙色点线是“事后挑最好的动作”的平均（约+0.005）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _img_edit import *  # noqa: E402,F401,F403

SRC = r"D:\paper\outputs\experiments\s2_bottleneck29_20261016\report_analysis\A1_01\figures\01_mother_terminal_effects.png"
OUT = Path(__file__).resolve().parents[1] / "figures" / "W23_单步干预逐场景SGM变化.png"

full = load(SRC)
# 原图图例的两个样例（圆点、方点），先拷出来
dot_blue = full.crop((870, 48, 910, 84))
dot_orange = full.crop((1445, 48, 1484, 84))
print("marker bboxes", nonwhite_bbox(full, (860, 30, 930, 100), thr=200), nonwhite_bbox(full, (1440, 30, 1500, 100), thr=200))

X0, Y0 = 1480, 150
im = full.crop((X0, Y0, full.width, 1090))
W, H = im.size
print("crop", W, H)


def bb(box):
    b = nonwhite_bbox(im, box, thr=170)
    print(box, "->", b)
    return b


# 子图标题
t = bb((600, 0, W, 70))
cover(im, (t[0] - 6, t[1] - 6, t[2] + 6, t[3] + 6))
put_text(im, "供需匹配程度的变化", (t[0] + t[2]) / 2, (t[1] + t[3]) / 2, 44, anchor="mm")
# 横轴标题
x = bb((400, 860, W, 940))
cover(im, (x[0] - 6, x[1] - 6, x[2] + 6, x[3] + 6))
put_text(im, "开发场景序号", (x[0] + x[2]) / 2, (x[1] + x[3]) / 2, 40, anchor="mm")
# 纵轴标题
y = bb((0, 60, 85, 860))
cover(im, (y[0] - 6, y[1] - 6, y[2] + 6, y[3] + 6))
put_text_rot(im, "最终供需匹配程度的变化（差值）", (y[0] + y[2]) / 2, (y[1] + y[3]) / 2, 40, angle=90)

# 顶部图例条
STRIP = 78
im = extend_canvas(im, top=STRIP)
f = font(38)
items = [(dot_blue, "奖励最大的单次干预"), (dot_orange, "事后选最好的动作（只作参照）")]
widths = [dot_blue.width + 12 + f.getlength(items[0][1]), dot_orange.width + 12 + f.getlength(items[1][1])]
gap = 70
x = (im.width - sum(widths) - gap) / 2
for (patch, s), w in zip(items, widths):
    im.paste(patch, (int(x), STRIP // 2 - patch.height // 2))
    put_text(im, s, x + patch.width + 12, STRIP // 2, 38, anchor="lm")
    x += w + gap
finish(im, OUT)
