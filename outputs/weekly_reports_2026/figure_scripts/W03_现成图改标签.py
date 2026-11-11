# -*- coding: utf-8 -*-
"""W03 五张“报告现成图”的中文改标签版本（训练逐回合原始记录本机没有，只有报告里的图）。

原图在 D:/paper/outputs/experiments/europe_adsb_20260928/ 下（本机转存在 .scratch_w1234/）：
  1. explanation/adsb_to_beams_example.png
       -> figures/W03_航班快照到波束需求.png
       只把副标题（写着样本的时刻）盖掉，其余原样（图中“不是模型训练效果”的说明保留）。
  2. curves/service_resource_curves.png（报告 03_欧洲ADS-B首轮训练结果.md 引用）
       -> figures/W03_训练和验证曲线.png
       去掉带实验代号的总标题；6 个子图的英文标题、图例、x 轴标签改成中文：
       Final demand satisfaction=最终需求满足率；Beams meeting 10% demand=达到10%需求的波束数；
       Supply-demand matching score=供需匹配度；Radiated power=实际发射功率；Episode return=回合回报；
       Allocated slots=分到的频率槽数；图例 Training episode / Training trailing mean (100 episodes) / Validation (20 scenes x 3)
       =训练每个回合的原值 / 训练最近100回合的平均 / 验证（20个场景，各3次动作）；Environment steps=环境交互步数。
  3. curves/learning_diagnostics.png
       -> figures/W03_学习诊断曲线.png
       去掉带实验代号的总标题；4 个子图标题、图例、x 轴标签改成中文
       （Actor loss/Critic 1 loss/Critic 2 loss/Temperature alpha；Logged value / Trailing mean (100 logged points)）。纵轴仍是原图的对称对数刻度。
  4. baseline_comparison_20260929/satisfaction_comparison.png（报告 04_欧洲模型与基线逐场景对比.md 引用）
       -> figures/W03_逐场景需求满足率对比.png     只盖掉带真实日期的两行标题。
  5. baseline_comparison_20260929/sgm_comparison.png
       -> figures/W03_逐场景供需匹配度对比.png     盖掉两行标题；y 轴标签“最终供需匹配评分 SGM”改成“最终供需匹配度”。
绘图区（曲线、阴影、网格、刻度数字）一个像素都不动；最后整张图等比缩小一次到 2400 px 宽以内（只有第 4、5 张超宽）。
没有用的现成图：curves/test_scene_distribution.png（图例在绘图区里，压着虚线，盖不干净）。
运行：python outputs/weekly_reports_2026/figure_scripts/W03_现成图改标签.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w01_04_tools import *
import numpy as np

SCR = os.path.normpath(os.path.join(HERE, "..", ".scratch_w1234"))


def src(name):
    return os.path.join(SCR, name)


def draw_legend(im, items, cx, cy, size=26, hw=60, gap=14, sep=50):
    """items: [(文字, 颜色, 线宽, 是否带点)]，整体居中于 (cx, cy)。"""
    f = font(size)
    d = ImageDraw.Draw(im)
    widths = [hw + gap + d.textlength(t, font=f) for t, *_ in items]
    x = cx - (sum(widths) + sep * (len(items) - 1)) / 2
    for (t, col, lw, dot), wd in zip(items, widths):
        d.line([(x, cy), (x + hw, cy)], fill=col, width=lw)
        if dot:
            d.ellipse([x + hw / 2 - 6, cy - 6, x + hw / 2 + 6, cy + 6], fill=col)
        d.text((x + hw + gap, cy), t, font=f, fill=BLACK, anchor="lm")
        x += wd + sep


DARK_BLUE = (18, 104, 154)
ORANGE = (204, 107, 40)
LIGHT = (176, 203, 226)


def fig_example():
    im = load_any(src("explanation__adsb_to_beams_example.png"))
    W, H = im.size
    bg = im.getpixel((5, 5))                    # 这张图的底色是浅蓝灰，不是纯白
    erase(im, (0, 0, W, 115), color=bg)         # 标题和写着样本时刻的副标题（图题写在文档里）
    im = im.crop((0, 100, W, H))
    save_png(im, "W03_航班快照到波束需求.png")

def fig_service():
    im = load_any(src("curves__service_resource_curves.png"))
    W, H = im.size
    erase(im, (0, 0, W, 60))                               # 总标题（带实验代号）
    erase(im, (470, 92, 1760, 156))                        # 图例（重画）
    draw_legend(im, [("训练每个回合的原值", LIGHT, 3, False), ("训练最近100回合的平均", DARK_BLUE, 4, False),
                     ("验证（20个场景，各3次动作）", ORANGE, 3, True)], 1113, 124)
    titles = [(600, 197, "最终需求满足率"), (1712, 197, "达到10%需求的波束数"),
              (600, 763, "供需匹配度"), (1712, 763, "实际发射功率"),
              (600, 1329, "回合回报"), (1712, 1329, "分到的频率槽数")]
    for cx, y0, t in titles:
        erase(im, (cx - 330, y0 - 8, cx + 330, y0 + 35))
        write(im, t, (cx, y0 + 13), 30)
    for cx in (600, 1712):
        erase(im, (cx - 160, 1895, cx + 160, 1934))
        write(im, "环境交互步数", (cx, 1915), 26)
    im = im.crop((0, 85, W, H))
    save_png(im, "W03_训练和验证曲线.png")


def fig_diag():
    im = load_any(src("curves__learning_diagnostics.png"))
    W, H = im.size
    sat = np.asarray(im)[87:132, 1000:1080]
    col = tuple(int(v) for v in np.median(sat[(sat.max(axis=2) - sat.min(axis=2)) > 25], axis=0))
    erase(im, (0, 0, W, 66))
    erase(im, (715, 78, 1510, 142))
    draw_legend(im, [("记录的原始值", LIGHT, 3, False), ("向后平均（最近100个记录点）", col, 4, False)], 1112, 109)
    titles = [(593, 197, "Actor损失"), (1700, 197, "第1个Critic损失"), (593, 775, "第2个Critic损失"), (1700, 775, "探索温度")]
    for cx, y0, t in titles:
        erase(im, (cx - 260, y0 - 8, cx + 260, y0 + 35))
        write(im, t, (cx, y0 + 13), 30)
    for cx in (594, 1700):
        erase(im, (cx - 160, 1360, cx + 160, 1398))
        write(im, "环境交互步数", (cx, 1379), 26)
    im = im.crop((0, 70, W, H))
    save_png(im, "W03_学习诊断曲线.png")


def fig_baseline():
    im = load_any(src("baseline_com__satisfaction_comparison.png"))
    W, H = im.size
    erase(im, (0, 0, W, 130))
    im = im.crop((0, 150, W, H))
    save_png(im, "W03_逐场景需求满足率对比.png")

    im = load_any(src("baseline_com__sgm_comparison.png"))
    erase(im, (0, 0, W, 130))
    erase(im, (100, 430, 146, 745))
    write(im, "最终供需匹配度", (121, 586), 28, rot=90)
    im = im.crop((0, 150, W, H))
    save_png(im, "W03_逐场景供需匹配度对比.png")


if __name__ == "__main__":
    fig_example()
    fig_service()
    fig_diag()
    fig_baseline()
