# -*- coding: utf-8 -*-
"""W12 训练过程图的共用改字函数（报告现成图改文字并裁出两个小图，不是重画）。

原图：D:\\paper\\outputs\\experiments\\first_stage_s1_01_20260930\\curves\\{R2,C2,P0}_training_diagnostics.png
（报告 第一阶段/实验报告/01 里的三张“原始训练与验证曲线”，webp 格式，转存成真 PNG，每张2400×2080，八个小图）。
原图缩到文档里字太小，所以只裁出最能说明问题的两个小图（第三行：验证需求覆盖、验证负载不均程度，对应固定的20个验证场景），
上下叠起来；裁剪不缩放，曲线、坐标轴一个像素没动。文字改动：
  1. 补一条白边放中文标题（原总标题“C2 / seed 51001 / no smoothing / linear scale”带实验代号，没有保留）；
  2. 两个小图的英文标题换成中文（“验证需求覆盖：20个场景平均”“验证负载不均程度：有效布局平均”）；
  3. 横轴标签“Environment interactions”→“环境交互步数”。
没有放进来的小图：学习回报、终态需求覆盖、终态负载CV、空束数（四张都是2048或4096个回合在固定几个步数上的散点）、价值损失、策略熵。
"""
from _patch_tools import *

PANEL_W = 1215
ROW_Y0, ROW_Y1 = 1095, 1572          # 第三行（两个验证曲线）的上下范围，含标题和横轴标签
LEFT_X0, RIGHT_X0 = 0, 1182          # 右边小图左移后，两个小图的纵轴左边线都落在 x=120 附近
TITLE_BOX_Y = (1103, 1136)           # 小图标题所在的行（小图上边线在 y=1140，不能擦到）
XLABEL_BOX_Y = (1540, 1568)          # 横轴标签所在的行


def _panel(im, x0, title):
    p = im.crop((x0, ROW_Y0, x0 + PANEL_W, ROW_Y1))
    erase(p, (0, TITLE_BOX_Y[0] - ROW_Y0, PANEL_W, TITLE_BOX_Y[1] - ROW_Y0))
    write(p, title, (646, 1121 - ROW_Y0), 27)
    erase(p, (200, XLABEL_BOX_Y[0] - ROW_Y0, 1100, XLABEL_BOX_Y[1] - ROW_Y0))
    write(p, "环境交互步数", (646, 1553 - ROW_Y0), 25)
    if x0 > 0:   # 右边小图的左缘会带进左边小图“400000”的最后一个字符，擦掉
        erase(p, (0, 1500 - ROW_Y0, 14, 1540 - ROW_Y0))
    return p


def patch_training(key, suptitle, out_name):
    src = EXP + rf"\first_stage_s1_01_20260930\curves\{key}_training_diagnostics.png"
    im = load_any(src)
    a = _panel(im, LEFT_X0, "验证需求覆盖（20个验证场景的平均）")
    b = _panel(im, RIGHT_X0, "验证负载不均程度（有效布局的平均）")
    top = 56
    out = Image.new("RGB", (PANEL_W, top + a.height + b.height + 8), WHITE)
    write(out, suptitle, (PANEL_W / 2, top / 2 + 4), 30)
    out.paste(a, (0, top))
    out.paste(b, (0, top + a.height + 8))
    save_png(out, out_name)
