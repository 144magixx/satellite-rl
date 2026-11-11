# -*- coding: utf-8 -*-
"""给报告现成图只改文字用的小工具（W09～W12 用）。

原则：绘图区（曲线、柱、点、网格）一个像素都不动、不缩放；只把标题、轴标签、图例、刻度标签里
带实验代号或英文的文字，用白底盖掉再写同字号的中文。有些图需要在下方/上方补白边来放新文字，
补白只是延长画布，不改动原有内容。
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
EXP = r"D:\paper\outputs\experiments"
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def load_any(path):
    """读图。注意报告里不少 .png 实际是 webp，Pillow 能读，转成 RGB。"""
    im = Image.open(path)
    im.load()
    return im.convert("RGB")


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_PATH, int(round(size)))


def erase(im, box, color=WHITE):
    """box=(x0,y0,x1,y1)，含端点。"""
    ImageDraw.Draw(im).rectangle(box, fill=color)


def write(im, text, xy, size, anchor="mm", rot=0, fill=BLACK, bold=False, spacing=4):
    """在 xy 处写文字。anchor 同 Pillow（多行文字用 'ma'/'mm' 等）；rot=90 写竖排（轴标签）。"""
    f = font(size, bold)
    if rot == 0:
        d = ImageDraw.Draw(im)
        if "\n" in text:
            d.multiline_text(xy, text, font=f, fill=fill, anchor=anchor, align="center", spacing=spacing)
        else:
            d.text(xy, text, font=f, fill=fill, anchor=anchor)
        return
    # 旋转：先写在透明层上再转
    tmp = Image.new("RGBA", (int(size * (len(text) + 2) * 1.2) + 20, int(size * 2.2) + 20), (255, 255, 255, 0))
    d = ImageDraw.Draw(tmp)
    d.text((tmp.width // 2, tmp.height // 2), text, font=f, fill=fill + (255,), anchor="mm")
    tmp = tmp.rotate(rot, expand=True, resample=Image.BICUBIC)
    # 锚点按中心处理
    cx, cy = xy
    im.paste(tmp, (int(cx - tmp.width / 2), int(cy - tmp.height / 2)), tmp)


def pad(im, left=0, top=0, right=0, bottom=0):
    """延长画布（白底），不改动原有像素。"""
    out = Image.new("RGB", (im.width + left + right, im.height + top + bottom), WHITE)
    out.paste(im, (left, top))
    return out


def text_boxes(im, region, thr=235, gapx=12, gapy=3):
    """在 region=(x0,y0,x1,y1) 里找成块的非白像素，返回按行排序的文字块框（全局坐标）。
    只用来找坐标，画图脚本里把结果写死成常数。"""
    x0, y0, x1, y1 = region
    a = np.asarray(im.crop(region)).min(axis=2) < thr
    rows = np.where(a.any(axis=1))[0]
    if rows.size == 0:
        return []
    lines, start, prev = [], rows[0], rows[0]
    for r in rows[1:]:
        if r - prev > gapy:
            lines.append((start, prev))
            start = r
        prev = r
    lines.append((start, prev))
    boxes = []
    for ra, rb in lines:
        cols = np.where(a[ra:rb + 1].any(axis=0))[0]
        s, p = cols[0], cols[0]
        for c in cols[1:]:
            if c - p > gapx:
                boxes.append((x0 + s, y0 + ra, x0 + p, y0 + rb))
                s = c
            p = c
        boxes.append((x0 + s, y0 + ra, x0 + p, y0 + rb))
    return boxes


def save_png(im, name, max_w=2400):
    """保存成真 PNG（Pillow 重存），检查格式、宽度和体积。"""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    im = im.convert("RGB")
    assert im.width <= max_w, f"{name} 宽 {im.width} 超过 {max_w}，请先裁掉白边，不要缩放"
    im.save(path, format="PNG", optimize=True)
    chk = Image.open(path)
    assert chk.format == "PNG"
    print(f"{name}: {chk.format} {chk.size} {os.path.getsize(path) / 1024:.0f} KB")
    return path
