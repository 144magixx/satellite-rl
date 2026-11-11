# -*- coding: utf-8 -*-
"""W01～W04 用的小工具：只改文字的图片编辑，和重画图的统一样式。

给报告现成图改文字的原则：绘图区（曲线、柱、点、网格）一个像素都不动；只把标题、轴标签、图例、
刻度标签里带实验代号或英文的文字，用白底盖掉再写同字号的中文。需要缩到 2400 px 宽以内时，
最后整张图等比缩小一次（不裁绘图区、不改比例）。

原始数据在 D:/paper/outputs/experiments、D:/paper/outputs/verification（只读）。
图写到 outputs/weekly_reports_2026/figures/。
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
EXP = r"D:\paper\outputs\experiments"
VER = r"D:\paper\outputs\verification"
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def load_any(path):
    """读图。报告里的 .png 有的实际是 webp/jpeg，Pillow 都能读，统一转 RGB。"""
    im = Image.open(path)
    im.load()
    return im.convert("RGB")


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_PATH, int(round(size)))


def erase(im, box, color=WHITE):
    """box=(x0,y0,x1,y1)，含端点。"""
    ImageDraw.Draw(im).rectangle(box, fill=color)


def erase_keep_vgrid(im, box, ref_y):
    """盖掉 box，但把被盖住的竖向网格线补回去（取 ref_y 那一行的像素往下复制）。用于绘图区内的图例。"""
    x0, y0, x1, y1 = box
    ref = im.crop((x0, ref_y, x1 + 1, ref_y + 1))
    erase(im, box)
    col = ref.resize((x1 - x0 + 1, y1 - y0 + 1), Image.NEAREST)
    arr_new = np.asarray(col).copy()
    arr_old = np.asarray(ref)[0]
    for i in range(arr_old.shape[0]):
        if arr_old[i].min() > 245:  # 白色，不是网格
            arr_new[:, i, :] = 255
    im.paste(Image.fromarray(arr_new), (x0, y0))


def write(im, text, xy, size, anchor="mm", rot=0, fill=BLACK, bold=False):
    """在 xy 处写文字。rot=90 写竖排（y 轴标签），xy 为文字块中心。"""
    f = font(size, bold)
    if rot == 0:
        ImageDraw.Draw(im).text(xy, text, font=f, fill=fill, anchor=anchor)
        return
    tmp = Image.new("RGBA", (int(size * (len(text) + 2) * 1.3) + 20, int(size * 2.4) + 20), (255, 255, 255, 0))
    ImageDraw.Draw(tmp).text((tmp.width // 2, tmp.height // 2), text, font=f, fill=fill + (255,), anchor="mm")
    tmp = tmp.rotate(rot, expand=True, resample=Image.BICUBIC)
    cx, cy = xy
    im.paste(tmp, (int(cx - tmp.width / 2), int(cy - tmp.height / 2)), tmp)


def text_boxes(im, region, thr=235, gapx=12, gapy=3):
    """在 region 里找成块的非白像素，返回按行排序的文字块框（全局坐标）。只用来找坐标。"""
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
    """保存成真 PNG（Pillow 重存）。宽度超过 max_w 时整张等比缩小一次。"""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    im = im.convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    im.save(path, format="PNG", optimize=True)
    chk = Image.open(path)
    assert chk.format == "PNG"
    print(f"{name}: {chk.format} {chk.size} {os.path.getsize(path) / 1024:.0f} KB")
    return path


# ---- 重画图的统一样式（缩到 14 cm 宽后字号约 8.5～9.5 pt）----
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 10.5
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9.5
    plt.rcParams["ytick.labelsize"] = 9.5
    plt.rcParams["legend.fontsize"] = 9.5
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["grid.color"] = "#e6e6e6"
    plt.rcParams["grid.linewidth"] = 0.6
    return plt


def save_fig(fig, name, dpi=220):
    """matplotlib 图保存成真 PNG，并检查宽度和体积。"""
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    im = Image.open(path).convert("RGB")
    if im.width > 2400:
        im = im.resize((2400, round(im.height * 2400 / im.width)), Image.LANCZOS)
    im.save(path, format="PNG", optimize=True)
    chk = Image.open(path)
    assert chk.format == "PNG"
    print(f"{name}: {chk.format} {chk.size} {os.path.getsize(path) / 1024:.0f} KB")
    return path


# 配色：灰=规则方法，蓝=学习方法，橙/红=对照或出问题
C_GREY = "#8c96a8"
C_BLUE = "#3a6ea5"
C_TEAL = "#2a9d8f"
C_ORANGE = "#e07a5f"
C_RED = "#c0392b"
C_GOLD = "#e9b44c"
C_PURPLE = "#7b5ea7"
