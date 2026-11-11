# -*- coding: utf-8 -*-
"""W05-W08 报告现成图改文字用的小工具：只盖白底、重写中文，不碰绘图区。"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"
WHITE = (255, 255, 255)

def font(px, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_PATH, int(round(px)))

def load(path):
    return Image.open(path).convert("RGB")

def nonwhite_bbox(im, box, thr=245):
    """box=(x0,y0,x1,y1) 内非白像素的外框，返回 (x0,y0,x1,y1) 或 None"""
    x0, y0, x1, y1 = box
    a = np.asarray(im.crop(box)).astype(int)
    m = (a.min(axis=2) < thr)
    if not m.any():
        return None
    ys, xs = np.where(m)
    return (x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1)

def cover(im, box, color=WHITE):
    ImageDraw.Draw(im).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=color)

def text_width(s, px, bold=False):
    return font(px, bold).getlength(s)

def fit_size(s, width_px, bold=False):
    best, bd = None, 1e9
    for px in range(8, 120):
        d = abs(font(px, bold).getlength(s) - width_px)
        if d < bd:
            best, bd = px, d
    return best

def put_text(im, s, x, y, px, anchor="lm", fill=(0, 0, 0), bold=False):
    """anchor 同 PIL：l/m/r + a/m/s/d，默认左中。"""
    ImageDraw.Draw(im).text((x, y), s, font=font(px, bold), fill=fill, anchor=anchor)

def put_text_rot(im, s, cx, cy, px, angle=90, fill=(0, 0, 0), bold=False):
    """旋转文字（纵轴标题用），中心对齐 (cx,cy)。"""
    f = font(px, bold)
    w = int(f.getlength(s)) + 8
    h = int(px * 1.5)
    t = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    ImageDraw.Draw(t).text((w / 2, h / 2), s, font=f, fill=fill + (255,), anchor="mm")
    t = t.rotate(angle, expand=True, resample=Image.BICUBIC)
    im.paste(t, (int(cx - t.width / 2), int(cy - t.height / 2)), t)

def crop_save(path_in, box, path_out, scale=1.0):
    im = Image.open(path_in).convert("RGB").crop(box)
    if scale != 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    im.save(path_out)


# ---------- 下面是第二批工具：替换文字、加图例条 ----------
TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

def replace_text(im, search_box, new_text, old_text=None, px=None, rotate=0, align="center", thr=235, pad=2, scale=1.0, bold=False):
    """把 search_box 里的文字盖白，用同字号中文重写。
    old_text 用来按宽度估计字号（旧文字内容）；px 直接给字号则不估计。
    rotate=90 表示纵轴标题（旧文字竖排，bbox 的高就是文字宽）。
    返回 (bbox, px)。"""
    bb = nonwhite_bbox(im, search_box, thr=thr)
    assert bb is not None, ("找不到文字", search_box)
    w = (bb[3] - bb[1]) if rotate else (bb[2] - bb[0])
    if px is None:
        px = fit_size(old_text, w, bold) if old_text else 20
    px = px * scale
    reg = np.asarray(im.crop(bb)).astype(int)
    k = np.unravel_index(reg.sum(axis=2).argmin(), reg.shape[:2])
    col = tuple(int(v) for v in reg[k[0], k[1]])
    cover(im, (bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad))
    cx = (bb[0] + bb[2]) / 2; cy = (bb[1] + bb[3]) / 2
    if rotate:
        put_text_rot(im, new_text, cx, cy, px, angle=rotate, bold=bold, fill=col)
    else:
        if align == "center":
            put_text(im, new_text, cx, cy, px, anchor="mm", bold=bold, fill=col)
        elif align == "left":
            put_text(im, new_text, bb[0], cy, px, anchor="lm", bold=bold, fill=col)
        else:
            put_text(im, new_text, bb[2], cy, px, anchor="rm", bold=bold, fill=col)
    return bb, px

def text_segments(im, band, gap=40, thr=235, axis="x"):
    """在 band=(x0,y0,x1,y1) 里按横向（或纵向）间隙切出多个文字块，返回各块的 bbox 列表。"""
    x0, y0, x1, y1 = band
    a = np.asarray(im.crop(band)).astype(int)
    m = a.min(axis=2) < thr
    prof = m.any(axis=0) if axis == "x" else m.any(axis=1)
    idx = np.where(prof)[0]
    segs = []
    if len(idx) == 0:
        return segs
    s = p = idx[0]
    for i in idx[1:]:
        if i - p > gap:
            segs.append((s, p)); s = i
        p = i
    segs.append((s, p))
    out = []
    for s, p in segs:
        if axis == "x":
            sub = (x0 + s, y0, x0 + p + 1, y1)
        else:
            sub = (x0, y0 + s, x1, y0 + p + 1)
        bb = nonwhite_bbox(im, sub, thr=thr)
        out.append(bb)
    return out

def extend_canvas(im, bottom=0, top=0, left=0, right=0):
    new = Image.new("RGB", (im.width + left + right, im.height + top + bottom), WHITE)
    new.paste(im, (left, top))
    return new

def legend_strip(im, entries, px, where="bottom", strip_h=None, rows=1, handle_w=44, gap=34, line_w=4, marker=True, y_pad=14, style="line"):
    """在图下方（或上方）加一条图例。entries=[(颜色hex或tuple, 文字, 线型可选)]。
    返回新图（画布加高，原图一个像素不动）。"""
    f = font(px)
    per_row = -(-len(entries) // rows)
    row_h = int(px * 1.5)
    if strip_h is None:
        strip_h = row_h * rows + y_pad * 2
    new = extend_canvas(im, bottom=strip_h) if where == "bottom" else extend_canvas(im, top=strip_h)
    y_base = im.height if where == "bottom" else 0
    d = ImageDraw.Draw(new)
    for r in range(rows):
        row = entries[r * per_row:(r + 1) * per_row]
        widths = [handle_w + 10 + f.getlength(e[1]) for e in row]
        total = sum(widths) + gap * (len(row) - 1)
        x = (new.width - total) / 2
        cy = y_base + y_pad + row_h * r + row_h / 2
        for e, w in zip(row, widths):
            col = hexrgb(e[0]) if isinstance(e[0], str) else e[0]
            dash = len(e) > 2 and e[2] == "dash"
            if style == "box":
                d.rectangle([x, cy - px * 0.4, x + handle_w * 0.6, cy + px * 0.4], fill=col)
            elif dash:
                for k in range(0, handle_w, 12):
                    d.line([x + k, cy, x + min(k + 7, handle_w), cy], fill=col, width=line_w)
            else:
                d.line([x, cy, x + handle_w, cy], fill=col, width=line_w)
                if marker:
                    d.ellipse([x + handle_w / 2 - 4, cy - 4, x + handle_w / 2 + 4, cy + 4], fill=col)
            d.text((x + handle_w + 10, cy), e[1], font=f, fill=(0, 0, 0), anchor="lm")
            x += w + gap
    return new

def cover_legend_box(im, box):
    """把图内带代号的图例整块盖白（图例框本来就盖在绘图区上，只含图例本身）。"""
    cover(im, box)

def finish(im, path, max_w=2400):
    """保存为真 PNG；超过 max_w 时按比例整体缩小（不拉伸）。"""
    if im.width > max_w:
        h = round(im.height * max_w / im.width)
        im = im.resize((max_w, h), Image.LANCZOS)
    im.save(path, format="PNG", optimize=True)
    import os
    print(os.path.basename(path), im.size, round(os.path.getsize(path) / 1024), "KB")


def replace_text_gray(im, search_box, new_text, old_text, align="left", gray_thr=225, sat=30, pad=2):
    """图内注释文字用：只把灰黑色的字形像素擦成白色，彩色曲线像素不动，再在原位置重写。"""
    x0, y0, x1, y1 = search_box
    a = np.asarray(im.crop(search_box)).astype(int)
    mn = a.min(axis=2); mx = a.max(axis=2)
    m = (mn < gray_thr) & ((mx - mn) < sat)
    ys, xs = np.where(m)
    assert len(xs) > 0, ("找不到文字", search_box)
    bx0, by0, bx1, by1 = x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1
    px = fit_size(old_text, bx1 - bx0)
    arr = np.asarray(im).copy()
    sub = arr[y0:y1, x0:x1]
    sub[m] = 255
    im.paste(Image.fromarray(arr), (0, 0))
    cy = (by0 + by1) / 2
    if align == "left":
        put_text(im, new_text, bx0, cy, px, anchor="lm")
    else:
        put_text(im, new_text, (bx0 + bx1) / 2, cy, px, anchor="mm")
    return (bx0, by0, bx1, by1), px
