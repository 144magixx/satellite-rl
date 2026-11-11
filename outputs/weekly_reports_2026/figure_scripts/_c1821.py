# -*- coding: utf-8 -*-
"""W18~W21 画图公共部分：字体、统一保存成真 PNG、读 csv / zip 内 csv 的小工具。

画布宽 7.2 英寸，插进 Word 时缩到 14 cm（约 0.77 倍），所以图里字号取 10.5~12，
缩小后不小于 8 pt。
"""
import csv
import io
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

MIN_FONT = 10.2
ROOT = Path(__file__).resolve().parents[1]          # outputs/weekly_reports_2026
FIG_DIR = ROOT / "figures"
DATA_DIR = Path(__file__).resolve().parent / "data"  # 读点/中间小数据
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# 真实实验数据所在位置（只读）
D_EXP = Path(r"D:\paper\outputs\experiments")
E_EXP = Path(r"E:\paper_archives\outputs\experiments")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# 配色
C_GREY = "#6b6b6b"
C_BLUE = "#3b74ac"
C_ORANGE = "#d98a1f"
C_TEAL = "#3f9f9c"
C_PURPLE = "#8f74b8"
C_PINK = "#cf6e7e"
C_RED = "#c0392b"
C_GREEN = "#5a9e4b"
C_LIGHT = "#b9c3d1"


def save_png(fig, name, dpi=200, max_width_px=2400, max_bytes=1_500_000):
    """保存为真正的 PNG（出图后再用 Pillow 重存一遍），并检查格式/宽度/大小。"""
    path = FIG_DIR / name
    # 小字统一抬到 10.2 pt 以上（插进 Word 缩到 14 cm 后才不会小于 8 pt）
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text() and t.get_fontsize() < MIN_FONT:
            t.set_fontsize(MIN_FONT)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    if im.width > max_width_px:
        h = round(im.height * max_width_px / im.width)
        im = im.resize((max_width_px, h), Image.LANCZOS)
    im.save(path, format="PNG", optimize=True)
    chk = Image.open(path)
    size = path.stat().st_size
    assert chk.format == "PNG", chk.format
    eff = MIN_FONT * 5.512 / (chk.size[0] / dpi)   # 缩到14cm后最小字号(pt)
    print(f"{name}: {chk.size[0]}x{chk.size[1]}px, {size/1024:.0f} KB, {chk.format}, 缩到14cm后最小字约{eff:.1f}pt")
    if eff < 8.0:
        print("  警告：缩到14cm后小字不足8pt，请收窄画布或换行标签")
    if size > max_bytes:
        print("  警告：超过 1.5MB")
    return path


def read_csv_rows(path_or_zip, member=None):
    """读取 csv，返回 list[dict]。path_or_zip 是 zip 时用 member 指定成员名。"""
    p = Path(path_or_zip)
    if member is None:
        with open(p, encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    with zipfile.ZipFile(p) as z, z.open(member) as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig", newline="")))


def read_json(path_or_zip, member=None):
    p = Path(path_or_zip)
    if member is None:
        return json.loads(p.read_text(encoding="utf-8"))
    with zipfile.ZipFile(p) as z:
        return json.loads(z.read(member).decode("utf-8"))


def fnum(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default
