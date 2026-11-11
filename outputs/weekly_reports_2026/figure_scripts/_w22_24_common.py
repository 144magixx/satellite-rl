"""W22、W23、W24 画图共用的小工具：字体、保存成真 PNG、颜色。
用法：在各图脚本里 `from _w22_24_common import *`。
"""
import io
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]           # 仓库根目录 E:\satellite-rl
FIG_DIR = ROOT / "outputs" / "weekly_reports_2026" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 真实实验数据所在的位置（本机）
D_EXP = Path(r"D:\paper\outputs\experiments")
ARCHIVE_T24 = Path(r"E:\satellite_rl_archives\s2_t0_fair24_20261013\publication_t24_v4")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})

W = 5.8   # 画布宽（英寸），插入文档时宽14 cm，几乎不缩放

# 颜色（色盲友好）
BLUE, ORANGE, GREEN, RED, PURPLE, BROWN, PINK, GRAY, YELLOW, CYAN = (
    "#0072B2", "#D55E00", "#009E73", "#C0392B", "#7B4FA3", "#8C564B", "#CC79A7", "#6E6E6E", "#E69F00", "#17A2B8")


def save_png(fig, name, dpi=250, max_px=2400, max_bytes=1_500_000):
    """保存成真 PNG（先写 matplotlib 的 PNG，再用 Pillow 重存），检查格式、宽度和体积。"""
    path = FIG_DIR / name
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    if img.width > max_px:
        h = round(img.height * max_px / img.width)
        img = img.resize((max_px, h), Image.LANCZOS)
    img.save(path, format="PNG", optimize=True)
    size = path.stat().st_size
    if size > max_bytes:
        q = img.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.NONE)
        q.save(path, format="PNG", optimize=True)
        size = path.stat().st_size
    chk = Image.open(path)
    assert chk.format == "PNG", chk.format
    print(f"saved {path.name}: {chk.width}x{chk.height}px, {size/1024:.0f} KB, format={chk.format}")
    return path


def save_pil_png(img, name, max_px=2400, max_bytes=1_500_000):
    """把 Pillow 图片存成真 PNG（用于改标签的报告原图）。"""
    path = FIG_DIR / name
    img = img.convert("RGB")
    if img.width > max_px:
        h = round(img.height * max_px / img.width)
        img = img.resize((max_px, h), Image.LANCZOS)
    img.save(path, format="PNG", optimize=True)
    if path.stat().st_size > max_bytes:
        img.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.NONE).save(path, format="PNG", optimize=True)
    chk = Image.open(path)
    assert chk.format == "PNG", chk.format
    print(f"saved {path.name}: {chk.width}x{chk.height}px, {path.stat().st_size/1024:.0f} KB, format={chk.format}")
    return path


def moving_average(y, n):
    import numpy as np
    y = np.asarray(y, dtype=float)
    if len(y) < n:
        return y
    k = np.ones(n) / n
    sm = np.convolve(y, k, mode="valid")
    pad = np.full(n - 1, np.nan)
    return np.concatenate([pad, sm])
