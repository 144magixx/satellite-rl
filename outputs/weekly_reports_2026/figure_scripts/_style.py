"""周报插图共用的画图设置。字体、字号、保存方式都放在这里，各周的画图脚本 import 它。

插入 Word 时图宽 14 cm，所以画布宽约 6.5 英寸、字号 10 左右，缩放后字号约 8.5 pt。
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
DATA_DIR = os.path.join(HERE, "data")

# 统一配色（色盲友好的一组）
BLUE = "#2f6db3"
ORANGE = "#e07b25"
GREEN = "#3a9a63"
RED = "#c8453b"
PURPLE = "#7b5ea7"
GREY = "#7a7a7a"
LIGHT = "#cfd6dd"
TEAL = "#2a9d9d"


def setup(font_size=10):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 0.5,
        "ytick.labelsize": font_size - 0.5,
        "legend.fontsize": font_size - 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e3e6ea",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.dpi": 100,
    })


def save(fig, name, dpi=220):
    """保存成真正的 PNG（再用 Pillow 重存一遍），宽度不超过 2400 px，尽量小于 1.5 MB。"""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    finalize_png(path)
    return path


def finalize_png(path):
    im = Image.open(path)
    im = im.convert("RGB")
    if im.width > 2400:
        h = round(im.height * 2400 / im.width)
        im = im.resize((2400, h), Image.LANCZOS)
    im.save(path, format="PNG", optimize=True)
    check = Image.open(path)
    assert check.format == "PNG", path
    size_mb = os.path.getsize(path) / 1e6
    print(f"{os.path.basename(path)}  {check.size}  {size_mb:.2f} MB")
    if size_mb > 1.5:
        q = im.quantize(colors=256, method=Image.MEDIANCUT)
        q.save(path, format="PNG", optimize=True)
        print("  量化后", f"{os.path.getsize(path) / 1e6:.2f} MB")
