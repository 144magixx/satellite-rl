# -*- coding: utf-8 -*-
"""W09～W12 按表格重画图的公共样式。画布约 6 英寸宽，字号 9～10，插进文档缩成 14 cm 后字还有 8 pt 以上。"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9.5
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 9.5
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#e6e6e6"
plt.rcParams["grid.linewidth"] = 0.6
plt.rcParams["axes.axisbelow"] = True

C_BASE = "#8c96a8"     # 灰：基线/原分配
C_A = "#3a6ea5"        # 蓝
C_B = "#e07a5f"        # 橙
C_C = "#2a9d8f"        # 青绿
C_D = "#e9b44c"        # 黄
C_E = "#7b5ea7"        # 紫
C_F = "#c0392b"        # 红
BUDGET_COLORS = ["#3a6ea5", "#e9b44c"]   # 两种功率预算


def save(fig, name, dpi=220):
    """保存成真 PNG（Pillow 重存），检查格式、宽度和体积。"""
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    im = Image.open(path).convert("RGB")
    assert im.width <= 2400, im.width
    im.save(path, format="PNG", optimize=True)
    chk = Image.open(path)
    assert chk.format == "PNG"
    print(f"{name}: {chk.format} {chk.size} {os.path.getsize(path) / 1024:.0f} KB")
    return path
