# -*- coding: utf-8 -*-
"""W25/W26 画图公共函数：路径、中文字体、读逐景评价 CSV、保存真 PNG。

原始数据在 D:/paper/outputs/experiments（只读）。运行方式：
    python outputs/weekly_reports_2026/figure_scripts/W25_xxx.py
图写到 outputs/weekly_reports_2026/figures/。
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

EXP = r"D:\paper\outputs\experiments"
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 9.5
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["xtick.labelsize"] = 8.5
plt.rcParams["ytick.labelsize"] = 8.5
plt.rcParams["legend.fontsize"] = 8.5
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# 配色：灰=规则方法，蓝绿=学习方法，橙=搜索/对照，红=重点
C_RULE = "#8c96a8"
C_LEARN = "#2a9d8f"
C_SEARCH = "#e07a5f"
C_KEY = "#c0392b"
C_BLUE = "#3a6ea5"
C_GOLD = "#e9b44c"
C_PURPLE = "#7b5ea7"
PALETTE = [C_BLUE, C_LEARN, C_SEARCH, C_PURPLE, C_GOLD, "#6c757d", "#d1495b", "#00798c", "#8d6a9f", "#a3b18a"]


def read_eval(path):
    """读一份逐景评价 CSV（每行一景）。"""
    return pd.read_csv(path)


def seed_mean(paths, cols=("SGM", "W", "worst10", "power_w", "slots", "interference_loss", "adj_ratio")):
    """每个运行先对场景取平均，再对种子取平均；返回 (均值字典, 标准差字典, 每个种子的字典列表)。"""
    per = []
    for p in paths:
        df = read_eval(p)
        per.append({c: float(df[c].mean()) for c in cols if c in df.columns})
    mean = {c: float(np.mean([d[c] for d in per])) for c in per[0]}
    std = {c: float(np.std([d[c] for d in per])) for c in per[0]}
    return mean, std, per


def scene_mean(paths, col):
    """同一方法多个种子：逐景取种子平均。按 key 对齐，返回 Series(index=key)。"""
    ss = []
    for p in paths:
        df = read_eval(p).set_index("key")
        ss.append(df[col])
    return pd.concat(ss, axis=1).mean(axis=1)


def find_run(round_dir, run, fname):
    """在某一轮实验目录下找运行的文件（运行分散在 account_A / account_B 的 runs* 里）。"""
    hits = glob.glob(os.path.join(EXP, round_dir, "account_*", "runs*", run, fname))
    assert len(hits) >= 1, (round_dir, run, fname)
    return hits[0]


def run_paths(round_dir, prefix, name, fname="eval_confirm_best_sgm.csv", seeds=(1, 2, 3)):
    """name 形如 't14-in-delta'，返回 3 个种子的文件路径。"""
    return [find_run(round_dir, f"{prefix}{name}-s{i}", fname) for i in seeds]


R1 = "s2lab_20261024"
R2 = "s2lab_round2_20261025"
R3 = "s2lab_round3_20261027"
R4 = "s2lab_round4_20261028"
S3 = "s3lab_20261029"


def greedy_confirm():
    """旧确认集287景上容量感知贪心的逐景结果（各轮共用）。"""
    return os.path.join(EXP, R1, "eval", "confirm", "rule-greedy.csv")


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def dev_curve(paths, field="SGM"):
    """多个种子的开发集曲线：返回 (steps, 均值, 最小, 最大)。"""
    cs = []
    steps = None
    for p in paths:
        rows = read_jsonl(p)
        s = np.array([r["step"] for r in rows], dtype=float)
        v = np.array([r[field] for r in rows], dtype=float)
        cs.append((s, v))
    n = min(len(c[0]) for c in cs)
    steps = cs[0][0][:n]
    arr = np.stack([c[1][:n] for c in cs])
    return steps, arr.mean(0), arr.min(0), arr.max(0)


def save_png(fig, name, dpi=210):
    """保存成真 PNG（Pillow 再存一遍），并检查宽度和体积。"""
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    im = Image.open(path)
    im = im.convert("RGB")
    if im.width > 2400:
        h = int(im.height * 2400 / im.width)
        im = im.resize((2400, h), Image.LANCZOS)
    im.save(path, format="PNG", optimize=True)
    chk = Image.open(path)
    size = os.path.getsize(path)
    print(f"{name}: {chk.format} {chk.size} {size/1024:.0f} KB")
    assert chk.format == "PNG"
    return path
