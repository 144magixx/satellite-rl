"""W24 画图共用：在 _w22_24_common 的字体和保存工具上，把字号调到“宽 7.2 英寸、插入文档 14 cm”后仍不小于 8.5 pt。
同时放第一阶段最忙波束数据的读取函数（图 W24_最忙波束对比 / W24_最忙波束随预算 共用）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _w22_24_common import *  # noqa: F401,F403,E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11.5,
    "axes.labelsize": 11,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "axes.titleweight": "normal",
})

FIGSIZE = (7.2, 3.8)
DPI = 210
SAVE_KW = dict(dpi=DPI)

# 颜色：FLOW-AO 蓝系，文献方法暖色/灰，贪心灰，学习绿
C_FLOW, C_FLOW2 = BLUE, "#6FA8DC"
C_HEX, C_CLU, C_VOR = "#B2B2B2", ORANGE, YELLOW
C_GREEDY, C_TABU, C_PURE, C_RAND, C_LEARN, C_TSAC = "#7F7F7F", GREEN, ORANGE, "#B9B9B9", PURPLE, RED

S1FIX = D_EXP / "s1_fix_20261017"
LITCMP = D_EXP / "litcmp_20261018"
PURE = D_EXP / "s2_pure_tabu_20261021" / "eval"


def stage1_peaks():
    """每个场景一行（文献方法三个随机种子先取平均）：最忙波束飞机数 peak，按 scene_set、N_budget、arm 排。
    FLOW-AO 与 FLOW-AO-no-spacing 取 effort==1，来自 s1_fix 的 A/static_results.csv；
    A1 蜂窝网格、A2 聚类、A3 Voronoi 来自 litcmp 的 stage1/static.csv。
    """
    lit = pd.read_csv(LITCMP / "stage1" / "static.csv")
    ref = pd.read_csv(S1FIX / "A" / "static_results.csv")
    k = ["scene_set", "date", "frame_time", "N_budget"]
    lit = lit[lit.feasible].groupby(k + ["arm", "n"], as_index=False)[["peak", "spacing_violations"]].mean()
    ref = ref[ref.feasible & (ref.effort == 1) & ref.arm.isin(["FLOW-AO", "FLOW-AO-no-spacing"])]
    ref = ref.assign(spacing_violations=0).groupby(k + ["arm", "n"], as_index=False)[["peak", "spacing_violations"]].mean()
    rows = pd.concat([lit, ref], ignore_index=True)
    rows["period"] = np.where(rows.n >= 1200, "day", "night")
    rows["LB"] = np.ceil(rows.n / rows.N_budget)
    wide = rows.pivot_table(index=k + ["n", "period", "LB"], columns="arm", values="peak").reset_index()
    return rows, wide
