"""参数量组对 d128L2（主表新版 T-SAC 3 种子）的逐景配对差。"""
from pathlib import Path

from report_figures2 import Data, paired

data = Data(Path('D:/paper/outputs/experiments/s2lab_round2_20261025'), 'best_sgm')
base = [data.rows(f'main-tsac-s{s}') for s in (1, 2, 3)]
for label in ('d32L1', 'd64L2', 'd128L4', 'd128L8', 'd256L2', 'd256L4', 'd512L4'):
    runs = data.available([f'size-{label}-s{s}' for s in (1, 2, 3)])
    p = paired([data.rows(r) for r in runs], base, 'SGM')
    w = paired([data.rows(r) for r in runs], base, 'W')
    print(f"{label} n={len(runs)} SGM {p['diff']:+.3f} [{p['lo']:+.3f},{p['hi']:+.3f}] W {w['diff']:+.3f}")
