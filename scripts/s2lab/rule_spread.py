"""看规则方法在开发集逐景结果的分布（均值、最小、最大、按档），用于解释参考线。"""
import csv
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'D:/paper/outputs/experiments/s2lab_20261024/eval/dev')
for name in ('greedy', 'fixed', 'random'):
    rows = list(csv.DictReader((root / f'rule-{name}.csv').open(encoding='utf-8')))
    W = np.array([float(r['W']) for r in rows])
    S = np.array([float(r['SGM']) for r in rows])
    print(f'{name:7s} n={len(rows)}  W 均值 {W.mean():.3f} 最小 {W.min():.3f} 最大 {W.max():.3f} 标准差 {W.std():.3f} | '
          f'SGM 均值 {S.mean():.3f} 最小 {S.min():.3f} 最大 {S.max():.3f}')
    for s in ('ample', 'marginal', 'short'):
        sub = [r for r in rows if r['stratum'] == s]
        print(f"    {s:8s} n={len(sub)} W={np.mean([float(r['W']) for r in sub]):.3f} SGM={np.mean([float(r['SGM']) for r in sub]):.3f}")
