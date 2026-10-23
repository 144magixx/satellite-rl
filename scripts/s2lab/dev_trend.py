"""打印各运行开发集曲线（W/SGM/SI 按步），对照标定表里的规则水平。用于巡检。

  python scripts/s2lab/dev_trend.py [--root D:/paper/outputs/experiments/s2lab_20261024]
"""
import argparse
import csv
import json
from pathlib import Path

from _common import ROOT  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='D:/paper/outputs/experiments/s2lab_20261024')
    ap.add_argument('--every', type=int, default=2)
    args = ap.parse_args()
    root = Path(args.root)
    for run_dir in sorted((root / 'runs').glob('*/')):
        dev = run_dir / 'dev.jsonl'
        if not dev.exists():
            continue
        rows = [json.loads(l) for l in dev.read_text(encoding='utf-8').splitlines() if l.strip()]
        status = json.loads((run_dir / 'status.json').read_text(encoding='utf-8')) if (run_dir / 'status.json').exists() else {}
        best = max(rows, key=lambda r: r['score'])
        print(f"{run_dir.name:22s} {status.get('state', '?'):8s} step={status.get('step', 0):>7} sps={status.get('sps', 0):.0f} "
              f"best={best['score']:.3f}@{best['step']} last W={rows[-1]['W']:.3f} SGM={rows[-1]['SGM']:.3f} SI={rows[-1].get('SI', float('nan')):.3f} "
              f"P={rows[-1]['power_w']:.0f} intf={rows[-1]['interference_loss']:.3f} adj={rows[-1]['adj_ratio_pooled']:.2f}")
        print('   ' + ' '.join(f"{r['step'] // 1000}k:{r['W']:.2f}/{r['SGM']:.2f}" for r in rows[::args.every]))
    calib = root / 'data' / 'v1' / 'calibration.csv'
    if calib.exists():
        with calib.open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        print('\n标定（开发集，规则直出）:')
        for r in rows:
            if r['subset'] == 'all':
                print(f"  {r['rule']:7s} {float(r['per_aircraft_mbps']):.0f} Mbps  W={float(r['W']):.3f} SGM={float(r['SGM']):.3f} intf={float(r['interference_loss']):.3f}")


if __name__ == '__main__':
    main()
