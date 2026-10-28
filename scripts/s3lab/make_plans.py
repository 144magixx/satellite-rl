"""生成 8 套第二阶段方案（开发集、旧确认集、新确认集）并回放核对直出指标。

  python scripts/s3lab/make_plans.py --out D:/paper/outputs/experiments/s3lab_20261029/plans
输出 <out>/<集>/<方案>.pkl；回放核对结果打印并写 <out>/replay_check.json。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from satellite_rl.s2lab.scenes import load_scene, read_config  # noqa: E402
from satellite_rl.s3lab.plans import PLANS, generate, save  # noqa: E402
from satellite_rl.s3lab.problem import FixedPlan  # noqa: E402

SETS = {
    'dev': ('D:/paper/outputs/experiments/s2lab_round3_20261027/data/v1', 'dev'),
    'old': ('D:/paper/outputs/experiments/s2lab_round3_20261027/data/v1', 'confirm'),
    'new': ('D:/paper/outputs/experiments/s2lab_round4_20261028/data/c2', 'confirm'),
}


def replay(dataset_dir, obj):
    config = read_config(dataset_dir)
    physics = config.physics_obj()
    worst = {}
    for r in obj['records']:
        sc = load_scene(dataset_dir, r['key'])
        fp = FixedPlan(sc, physics, r['demand'], r['groups'], r['starts'], r['counts'], config.power_budget_w, config.beam_power_cap_w)
        m = fp.metrics(r['power'][fp.idx])
        for k in ('W', 'SGM', 'SI', 'worst10', 'power_w'):
            d = abs(m[k] - r['direct'][k]) / (1. if k != 'power_w' else 100.)
            worst[k] = max(worst.get(k, 0.), d)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--models', default='D:/paper/outputs/experiments/s3lab_20261029/stage2_models')
    ap.add_argument('--sets', nargs='*', default=list(SETS))
    ap.add_argument('--plans', nargs='*', default=list(PLANS))
    a = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(a.out)
    check = {}
    for s in a.sets:
        ds, split = SETS[s]
        for pid in a.plans:
            t0 = time.time()
            obj = generate(ds, split, pid, a.models, device)
            save(obj, out / s / f'{pid}.pkl')
            w = replay(ds, obj)
            check[f'{s}/{pid}'] = w
            d = np.mean([r['direct']['SGM'] for r in obj['records']])
            print(f'{s:4s} {pid:10s} n={len(obj["records"])} SGM={d:.4f} W={np.mean([r["direct"]["W"] for r in obj["records"]]):.4f} '
                  f'回放最大差 {json.dumps({k: round(v, 6) for k, v in w.items()})} {time.time() - t0:.0f}s', flush=True)
    (out / 'replay_check.json').write_text(json.dumps(check, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
