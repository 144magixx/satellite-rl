"""标定扫描：规则方法在 单机需求 × 功率预算 网格上的直出指标，用来定主工况（需求区间、预算、分层阈值）。

  python scripts/s2lab/calibrate.py --dataset-dir D --split dev --demands 20e6 40e6 60e6 80e6 100e6 --budgets 3000 2000 1500 1000
输出 <dataset>/calibration.csv 与 calibration.md。
"""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from _common import default_device
from satellite_rl.s2lab.env import EnvOptions, SceneBank
from satellite_rl.s2lab.evaluate import evaluate_rule, summarize
from satellite_rl.s2lab.scenes import read_config, read_splits, read_index, load_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--split', default='dev')
    ap.add_argument('--max-scenes', type=int, default=60)
    ap.add_argument('--demands', type=float, nargs='+', default=[20e6, 40e6, 60e6, 80e6, 100e6])
    ap.add_argument('--budgets', type=float, nargs='+', default=[3000., 2000., 1500., 1000.])
    ap.add_argument('--rules', nargs='+', default=['greedy', 'fixed'])
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    device = torch.device(args.device or default_device())
    dataset_dir = Path(args.dataset_dir)
    config = read_config(dataset_dir)
    physics = config.physics_obj()
    rows_index = {r['key']: r for r in read_index(dataset_dir)}
    keys = sorted({it['key'] for it in read_splits(dataset_dir)[args.split]})
    # 按飞机数均匀挑场景，覆盖忙闲
    keys = sorted(keys, key=lambda k: rows_index[k]['aircraft'])
    if len(keys) > args.max_scenes:
        keys = [keys[int(i)] for i in np.linspace(0, len(keys) - 1, args.max_scenes)]
    bank = SceneBank(dataset_dir, keys, physics, device)
    out = []
    for budget in args.budgets:
        cfg = replace(config, power_budget_w=budget)
        for d in args.demands:
            items = [dict(key=k, per_aircraft_bps=d) for k in keys]
            for rule in args.rules:
                results = evaluate_rule(rule, bank, cfg, items, EnvOptions(batch_size=64), device)
                busy = [r for r in results if r['aircraft'] >= 1000]
                for name, subset in (('all', results), ('busy', busy)):
                    if not subset:
                        continue
                    s = summarize(subset)['all']
                    out.append(dict(rule=rule, budget_w=budget, per_aircraft_mbps=d / 1e6, subset=name, n=s['n'],
                                    load_ratio=s['load_ratio'], W=s['W'], SGM=s['SGM'], worst10=s['worst10'],
                                    power_used=s['power_w'] / budget, slots=s['slots'], interference_loss=s['interference_loss'],
                                    mean_in_db=s['mean_in_db'], unserved=s['unserved'], copol_overlap=s['copol_overlap']))
                    print(json.dumps(out[-1]), flush=True)
    with (dataset_dir / 'calibration.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out[0]))
        writer.writeheader(); writer.writerows(out)
    lines = ['| 规则 | 预算W | 单机Mbps | 子集 | n | ρ | W | SGM | 最差10% | 功率用满 | 用槽 | 干扰损失 | I/N dB | 未服务 |',
             '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in out:
        lines.append(f"| {r['rule']} | {r['budget_w']:.0f} | {r['per_aircraft_mbps']:.0f} | {r['subset']} | {r['n']} | {r['load_ratio']:.2f} | {r['W']:.3f} | {r['SGM']:.3f} | {r['worst10']:.3f} | {r['power_used']:.2f} | {r['slots']:.0f} | {r['interference_loss']:.3f} | {r['mean_in_db']:.1f} | {r['unserved']:.1f} |")
    (dataset_dir / 'calibration.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
