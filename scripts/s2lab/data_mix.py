"""核对数据集的负载构成：各集合场景来源、三档占比，以及训练抽样器实际抽到的三档比例。

  python scripts/s2lab/data_mix.py --dataset-dir D:/paper/outputs/experiments/s2lab_20261024/data/v1
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401
from satellite_rl.s2lab.scenes import read_config, read_index, read_splits, load_ratio, stratum_of, STRATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--draws', type=int, default=20000)
    args = ap.parse_args()
    d = Path(args.dataset_dir)
    config = read_config(d)
    rows = {r['key']: r for r in read_index(d)}
    splits = read_splits(d)
    print(f"物理 {config.physics['version']}，预算 {config.power_budget_w} W，单机需求区间 {[v / 1e6 for v in config.per_aircraft_bps_range]} Mbps，阈值 ρ {config.load_thresholds}")
    for name, items in splits.items():
        src = Counter(rows[it['key']]['source'] for it in items)
        dates = Counter(rows[it['key']]['date'] for it in items)
        print(f"\n[{name}] {len(items)} 景；来源 {dict(src)}；日期 {dict(dates)}")
        if name == 'train':
            lo, hi = config.per_aircraft_bps_range
            totals = np.array([rows[it['key']]['members_total'] for it in items])
            # 复现 ItemSampler 的接受概率
            rng = np.random.default_rng(0)
            natural = np.zeros(3)
            for _ in range(4000):
                i = rng.integers(len(items)); v = rng.uniform(lo, hi)
                natural[STRATA.index(stratum_of(load_ratio(totals[i], v, config), config))] += 1
            natural = np.maximum(natural / natural.sum(), 1e-3)
            weight = np.asarray(config.train_strata_share) / natural
            accept = weight / weight.max()
            got = np.zeros(3)
            n = 0
            while n < args.draws:
                i = rng.integers(len(items)); v = rng.uniform(lo, hi)
                s = STRATA.index(stratum_of(load_ratio(totals[i], v, config), config))
                if rng.uniform() <= accept[s]:
                    got[s] += 1; n += 1
            print(f"  均匀抽样时三档自然占比 宽裕/勉强/不够 = {np.round(natural, 2)}；重采样后实际训练占比 = {np.round(got / got.sum(), 2)}（目标 {config.train_strata_share}）")
            busy = (totals >= 1000).sum()
            print(f"  场景飞机数：≥1000 架 {busy} 景，<1000 架 {len(items) - busy} 景")
        else:
            strata = Counter(it['stratum'] for it in items)
            ratios = np.array([it['load_ratio'] for it in items])
            dem = np.array([it['per_aircraft_bps'] for it in items]) / 1e6
            print(f"  三档 {dict(strata)}；ρ 范围 {ratios.min():.2f}—{ratios.max():.2f}；单机需求 {dem.min():.0f}—{dem.max():.0f} Mbps")


if __name__ == '__main__':
    main()
