"""构建数据集：复用 S2-FIX 布局 + 按日期生成新布局（FLOW-AO v2，CPU 并行）→ 物理v3张量 → 集合划分。

用法（可重复运行，已有场景跳过）：
  python scripts/s2lab/build_data.py --dataset-dir <D> --s2fix-data <S2FIX/data> --frames-dir <frames> --workers 12
  python scripts/s2lab/build_data.py --dataset-dir <D> --splits-only          # 只重做划分
参数可覆盖口径：--power-budget --demand-lo --demand-hi --thresholds t0 t1 --sidelobe --xpd --cap
"""
import argparse
from dataclasses import asdict
import json
import time
from pathlib import Path

from _common import ROOT  # noqa: F401
from satellite_rl.s2lab.physics import PhysicsV3
from satellite_rl.s2lab.scenes import DatasetConfig, build_scenes, make_splits, read_config, read_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--name', default='v1')
    ap.add_argument('--s2fix-data')
    ap.add_argument('--frames-dir')
    ap.add_argument('--dates', nargs='*', help='本次只生成这些日期（口径日期集合不变；多容器分日期并行）')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--per-date-limit', type=int, help='每个日期只取均匀分布的 k 个时刻（冒烟用）')
    ap.add_argument('--splits-only', action='store_true')
    ap.add_argument('--power-budget', type=float)
    ap.add_argument('--demand-lo', type=float)
    ap.add_argument('--demand-hi', type=float)
    ap.add_argument('--thresholds', type=float, nargs=2)
    ap.add_argument('--sidelobe', type=float)
    ap.add_argument('--xpd', type=float)
    ap.add_argument('--cap', type=float)
    ap.add_argument('--dev-per-stratum', type=int)
    ap.add_argument('--confirm-per-stratum', type=int)
    args = ap.parse_args()
    dataset_dir = Path(args.dataset_dir)
    if (dataset_dir / 'config.json').exists():
        config = read_config(dataset_dir)
    else:
        config = DatasetConfig(name=args.name)
    physics = dict(config.physics)
    if args.sidelobe is not None: physics['sidelobe_level_dB'] = args.sidelobe
    if args.xpd is not None: physics['xpd_dB'] = args.xpd
    if args.cap is not None: physics['spectral_efficiency_cap'] = args.cap
    config.physics = PhysicsV3(**physics).to_dict()
    if args.power_budget is not None: config.power_budget_w = args.power_budget
    if args.demand_lo is not None or args.demand_hi is not None:
        lo, hi = config.per_aircraft_bps_range
        config.per_aircraft_bps_range = (args.demand_lo or lo, args.demand_hi or hi)
    if args.thresholds: config.load_thresholds = tuple(args.thresholds)
    if args.dev_per_stratum: config.dev_per_stratum = args.dev_per_stratum
    if args.confirm_per_stratum: config.confirm_per_stratum = args.confirm_per_stratum
    started = time.time()
    if not args.splits_only:
        results = build_scenes(dataset_dir, config, args.s2fix_data, args.frames_dir, args.workers, args.limit, args.per_date_limit, args.dates)
        print(json.dumps(dict(built=len(results), seconds=time.time() - started), ensure_ascii=False), flush=True)
        if args.dates:
            return
    else:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'config.json').write_text(json.dumps(asdict(config), ensure_ascii=False, indent=1), encoding='utf-8')
    rows = read_index(dataset_dir)
    splits = make_splits(dataset_dir, config)
    summary = {k: len(v) for k, v in splits.items()}
    strata = {k: {s: sum(1 for it in v if it.get('stratum') == s) for s in ('ample', 'marginal', 'short')} for k, v in splits.items() if k != 'train'}
    print(json.dumps(dict(scenes=len(rows), splits=summary, strata=strata), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
