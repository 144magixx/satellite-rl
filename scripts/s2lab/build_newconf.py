"""第四轮新确认集（c2）：口径与 v1 完全相同（物理、预算、单机需求区间与抽样、分档阈值、每档 100 景、随机种子），只换日期。
本机（DQN 环境）运行：读 frames4/ 里的帧文件 → 生成布局与张量 → 划分 → 打包成 c2.tgz。
帧文件的拉取与 c2.tgz 的上传用 xfer.py（base 环境）。
  python scripts/s2lab/build_newconf.py --dates 2024-06-12 2024-08-14 2024-10-09 [--per-date 150] [--workers 12]
"""
import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault('S2LAB_ROUND', '4')
sys.path.insert(0, str(Path(__file__).parent))
from _common import ROOT  # noqa: E402,F401
from satellite_rl.s2lab.scenes import build_scenes, make_splits, read_config  # noqa: E402

LOCAL = Path('D:/paper/outputs/experiments/s2lab_round4_20261028')
V1 = Path('D:/paper/outputs/experiments/s2lab_round3_20261027/data/v1')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dates', nargs='+', required=True)
    ap.add_argument('--per-date', type=int, default=150)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--out', default=str(LOCAL / 'data' / 'c2'))
    a = ap.parse_args()
    frames = LOCAL / 'frames4'
    out = Path(a.out)
    config = read_config(V1)
    config.name = out.name
    config.split_dates = dict(confirm=list(a.dates))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(asdict(config), ensure_ascii=False, indent=1), encoding='utf-8')
    build_scenes(out, config, None, str(frames), a.workers, None, a.per_date)
    splits = make_splits(out, config)
    print({k: len(v) for k, v in splits.items()},
          {s: sum(1 for it in splits['confirm'] if it['stratum'] == s) for s in ('ample', 'marginal', 'short')}, flush=True)
    tar = LOCAL / f'{out.name}.tgz'
    subprocess.run(['tar', 'czf', str(tar), '-C', str(out.parent), out.name], check=True)
    print('打包', tar, flush=True)


if __name__ == '__main__':
    main()
