"""批量重画训练过程图与按组的开发集比较图。

  python scripts/s2lab/plot_runs.py --runs-dir <runs>            # 每个运行 training.png；每组 _figures/dev_<group>.png
可在远端 ~/s2lab/runs 或本地 fetch 后的 outputs/experiments/s2lab_20261024/runs 上运行。
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from _common import ROOT  # noqa: F401
from satellite_rl.s2lab.plots import plot_training, plot_group_dev, setup_chinese_font


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--window', type=int, default=5)
    ap.add_argument('--only-done', action='store_true')
    args = ap.parse_args()
    setup_chinese_font()
    runs_dir = Path(args.runs_dir)
    matrix = {r['run_id']: r for r in json.loads((runs_dir / 'matrix.json').read_text(encoding='utf-8'))} if (runs_dir / 'matrix.json').exists() else {}
    groups = defaultdict(list)
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir() and not p.name.startswith('_')):
        status_path = run_dir / 'status.json'
        state = json.loads(status_path.read_text(encoding='utf-8')).get('state') if status_path.exists() else 'new'
        if args.only_done and state != 'done':
            continue
        cfg = matrix.get(run_dir.name, {})
        path = plot_training(run_dir, args.window, title=f"{run_dir.name}（{cfg.get('label', '')}，seed {cfg.get('seed', '')}，{state}）")
        if path:
            print(path)
        groups[cfg.get('group', run_dir.name.split('-')[0])].append(run_dir)
    reference = {}
    summary = runs_dir.parent / 'eval' / 'dev' / 'summary.json'
    if summary.exists():
        for name, s in json.loads(summary.read_text(encoding='utf-8')).items():
            if name.startswith('rule-'):
                reference[name[5:]] = {k: s['all'][k] for k in ('SGM', 'W', 'SI')}
    for group, dirs in groups.items():
        out = plot_group_dev(dirs, runs_dir / '_figures' / f'dev_{group}.png', title=f'开发集曲线：{group}', reference=reference)
        if out:
            print(out)


if __name__ == '__main__':
    main()
