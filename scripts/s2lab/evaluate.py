"""评价：学习模型（policy_best/final）与规则在 dev/confirm 上的直出指标，逐景 CSV + 汇总 JSON。

  python scripts/s2lab/evaluate.py --dataset-dir D --runs-dir R --split confirm --out E [--run-ids ...] [--rules greedy fixed random] [--which best|final] [--timing]
每个模型/规则写 E/<split>/<name>.csv；汇总写 E/<split>/summary.json。学习模型的掩码/组模式按其训练配置。
"""
import argparse
import json
from pathlib import Path

import torch

from _common import default_device
from satellite_rl.s2lab.agents import load_policy
from satellite_rl.s2lab.env import EnvOptions, SceneBank
from satellite_rl.s2lab.evaluate import run_policy, single_scene_timing, summarize, write_results
from satellite_rl.s2lab.matrix import read_matrix
from satellite_rl.s2lab.rules import make_rule
from satellite_rl.s2lab.scenes import read_config, read_splits
from satellite_rl.s2lab.train import env_options


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--runs-dir')
    ap.add_argument('--split', default='confirm')
    ap.add_argument('--out', required=True)
    ap.add_argument('--run-ids', nargs='*')
    ap.add_argument('--rules', nargs='*', default=[])
    ap.add_argument('--which', default='best', choices=['best', 'final', 'last'])
    ap.add_argument('--timing', action='store_true')
    ap.add_argument('--device', default=None)
    ap.add_argument('--batch', type=int, default=64)
    args = ap.parse_args()
    device = torch.device(args.device or default_device())
    dataset_dir = Path(args.dataset_dir)
    config = read_config(dataset_dir)
    physics = config.physics_obj()
    items = read_splits(dataset_dir)[args.split]
    bank = SceneBank(dataset_dir, sorted({it['key'] for it in items}), physics, device)
    out_dir = Path(args.out) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}

    def record(name, results, extra, timing):
        write_results(out_dir / f'{name}.csv', results, extra)
        s = summarize(results)
        s['timing_s'] = timing
        s.update(extra)
        summary[name] = s
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
        a = s['all']
        print(f"{name:28s} n={a['n']:3d} W={a['W']:.4f} SGM={a['SGM']:.4f} worst10={a['worst10']:.3f} P={a['power_w']:.0f} "
              f"slots={a['slots']:.0f} intf_loss={a['interference_loss']:.3f} adj={a['adj_ratio_pooled']:.2f} unserved={a['unserved']:.1f}"
              + (f" t={timing[0]:.2f}s" if timing else ''), flush=True)

    for rule in args.rules:
        for mode in ('full',):
            opts = EnvOptions(batch_size=args.batch, mask_mode=mode)
            results = run_policy(lambda env, r=rule: make_rule(r, env, 0), bank, config, items, opts, device)
            timing = single_scene_timing(lambda env, r=rule: make_rule(r, env, 0), bank, config, items, opts, device) if args.timing else None
            record(f'rule-{rule}', results, dict(group='rule', label=rule, seed=0, mask_mode=mode), timing)
    if args.runs_dir and args.run_ids is not None:
        matrix = {r['run_id']: r for r in read_matrix(args.runs_dir)}
        wanted = args.run_ids or list(matrix)
        for run_id in wanted:
            run = matrix[run_id]
            path = Path(args.runs_dir) / run_id / f'policy_{args.which}.pt'
            if not path.exists():
                print(f'{run_id}: no {path.name}', flush=True)
                continue
            agent = load_policy(path, device)
            opts = env_options(dict(run, batch_envs=args.batch))
            results = run_policy(lambda env, a=agent: a, bank, config, items, opts, device)
            timing = single_scene_timing(lambda env, a=agent: a, bank, config, items, opts, device) if args.timing else None
            record(run_id, results, dict(group=run['group'], label=run.get('label', ''), seed=run['seed'], mask_mode=run['mask_mode'], which=args.which), timing)


if __name__ == '__main__':
    main()
