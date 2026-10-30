"""训练一个运行（从 matrix.json 取配置；目录已有 checkpoint 则自动续跑）。

  python scripts/s2lab/train_run.py --runs-dir R --dataset-dir D --run-id main-tsac-s1 --device cuda --max-hours 5
也可 --config some.json 直接给运行配置（冒烟用）。
"""
import argparse
import json
import traceback
from pathlib import Path

from _common import default_device
from satellite_rl.s2lab.matrix import read_matrix
from satellite_rl.s2lab.train import train, write_json, now


def auto_eval(run_dir, dataset_dir, run, device, log):
    """训完立即用 policy_best_sgm 与 policy_best 在开发集、确认集各评一遍，写 run_dir/eval_<split>_<which>.csv 与 eval_summary.json。"""
    import torch
    from satellite_rl.s2lab.agents import load_policy
    from satellite_rl.s2lab.env import SceneBank
    from satellite_rl.s2lab.evaluate import run_policy, summarize, write_results
    from satellite_rl.s2lab.scenes import read_config, read_splits
    from satellite_rl.s2lab.train import env_options
    device = torch.device(device)
    config = read_config(dataset_dir)
    splits = read_splits(dataset_dir)
    out = {}
    for split in ('dev', 'confirm'):
        items = splits[split]
        bank = SceneBank(dataset_dir, sorted({it['key'] for it in items}), config.physics_obj(), device)
        for which in ('best_sgm', 'best'):
            agent = load_policy(run_dir / f'policy_{which}.pt', device)
            results = run_policy(lambda env, a=agent: a, bank, config, items, env_options(dict(run, batch_envs=64)), device)
            extra = dict(group=run['group'], label=run.get('label', ''), seed=run['seed'], mask_mode=run['mask_mode'], which=which)
            write_results(run_dir / f'eval_{split}_{which}.csv', results, extra)
            out[f'{split}_{which}'] = summarize(results)
            a = out[f'{split}_{which}']['all']
            log(f'[{now()}] eval {split} {which}: W={a["W"]:.4f} SGM={a["SGM"]:.4f} worst10={a["worst10"]:.3f} P={a["power_w"]:.0f} '
                f'intf={a["interference_loss"]:.3f} adj={a["adj_ratio_pooled"]:.2f}')
    write_json(run_dir / 'eval_summary.json', out)
    status_path = run_dir / 'status.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    status.update(eval_done=now(), confirm_sgm=out['confirm_best_sgm']['all']['SGM'], confirm_w=out['confirm_best_sgm']['all']['W'])
    write_json(status_path, status)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--run-id')
    ap.add_argument('--config')
    ap.add_argument('--device', default=None)
    ap.add_argument('--max-hours', type=float)
    ap.add_argument('--lane', default='')
    ap.add_argument('--total-steps', type=int, help='覆盖步数（冒烟用）')
    ap.add_argument('--no-eval', action='store_true', help='训完不自动评开发集/确认集')
    args = ap.parse_args()
    if args.config:
        run = json.loads(Path(args.config).read_text(encoding='utf-8'))
    else:
        run = next(r for r in read_matrix(args.runs_dir) if r['run_id'] == args.run_id)
    if args.total_steps:
        run['total_steps'] = args.total_steps
    run['lane'] = args.lane
    run_dir = Path(args.runs_dir) / run['run_id']
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / 'train.log'

    def log(message):
        print(message, flush=True)
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(message + '\n')
    try:
        device = args.device or default_device()
        result = train(run_dir, args.dataset_dir, run, device, args.max_hours, log)
        print(json.dumps(dict(run_id=run['run_id'], result=result)), flush=True)
        if result == 'done' and not args.no_eval:
            try:
                auto_eval(run_dir, args.dataset_dir, run, device, log)
            except Exception:                      # 评价失败不把训练标成失败，事后补评
                log(f'[{now()}] auto_eval FAILED\n' + traceback.format_exc())
    except Exception:
        log(f'[{now()}] FAILED\n' + traceback.format_exc())
        status_path = run_dir / 'status.json'
        status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else dict(run_id=run['run_id'])
        status.update(state='failed', updated_at=now(), error=traceback.format_exc()[-2000:])
        write_json(status_path, status)
        raise


if __name__ == '__main__':
    main()
