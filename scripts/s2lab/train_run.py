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
        result = train(run_dir, args.dataset_dir, run, args.device or default_device(), args.max_hours, log)
        print(json.dumps(dict(run_id=run['run_id'], result=result)), flush=True)
    except Exception:
        log(f'[{now()}] FAILED\n' + traceback.format_exc())
        status_path = run_dir / 'status.json'
        status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else dict(run_id=run['run_id'])
        status.update(state='failed', updated_at=now(), error=traceback.format_exc()[-2000:])
        write_json(status_path, status)
        raise


if __name__ == '__main__':
    main()
