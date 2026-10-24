"""单卡排队器：按 queue.json 顺序认领可运行的 run，逐个（或并行 k 个）训练，窗口到点自动让训练存盘退出。

  python scripts/s2lab/lane.py --runs-dir R --dataset-dir D --lane 20654 --window-hours 21 --parallel 2
收到 SIGTERM 时转发给子进程（训练先存后退），然后退出。多个容器共享同一 runs 目录时各自认领不同 run。
failed 的 run 不自动重试：看过 train.log 后把 status.json 的 state 改回 paused 即可重新排队。
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from _common import ROOT, default_device
from satellite_rl.s2lab.matrix import read_queue, run_status, claimable
from satellite_rl.s2lab.train import write_json, now


def claim(runs_dir, run_id, lane):
    """原子认领：O_EXCL 创建 <run>/.claim 锁文件，先到先得（10-24 size-d512L4 曾被两卡同时认领，故改锁文件）。
    锁已存在但该运行可认领（new/paused/心跳过期）时视为陈旧锁，删掉重建。"""
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / '.claim'
    status = run_status(runs_dir, run_id)
    if lock.exists():
        if not claimable(status):
            return False
        lock.unlink()
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(lane=lane, claimed_at=now())))
    status.update(state='claimed', lane=lane, claimed_at=now(), updated_at=now())
    write_json(run_dir / 'status.json', status)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--lane', required=True)
    ap.add_argument('--window-hours', type=float, required=True, help='本窗口可用小时数（调用方已减去安全余量）')
    ap.add_argument('--parallel', type=int, default=1)
    ap.add_argument('--device', default=None)
    ap.add_argument('--only', nargs='*', help='只认领这些前缀的 run')
    ap.add_argument('--min-hours', type=float, default=.3)
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)
    logs = runs_dir / '_logs'
    logs.mkdir(parents=True, exist_ok=True)
    lane_log = (logs / f'lane_{args.lane}.log').open('a', encoding='utf-8')

    def log(message):
        line = f'[{now()}] {message}'
        print(line, flush=True)
        lane_log.write(line + '\n'); lane_log.flush()
    deadline = time.time() + args.window_hours * 3600
    children = {}
    stopping = {'flag': False}

    def handler(signum, frame):
        stopping['flag'] = True
        for proc in children.values():
            proc.send_signal(signal.SIGTERM)
        log(f'signal {signum}: forwarded to {len(children)} children')
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    lane_state = runs_dir / '_lanes' / f'{args.lane}.json'
    lane_state.parent.mkdir(exist_ok=True)
    log(f'lane {args.lane} start, window {args.window_hours:.2f} h, parallel {args.parallel}')
    while True:
        for run_id, proc in list(children.items()):
            if proc.poll() is not None:
                log(f'{run_id} exited with code {proc.returncode}')
                del children[run_id]
        remaining = (deadline - time.time()) / 3600
        write_json(lane_state, dict(lane=args.lane, pid=os.getpid(), running=list(children), remaining_hours=remaining,
                                    updated_at=now(), stopping=stopping['flag']))
        if stopping['flag'] or remaining < args.min_hours:
            if not children:
                break
            time.sleep(10)
            continue
        if len(children) < args.parallel:
            picked = None
            for run_id in read_queue(runs_dir):
                if args.only and not any(run_id.startswith(p) for p in args.only):
                    continue
                if run_id in children:
                    continue
                if claimable(run_status(runs_dir, run_id)) and claim(runs_dir, run_id, args.lane):
                    picked = run_id
                    break
            if picked is None:
                if not children:
                    log('queue empty for this lane; exit')
                    break
                time.sleep(30)
                continue
            cmd = [sys.executable, str(ROOT / 'scripts' / 's2lab' / 'train_run.py'), '--runs-dir', str(runs_dir),
                   '--dataset-dir', args.dataset_dir, '--run-id', picked, '--device', args.device or default_device(),
                   '--max-hours', f'{remaining:.3f}', '--lane', args.lane]
            out = (runs_dir / picked / 'stdout.log').open('a', encoding='utf-8')
            children[picked] = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=str(ROOT))
            log(f'started {picked} (pid {children[picked].pid}, {remaining:.2f} h left)')
            time.sleep(5)
            continue
        time.sleep(20)
    write_json(lane_state, dict(lane=args.lane, pid=os.getpid(), running=[], remaining_hours=(deadline - time.time()) / 3600,
                                updated_at=now(), stopping=stopping['flag'], finished=True))
    log('lane finished')


if __name__ == '__main__':
    main()
