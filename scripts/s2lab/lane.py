"""鍗曞崱鎺掗槦鍣細鎸?queue.json 椤哄簭璁ら鍙繍琛岀殑 run锛岄€愪釜锛堟垨骞惰 k 涓級璁粌锛岀獥鍙ｅ埌鐐硅嚜鍔ㄨ璁粌瀛樼洏閫€鍑恒€?
  python scripts/s2lab/lane.py --runs-dir R --dataset-dir D --lane 20654 --window-hours 21 --parallel 2
鏀跺埌 SIGTERM 鏃惰浆鍙戠粰瀛愯繘绋嬶紙璁粌鍏堝瓨鍚庨€€锛夛紝鐒跺悗閫€鍑恒€傚涓鍣ㄥ叡浜悓涓€ runs 鐩綍鏃跺悇鑷棰嗕笉鍚?run銆?"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from _common import ROOT, default_device
from satellite_rl.s2lab.matrix import read_queue, read_matrix, run_status, claimable
from satellite_rl.s2lab.train import write_json, now


def claim(runs_dir, run_id, lane):
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / 'status.json'
    status = run_status(runs_dir, run_id)
    status.update(state='claimed', lane=lane, claimed_at=now(), updated_at=now())
    write_json(path, status)
    time.sleep(2 + hash(lane) % 3)
    status = json.loads(path.read_text(encoding='utf-8'))
    return status.get('lane') == lane and status.get('state') == 'claimed'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--lane', required=True)
    ap.add_argument('--window-hours', type=float, required=True, help='鏈獥鍙ｅ彲鐢ㄥ皬鏃舵暟锛堣皟鐢ㄦ柟宸插噺鍘诲畨鍏ㄤ綑閲忥級')
    ap.add_argument('--parallel', type=int, default=1)
    ap.add_argument('--device', default=None)
    ap.add_argument('--only', nargs='*', help='鍙棰嗚繖浜涘墠缂€鐨?run')
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
