"""远端四卡操作（paramiko，口令只读环境变量/注册表 BUPT_PASSWORD）。本地用 base anaconda 的 python 运行。

  python scripts/s2lab/remote.py check                       # GPU、解释器、数据目录
  python scripts/s2lab/remote.py sync                        # 把本地 HEAD 同步到远端 repo（需先提交）
  python scripts/s2lab/remote.py put-frames --dates ...      # 上传 frames_<date>.npz（缺的才传）
  python scripts/s2lab/remote.py build-data --port 20654 --workers 16 [--extra "--dates 2024-01-15"]
  python scripts/s2lab/remote.py calibrate --port 20654 [--extra "..."]
  python scripts/s2lab/remote.py launch --window-hours 21 [--parallel 2] [--ports ...]
  python scripts/s2lab/remote.py pause                       # 训练先存后退、排队器退出
  python scripts/s2lab/remote.py status                      # 汇总 runs 状态，写本地 STATE
  python scripts/s2lab/remote.py fetch                       # 下载曲线/开发集/状态/评价结果到本地
  python scripts/s2lab/remote.py evaluate --port 20654 --split confirm [--extra "..."]
  python scripts/s2lab/remote.py ps | tail --port P --run-id X | sh --port P --cmd "..."
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from _common import (ROOT, REMOTE_HOST, REMOTE_USER, REMOTE_PORTS, REMOTE_PY, REMOTE_ROOT, REMOTE_S2FIX_DATA, LOCAL_OUT, DATASET_NAME)

sys.path.insert(0, str(ROOT / 'src'))
from satellite_rl.s2lab.matrix import render_state_markdown, heartbeat_stale  # noqa: E402

REMOTE_REPO = f'{REMOTE_ROOT}/repo'
REMOTE_DATA = f'{REMOTE_ROOT}/data/{DATASET_NAME}'
REMOTE_RUNS = f'{REMOTE_ROOT}/runs'
REMOTE_FRAMES = f'{REMOTE_ROOT}/frames'
REMOTE_EVAL = f'{REMOTE_ROOT}/eval'
STATE_DOC = ROOT / 'docs' / 'Claude维护' / 'S2-LAB' / '01_接手状态.md'
ENV = f'export PYTHONPATH={REMOTE_REPO}/src; export OMP_NUM_THREADS=4; cd {REMOTE_REPO}; '


def read_password():
    password = os.environ.get('BUPT_PASSWORD')
    if not password and os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
            password = winreg.QueryValueEx(key, 'BUPT_PASSWORD')[0]
    if not password:
        raise RuntimeError('BUPT_PASSWORD 不可用')
    return password


def connect(port):
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(REMOTE_HOST, port=port, username=REMOTE_USER, password=read_password(), allow_agent=False,
                   look_for_keys=False, timeout=20, banner_timeout=30, auth_timeout=30)
    client.get_transport().set_keepalive(20)
    return client


def run(client, command, timeout=600, check=True):
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out, err = stdout.read().decode(errors='replace'), stderr.read().decode(errors='replace')
    status = stdout.channel.recv_exit_status()
    if check and status:
        raise RuntimeError(f'远端命令失败({status}): {command}\n{err or out}')
    return out + (('\n[stderr]\n' + err) if err.strip() else '')


def background(client, command, log_path):
    full = f'mkdir -p {REMOTE_ROOT}/logs; nohup bash -c {shlex.quote(ENV + command)} > {log_path} 2>&1 < /dev/null & echo $!'
    return run(client, full).strip()


def cmd_check(args):
    for port in args.ports:
        with connect(port) as c:
            gpu = run(c, 'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader', check=False).strip()
            py = run(c, f'{REMOTE_PY} -c "import torch,sys;print(sys.version.split()[0],torch.__version__,torch.cuda.is_available())"', check=False).strip()
            data = run(c, f'ls {REMOTE_S2FIX_DATA} 2>/dev/null | head; ls {REMOTE_ROOT} 2>/dev/null; ls {REMOTE_FRAMES} 2>/dev/null; nproc; df -h /home | tail -1', check=False).strip()
            procs = run(c, 'ps -eo pid,etime,cmd | grep -E "s2lab|s2fix|python" | grep -v grep | head -20', check=False).strip()
            print(f'== {port}\n{gpu}\n{py}\n{data}\n-- processes\n{procs}\n')


def cmd_sync(args):
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'scripts', 'tests'], cwd=ROOT).decode().strip()
    if dirty:
        raise SystemExit('有未提交改动，先提交：\n' + dirty)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    with connect(args.ports[0]) as c:
        run(c, f'mkdir -p {REMOTE_ROOT}/setup {REMOTE_RUNS} {REMOTE_ROOT}/logs {REMOTE_FRAMES}')
        bundle_remote = f'{REMOTE_ROOT}/setup/source_{revision}.bundle'
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / 'source.bundle'
            subprocess.run(['git', 'bundle', 'create', str(bundle), 'HEAD'], cwd=ROOT, check=True, capture_output=True)
            with c.open_sftp() as sftp:
                sftp.put(str(bundle), bundle_remote)
        script = (f'set -e; if [ ! -d {REMOTE_REPO}/.git ]; then git clone -q {bundle_remote} {REMOTE_REPO}; fi; '
                  f'cd {REMOTE_REPO}; git fetch -q {bundle_remote} HEAD; git checkout -q --detach {revision}; git rev-parse HEAD')
        out = run(c, script).strip().splitlines()[-1]
        if out != revision:
            raise RuntimeError(f'远端提交 {out} ≠ 本地 {revision}')
        print(json.dumps(dict(synced=revision)))


def cmd_put_frames(args):
    local = Path('D:/paper/outputs/experiments/s1_fix_20261017/data')
    with connect(args.ports[0]) as c:
        run(c, f'mkdir -p {REMOTE_FRAMES}')
        have = set(run(c, f'ls {REMOTE_FRAMES}').split())
        with c.open_sftp() as sftp:
            for date in args.dates:
                name = f'frames_{date}.npz'
                if name in have:
                    print(f'{name} 已存在'); continue
                src = local / name
                if not src.exists():
                    print(f'{name} 本地不存在'); continue
                sftp.put(str(src), f'{REMOTE_FRAMES}/{name}')
                print(f'已上传 {name} ({src.stat().st_size // 1_000_000} MB)')


def cmd_build_data(args):
    with connect(args.port) as c:
        pid = background(c, f'{REMOTE_PY} scripts/s2lab/build_data.py --dataset-dir {REMOTE_DATA} --s2fix-data {REMOTE_S2FIX_DATA} '
                            f'--frames-dir {REMOTE_FRAMES} --workers {args.workers} {args.extra}', f'{REMOTE_ROOT}/logs/build_data_{args.port}.log')
        print(f'build_data started on {args.port}, pid {pid}; log {REMOTE_ROOT}/logs/build_data_{args.port}.log')


def cmd_calibrate(args):
    with connect(args.port) as c:
        pid = background(c, f'{REMOTE_PY} scripts/s2lab/calibrate.py --dataset-dir {REMOTE_DATA} {args.extra}', f'{REMOTE_ROOT}/logs/calibrate.log')
        print(f'calibrate started on {args.port}, pid {pid}; log {REMOTE_ROOT}/logs/calibrate.log')


def cmd_launch(args):
    with connect(args.ports[0]) as c:
        if not args.keep_queue:
            run(c, f'{ENV}{REMOTE_PY} -c "from satellite_rl.s2lab.matrix import write_matrix; write_matrix(\'{REMOTE_RUNS}\')"')
    for port in args.ports:
        with connect(port) as c:
            only = (' --only ' + ' '.join(args.only)) if args.only else ''
            pid = background(c, f'{REMOTE_PY} scripts/s2lab/lane.py --runs-dir {REMOTE_RUNS} --dataset-dir {REMOTE_DATA} '
                                f'--lane {port} --window-hours {args.window_hours} --parallel {args.parallel}{only}',
                             f'{REMOTE_ROOT}/logs/lane_{port}_{datetime.now():%m%d_%H%M}.log')
            print(f'lane {port} started pid {pid}')
    window = dict(launched_at=datetime.now().astimezone().isoformat(timespec='minutes'), window_hours=args.window_hours,
                  ends_at=datetime.fromtimestamp(time.time() + args.window_hours * 3600).astimezone().isoformat(timespec='minutes'),
                  ports=args.ports, parallel=args.parallel)
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    (LOCAL_OUT / 'window.json').write_text(json.dumps(window, indent=1), encoding='utf-8')
    print(json.dumps(window))


def cmd_pause(args):
    for port in args.ports:
        with connect(port) as c:
            out = run(c, 'pkill -TERM -f "scripts/s2lab/lane.py" ; pkill -TERM -f "scripts/s2lab/train_run.py"; sleep 1; '
                         'ps -eo pid,etime,cmd | grep -E "s2lab/(lane|train_run)" | grep -v grep', check=False)
            print(f'== {port} 已发送 SIGTERM；仍在运行（存盘中）：\n{out.strip()}')
    print('训练进程会在当前步结束后保存断点再退出；稍后用 status 确认全部为 paused。')


def cmd_ps(args):
    for port in args.ports:
        with connect(port) as c:
            out = run(c, 'ps -eo pid,etime,%cpu,%mem,cmd | grep -E "s2lab" | grep -v grep', check=False)
            gpu = run(c, 'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader', check=False).strip()
            print(f'== {port} GPU {gpu}\n{out.strip()}\n')


def fetch_status_rows(c):
    out = run(c, f'cd {REMOTE_RUNS} 2>/dev/null && for d in */; do f="$d/status.json"; [ -f "$f" ] && echo "=== $d" && cat "$f"; done; '
                 f'for f in _lanes/*.json; do [ -f "$f" ] && echo "=== $f" && cat "$f"; done', check=False)
    statuses, lanes = {}, {}
    current, buf = None, []

    def flush():
        if current and buf:
            try:
                value = json.loads('\n'.join(buf))
            except json.JSONDecodeError:
                return
            (lanes if current.startswith('_lanes') else statuses)[current.strip('/').split('/')[-1].replace('.json', '')] = value
    for line in out.splitlines():
        if line.startswith('=== '):
            flush(); current, buf = line[4:], []
        else:
            buf.append(line)
    flush()
    return statuses, lanes


def cmd_status(args):
    with connect(args.ports[0]) as c:
        statuses, lanes = fetch_status_rows(c)
        matrix = json.loads(run(c, f'cat {REMOTE_RUNS}/matrix.json', check=False) or '[]')
        calib = run(c, f'cat {REMOTE_DATA}/calibration.md 2>/dev/null | head -60', check=False)
        splits = run(c, f'{REMOTE_PY} -c "import json;s=json.load(open(\'{REMOTE_DATA}/splits.json\'));print({{k:len(v) for k,v in s.items()}})"', check=False).strip()
    rows = []
    for run_cfg in matrix:
        st = statuses.get(run_cfg['run_id'], dict(state='new', step=0))
        rows.append(dict(run_id=run_cfg['run_id'], group=run_cfg['group'], label=run_cfg.get('label', ''), seed=run_cfg['seed'],
                         state=st.get('state', 'new'), step=st.get('step', 0), total=run_cfg['total_steps'],
                         elapsed_hours=st.get('elapsed_hours', 0.), lane=st.get('lane', ''), updated_at=st.get('updated_at', ''),
                         best_dev=st.get('best_dev'), last_dev=st.get('last_dev'),
                         stale=heartbeat_stale(st) if st.get('state') in ('running', 'starting', 'claimed') else False,
                         pause_reason=st.get('pause_reason', '')))
    lane_lines = ['## 排队器（每卡一个）', '', '| 卡 | 进程 | 在跑 | 剩余小时 | 更新 | 结束 |', '| --- | --- | --- | --- | --- | --- |']
    for name, lane in sorted(lanes.items()):
        lane_lines.append(f"| {lane['lane']} | {lane['pid']} | {', '.join(lane['running'])} | {lane['remaining_hours']:.2f} | {lane['updated_at'][:16]} | {'是' if lane.get('finished') else ''} |")
    window = (LOCAL_OUT / 'window.json')
    note = f'数据集 {REMOTE_DATA}：集合规模 {splits}。' + (f" 最近启动窗口：{window.read_text(encoding='utf-8')}" if window.exists() else '')
    extra = '\n'.join(lane_lines) + '\n\n## 标定表（规则直出，来自 calibration.md）\n\n' + calib
    text = render_state_markdown(rows, note, extra)
    STATE_DOC.parent.mkdir(parents=True, exist_ok=True)
    header = (STATE_DOC.read_text(encoding='utf-8').split('<!-- AUTO_STATE_START -->')[0] if STATE_DOC.exists() else '')
    STATE_DOC.write_text(header + '<!-- AUTO_STATE_START -->\n' + text + '<!-- AUTO_STATE_END -->\n', encoding='utf-8')
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    (LOCAL_OUT / 'state.json').write_text(json.dumps(dict(fetched_at=datetime.now().isoformat(timespec='minutes'), runs=rows, lanes=lanes), ensure_ascii=False, indent=1), encoding='utf-8')
    print(text)


def cmd_fetch(args):
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    with connect(args.ports[0]) as c:
        archive = f'{REMOTE_ROOT}/fetch.tgz'
        run(c, f'cd {REMOTE_ROOT} && tar czf {archive} --ignore-failed-read runs/matrix.json runs/queue.json runs/_lanes runs/_logs '
               f'$(ls -d runs/*/ 2>/dev/null | sed "s#/\\$##" | while read d; do for f in status.json config.json curve.jsonl dev.jsonl train.log; do [ -f "$d/$f" ] && echo "$d/$f"; done; done) '
               f'eval data/{DATASET_NAME}/config.json data/{DATASET_NAME}/splits.json data/{DATASET_NAME}/scenes.csv data/{DATASET_NAME}/calibration.csv data/{DATASET_NAME}/calibration.md logs 2>/dev/null; ls -la {archive}', timeout=900)
        with c.open_sftp() as sftp:
            sftp.get(archive, str(LOCAL_OUT / 'fetch.tgz'))
    subprocess.run(['tar', 'xzf', str(LOCAL_OUT / 'fetch.tgz'), '-C', str(LOCAL_OUT)], check=True)
    print(f'已下载到 {LOCAL_OUT}')


def cmd_evaluate(args):
    with connect(args.port) as c:
        pid = background(c, f'{REMOTE_PY} scripts/s2lab/evaluate.py --dataset-dir {REMOTE_DATA} --runs-dir {REMOTE_RUNS} --split {args.split} '
                            f'--out {REMOTE_EVAL} --rules greedy fixed random --run-ids {args.extra}', f'{REMOTE_ROOT}/logs/evaluate_{args.split}.log')
        print(f'evaluate started on {args.port}, pid {pid}')


def cmd_tail(args):
    with connect(args.port) as c:
        if args.run_id:
            print(run(c, f'tail -n {args.lines} {REMOTE_RUNS}/{args.run_id}/train.log', check=False))
        else:
            print(run(c, f'ls -t {REMOTE_ROOT}/logs | head -5; for f in $(ls -t {REMOTE_ROOT}/logs | head -{args.count}); do echo "=== $f"; tail -n {args.lines} {REMOTE_ROOT}/logs/$f; done', check=False))


def cmd_sh(args):
    with connect(args.port) as c:
        print(run(c, ENV + args.cmd, timeout=args.timeout, check=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)

    def common(p):
        p.add_argument('--ports', type=int, nargs='*', default=list(REMOTE_PORTS))
        p.add_argument('--port', type=int, default=REMOTE_PORTS[0])
        p.add_argument('--extra', default='')
    for name in ('check', 'sync', 'pause', 'status', 'fetch', 'ps'):
        common(sub.add_parser(name))
    p = sub.add_parser('put-frames'); common(p); p.add_argument('--dates', nargs='+', required=True)
    p = sub.add_parser('build-data'); common(p); p.add_argument('--workers', type=int, default=16)
    p = sub.add_parser('calibrate'); common(p)
    p = sub.add_parser('launch'); common(p); p.add_argument('--window-hours', type=float, required=True); p.add_argument('--parallel', type=int, default=1)
    p.add_argument('--only', nargs='*'); p.add_argument('--keep-queue', action='store_true')
    p = sub.add_parser('evaluate'); common(p); p.add_argument('--split', default='confirm')
    p = sub.add_parser('tail'); common(p); p.add_argument('--run-id'); p.add_argument('--lines', type=int, default=30); p.add_argument('--count', type=int, default=3)
    p = sub.add_parser('sh'); common(p); p.add_argument('--cmd', required=True); p.add_argument('--timeout', type=int, default=600)
    args = ap.parse_args()
    globals()['cmd_' + args.command.replace('-', '_')](args)


if __name__ == '__main__':
    main()
