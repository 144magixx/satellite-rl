"""第二轮远端操作：两个账号（A=u2025140002、B=u2024018003）各 8 卡，同一口令 BUPT_PASSWORD，/home 各自独立。
运行目录 <root>/runs2，数据 <root>/data/v1。本地用 base anaconda 的 python 运行。

  python scripts/s2lab/remote2.py check                         # 两账号全部卡：GPU、进程
  python scripts/s2lab/remote2.py sync                          # 本地 HEAD 同步到两账号 repo（需先提交）
  python scripts/s2lab/remote2.py push-data                     # A 的 data/v1 打包 → 本地 → B（B 缺时用）
  python scripts/s2lab/remote2.py write-queue                   # 两账号各写 runs2/matrix.json、queue.json（按账号分）
  python scripts/s2lab/remote2.py launch --window-hours 22 [--accounts A B] [--parallel 2]
  python scripts/s2lab/remote2.py status                        # 两账号运行状态 → 本地 state2.json 与 01_接手状态.md 第二轮区
  python scripts/s2lab/remote2.py fetch                         # 两账号 runs2 轻量文件（曲线、开发集、评价 CSV、图）→ 本地
  python scripts/s2lab/remote2.py pause [--accounts A B]        # 训练先存后退
  python scripts/s2lab/remote2.py sh --port P --cmd "..."      # 在某卡上执行（已 cd repo、设好 PYTHONPATH）
  python scripts/s2lab/remote2.py tail --port P --run-id X
"""
import argparse
import json
import shlex
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from _common import ROOT, REMOTE_HOST, ACCOUNTS, LOCAL_OUT2
from remote import read_password, run
from satellite_rl.s2lab.matrix import heartbeat_stale

STATE_DOC = ROOT / 'docs' / 'Claude维护' / 'S2-LAB' / '01_接手状态.md'


def account_of(port):
    return next(k for k, a in ACCOUNTS.items() if port in a['ports'])


def paths(acc):
    a = ACCOUNTS[acc]
    root = a['root']
    return dict(user=a['user'], py=a['py'], root=root, repo=f'{root}/repo', data=f'{root}/data/v1', runs=f'{root}/runs2',
                env=f'export PYTHONPATH={root}/repo/src; export OMP_NUM_THREADS=4; cd {root}/repo; ')


def connect(port):
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(REMOTE_HOST, port=port, username=ACCOUNTS[account_of(port)]['user'], password=read_password(), allow_agent=False,
                   look_for_keys=False, timeout=20, banner_timeout=40, auth_timeout=30)
    client.get_transport().set_keepalive(20)
    return client


def first_port(acc):
    return ACCOUNTS[acc]['ports'][0]


def background(client, p, command, log_path):
    full = f'mkdir -p {p["root"]}/logs; nohup bash -c {shlex.quote(p["env"] + command)} > {log_path} 2>&1 < /dev/null & echo $!'
    return run(client, full).strip()


def cmd_check(args):
    for acc in args.accounts:
        for port in ACCOUNTS[acc]['ports']:
            try:
                with connect(port) as c:
                    gpu = run(c, 'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader', check=False).strip()
                    procs = run(c, 'ps -eo pid,etime,cmd | grep -E "s2lab/(lane|train_run|evaluate)" | grep -v grep | wc -l', check=False).strip()
                    print(f'{acc} {port}: GPU {gpu}; s2lab 进程 {procs}', flush=True)
            except Exception as e:
                print(f'{acc} {port}: FAIL {type(e).__name__} {e}', flush=True)


def cmd_sync(args):
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'scripts', 'tests'], cwd=ROOT).decode().strip()
    if dirty:
        raise SystemExit('有未提交改动，先提交：\n' + dirty)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / 'source.bundle'
        subprocess.run(['git', 'bundle', 'create', str(bundle), 'HEAD'], cwd=ROOT, check=True, capture_output=True)
        for acc in args.accounts:
            p = paths(acc)
            with connect(first_port(acc)) as c:
                run(c, f'mkdir -p {p["root"]}/setup {p["runs"]} {p["root"]}/logs')
                remote_bundle = f'{p["root"]}/setup/source_{revision}.bundle'
                with c.open_sftp() as sftp:
                    sftp.put(str(bundle), remote_bundle)
                script = (f'set -e; if [ ! -d {p["repo"]}/.git ]; then git clone -q {remote_bundle} {p["repo"]}; fi; '
                          f'cd {p["repo"]}; git fetch -q {remote_bundle} HEAD; git -c advice.detachedHead=false checkout -q --detach {revision}; git rev-parse HEAD')
                out = [l.strip() for l in run(c, script).splitlines() if len(l.strip()) == 40][-1]
                if out != revision:
                    raise RuntimeError(f'{acc} 远端提交 {out} ≠ 本地 {revision}')
                print(json.dumps(dict(account=acc, synced=revision)), flush=True)


def cmd_push_data(args):
    src, dst = paths('A'), paths('B')
    local = LOCAL_OUT2 / 'transfer'
    local.mkdir(parents=True, exist_ok=True)
    with connect(first_port('A')) as c:
        run(c, f'cd {src["root"]}/data && tar czf {src["root"]}/data_v1.tgz v1/config.json v1/splits.json v1/scenes.csv v1/scenes && ls -la {src["root"]}/data_v1.tgz', timeout=1800)
        with c.open_sftp() as sftp:
            sftp.get(f'{src["root"]}/data_v1.tgz', str(local / 'data_v1.tgz'))
    print('下载完成', (local / 'data_v1.tgz').stat().st_size // 1_000_000, 'MB', flush=True)
    with connect(first_port('B')) as c:
        run(c, f'mkdir -p {dst["root"]}/data')
        with c.open_sftp() as sftp:
            sftp.put(str(local / 'data_v1.tgz'), f'{dst["root"]}/data/data_v1.tgz')
        out = run(c, f'cd {dst["root"]}/data && tar xzf data_v1.tgz && ls v1/scenes | wc -l && rm data_v1.tgz', timeout=1800)
        print('B 数据景数', out.strip(), flush=True)


def cmd_write_queue(args):
    for acc in args.accounts:
        p = paths(acc)
        with connect(first_port(acc)) as c:
            out = run(c, p['env'] + f'{p["py"]} scripts/s2lab/write_matrix.py --runs-dir {p["runs"]} --round 2 --account {acc}')
            queue = json.loads(run(c, f'cat {p["runs"]}/queue.json'))['order']
            print(f'{acc}: 矩阵 {out.strip()} 个运行，本账号队列 {len(queue)} 个：{" ".join(queue[:6])} …', flush=True)


def cmd_launch(args):
    windows = {}
    for acc in args.accounts:
        p = paths(acc)
        for port in (args.ports or ACCOUNTS[acc]['ports']):
            if account_of(port) != acc:
                continue
            with connect(port) as c:
                only = (' --only ' + ' '.join(args.only)) if args.only else ''
                pid = background(c, p, f'{p["py"]} scripts/s2lab/lane.py --runs-dir {p["runs"]} --dataset-dir {p["data"]} '
                                       f'--lane {port} --window-hours {args.window_hours} --parallel {args.parallel}{only}',
                                 f'{p["root"]}/logs/lane2_{port}_{datetime.now():%m%d_%H%M}.log')
                print(f'{acc} lane {port} started pid {pid}', flush=True)
        windows[acc] = dict(launched_at=datetime.now().astimezone().isoformat(timespec='minutes'), window_hours=args.window_hours,
                            ends_at=datetime.fromtimestamp(time.time() + args.window_hours * 3600).astimezone().isoformat(timespec='minutes'),
                            parallel=args.parallel)
    LOCAL_OUT2.mkdir(parents=True, exist_ok=True)
    old = json.loads((LOCAL_OUT2 / 'window.json').read_text(encoding='utf-8')) if (LOCAL_OUT2 / 'window.json').exists() else {}
    old.update(windows)
    (LOCAL_OUT2 / 'window.json').write_text(json.dumps(old, indent=1), encoding='utf-8')
    print(json.dumps(windows), flush=True)


def cmd_pause(args):
    for acc in args.accounts:
        for port in ACCOUNTS[acc]['ports']:
            with connect(port) as c:
                # 方括号写法避免 pkill 匹配到执行本命令的 shell 自身
                out = run(c, 'pkill -TERM -f "[s]cripts/s2lab/lane.py" ; pkill -TERM -f "[s]cripts/s2lab/train_run.py"; sleep 1; '
                             'ps -eo pid,cmd | grep -E "[s]2lab/(lane|train_run)" | wc -l', check=False)
                print(f'{acc} {port}: 已发 SIGTERM，仍在存盘的进程 {out.strip()}', flush=True)


def read_statuses(c, p):
    out = run(c, f'cd {p["runs"]} 2>/dev/null && for d in */; do f="$d/status.json"; [ -f "$f" ] && printf "\\n=== %s\\n" "$d" && cat "$f"; done; '
                 f'for f in _lanes/*.json; do [ -f "$f" ] && printf "\\n=== %s\\n" "$f" && cat "$f"; done', check=False)
    statuses, lanes, current, buf = {}, {}, None, []

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
    matrix = json.loads(run(c, f'cat {p["runs"]}/matrix.json', check=False) or '[]')
    queue = json.loads(run(c, f'cat {p["runs"]}/queue.json', check=False) or '{"order": []}')['order']
    return statuses, lanes, matrix, queue


def cmd_status(args):
    rows, lanes_all = [], {}
    for acc in args.accounts:
        p = paths(acc)
        with connect(first_port(acc)) as c:
            statuses, lanes, matrix, queue = read_statuses(c, p)
        cfg = {r['run_id']: r for r in matrix}
        for run_id in queue:
            st = statuses.get(run_id, dict(state='new', step=0))
            stale = heartbeat_stale(st) if st.get('state') in ('running', 'starting', 'claimed') else False
            rows.append(dict(account=acc, run_id=run_id, group=cfg[run_id]['group'], label=cfg[run_id].get('label', ''), seed=cfg[run_id]['seed'],
                             state=st.get('state', 'new') + ('(过期)' if stale else ''), step=st.get('step', 0), total=cfg[run_id]['total_steps'],
                             hours=st.get('elapsed_hours', 0.), lane=st.get('lane', ''), sps=st.get('sps'),
                             dev_sgm=(st.get('last_dev') or {}).get('SGM'), dev_w=(st.get('last_dev') or {}).get('W'),
                             best_sgm=st.get('best_dev_sgm'), confirm_sgm=st.get('confirm_sgm'), confirm_w=st.get('confirm_w'),
                             eval_done=bool(st.get('eval_done'))))
        for k, v in lanes.items():
            lanes_all[f'{acc}:{k}'] = v
    count = {}
    for r in rows:
        count[r['state']] = count.get(r['state'], 0) + 1
    def f(v, d=3):
        return '' if v is None else f'{v:.{d}f}'
    lines = [f'### 第二轮自动状态（{datetime.now():%Y-%m-%d %H:%M}）', '', f'共 {len(rows)} 个运行：' + '，'.join(f'{k} {v}' for k, v in sorted(count.items())), '',
             '| 账号 | 运行 | 组 | 状态 | 步数 | 用时h | 卡 | 步/秒 | 开发集 W/SGM（最近） | 开发集最好SGM | 确认集 W/SGM（best_sgm） |',
             '| --- | --- | --- | --- | --- | ---: | --- | ---: | --- | ---: | --- |']
    for r in rows:
        lines.append(f"| {r['account']} | {r['run_id']} | {r['group']} | {r['state']} | {r['step']}/{r['total']} | {r['hours']:.2f} | {r['lane']} | "
                     f"{f(r['sps'], 0)} | {f(r['dev_w'])}/{f(r['dev_sgm'])} | {f(r['best_sgm'])} | {f(r['confirm_w'])}/{f(r['confirm_sgm'])} |")
    lines += ['', '| 排队器 | 在跑 | 剩余小时 | 更新 | 结束 |', '| --- | --- | ---: | --- | --- |']
    for k, v in sorted(lanes_all.items()):
        lines.append(f"| {k} | {', '.join(v.get('running', []))} | {v.get('remaining_hours', 0):.2f} | {v.get('updated_at', '')[:16]} | {'是' if v.get('finished') else ''} |")
    text = '\n'.join(lines) + '\n'
    LOCAL_OUT2.mkdir(parents=True, exist_ok=True)
    (LOCAL_OUT2 / 'state2.json').write_text(json.dumps(dict(fetched_at=datetime.now().isoformat(timespec='minutes'), runs=rows, lanes=lanes_all),
                                                       ensure_ascii=False, indent=1), encoding='utf-8')
    doc = STATE_DOC.read_text(encoding='utf-8')
    start, end = '<!-- AUTO_STATE2_START -->', '<!-- AUTO_STATE2_END -->'
    block = f'{start}\n{text}{end}'
    doc = doc.split(start)[0] + block + doc.split(end)[1] if start in doc else doc.rstrip() + '\n\n' + block + '\n'
    STATE_DOC.write_text(doc, encoding='utf-8')
    print(text)


def cmd_fetch(args):
    for acc in args.accounts:
        p = paths(acc)
        dest = LOCAL_OUT2 / f'account_{acc}'
        dest.mkdir(parents=True, exist_ok=True)
        with connect(first_port(acc)) as c:
            archive = f'{p["root"]}/fetch2.tgz'
            run(c, f'cd {p["root"]} && tar czf {archive} --ignore-failed-read runs2/matrix.json runs2/queue.json runs2/_lanes runs2/_logs '
                   f'$(ls -d runs2/*/ 2>/dev/null | sed "s#/\\$##" | while read d; do for f in status.json config.json curve.jsonl dev.jsonl train.log '
                   f'training.png eval_summary.json eval_dev_best_sgm.csv eval_dev_best.csv eval_confirm_best_sgm.csv eval_confirm_best.csv; '
                   f'do [ -f "$d/$f" ] && echo "$d/$f"; done; done) $(ls -d eval2 eval2_meta 2>/dev/null) 2>/dev/null; ls -la {archive}', timeout=900)
            with c.open_sftp() as sftp:
                sftp.get(archive, str(dest / 'fetch2.tgz'))
        subprocess.run(['tar', 'xzf', str(dest / 'fetch2.tgz'), '-C', str(dest)], check=True)
        print(f'{acc} 已下载到 {dest}', flush=True)


def cmd_sh(args):
    p = paths(account_of(args.port))
    with connect(args.port) as c:
        print(run(c, p['env'] + args.cmd, timeout=args.timeout, check=False))


def cmd_tail(args):
    p = paths(account_of(args.port))
    with connect(args.port) as c:
        print(run(c, f'tail -n {args.lines} {p["runs"]}/{args.run_id}/train.log', check=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)

    def common(p):
        p.add_argument('--accounts', nargs='*', default=['A', 'B'])
    for name in ('check', 'sync', 'push-data', 'write-queue', 'pause', 'status', 'fetch'):
        common(sub.add_parser(name))
    p = sub.add_parser('launch'); common(p)
    p.add_argument('--window-hours', type=float, required=True); p.add_argument('--parallel', type=int, default=2)
    p.add_argument('--ports', type=int, nargs='*'); p.add_argument('--only', nargs='*')
    p = sub.add_parser('sh'); p.add_argument('--port', type=int, required=True); p.add_argument('--cmd', required=True); p.add_argument('--timeout', type=int, default=600)
    p = sub.add_parser('tail'); p.add_argument('--port', type=int, required=True); p.add_argument('--run-id', required=True); p.add_argument('--lines', type=int, default=30)
    args = ap.parse_args()
    globals()['cmd_' + args.command.replace('-', '_')](args)


if __name__ == '__main__':
    main()
