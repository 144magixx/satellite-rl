"""第四轮条件组：按事先定好的开发集门槛 G 决定追加哪些运行（只看开发集，不看两套确认集）。

门槛 G(候选, 父组)：同种子配对的开发集 SGM（20 万步内按开发集 SGM 选出的模型，即 eval_dev_best_sgm.csv）平均至少 +0.005，
且 3 个种子中至少 2 个提高。
  - ＋L＋B′：＋A 对探索组合平均 ≤ −0.005，且 ＋A＋B′ 对探索组合未过 G。
  - ＋指针＋γ：指针、以及至少一种 γ 各自对探索组合过 G；两种 γ 都过就取开发集均值高的，相同取 0.995。
  - 补种子 4—6：除上面条件组之外，所有对探索组合过 G 的配置里开发集均值最高的一个，与探索组合一起补种子 4、5、6；
    启动时剩余窗口要不少于"两者已实测的较慢单运行耗时＋3 小时"。
  python scripts/s2lab/round4_followup.py plan [--stage cond|seeds]
  python scripts/s2lab/round4_followup.py launch --stage cond|seeds --window-hours H
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ['S2LAB_ROUND'] = '4'
sys.path.insert(0, str(Path(__file__).parent))
from _common import ROOT  # noqa: E402,F401
from report_figures2 import read_csv  # noqa: E402
from satellite_rl.s2lab.matrix import ROUND4, ROUND4_COND, SEEDS, r4_run  # noqa: E402

LOCAL = Path('D:/paper/outputs/experiments/s2lab_round4_20261028')


def run_dirs():
    base = LOCAL / 'account_B' / 'runs4'
    return {d.name: d for d in base.iterdir() if d.is_dir() and not d.name.startswith('_')} if base.exists() else {}


def dev_sgm(d):
    p = d / 'eval_dev_best_sgm.csv'
    return float(np.mean([r['SGM'] for r in read_csv(p)])) if p.exists() else None


def gate(name, parent, dirs, seeds=SEEDS):
    a = [dev_sgm(dirs[f'{name}-s{s}']) if f'{name}-s{s}' in dirs else None for s in seeds]
    b = [dev_sgm(dirs[f'{parent}-s{s}']) if f'{parent}-s{s}' in dirs else None for s in seeds]
    if any(v is None for v in a + b):
        return None
    d = [x - y for x, y in zip(a, b)]
    return dict(ok=bool(np.mean(d) >= .005 and sum(v > 0 for v in d) >= 2), diff=float(np.mean(d)), seeds=[round(v, 4) for v in d],
                dev=float(np.mean(a)))


def hours(d):
    """单运行实际耗时（行为克隆组加上示范与克隆两段，它们不在 elapsed_hours 里）。"""
    s = json.loads((d / 'status.json').read_text(encoding='utf-8'))
    h = float(s.get('elapsed_hours', 0))
    if (d / 'bc.json').exists():
        bc = json.loads((d / 'bc.json').read_text(encoding='utf-8'))
        h += (bc.get('demo_seconds', 0) + bc.get('clone_seconds', 0)) / 3600
    return h


def ptrg_name(gamma):
    return f"r4-ptrg{gamma:g}".replace('.', '')             # r4-ptrg0995 / r4-ptrg1：两种 γ 的组合名字不同，避免互相顶替


def plan(stage):
    """10-28 按 Astra 代码复核修正：比较对象的三种子结果必须齐全才下结论（缺就"待定"、不追加）；
    补种子按事先顺序：先在全部候选里取开发集均值最高者，再看它过不过 G，不过就不补，不换次名。"""
    dirs = run_dirs()
    g = {n: gate(n, 'r4-base', dirs) for n in ROUND4 if n != 'r4-base'}
    g['r4-AB/A'] = gate('r4-AB', 'r4-A', dirs)
    out = dict(gates=g, runs=[], pending=[])
    for k, v in g.items():
        print(f"{k:12s} " + (f"开发集 dSGM {v['diff']:+.4f} {v['seeds']} → {'过' if v['ok'] else '不过'}" if v else '数据不全'))
    if stage == 'cond':
        a, ab = g['r4-A'], g['r4-AB']
        if a is None or ab is None:
            out['pending'].append('r4-LB')
        elif a['diff'] <= -.005 and not ab['ok']:
            out['runs'].append(('r4-LB', ROUND4_COND['r4-LB'], {}))
        need = ('r4-ptr', 'r4-g995', 'r4-g1')
        if any(g[n] is None for n in need):
            out['pending'].append('r4-ptrg')
        else:
            gammas = [(n, g[n]) for n in ('r4-g995', 'r4-g1') if g[n]['ok']]
            if g['r4-ptr']['ok'] and gammas:
                best = max(gammas, key=lambda nv: (nv[1]['dev'], nv[0] == 'r4-g995'))[0]   # 同分取 0.995
                gamma = ROUND4[best][3]['gamma']
                out['runs'].append((ptrg_name(gamma), ('struct', f'＋块指针头＋γ={gamma}', 'r4-base', dict(gamma=gamma), {}, dict(pointer=True)), {}))
    else:
        cond = json.loads((LOCAL / 'followup_cond.json').read_text(encoding='utf-8')) if (LOCAL / 'followup_cond.json').exists() else None
        if cond is None or cond['pending']:
            out['pending'].append('条件组尚未定')
        candidates = [n for n in ROUND4 if n != 'r4-base'] + [r[0] for r in (cond or {}).get('runs', [])]
        for n in candidates:
            if n not in g:
                g[n] = gate(n, 'r4-base', dirs)
        missing = [n for n in candidates if g[n] is None]
        if missing:
            out['pending'] += missing
        elif not out['pending']:
            best = max(candidates, key=lambda n: (g[n]['dev'], -candidates.index(n)))          # 同分按方案表先后
            out['candidate'], out['candidate_gate'] = best, g[best]
            if g[best]['ok']:
                out['runs'].append(('seeds', best, {}))
    print('待定：', out['pending'], '追加：', [r[0] if r[0] != 'seeds' else f'补种子 {r[1]}' for r in out['runs']])
    (LOCAL / f'followup_{stage}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    return out, dirs


def launch(stage, window_hours):
    import remote2
    out, dirs = plan(stage)
    if stage == 'seeds' and out['pending']:
        print('有比较对象结果未齐，先不补种子')
        return
    if out['pending']:
        print('这些条件组的比较对象还没齐，下次再判：', out['pending'])   # 已能判定的先追加；重复追加按运行名去重
    runs = []
    for name, spec, _ in out['runs']:
        if name == 'seeds':
            cand = spec
            defs = dict(ROUND4)
            if cand == 'r4-LB':
                defs[cand] = ROUND4_COND[cand]
            if cand.startswith('r4-ptrg'):
                defs[cand] = json.loads((dirs[f'{cand}-s1'] / 'config.json').read_text(encoding='utf-8'))
            slow = max(hours(dirs[f'{n}-s{s}']) for n in (cand, 'r4-base') for s in SEEDS)
            if window_hours < slow + 3:
                print(f'剩余 {window_hours:.1f} h 不足（需要 {slow + 3:.1f} h），不补种子')
                return
            for n in (cand, 'r4-base'):
                for s in (4, 5, 6):
                    if isinstance(defs[n], dict):          # 条件组从已跑配置复制
                        run = dict(defs[n], run_id=f'{n}-s{s}', seed=s)
                    else:
                        grp, label, parent, agent, size, other = defs[n]
                        run = r4_run(n, s, grp, label, parent, agent, size, **other)
                    runs.append(run)
        else:
            grp, label, parent, agent, size, other = spec
            runs += [r4_run(name, s, grp, label, parent, agent, size, **other) for s in SEEDS]
    if not runs:
        print('没有要追加的运行')
        return
    p = remote2.paths('B')
    extra_path = f"{p['runs']}/extra_runs.json"
    with remote2.connect(remote2.first_port('B')) as c:
        with c.open_sftp() as sftp:
            try:
                with sftp.open(extra_path) as f:
                    old = json.loads(f.read())
            except IOError:
                old = []
            ids = {r['run_id'] for r in old}
            merged = old + [r for r in runs if r['run_id'] not in ids]
            with sftp.open(extra_path, 'w') as f:
                f.write(json.dumps(merged, ensure_ascii=False))
        print(remote2.run(c, p['env'] + f"{p['py']} scripts/s2lab/write_matrix.py --runs-dir {p['runs']} --round 4 --account B --extra {extra_path}"))
    remote2.cmd_launch(argparse.Namespace(accounts=['B'], window_hours=window_hours, parallel=2, ports=None, only=None))


def add_runs(runs, window_hours):
    """把运行追加进 B 账号的 extra_runs.json（按运行名去重）、重写队列；排队器在跑就自己取，没在跑就启动。"""
    import remote2
    p = remote2.paths('B')
    extra_path = f"{p['runs']}/extra_runs.json"
    with remote2.connect(remote2.first_port('B')) as c:
        with c.open_sftp() as sftp:
            try:
                with sftp.open(extra_path) as f:
                    old = json.loads(f.read())
            except IOError:
                old = []
            ids = {r['run_id'] for r in old}
            merged = old + [r for r in runs if r['run_id'] not in ids]
            with sftp.open(extra_path, 'w') as f:
                f.write(json.dumps(merged, ensure_ascii=False))
        print(remote2.run(c, p['env'] + f"{p['py']} scripts/s2lab/write_matrix.py --runs-dir {p['runs']} --round 4 --account B --extra {extra_path}"))
    remote2.cmd_launch(argparse.Namespace(accounts=['B'], window_hours=window_hours, parallel=2, ports=None, only=None))


def explore(names, window_hours):
    from satellite_rl.s2lab.matrix import ROUND4_EXPLORE
    runs = []
    for name in names:
        grp, label, parent, agent, size, other = ROUND4_EXPLORE[name]
        runs += [r4_run(name, s, grp, label, parent, agent, size, **other) for s in SEEDS]
    add_runs(runs, window_hours)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('explore'); p.add_argument('--names', nargs='+', required=True); p.add_argument('--window-hours', type=float, required=True)
    p = sub.add_parser('plan'); p.add_argument('--stage', default='cond', choices=('cond', 'seeds'))
    p = sub.add_parser('launch'); p.add_argument('--stage', required=True, choices=('cond', 'seeds')); p.add_argument('--window-hours', type=float, required=True)
    a = ap.parse_args()
    if a.cmd == 'explore':
        explore(a.names, a.window_hours)
    else:
        plan(a.stage) if a.cmd == 'plan' else launch(a.stage, a.window_hours)


if __name__ == '__main__':
    main()
