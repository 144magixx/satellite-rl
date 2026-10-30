"""第三轮后续：按方案第 1 节事先冻结的规则，用开发集决定"续训到 30 万步"和"组合版"，并在远端排队。

  python scripts/s2lab/round3_followup.py plan            # 读本地 fetch 下来的数据，打印决定并写 followup_plan.json
  python scripts/s2lab/round3_followup.py launch --window-hours H   # 按 plan 在远端复制续训目录、写队列、起排队器

规则（只看开发集，不看确认集）：
- 续训：组内各种子"去重后最后 5 个开发集节点均值 − 再往前 5 个"的均值 ≥ 0.01，且最后 8 点斜率均值 > 0 → 该组与其父组
  三种子都从 20 万步断点续到 30 万步（复制成 <运行>-c30，不覆盖 20 万步结果）。
- 组合版：与父组同种子比，开发集 SGM 均值至少高 0.005、至少两个种子同向，W 和最差 10% 都不降超过 0.005。
  动作/功率链（L→A→B′、A→P1→P2→P3）中，只有父组也被接受（或父组是 R0）的环节才算；取链上开发集 SGM 最高者。
  其余单项（训练、状态补充、去除）对 R0 判定，被接受的都合进来；"去全局令牌"与需要全局令牌的特征冲突时保留特征。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import ROOT, ACCOUNTS  # noqa: E402
from report_figures2 import read_csv, read_jsonl  # noqa: E402
from satellite_rl.s2lab.matrix import ROUND3, r3_run, SEEDS  # noqa: E402

LOCAL = Path('D:/paper/outputs/experiments/s2lab_round3_20261027')
CHAIN = ('r3-legal', 'r3-cond', 'r3-condtop8', 'r3-p1', 'r3-p2', 'r3-p3')


def run_dirs():
    out = {}
    for acc in ('A', 'B'):
        base = LOCAL / f'account_{acc}' / 'runs3'
        if base.exists():
            for d in base.iterdir():
                if d.is_dir() and not d.name.startswith('_'):
                    out[d.name] = (acc, d)
    return out


def dev_metrics(d):
    p = d / 'eval_dev_best_sgm.csv'
    if not p.exists():
        return None
    rows = read_csv(p)
    return {c: float(np.nanmean([r[c] for r in rows])) for c in ('SGM', 'W', 'worst10')}


def trend(d):
    dev = list({r['step']: r for r in read_jsonl(d / 'dev.jsonl')}.values())
    s = np.array([r['step'] for r in dev], float)
    v = np.array([r['SGM'] for r in dev])
    if len(v) < 10:
        return None
    return float(v[-5:].mean() - v[-10:-5].mean()), float(np.polyfit(s[-8:] / 1e5, v[-8:], 1)[0])


def passes(name, ref, dirs):
    a = [dev_metrics(dirs[f'{name}-s{s}'][1]) for s in SEEDS if f'{name}-s{s}' in dirs]
    b = [dev_metrics(dirs[f'{ref}-s{s}'][1]) for s in SEEDS if f'{ref}-s{s}' in dirs]
    if len(a) < 3 or len(b) < 3 or any(x is None for x in a + b):
        return None
    d = {c: [x[c] - y[c] for x, y in zip(a, b)] for c in ('SGM', 'W', 'worst10')}
    ok = (np.mean(d['SGM']) >= .005 and sum(v > 0 for v in d['SGM']) >= 2
          and np.mean(d['W']) >= -.005 and np.mean(d['worst10']) >= -.005)
    return dict(ok=bool(ok), dSGM=float(np.mean(d['SGM'])), dW=float(np.mean(d['W'])), d10=float(np.mean(d['worst10'])),
                seeds=[round(v, 4) for v in d['SGM']], devSGM=float(np.mean([x['SGM'] for x in a])))


def plan():
    dirs = run_dirs()
    done = {k for k, (_, d) in dirs.items() if (d / 'eval_dev_best_sgm.csv').exists()}
    names = [n for n in ROUND3 if all(f'{n}-s{s}' in done for s in SEEDS)]
    print(f'已完成组 {len(names)}/{len(ROUND3)}')
    # 续训
    cont = set()
    trends = {}
    for n in names:
        t = [trend(dirs[f'{n}-s{s}'][1]) for s in SEEDS]
        if any(x is None for x in t):
            continue
        g, sl = float(np.mean([x[0] for x in t])), float(np.mean([x[1] for x in t]))
        trends[n] = (g, sl)
        if g >= .01 and sl > 0:
            cont.add(n)
            if ROUND3[n][2]:
                cont.add(ROUND3[n][2])
    # 组合版
    verdict = {}
    for n in names:
        parent = ROUND3[n][2]
        if parent and parent in names and ROUND3[n][0] != 'algo':
            verdict[n] = passes(n, parent, dirs)
    accepted_chain = []
    for n in CHAIN:
        v = verdict.get(n)
        parent = ROUND3[n][2]
        if v and v['ok'] and (parent == 'r3-r0' or parent in accepted_chain):
            accepted_chain.append(n)
    best_chain = max(accepted_chain, key=lambda n: verdict[n]['devSGM']) if accepted_chain else None
    singles = [n for n in names if n not in CHAIN and n != 'r3-r0' and ROUND3[n][0] != 'algo' and verdict.get(n) and verdict[n]['ok']]
    agent, size = {}, {}
    if best_chain:
        agent.update(ROUND3[best_chain][3])
    feats = []
    for n in singles:
        agent.update(ROUND3[n][3])
        s = dict(ROUND3[n][4])
        feats += s.pop('feats', [])
        size.update(s)
    feats = list(dict.fromkeys(feats))
    if size.get('no_global') and ({'future', 'gspec'} & set(feats)):
        size.pop('no_global')
    if feats:
        size['feats'] = feats
    combo = dict(agent=agent, size=size, members=([best_chain] if best_chain else []) + singles)
    # 探索组合（10-27 09:25 加，不属于事先冻结的规则，报告中单列标明）：冻结规则要求 W、最差10% 都不降超过 0.005，
    # 结果一项都没通过，组合版为空。探索规则放宽为：对 R0 开发集 SGM 至少高 0.005、至少两个种子同向，W 降不超过 0.01、最差10% 降不超过 0.02。
    loose = {}
    for n in names:
        if n == 'r3-r0' or ROUND3[n][0] == 'algo':
            continue
        v = passes(n, 'r3-r0', dirs)
        if v:
            v['ok'] = bool(v['dSGM'] >= .005 and sum(x > 0 for x in v['seeds']) >= 2 and v['dW'] >= -.01 and v['d10'] >= -.02)
            loose[n] = v
    x_chain = [n for n in CHAIN if loose.get(n, {}).get('ok')]
    x_best_chain = max(x_chain, key=lambda n: loose[n]['devSGM']) if x_chain else None
    x_singles = [n for n in names if n not in CHAIN and n in loose and loose[n]['ok']]
    x_agent, x_size, x_feats = {}, {}, []
    if x_best_chain:
        x_agent.update(ROUND3[x_best_chain][3])
    for n in x_singles:
        x_agent.update(ROUND3[n][3])
        s = dict(ROUND3[n][4])
        x_feats += s.pop('feats', [])
        x_size.update(s)
    x_feats = list(dict.fromkeys(x_feats))
    if x_size.get('no_global') and ({'future', 'gspec'} & set(x_feats)):
        x_size.pop('no_global')
    if x_feats:
        x_size['feats'] = x_feats
    explore = dict(agent=x_agent, size=x_size, members=([x_best_chain] if x_best_chain else []) + x_singles, verdict=loose)
    out = dict(trends=trends, continue_groups=sorted(cont), verdict=verdict, combo=combo, explore=explore)
    (LOCAL / 'followup_plan.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('续训到 30 万步的组：', sorted(cont))
    for n, v in verdict.items():
        print(f"  {n:16s} 对父组 开发集 dSGM {v['dSGM']:+.4f} {v['seeds']} dW {v['dW']:+.4f} d10 {v['d10']:+.4f} → {'接受' if v['ok'] else '不接受'}"
              if v else f'  {n:16s} 数据不全')
    print('组合版（冻结规则）：', combo)
    for n, v in loose.items():
        print(f"  [探索] {n:16s} 对 R0 dSGM {v['dSGM']:+.4f} {v['seeds']} dW {v['dW']:+.4f} d10 {v['d10']:+.4f} → {'接受' if v['ok'] else '不接受'}")
    print('探索组合：', {k: v for k, v in explore.items() if k != 'verdict'})
    return out


def launch(window_hours):
    import os
    os.environ['S2LAB_ROUND'] = '3'
    import remote2
    p = json.loads((LOCAL / 'followup_plan.json').read_text(encoding='utf-8'))
    dirs = run_dirs()
    extra = {'A': [], 'B': []}
    for n in p['continue_groups']:
        for s in SEEDS:
            acc = dirs[f'{n}-s{s}'][0]
            grp, label, parent, agent, size, other = ROUND3[n]
            other = dict(other)
            algo = other.pop('algo', 'sac3')
            run = r3_run(n, s, grp, label, parent, agent, size, algo=algo, **other)
            run.update(run_id=f'{n}-s{s}-c30', total_steps=300000, continued_from=f'{n}-s{s}', account=acc, label=f'{label}（30万步）')
            extra[acc].append(run)
    if len(p['combo']['members']) > 1:          # 只接受一项时组合版就是该组本身，不重跑
        for i, s in enumerate(SEEDS):
            acc = 'AB'[i % 2]
            run = r3_run('r3-combo', s, 'combo', '组合版', 'r3-r0', p['combo']['agent'], p['combo']['size'])
            run.update(account=acc, combo_members=p['combo']['members'])
            extra[acc].append(run)
    if p['explore']['members']:
        for i, s in enumerate(SEEDS):
            acc = 'BA'[i % 2]
            run = r3_run('r3-xcombo', s, 'combo', '探索组合（放宽规则）', 'r3-r0', p['explore']['agent'], p['explore']['size'])
            run.update(account=acc, combo_members=p['explore']['members'])
            extra[acc].append(run)
    for acc in ('A', 'B'):
        pa = remote2.paths(acc)
        with remote2.connect(remote2.first_port(acc)) as c:
            for run in extra[acc]:
                if 'continued_from' in run:
                    src, dst = f"{pa['runs']}/{run['continued_from']}", f"{pa['runs']}/{run['run_id']}"
                    remote2.run(c, f"rm -rf {dst} && cp -r {src} {dst} && rm -f {dst}/status.json {dst}/eval_* {dst}/.claim")
            with c.open_sftp() as sftp:
                with sftp.open(f"{pa['runs']}/extra_runs.json", 'w') as f:
                    f.write(json.dumps(extra[acc], ensure_ascii=False))
            out = remote2.run(c, pa['env'] + f"{pa['py']} scripts/s2lab/write_matrix.py --runs-dir {pa['runs']} --round 3 --account {acc} "
                                             f"--extra {pa['runs']}/extra_runs.json")
            print(acc, '矩阵', out.strip(), '追加', [r['run_id'] for r in extra[acc]], flush=True)
    args = argparse.Namespace(accounts=['A', 'B'], window_hours=window_hours, parallel=2, ports=None, only=None)
    remote2.cmd_launch(args)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('plan')
    p = sub.add_parser('launch'); p.add_argument('--window-hours', type=float, required=True)
    a = ap.parse_args()
    plan() if a.cmd == 'plan' else launch(a.window_hours)


if __name__ == '__main__':
    main()
