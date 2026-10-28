"""S3-LAB 运行器：对冻结输入（集 × 方案 × 场景）跑第三阶段方法，逐行写结果。

  python scripts/s3lab/run_shard.py --mode main --sets old new --workers 18 --out D:/paper/outputs/experiments/s3lab_20261029/runs
模式：
  main      主矩阵：δ=0，十种（含不细调），两段式 3 个求解器种子，记 0.1/0.3/1/3/10 秒时间点
  tradeoff  取舍曲线全量：SCA，δ=0.005/0.01/0.02
  panel     小面板 24 景×8 套：SLSQP、两段式×3 种子在三个 δ 下；SCA、两段式×3、WMMSE、SLSQP 放到 30 秒（δ=0）
  calib     开发集：两段式温度/步长、目标容量迭代阻尼的候选（按各自原生准则选）
同一输入的全部方法在同一个进程里依次运行（顺序按输入编号打乱），每进程单线程。可断点续跑。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from satellite_rl.s2lab.scenes import load_scene, read_config  # noqa: E402
from satellite_rl.s3lab.methods import MARKS, METHODS, Clock, protect  # noqa: E402
from satellite_rl.s3lab.plans import PLANS, load, tier_of  # noqa: E402
from satellite_rl.s3lab.problem import FixedPlan  # noqa: E402

ROOT = Path('D:/paper/outputs/experiments/s3lab_20261029')
DATASETS = {'dev': 'D:/paper/outputs/experiments/s2lab_round3_20261027/data/v1',
            'old': 'D:/paper/outputs/experiments/s2lab_round3_20261027/data/v1',
            'new': 'D:/paper/outputs/experiments/s2lab_round4_20261028/data/c2'}
SOLVER_SEEDS = (101, 102, 103)
DELTAS = (.005, .01, .02)
MAIN = ['none', 'sca', 'aravanis', 'choi_chan', 'proportional', 'equal', 'target_cap', 'maxmin', 'wmmse', 'slsqp']

_cache = {}


def _plan(plans_dir, s, pid):
    k = (s, pid)
    if k not in _cache:
        _cache[k] = load(Path(plans_dir) / s / f'{pid}.pkl')
    return _cache[k]


def panel_scenes(plans_dir):
    """小面板：每套评价集×每个负载档，按负载排序取 12.5/37.5/62.5/87.5% 分位的 4 景（只看输入）。"""
    out = []
    for s in ('old', 'new'):
        recs = _plan(plans_dir, s, 'greedy')['records']
        for st in ('ample', 'marginal', 'short'):
            rows = sorted([j for j, r in enumerate(recs) if r['stratum'] == st], key=lambda j: recs[j]['load_ratio'])
            out += [(s, rows[int(q * (len(rows) - 1))]) for q in (.125, .375, .625, .875)]
    return out


def jobs_for(mode, sets, plans_dir, params=None):
    """每个输入一组 (方法, 求解器种子, δ, 时限, 时间点, 额外参数)。"""
    if mode == 'main':
        jobs = []
        for m in MAIN:
            if m == 'aravanis':
                jobs += [(m, sd, 0., 10., MARKS, {}) for sd in SOLVER_SEEDS]
            else:
                jobs.append((m, 0, 0., 10., MARKS, {}))
        return [(s, pid, j) for s in sets for pid in PLANS for j in range(len(_plan(plans_dir, s, pid)['records']))], jobs
    if mode == 'tradeoff':
        return ([(s, pid, j) for s in sets for pid in PLANS for j in range(len(_plan(plans_dir, s, pid)['records']))],
                [('sca', 0, d, 10., (10.,), {}) for d in DELTAS])
    if mode == 'panel':
        jobs = []
        for d in DELTAS:
            jobs.append(('slsqp', 0, d, 10., (10.,), {}))
            jobs += [('aravanis', sd, d, 10., (10.,), {}) for sd in SOLVER_SEEDS]
        marks30 = MARKS + (30.,)
        jobs += [('sca', 0, 0., 30., marks30, {'tag': 't30'}), ('wmmse', 0, 0., 30., marks30, {'tag': 't30'}),
                 ('slsqp', 0, 0., 30., marks30, {'tag': 't30'})]
        jobs += [('aravanis', sd, 0., 30., marks30, {'tag': 't30'}) for sd in SOLVER_SEEDS]
        return [(s, pid, j) for (s, j) in panel_scenes(plans_dir) for pid in PLANS], jobs
    if mode == 'calib':
        jobs = []
        for T0 in (1e-5, 1e-4, 1e-3):
            for step0 in (2., 5., 20.):
                jobs += [('aravanis', sd, 0., 10., (10.,), dict(T0=T0, T1=T0 / 1000, step0=step0, step1=step0 / 10,
                                                               tag=f'T{T0:g}_s{step0:g}')) for sd in SOLVER_SEEDS]
        jobs += [('target_cap', 0, 0., 10., (10.,), dict(damping=dmp, tag=f'd{dmp:g}')) for dmp in (.3, .5, .8)]
        return [(s, pid, j) for s in sets for pid in PLANS for j in range(len(_plan(plans_dir, s, pid)['records']))], jobs
    raise ValueError(mode)


def run_input(args):
    plans_dir, s, pid, j, jobs, params = args
    obj = _plan(plans_dir, s, pid)
    r = obj['records'][j]
    ds = DATASETS[s]
    config = read_config(ds)
    t0 = time.perf_counter()
    fp = FixedPlan(load_scene(ds, r['key']), config.physics_obj(), r['demand'], r['groups'], r['starts'], r['counts'],
                   config.power_budget_w, config.beam_power_cap_w)
    t_build = time.perf_counter() - t0
    p0 = r['power'][fp.idx].astype(np.float64)
    m0 = fp.metrics(p0)
    F0 = m0['SGM']
    base = dict(set=s, plan=pid, tier=tier_of(pid), idx=j, key=r['key'], stratum=r['stratum'], load_ratio=r['load_ratio'],
                per_aircraft=r['per_aircraft_bps'], N=fp.N, M=fp.M, Q=fp.Q, F0=F0, W0=m0['W'], worst10_0=m0['worst10'],
                SI0=m0['SI'], P0=m0['power_w'], t_build=t_build)
    rows, powers = [], {}
    if params.get('bounds'):
        rows.append(dict(base, method='bounds', **fp.bounds()))
    order = np.random.default_rng(hash((s, pid, j)) % 2 ** 32).permutation(len(jobs))
    for k in order:
        name, seed, delta, limit, marks, extra = jobs[k]
        kw = {kk: v for kk, v in extra.items() if kk != 'tag'}
        if name == 'aravanis':
            kw['seed'] = seed
        clock = Clock(limit)
        clock.access = lambda p: protect(fp, p0, p, F0, delta)    # 运行中每个新最好点当场接入，耗时计入同一时限
        try:
            res = METHODS[name](fp, p0, F0, delta, clock, marks=marks, **kw)
            status = res['status']
        except Exception as e:                                # 真正没有输出才算失败，回用原功率
            res = dict(p=p0, status='fail', iters=0, evals=0, t_best=0., snaps={m: p0 for m in marks}, acc=[(0., p0, 0.)], error=repr(e))
            status = 'fail'
        t_native = clock.elapsed()

        def access_at(mark):
            """时间点 mark 能拿到的接入结果：接入实际完成时刻不晚于 mark 的最后一条记录。"""
            for t_done, pa_i, tau_i in reversed(res['acc']):
                if t_done <= mark:
                    return pa_i, tau_i, t_done
            return p0, 0., 0.

        pa, tau, t_avail = access_at(limit)
        t_access = 0.
        mn, ma = fp.metrics(res['p']), fp.metrics(pa)
        row = dict(base, method=name, seed=seed, delta=delta, limit=limit, tag=extra.get('tag', ''), status=status,
                   iters=res['iters'], evals=res['evals'], t_native=t_native, t_best=res['t_best'], t_access=t_access,
                   t_avail=t_avail, tau=tau, **{f'n_{kk}': v for kk, v in mn.items()}, **{f'a_{kk}': v for kk, v in ma.items()})
        for kk in ('t', 't_upper', 'undetermined', 'error'):
            if kk in res:
                row[kk] = res[kk]
        if params.get('snaps'):
            snaps = {}
            for mk, ps in sorted(res['snaps'].items()):
                en = fp.evaluate(ps)
                ea = fp.evaluate(access_at(mk)[0])
                snaps[f'{mk:g}'] = [en.W, en.F, ea.W, ea.F]
            row['snaps'] = snaps
        rows.append(row)
        if params.get('powers'):
            tagk = f"{name}_{seed}_{delta:g}_{extra.get('tag', '')}"
            powers[f'n_{tagk}'] = res['p'].astype(np.float32)
            powers[f'a_{tagk}'] = pa.astype(np.float32)
    if params.get('powers'):
        powers['p0'] = p0.astype(np.float32)
        powers['idx'] = fp.idx.astype(np.int16)
    return f'{s}/{pid}/{j}', rows, powers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=('main', 'tradeoff', 'panel', 'calib'))
    ap.add_argument('--sets', nargs='+', default=['old', 'new'])
    ap.add_argument('--plans-dir', default=str(ROOT / 'plans'))
    ap.add_argument('--out', default=str(ROOT / 'runs'))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit-inputs', type=int, default=0, help='只跑前若干个输入（冒烟）')
    a = ap.parse_args()
    inputs, jobs = jobs_for(a.mode, a.sets, a.plans_dir)
    if a.limit_inputs:
        inputs = inputs[:: max(1, len(inputs) // a.limit_inputs)][:a.limit_inputs]
    params = dict(bounds=a.mode == 'main', snaps=a.mode in ('main', 'panel'), powers=a.mode == 'main')
    out = Path(a.out) / a.mode
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / 'rows.jsonl'
    done = set()
    if res_path.exists():
        with res_path.open(encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                done.add(f"{d['set']}/{d['plan']}/{d['idx']}")
    todo = [x for x in inputs if f'{x[0]}/{x[1]}/{x[2]}' not in done]
    print(f'{a.mode}: {len(inputs)} 个输入，已完成 {len(done)}，待跑 {len(todo)}，每输入 {len(jobs)} 次求解，{a.workers} 进程', flush=True)
    (out / 'jobs.json').write_text(json.dumps([list(j[:5]) + [j[5]] for j in jobs], default=str), encoding='utf-8')
    started = time.time()
    from multiprocessing import Pool
    pw_dir = out / 'powers'
    pw_dir.mkdir(exist_ok=True)
    with Pool(a.workers) as pool, res_path.open('a', encoding='utf-8') as f:
        for n, (iid, rows, powers) in enumerate(pool.imap_unordered(run_input, [(a.plans_dir, s, pid, j, jobs, params) for s, pid, j in todo]), 1):
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
            if powers:
                np.savez_compressed(pw_dir / (iid.replace('/', '_') + '.npz'), **powers)
            if n % 50 == 0 or n == len(todo):
                el = time.time() - started
                print(f'  {n}/{len(todo)}  {el / 60:.1f} min，预计还要 {el / n * (len(todo) - n) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
