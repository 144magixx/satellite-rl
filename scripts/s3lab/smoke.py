"""本机冒烟：开发集若干输入上跑全部方法，看结果和耗时。"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from satellite_rl.s2lab.scenes import load_scene, read_config  # noqa: E402
from satellite_rl.s3lab.methods import METHODS, Clock, protect  # noqa: E402
from satellite_rl.s3lab.plans import load  # noqa: E402
from satellite_rl.s3lab.problem import FixedPlan  # noqa: E402

DS = 'D:/paper/outputs/experiments/s2lab_round3_20261027/data/v1'
PL = 'D:/paper/outputs/experiments/s3lab_20261029/plans/dev'
cfg = read_config(DS)
ph = cfg.physics_obj()
plans = sys.argv[1].split(',') if len(sys.argv) > 1 else ['ptr-s1', 'greedy', 'fixed', 'random-s1']
which = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0, 25, 50]
methods = sys.argv[3].split(',') if len(sys.argv) > 3 else list(METHODS)
limit = float(sys.argv[4]) if len(sys.argv) > 4 else 10.
for pid in plans:
    obj = load(f'{PL}/{pid}.pkl')
    for j in which:
        r = obj['records'][j]
        t0 = time.perf_counter()
        fp = FixedPlan(load_scene(DS, r['key']), ph, r['demand'], r['groups'], r['starts'], r['counts'])
        tb = time.perf_counter() - t0
        p0 = r['power'][fp.idx]
        e0 = fp.evaluate(p0)
        bd = fp.bounds()
        print(f"== {pid} #{j} {r['stratum']} N={fp.N} M={fp.M} Q={fp.Q} 建模 {tb:.2f}s  直出 W={e0.W:.4f} SGM={e0.F:.4f} P={e0.power:.0f}  "
              f"上界 W_H={bd['W_H']:.3f} W_U={bd['W_U']:.3f} 缺槽束 {bd['slot_short_beams']}", flush=True)
        for name in methods:
            if name == 'none':
                continue
            clock = Clock(limit)
            kw = dict(seed=1) if name == 'aravanis' else {}
            res = METHODS[name](fp, p0, e0.F, 0., clock, **kw)
            tm = clock.elapsed()
            en = fp.evaluate(res['p'])
            t1 = time.perf_counter()
            pa, tau = protect(fp, p0, res['p'], e0.F, 0.)
            ta = time.perf_counter() - t1
            ea = fp.metrics(pa)
            print(f"  {name:12s} {res['status']:5s} it={res['iters']:3d} ev={res['evals']:4d} {tm:5.2f}s | 原生 W={en.W:.4f} SGM={en.F:.4f} P={en.power:.0f}"
                  f" | 接入 τ={tau:.2f} W={ea['W']:.4f} SGM={ea['SGM']:.4f} 10%={ea['worst10']:.3f} P={ea['power_w']:.0f} ({ta:.2f}s)", flush=True)
