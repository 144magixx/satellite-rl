"""S3-LAB 测试：物理与 S2-LAB 参考实现一致；梯度；冻结干扰下的最小功率；各方法输出合法、接入后 SGM 不降；
最大最小公平的可行证明；WMMSE 的下界在当前点取等、真实 W 不降。"""
import unittest

import numpy as np

from satellite_rl.s2lab.physics import reference_capacities
from satellite_rl.s2lab.scenes import load_scene, read_config
from satellite_rl.s3lab import methods as M
from satellite_rl.s3lab.plans import load
from satellite_rl.s3lab.problem import FixedPlan

DS = 'D:/paper/outputs/experiments/s2lab_round3_20261027/data/v1'
PL = 'D:/paper/outputs/experiments/s3lab_20261029/plans/dev'


class S3LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = read_config(DS)
        cls.ph = cfg.physics_obj()
        cls.cases = []
        for pid, j in (('ptr-s1', 0), ('greedy', 25), ('fixed', 50), ('random-s1', 10)):
            r = load(f'{PL}/{pid}.pkl')['records'][j]
            sc = load_scene(DS, r['key'])
            fp = FixedPlan(sc, cls.ph, r['demand'], r['groups'], r['starts'], r['counts'])
            cls.cases.append((sc, r, fp, r['power'][fp.idx].astype(np.float64)))

    def test_capacity_matches_reference(self):
        rng = np.random.default_rng(0)
        for sc, r, fp, p0 in self.cases:
            for p in (p0, rng.uniform(.5, 50, fp.M)):
                full = fp.full(p)
                ref = reference_capacities(sc, r['groups'], r['starts'], r['counts'], full, self.ph)
                self.assertTrue(np.allclose(fp.evaluate(p).C, ref, rtol=1e-5, atol=1.))

    def test_gradients(self):
        rng = np.random.default_rng(1)
        for _, _, fp, p0 in self.cases:
            e = fp.evaluate(p0)
            gW, gF, _ = fp.gradients(e)
            for k in rng.choice(fp.M, 5, replace=False):
                h = 1e-5
                q = p0.copy(); q[k] += h
                e2 = fp.evaluate(q)
                self.assertAlmostEqual(gF[k], (e2.F - e.F) / h, delta=1e-3 * abs(gF[k]) + 1e-9)
                if not ((e2.C[fp.idx] >= fp.D_act) != (e.C[fp.idx] >= fp.D_act)).any():
                    self.assertAlmostEqual(gW[k], (e2.W - e.W) / h, delta=1e-3 * abs(gW[k]) + 1e-9)

    def test_min_power_for(self):
        for _, _, fp, p0 in self.cases:
            s = fp.snr_frozen(p0)
            target = np.minimum(fp.D_act, fp.H) * .5
            q, reach = fp.min_power_for(s, target)
            got = fp.beam_rate(s, q)
            ok = reach & (q > fp.lo + 1e-6)
            self.assertTrue((got[reach] >= target[reach] * (1 - 1e-6)).all())
            self.assertTrue((fp.beam_rate(s, q * (1 - 1e-4))[ok] < target[ok]).all())

    def test_methods_feasible_and_protected(self):
        for _, _, fp, p0 in self.cases:
            e0 = fp.evaluate(p0)
            for name, fn in M.METHODS.items():
                res = fn(fp, p0, e0.F, 0., M.Clock(10.), **({'seed': 1} if name == 'aravanis' else {}))
                p = res['p']
                self.assertTrue((p >= fp.lo - 1e-9).all() and (p <= fp.hi + 1e-9).all(), name)
                self.assertLessEqual(p.sum(), fp.budget + 1e-6, name)
                pa, tau = M.protect(fp, p0, p, e0.F, 0.)
                ea = fp.evaluate(pa)
                self.assertGreaterEqual(ea.F, e0.F - 1e-12, name)
                self.assertGreaterEqual(ea.W, e0.W - 1e-12, name)
                if name in ('sca', 'slsqp', 'aravanis'):
                    self.assertGreaterEqual(fp.evaluate(p).F, e0.F - 1e-12, name)

    def test_maxmin_certificate(self):
        for _, _, fp, p0 in self.cases:
            res = M.maxmin(fp, p0, 0., 0., M.Clock(10.))
            C = fp.evaluate(res['p']).C[fp.idx]
            self.assertTrue((C >= res['t'] * fp.D_act * (1 - 1e-6)).all())
            self.assertLessEqual(res['t'], res['t_upper'] + 1e-12)

    def test_maxmin_keeps_known_feasible(self):
        """Astra 反例：两束各一槽、需求 25 Mbps，SINR_i = 100 p_i/(1+100 p_j)；(50,50) 已有满足比例 0.99986，结果不得更差、上界不得低于它。"""
        from scipy import sparse
        fp = object.__new__(FixedPlan)
        fp.N = fp.M = fp.Q = 2
        fp.demand = fp.D_act = np.array([25e6, 25e6])
        fp.D_total, fp.beta = 50e6, 25e6
        fp.members = np.array([1, 1])
        fp.idx = fp.owner = np.arange(2)
        fp.lo, fp.hi = np.full(2, .5), np.full(2, 50.)
        fp.budget, fp.cap, fp.floor = 2000., 50., .5
        fp.B, fp.eta = 25e6, 5.5
        fp.a = np.array([100., 100.])
        fp.b = sparse.csr_matrix(np.array([[0., 100.], [100., 0.]]))
        fp.R = sparse.csr_matrix(np.eye(2))
        fp.H = np.full(2, 25e6 * 5.5)
        p0 = np.array([50., 50.])
        t0 = (fp.evaluate(p0).C / fp.demand).min()
        res = M.maxmin(fp, p0, 0., 0., M.Clock(10.))
        self.assertGreaterEqual(res['t'], t0 - 1e-12)
        self.assertGreaterEqual(res['t_upper'], t0 - 1e-12)
        C = fp.evaluate(res['p']).C
        self.assertTrue((C >= res['t'] * fp.demand * (1 - 1e-6)).all())
        # 无干扰、起点已 100% 满足：仍应求出同比例下更省功率的解
        fp.b = sparse.csr_matrix(np.zeros((2, 2)))
        fp.a = np.array([1., 1.])
        res = M.maxmin(fp, p0, 0., 0., M.Clock(10.))
        self.assertAlmostEqual(res['t'], 1., places=6)
        self.assertLess(res['p'].sum(), 10.)

    def test_access_recorded_during_run(self):
        """接入在运行中完成、时刻记在 acc 里且不超过时限；时限后的候选不进记录。"""
        _, _, fp, p0 = self.cases[1]
        e0 = fp.evaluate(p0)
        clock = M.Clock(10.)
        clock.access = lambda p: M.protect(fp, p0, p, e0.F, 0.)
        res = M.sca(fp, p0, e0.F, 0., clock)
        self.assertGreater(len(res['acc']), 1)
        times = [t for t, _, _ in res['acc']]
        self.assertEqual(times, sorted(times))
        self.assertLessEqual(times[-1], clock.elapsed())
        clock2 = M.Clock(0.)
        clock2.access = clock.access
        tr = M.Tracker(clock2, p0, (0.,), M.MARKS)
        import time as _t
        _t.sleep(.01)
        self.assertFalse(tr.offer(p0 * .9, (1.,)))
        self.assertEqual(len(tr.acc), 1)

    def test_wmmse_bound_and_monotone(self):
        _, _, fp, p0 = self.cases[0]
        x = np.sqrt(p0)
        I = 1. + fp.b @ p0
        sig = fp.a * p0[fp.owner]
        u = np.sqrt(fp.a) * x[fp.owner] / (I + sig)
        w = (I + sig) / I
        kappa = (np.log(w) - w * (1 + u * u) + 1) / M.LN2
        beta = 2 * w * u * np.sqrt(fp.a) / M.LN2
        L = kappa + beta * x[fp.owner] - (w * u * u * fp.a / M.LN2) * p0[fp.owner] - (w * u * u / M.LN2) * (fp.b @ p0)
        self.assertTrue(np.allclose(L, np.log2(1 + sig / I), rtol=1e-8, atol=1e-8))
        socp = M._WmmseSOCP(fp)
        Ws = []
        for _ in range(5):
            p = x * x
            I = 1. + fp.b @ p
            sig = fp.a * p[fp.owner]
            u = np.sqrt(fp.a) * x[fp.owner] / (I + sig)
            w = (I + sig) / I
            xn = socp.solve((np.log(w) - w * (1 + u * u) + 1) / M.LN2, 2 * w * u * np.sqrt(fp.a) / M.LN2,
                            w * u * u * fp.a / M.LN2, (w * u * u / M.LN2)[socp.b_row] * socp.b_val)
            x = xn
            Ws.append(fp.evaluate(x * x).W)
        self.assertGreaterEqual(Ws[0], fp.evaluate(p0).W - 1e-6)
        self.assertTrue(all(b >= a - 1e-6 for a, b in zip(Ws, Ws[1:])))

    def test_bounds(self):
        for _, _, fp, p0 in self.cases:
            bd = fp.bounds()
            e = fp.evaluate(p0)
            self.assertLessEqual(e.W, bd['W_U'] + 1e-12)
            self.assertLessEqual(bd['W_U'], bd['W_H'] + 1e-12)


if __name__ == '__main__':
    unittest.main()
