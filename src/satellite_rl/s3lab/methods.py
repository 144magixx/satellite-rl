"""九种第三阶段功率方法与统一的"保 SGM 接入"。

问题：max W(p)  s.t.  F(p) ≥ F0−δ，有槽束 0.5≤p≤50 W，Σp≤2000 W（见 03_S3-LAB方案定稿）。
每个方法返回 Result：原生输出（按方法自身准则保留的最好已核验点）、状态、迭代与评价次数、各时间点当时的原生输出。
"""
import math
import time

import clarabel
import numpy as np
from scipy import sparse
from scipy.optimize import linprog, minimize

LN2 = math.log(2.)
MARKS = (0.1, 0.3, 1., 3., 10.)


class Clock:
    def __init__(self, limit):
        self.t0 = time.perf_counter()
        self.limit = limit

    def elapsed(self):
        return time.perf_counter() - self.t0

    def over(self):
        return self.elapsed() >= self.limit


class Tracker:
    """按原生准则（key 越大越好）保留最好的已核验点；在每个时间点记下当时的最好点。"""

    def __init__(self, clock, p, key, marks):
        self.clock, self.marks = clock, [m for m in marks if m <= clock.limit + 1e-9]
        self.p, self.key, self.t_best = p.copy(), key, 0.
        self.snap = {}
        self.hist = [(0., p.copy())]                          # 每次原生最好点更新的 (完成时刻, 功率)，运行器据此按时间点做接入

    def _fill(self, upto):
        for m in self.marks:
            if m not in self.snap and upto >= m:
                self.snap[m] = self.p.copy()

    def offer(self, p, key):
        e = self.clock.elapsed()
        if e > self.clock.limit:
            return False
        self._fill(e)
        if key > self.key:
            self.p, self.key, self.t_best = p.copy(), key, e
            self.hist.append((e, self.p))
            return True
        return False

    def finish(self):
        self._fill(float('inf'))
        return self.snap


class Result(dict):
    pass


def _result(tracker, status, iters, evals, **extra):
    return Result(p=tracker.p, key=tracker.key, status=status, iters=iters, evals=evals, t_best=tracker.t_best,
                  snaps=tracker.finish(), hist=tracker.hist, **extra)


# ---------------------------------------------------------------- 共用小工具
def project_budget(fp, v):
    """投影到 {lo≤p≤hi, Σp≤预算}：先夹到上下限，超预算则整体平移 clip(v−λ)。"""
    q = np.clip(v, fp.lo, fp.hi)
    if q.sum() <= fp.budget:
        return q
    a, b = 0., float(v.max() - fp.lo.min())
    for _ in range(60):
        lam = (a + b) / 2
        if np.clip(v - lam, fp.lo, fp.hi).sum() > fp.budget:
            a = lam
        else:
            b = lam
    return np.clip(v - b, fp.lo, fp.hi)


def waterfill(fp, s, price=0.):
    """冻结每槽系数 s：max Σmin(R_i(p_i), D_i) − price·Σp，盒约束＋总预算（预算用不满就留空）。price 单位 bps/W。"""
    D = fp.D_act
    pD, _ = fp.min_power_for(s, D)                       # 达到需求的最小功率（达不到取 50 W）

    def p_of(mu):
        if mu <= 0:
            return pD.copy()
        a, b = fp.lo.copy(), pD.copy()
        for _ in range(50):
            mid = (a + b) / 2
            up = fp.beam_rate_slope(s, mid) >= mu
            a = np.where(up, mid, a)
            b = np.where(up, b, mid)
        low_slope = fp.beam_rate_slope(s, fp.lo) < mu
        return np.where(low_slope, fp.lo, a)

    p = p_of(price)
    if p.sum() <= fp.budget:
        return p
    lo_mu, hi_mu = price, max(price, float(fp.beam_rate_slope(s, fp.lo).max())) * 2 + 1.
    for _ in range(60):
        mid = (lo_mu + hi_mu) / 2
        if p_of(mid).sum() > fp.budget:
            lo_mu = mid
        else:
            hi_mu = mid
    return project_budget(fp, p_of(hi_mu))


# ---------------------------------------------------------------- 0/3/4/5 直接给出
def direct(fp, p0, F0, delta, clock, marks=MARKS):
    tr = Tracker(clock, p0, 0., marks)
    return _result(tr, 'ok', 0, 0)


def _closed(fp, p0, clock, marks, p):
    tr = Tracker(clock, p0, 0., marks)
    tr.offer(p, 1.)
    return _result(tr, 'ok', 1, 1)


def choi_chan(fp, p0, F0, delta, clock, marks=MARKS):
    return _closed(fp, p0, clock, marks, waterfill(fp, fp.snr_frozen(None), 0.))


def proportional(fp, p0, F0, delta, clock, marks=MARKS):
    D = fp.D_act
    total = min(fp.budget, fp.cap * fp.M)
    a, b = 0., fp.cap / D.min() * 2
    for _ in range(80):
        lam = (a + b) / 2
        if np.clip(lam * D, fp.lo, fp.hi).sum() < total:
            a = lam
        else:
            b = lam
    return _closed(fp, p0, clock, marks, np.clip(b * D, fp.lo, fp.hi))


def equal(fp, p0, F0, delta, clock, marks=MARKS):
    v = min(fp.cap, fp.budget / fp.M)
    return _closed(fp, p0, clock, marks, np.clip(np.full(fp.M, v), fp.lo, fp.hi))


# ---------------------------------------------------------------- 1 SCA（本研究）
def _sca_lp(fp, p, ev, J, gF, thr, r, margin):
    """线性子问题：max Σ w_i，w_i ≤ C_i + J_i·Δ，w_i ≤ D_i，F + gF·Δ ≥ thr + margin，盒＋预算＋信赖域。单位 Mbps。"""
    M = fp.M
    C = ev.C[fp.idx] / 1e6
    D = fp.D_act / 1e6
    lo = np.maximum(fp.lo - p, -r)
    hi = np.minimum(fp.hi - p, r)
    sF = 1. / max(np.abs(gF).max(), 1e-12)
    A = np.zeros((M + 2, 2 * M))
    A[:M, :M] = -J / 1e6
    A[np.arange(M), M + np.arange(M)] = 1.
    A[M, :M] = -gF * sF
    A[M + 1, :M] = 1.
    bub = np.concatenate([C, [(ev.F - thr - margin) * sF, fp.budget - p.sum()]])
    c = np.concatenate([np.zeros(M), -np.ones(M)])
    bounds = list(zip(lo, hi)) + [(None, d) for d in D]
    res = linprog(c, A_ub=A, b_ub=bub, bounds=bounds, method='highs')
    if res.status != 0:
        return None
    d = res.x[:M]
    pred = (-res.fun - np.minimum(C, D).sum()) * 1e6 / fp.D_total
    return d, pred


def sca(fp, p0, F0, delta, clock, marks=MARKS, max_iter=200, max_evals=2000, r0=5.):
    thr = F0 - delta - 1e-12
    p = p0.copy()
    ev = fp.evaluate(p)
    evals = 1
    tr = Tracker(clock, p, (ev.W, -ev.power), marks)
    r, cm, status, it = r0, 0., 'ok', 0
    grad_at = None
    while it < max_iter:
        it += 1
        if clock.over() or evals >= max_evals:
            status = 'limit'
            break
        if grad_at is not ev:
            gW, gF, J = fp.gradients(ev)
            grad_at = ev
        step = _sca_lp(fp, p, ev, J, gF, thr, r, cm * r * r)
        if step is None or step[1] < 1e-7:
            if cm > 0:                                           # 余量太大导致无解或无预测收益：去掉余量再试
                cm = 0.
                continue
            if step is not None:
                break                                            # 线性模型已无提升：收敛
            r /= 2
            if r < 1e-3:
                break
            continue
        d, pred = step
        pc = p + d
        evc = fp.evaluate(pc)
        evals += 1
        if evc.W > ev.W + 1e-12 and evc.F >= thr:
            p, ev = pc, evc
            tr.offer(p, (ev.W, -ev.power))
            r, cm = min(2 * r, fp.cap), cm / 2
            continue
        if evc.F < thr:
            err = ev.F + gF @ d - evc.F                       # 线性化高估的 SGM：按实测弯曲给后续余量
            cm = max(2 * cm, err / max(np.abs(d).max(), 1e-9) ** 2)
            if evc.W > ev.W:                                    # 修正步：在候选点补回 SGM 缺口，仍以 W 为目标
                _, gF2, J2 = fp.gradients(evc)
                st2 = _sca_lp(fp, pc, evc, J2, gF2, thr, r / 2, cm * (r / 2) ** 2)
                if st2 is not None:
                    pc2 = pc + st2[0]
                    ev2 = fp.evaluate(pc2)
                    evals += 1
                    if ev2.W > ev.W + 1e-12 and ev2.F >= thr:
                        p, ev = pc2, ev2
                        tr.offer(p, (ev.W, -ev.power))
                        continue
        r /= 2
        if r < 1e-3:
            break
    return _result(tr, status, it, evals)


# ---------------------------------------------------------------- 9 SLSQP
class _Stop(Exception):
    pass


def slsqp(fp, p0, F0, delta, clock, marks=MARKS, max_iter=300, max_evals=2000):
    thr = F0 - delta - 1e-12
    S = fp.cap
    ev0 = fp.evaluate(p0)
    tr = Tracker(clock, p0, (ev0.W, -ev0.power), marks)
    cache = {'key': None}
    count = {'evals': 1, 'iters': 0}

    def at(u):
        k = u.tobytes()
        if cache['key'] != k:
            if clock.over() or count['evals'] >= max_evals:
                raise _Stop
            p = np.clip(u * S, fp.lo, fp.hi)
            e = fp.evaluate(p)
            count['evals'] += 1
            gW, gF, _ = fp.gradients(e)
            if e.F >= thr and p.sum() <= fp.budget + 1e-6:
                tr.offer(p, (e.W, -e.power))
            cache.update(key=k, e=e, gW=gW, gF=gF)
        return cache

    def cb(u):
        count['iters'] += 1

    aim = thr + 1e-8                                         # SLSQP 只把约束满足到约 1e-12，给它一点余量，接受仍按 thr
    cons = [dict(type='ineq', fun=lambda u: (at(u)['e'].F - aim) * 1e2, jac=lambda u: at(u)['gF'] * S * 1e2),
            dict(type='ineq', fun=lambda u: (fp.budget - u.sum() * S) / S, jac=lambda u: -np.ones(fp.M))]
    status = 'ok'
    try:
        minimize(lambda u: -at(u)['e'].W * 1e2, p0 / S, jac=lambda u: -at(u)['gW'] * S * 1e2, method='SLSQP',
                 bounds=list(zip(fp.lo / S, fp.hi / S)), constraints=cons, callback=cb,
                 options=dict(maxiter=max_iter, ftol=1e-10))
    except _Stop:
        status = 'limit'
    return _result(tr, status, count['iters'], count['evals'])


# ---------------------------------------------------------------- 2 Aravanis 两段式
def aravanis(fp, p0, F0, delta, clock, marks=MARKS, seed=0, K=320, frac=.7, T0=1e-5, T1=1e-8, step0=20., step1=2.):
    """温度与步长在开发集上按本方法原生代价选定（9 组候选里最低）。"""
    thr = F0 - delta - 1e-12
    omega = .001 * fp.D_total / fp.budget                  # 功率价格（bps/W）：用满 2000 W 相当于损失 0.1% 需求
    rng = np.random.default_rng(seed)

    def cost(e):
        return 1. - e.W + .001 * e.power / fp.budget

    p = p0.copy()
    ev = fp.evaluate(p)
    cur = cost(ev)
    tr = Tracker(clock, p, -cur, marks)
    best_p, best = p.copy(), cur
    evals, status, k = 1, 'ok', 0
    for k in range(K):
        if clock.elapsed() >= frac * clock.limit:
            status = 'limit'
            break
        T = T0 * (T1 / T0) ** (k / K)
        step = step0 * (step1 / step0) ** (k / K) * rng.uniform(.1, 1.)
        i = int(rng.integers(fp.M))
        q = p.copy()
        move = int(rng.integers(3))
        if move == 0:
            q[i] = min(fp.hi[i], q[i] + min(step, max(fp.budget - q.sum(), 0.)))
        elif move == 1:
            q[i] = max(fp.lo[i], q[i] - step)
        else:
            j = int(rng.integers(fp.M))
            amt = min(step, q[j] - fp.lo[j], fp.hi[i] - q[i])
            q[i] += amt
            q[j] -= amt
        eq = fp.evaluate(q)
        evals += 1
        if eq.F < thr:
            continue
        cq = cost(eq)
        if cq <= cur or rng.random() < math.exp(-(cq - cur) / T):
            p, cur = q, cq
            if cur < best:
                best_p, best = p.copy(), cur
                tr.offer(best_p, -best)
    # 第二段：冻结粗搜终点的逐槽干扰，带功率价格注水
    if not clock.over():
        p2 = waterfill(fp, fp.snr_frozen(best_p), omega)
        e2 = fp.evaluate(p2)
        evals += 1
        if e2.F >= thr and cost(e2) < best:
            tr.offer(p2, -cost(e2))
    return _result(tr, status, k + 1, evals)


# ---------------------------------------------------------------- 6 目标容量迭代
def target_capacity(fp, p0, F0, delta, clock, marks=MARKS, damping=.3, max_iter=200):
    """阻尼在开发集上按本方法原生误差选定（0.3/0.5/0.8 里最好）。"""
    r = np.minimum(fp.D_act, fp.H)

    def key(e):
        C = e.C[fp.idx]
        return (-float(np.maximum(r - C, 0).sum() / r.sum()), -e.power)

    p = p0.copy()
    e = fp.evaluate(p)
    tr = Tracker(clock, p, key(e), marks)
    status, it, evals = 'ok', 0, 1
    while it < max_iter:
        it += 1
        if clock.over():
            status = 'limit'
            break
        v, _ = fp.min_power_for(fp.snr_frozen(p), r)
        q = project_budget(fp, (1 - damping) * p + damping * v)
        e = fp.evaluate(q)
        evals += 1
        tr.offer(q, key(e))
        if np.abs(q - p).max() < 1e-4:
            break
        p = q
    return _result(tr, status, it, evals)


# ---------------------------------------------------------------- 7 最大最小公平
def maxmin(fp, p0, F0, delta, clock, marks=MARKS, outer=20, inner=200):
    """二分满足比例 t。已证可行的 t 记入下界 a；已证不可行才收紧上界 b；内层未判定只把本轮试探上限 c 降下来，不动 b。
    起点：第二阶段功率本身的最小满足比例就是一个已证可行值（10-29 按 Astra 代码复核修正）。"""
    D = fp.D_act
    floor_p = fp.lo.copy()
    if fp.M < fp.N:                                          # 有无槽且有需求的束：最小值只能是 0，取最小功率解
        tr = Tracker(clock, p0, (-1., 0.), marks)
        tr.offer(floor_p, (0., -floor_p.sum()))
        return _result(tr, 'ok', 0, 0, t=0., t_upper=0., undetermined=0)
    e0 = fp.evaluate(p0)
    t0 = float(min(1., (e0.C[fp.idx] / D).min()))
    tr = Tracker(clock, p0, (t0, -p0.sum()), marks)
    U = fp.beam_rate(fp.snr_frozen(None), fp.hi)
    t_up = float(min(1., (U / D).min()))
    best_t, best_p = t0, p0
    count = {'evals': 1, 'undet': 0}

    def check(t):
        p = fp.lo.copy()
        target = t * D
        for _ in range(inner):
            if clock.over():
                return 'stop', None
            v, reach = fp.min_power_for(fp.snr_frozen(p), target)
            count['evals'] += 1
            if not reach.all():
                return 'no', None                            # 当前干扰（只会更大）下 50 W 都不够
            v = np.maximum(v, fp.lo)
            if v.sum() > fp.budget + 1e-9:
                return 'no', None
            if np.abs(v - p).max() < 1e-5:
                e = fp.evaluate(v)
                count['evals'] += 1
                if (e.C[fp.idx] >= target * (1 - 1e-6)).all():
                    return 'yes', v
                p = v
                continue
            p = v
        return 'unknown', None

    a, b, it = t0, t_up, 0
    c = b                                                    # 本轮试探上限（未判定时下调，不当作不可行）
    res = 'skip'
    if b > a + 1e-12:
        res, p = check(b)
        if res == 'yes':
            best_t, best_p, a = b, p, b
            tr.offer(best_p, (best_t, -best_p.sum()))
        elif res == 'no':
            c = b
        elif res == 'unknown':
            count['undet'] += 1
    status = 'limit' if res == 'stop' else 'ok'
    while res != 'stop' and c - a > 1e-3 and it < outer:
        it += 1
        mid = (a + c) / 2
        res, p = check(mid)
        if res == 'yes':
            a, best_t, best_p = mid, mid, p
            tr.offer(best_p, (best_t, -best_p.sum()))
        elif res == 'no':
            b = c = mid
        elif res == 'unknown':
            count['undet'] += 1
            c = mid
        else:
            status = 'limit'
    return _result(tr, status, it, count['evals'], t=best_t, t_upper=b, undetermined=count['undet'])


# ---------------------------------------------------------------- 8 WMMSE（需求封顶的和速率）
class _WmmseSOCP:
    """变量 v=[x(M), s(M), z(Q), y(M)]；max Σy。
    z_q ≤ κ_q + β_q·x_o(q) − Σ_k γ_qk·s_k，z_q ≤ η，y_i ≤ 25·Σ_{q∈i} z_q（Mbps），y_i ≤ D_i，Σs ≤ 预算，
    √lo ≤ x ≤ √hi，s ≤ hi，s_k ≥ x_k²（旋转二阶锥）。"""

    def __init__(self, fp):
        self.fp = fp
        M, Q = fp.M, fp.Q
        self.n = 3 * M + Q
        self.ix, self.iss, self.iz, self.iy = 0, M, 2 * M, 2 * M + Q
        bcoo = fp.b.tocoo()
        self.b_row, self.b_col, self.b_val = bcoo.row, bcoo.col, bcoo.data
        P = sparse.csc_matrix((self.n, self.n))
        self.P = P
        self.q = np.zeros(self.n)
        self.q[self.iy:] = -1.

    def solve(self, kappa, beta, g_own, g_cross):
        fp = self.fp
        M, Q = fp.M, fp.Q
        rows, cols, vals, rhs = [], [], [], []
        r = 0
        # z_q − β_q x_o + γ_own s_o + Σ γ_cross s_k ≤ κ_q
        qq = np.arange(Q)
        rows += [qq, qq, qq, self.b_row]
        cols += [self.iz + qq, self.ix + fp.owner, self.iss + fp.owner, self.iss + self.b_col]
        vals += [np.ones(Q), -beta, g_own, g_cross]
        rhs.append(kappa)
        r += Q
        rows.append(r + qq); cols.append(self.iz + qq); vals.append(np.ones(Q)); rhs.append(np.full(Q, fp.eta)); r += Q
        mm = np.arange(M)
        rows += [r + mm, r + fp.owner]
        cols += [self.iy + mm, self.iz + qq]
        vals += [np.ones(M), -np.full(Q, fp.B / 1e6)]
        rhs.append(np.zeros(M)); r += M
        rows.append(r + mm); cols.append(self.iy + mm); vals.append(np.ones(M)); rhs.append(fp.D_act / 1e6); r += M
        rows.append(np.full(M, r)); cols.append(self.iss + mm); vals.append(np.ones(M)); rhs.append(np.array([fp.budget])); r += 1
        rows.append(r + mm); cols.append(self.ix + mm); vals.append(np.ones(M)); rhs.append(np.sqrt(fp.hi)); r += M
        rows.append(r + mm); cols.append(self.ix + mm); vals.append(-np.ones(M)); rhs.append(-np.sqrt(fp.lo)); r += M
        rows.append(r + mm); cols.append(self.iss + mm); vals.append(np.ones(M)); rhs.append(fp.hi.copy()); r += M
        n_nonneg = r
        # 旋转锥 s_k ≥ x_k²：(s_k+1, 2x_k, s_k−1) ∈ SOC
        for k in range(M):
            rows += [np.array([r]), np.array([r + 1]), np.array([r + 2])]
            cols += [np.array([self.iss + k]), np.array([self.ix + k]), np.array([self.iss + k])]
            vals += [np.array([-1.]), np.array([-2.]), np.array([-1.])]
            rhs.append(np.array([1., 0., -1.]))
            r += 3
        A = sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, self.n))
        cones = [clarabel.NonnegativeConeT(n_nonneg)] + [clarabel.SecondOrderConeT(3)] * M
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = 100
        sol = clarabel.DefaultSolver(self.P, self.q, A, np.concatenate(rhs), cones, settings).solve()
        if str(sol.status) not in ('Solved', 'AlmostSolved'):
            return None
        x = np.asarray(sol.x)[self.ix:self.ix + M]
        return np.clip(x, np.sqrt(fp.lo), np.sqrt(fp.hi))


def wmmse(fp, p0, F0, delta, clock, marks=MARKS, max_iter=200, tol=1e-7):
    sq_a = np.sqrt(fp.a)
    socp = _WmmseSOCP(fp)
    x = np.sqrt(p0)
    e = fp.evaluate(p0)
    tr = Tracker(clock, p0, (e.W, -e.power), marks)
    status, it, evals, stall = 'ok', 0, 1, 0
    last = e.W
    while it < max_iter:
        it += 1
        if clock.over():
            status = 'limit'
            break
        p = x * x
        I = 1. + fp.b @ p
        sig = fp.a * p[fp.owner]
        u = sq_a * x[fp.owner] / (I + sig)
        w = (I + sig) / I
        kappa = (np.log(w) - w * (1 + u * u) + 1) / LN2
        beta = 2 * w * u * sq_a / LN2
        g_own = w * u * u * fp.a / LN2
        g_cross = (w * u * u / LN2)[socp.b_row] * socp.b_val
        xn = socp.solve(kappa, beta, g_own, g_cross)
        if xn is None or clock.over():
            status = 'limit' if xn is not None else 'subproblem_stop'    # 子问题没解出：带着已有有效输出停下，不算失败
            break
        pn = project_budget(fp, xn * xn)
        e = fp.evaluate(pn)
        evals += 1
        tr.offer(pn, (e.W, -e.power))
        stall = stall + 1 if e.W - last < tol else 0
        last = max(last, e.W)
        x = np.sqrt(pn)
        if stall >= 3:
            break
    return _result(tr, status, it, evals)


# ---------------------------------------------------------------- 保 SGM 接入
def protect(fp, p0, pm, F0, delta, n=21):
    """在 p0→pm 的连线上取 n 个点，选 F ≥ F0−δ 中 W 最高者（并列省功率）；τ=0 一定可行。"""
    thr = F0 - delta - 1e-12
    e0 = fp.evaluate(p0)
    best = (e0.W, -e0.power)
    best_p, best_tau = p0, 0.
    for tau in np.linspace(0, 1, n)[1:]:
        p = p0 + tau * (pm - p0)
        e = fp.evaluate(p)
        if e.F >= thr and (e.W, -e.power) > best:
            best, best_p, best_tau = (e.W, -e.power), p, float(tau)
    return best_p, best_tau


METHODS = {
    'none': direct, 'sca': sca, 'aravanis': aravanis, 'choi_chan': choi_chan, 'proportional': proportional,
    'equal': equal, 'target_cap': target_capacity, 'maxmin': maxmin, 'wmmse': wmmse, 'slsqp': slsqp,
}
METHOD_ZH = {
    'none': '不细调', 'sca': 'SCA（本研究）', 'aravanis': 'Aravanis 两段式', 'choi_chan': 'Choi-Chan 注水',
    'proportional': '按需求比例', 'equal': '等功率', 'target_cap': '目标容量迭代', 'maxmin': '最大最小公平',
    'wmmse': 'WMMSE', 'slsqp': 'SLSQP',
}
