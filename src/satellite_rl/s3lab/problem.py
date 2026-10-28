"""固定频率方案下的整束功率问题（S2-LAB 物理 v3，numpy float64）。

变量是"有槽波束"的整束功率 p（瓦），按本束槽数均摊到各槽。每个占用槽 q（属于波束 i、频率槽 t）：
    γ_q = a_q·p_i / (1 + Σ_k b_qk·p_k)          （已除以噪声功率）
    a_q = sig_i·path[i,t]·实现损耗 / c_i / 噪声
    b_qk = coupling[i,k]·极化系数·[k 占用 t]·path[i,t] / c_k / 噪声   （同极化 1，异极化 XPD 泄漏）
    C_i = 25 MHz · Σ_q min(log2(1+γ_q), 5.5)
与 S2-LAB 环境 capacities() 完全同一公式；SGM 的 β 取全部真实波束需求均值，无槽束容量记 0。
"""
import math

import numpy as np
from scipy import sparse

from satellite_rl.s2lab.physics import path_gain

LN2 = math.log(2.)


def sgm_and_slope(capacity, demand, beta):
    """每束 SGM 及其对容量（bps）的导数；容量 0 记 0。"""
    sgm = np.zeros_like(capacity)
    slope = np.zeros_like(capacity)
    pos = capacity > 0
    c, d = capacity[pos], demand[pos]
    ratio = c / d
    real = np.where(ratio >= 1, ratio - 1, 1 - 1 / ratio)
    real_p = np.where(ratio >= 1, 1 / d, d / (c * c))
    imag = (c - d) / beta
    mag = np.hypot(real, imag)
    decay = np.exp(-mag)
    sgm[pos] = 1 - (1 - decay) ** 3
    dm = np.divide(real * real_p + imag / beta, mag, out=np.zeros_like(c), where=mag > 0)
    slope[pos] = -3 * (1 - decay) ** 2 * decay * dm          # d/dc [1-(1-e^-m)^3] = -3(1-e^-m)^2·e^-m·dm/dc
    return sgm, slope


class Evaluation:
    __slots__ = ('p', 'C', 'W', 'F', 'SI', 'power', 'gamma', 'I', 'rate', 'capped')


class FixedPlan:
    """一个场景＋一份冻结的第二阶段分配（组号、起点、实际槽数）＋需求。"""

    def __init__(self, scene, physics, demand_bps, groups, starts, counts, budget=2000., cap=50., floor=.5):
        self.N = scene.n
        self.demand = np.asarray(demand_bps, np.float64)
        self.D_total = float(self.demand.sum())
        self.beta = float(self.demand.mean())
        self.members = np.asarray(scene.members, np.int64)
        counts = np.asarray(counts, np.int64)
        starts = np.asarray(starts, np.int64)
        groups = np.asarray(groups, np.int64)
        self.counts_all, self.starts_all, self.groups_all = counts, starts, groups
        self.idx = np.nonzero(counts > 0)[0]                 # 有槽波束（变量顺序）
        M = self.M = len(self.idx)
        self.budget, self.cap, self.floor = float(budget), float(cap), float(floor)
        self.lo = np.full(M, self.floor)
        self.hi = np.full(M, self.cap)
        self.B = float(physics.slot_Hz)
        self.eta = float(physics.spectral_efficiency_cap)
        noise = physics.noise_w
        path = path_gain(scene.distance_m, physics)          # [N,100]
        cnt = counts[self.idx].astype(np.float64)
        owner = np.concatenate([np.full(counts[i], k) for k, i in enumerate(self.idx)]) if M else np.zeros(0, np.int64)
        slot = np.concatenate([np.arange(starts[i], starts[i] + counts[i]) for i in self.idx]) if M else np.zeros(0, np.int64)
        beam = self.idx[owner]
        self.owner, self.slot = owner, slot
        self.Q = len(owner)
        self.a = scene.sig[beam].astype(np.float64) * path[beam, slot] * physics.impl_linear / cnt[owner] / noise
        occ = np.zeros((M, physics.slots), bool)
        for k, i in enumerate(self.idx):
            occ[k, starts[i]:starts[i] + counts[i]] = True
        parity = groups[self.idx] % 2
        pol = np.where(parity[owner][:, None] == parity[None, :], 1., physics.xpd_linear)
        b = (scene.coupling[beam][:, self.idx].astype(np.float64) * pol * occ[:, slot].T
             * path[beam, slot][:, None] / cnt[None, :] / noise)
        b[np.arange(self.Q), owner] = 0.
        self.b = sparse.csr_matrix(b)
        self.R = sparse.csr_matrix((np.ones(self.Q), (owner, np.arange(self.Q))), shape=(M, self.Q))   # 槽→束求和
        self.H = cnt * self.B * self.eta                      # 有槽束的缺槽上限（bps）
        self.D_act = self.demand[self.idx]

    # ------------------------------------------------------------ 评价
    def full(self, p):
        out = np.zeros(self.N)
        out[self.idx] = p
        return out

    def evaluate(self, p):
        ev = Evaluation()
        ev.p = p
        ev.I = 1. + self.b @ p
        ev.gamma = self.a * p[self.owner] / ev.I
        raw = np.log2(1. + ev.gamma)
        ev.capped = raw >= self.eta
        ev.rate = np.minimum(raw, self.eta)
        C = np.zeros(self.N)
        C[self.idx] = self.B * np.bincount(self.owner, ev.rate, self.M)
        ev.C = C
        ev.W = float(np.minimum(C, self.demand).sum() / self.D_total)
        sgm, _ = sgm_and_slope(C, self.demand, self.beta)
        ev.F = float(sgm.mean())
        ev.SI = float(np.minimum(C / self.demand, 1.).mean())
        ev.power = float(p.sum())
        return ev

    def jacobian(self, ev):
        """有槽束容量对有槽束功率的导数 J[i,k]（bps/W），封顶槽导数为 0。"""
        g = np.where(ev.capped, 0., self.B / LN2 / (1. + ev.gamma))
        own = np.bincount(self.owner, g * self.a / ev.I, self.M)
        cross = self.b.multiply((-g * ev.gamma / ev.I)[:, None]).tocsr()
        J = (self.R @ cross).toarray()
        J[np.arange(self.M), np.arange(self.M)] += own
        return J

    def gradients(self, ev, J=None):
        """W 的单侧梯度（容量低于需求的束才计）与 F 的梯度，均对有槽束功率。"""
        J = self.jacobian(ev) if J is None else J
        C = ev.C[self.idx]
        under = (C < self.D_act).astype(np.float64)
        gW = under @ J / self.D_total
        _, slope = sgm_and_slope(ev.C, self.demand, self.beta)
        gF = slope[self.idx] @ J / self.N
        return gW, gF, J

    def metrics(self, p):
        ev = self.evaluate(p)
        si = np.minimum(ev.C / self.demand, 1.)
        service = np.repeat(si, self.members)
        k = max(1, int(math.ceil(.1 * len(service))))
        worst10 = float(np.partition(service, k - 1)[:k].mean())
        return dict(W=ev.W, SGM=ev.F, SI=ev.SI, worst10=worst10, power_w=ev.power)

    # ------------------------------------------------------------ 冻结干扰下的逐束容量（注水、目标容量迭代用）
    def snr_frozen(self, p=None):
        """每槽"每瓦 SINR"，干扰按 p 冻结（p=None 表示不计干扰）。"""
        if p is None:
            return self.a.copy()
        return self.a / (1. + self.b @ p)

    def beam_rate(self, s, q):
        """冻结每槽系数 s 时，各有槽束取功率 q[M] 的容量（bps）。"""
        r = np.minimum(np.log2(1. + s * q[self.owner]), self.eta)
        return self.B * np.bincount(self.owner, r, self.M)

    def beam_rate_slope(self, s, q):
        x = s * q[self.owner]
        g = np.where(np.log2(1. + x) >= self.eta, 0., s / (1. + x))
        return self.B / LN2 * np.bincount(self.owner, g, self.M)

    def min_power_for(self, s, target, iters=40):
        """冻结 s 下各束达到 target（bps）的最小功率；夹在 [lo,hi]，hi 也达不到返回 hi 并标记。"""
        lo, hi = self.lo.copy(), self.hi.copy()
        reach_hi = self.beam_rate(s, hi) >= target * (1 - 1e-9)
        ok_lo = self.beam_rate(s, lo) >= target * (1 - 1e-9)
        a, b = lo.copy(), hi.copy()
        for _ in range(iters):
            mid = (a + b) / 2
            good = self.beam_rate(s, mid) >= target
            b = np.where(good, mid, b)
            a = np.where(good, a, mid)
        out = np.where(ok_lo, lo, np.where(reach_hi, b, hi))
        return out, reach_hi

    # ------------------------------------------------------------ 上界
    def bounds(self):
        """缺槽上限 H 与无干扰 50 W 上限 U（均为乐观上界）及对应的 W、SGM 上界；返回缺口统计。"""
        U = np.zeros(self.N)
        U[self.idx] = self.beam_rate(self.a, self.hi)
        H = np.zeros(self.N)
        H[self.idx] = self.H
        d = self.demand
        sgm_U, _ = sgm_and_slope(np.minimum(d, U), d, self.beta)
        slot_short = d > H
        link_short = (~slot_short) & (d > U)
        return dict(W_H=float(np.minimum(d, H).sum() / self.D_total), W_U=float(np.minimum(d, U).sum() / self.D_total),
                    SGM_U=float(sgm_U.mean()),
                    slot_short_beams=int(slot_short.sum()), slot_short_aircraft=int(self.members[slot_short].sum()),
                    slot_short_demand=float(d[slot_short].sum() / self.D_total),
                    link_short_beams=int(link_short.sum()), link_short_aircraft=int(self.members[link_short].sum()),
                    link_short_demand=float(d[link_short].sum() / self.D_total))
