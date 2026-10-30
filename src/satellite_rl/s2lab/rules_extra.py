"""另外三条规则基线：旧贪心（W26—W32）、原版论文贪心（首次提交 45427b4）、分散摆放规则（v2_free8）。

三条都按环境顺序逐束出动作 [B,3]，组号由规则自己定；组不在 env.allowed_groups() 里（或已满）时
改用允许组里空闲槽最多的那个。全掩码环境下起点必须是空闲槽、功率不超预算减预留，这两处是相对原实现的被迫改动。
"""
import numpy as np
import torch

from .rules import _levels_index


def _own_group(env, desired):
    allow = env.allowed_groups() & (~env.occ).any(-1)
    free = (~env.occ).sum(-1)
    fallback = torch.where(allow, free, torch.full_like(free, -1)).argmax(1)
    return torch.where(allow[env.arange_b, desired], desired, fallback)


def _one_hot(env, g):
    allow = torch.zeros(env.B, env.G, dtype=torch.bool, device=env.device)
    allow[env.arange_b, g] = True
    return allow


def _old_preference(env):
    """原版贪心的偏好：全部组合计占用最少的槽作起点，槽数 max(1, min(5, 10−该槽占用组数))，功率按需求 20/25/30 W。"""
    load = env.occ.sum(1)                                            # [B,T] 每槽被几个组占用
    start = load.argmin(1)
    count = (10 - load[env.arange_b, start]).clamp(min=1, max=5)
    d = env.demand[env.arange_b, env.cur]
    watts = torch.where(d >= 8e8, 30., torch.where(d >= 4e8, 25., 20.)).float()
    return start, count, watts


class OldFixedGreedy:
    """W26—W32 的旧贪心（t0_width/runner.py BackgroundPolicy('greedy')）：原版偏好映射到合法域——
    组 = 处理序号 % 8；在本组合法 (起点, 槽数) 里先取槽数最接近偏好、再取起点最接近偏好、再取起点小的；
    功率取最接近偏好的合法档（并列取小）。"""

    name = 'greedy_old'

    def __init__(self, env):
        self.env = env

    @torch.no_grad()
    def act(self, obs=None):
        env = self.env
        T = env.T
        g = _own_group(env, env.t % env.G)
        start, count, watts = _old_preference(env)
        free_run = env._free_run()
        fr = free_run[env.arange_b, g]                                # [B,T]
        cap = env.count_cap(_one_hot(env, g), free_run)
        most = torch.minimum(fr, cap[:, None])                        # 每个起点能放的最多槽数（占用处为 0）
        c = torch.minimum(count, most.amax(1)).clamp(min=1)
        key = (env.arange_t[None, :] - start[:, None]).abs() * T + env.arange_t[None, :]
        s = torch.where(most >= c[:, None], key, torch.full_like(key, 2 * T * T)).argmin(1)
        remaining = (env.budget - env.used_w - env.reserve * (env.n - env.t - 1).clamp(min=0).float()).clamp(min=0)
        legal = env.levels[None, :] <= remaining[:, None] + 1e-6
        gap = (env.levels[None, :] - watts[:, None]).abs()
        pw = torch.where(legal, gap, torch.full_like(gap, float('inf'))).argmin(1)
        return torch.stack([g * T + s, c - 1, pw], 1)


class PaperGreedy:
    """原版论文自带贪心（45427b4 evaluation/compare_metrics.py greedy_policy）：组 = 波束序号 % 8，
    起点 = 全部组合计占用最少的槽，槽数 max(1, min(5, 10−占用组数))，功率 20/25/30 W；原版环境把槽数截到
    本组从起点起的空闲前缀。起点在本组已占时原版该束得 0 槽，全掩码下改放本组离它最近的空闲槽。"""

    name = 'greedy_paper'

    def __init__(self, env):
        self.env = env

    @torch.no_grad()
    def act(self, obs=None):
        env = self.env
        T = env.T
        g = _own_group(env, env.t % env.G)
        start, count, watts = _old_preference(env)
        free_run = env._free_run()
        fr = free_run[env.arange_b, g]
        key = (env.arange_t[None, :] - start[:, None]).abs() * T + env.arange_t[None, :]
        s = torch.where(fr >= 1, key, torch.full_like(key, 2 * T * T)).argmin(1)
        cap = env.count_cap(_one_hot(env, g), free_run)
        c = torch.minimum(torch.minimum(count, fr[env.arange_b, s]), cap).clamp(min=1)
        pw = _levels_index(env, watts)
        return torch.stack([g * T + s, c - 1, pw], 1)


def proportional_power(demand, budget, cap):
    """S2-FIX greedy.proportional_power：按需求比例分预算，到单束上限的截住，余量再按比例分给其余波束。"""
    demand = np.asarray(demand, float)
    power = np.zeros(len(demand))
    free = np.ones(len(demand), bool)
    remaining = min(budget, cap * len(demand))
    while free.any():
        quota = remaining * demand[free] / demand[free].sum()
        indices = np.flatnonzero(free)
        hit = quota >= cap
        if not hit.any():
            power[indices] = quota
            break
        power[indices[hit]] = cap
        free[indices[hit]] = False
        remaining -= cap * hit.sum()
    power *= min(1., budget / np.nextafter(power.sum(), np.inf))
    return power


class DispersedRule:
    """分散摆放规则（outputs/experiments/s2_orig_compare_20261022/v2_free8.py allocation(fixed_groups=True, free_starts=True)）。
    开局一次算好：槽数 = 按"预算×需求占比（≤50 W）"估计、满足需求的最少槽数；功率 = 按需求比例注水；
    组 = 需求名次 % 8，某组槽数合计超过 T 时按需求配额压缩。逐束：在本组所有放得下的窗口里取与已放
    同极化异组波束"重叠槽数×对称耦合"之和最小的起点（并列取小）；一个窗口都放不下就缩到最长空段、放在它开头。"""

    name = 'dispersed'

    def __init__(self, env):
        self.env = env
        self.group = torch.zeros(env.B, env.N, dtype=torch.long, device=env.device)
        self.slots = torch.ones(env.B, env.N, dtype=torch.long, device=env.device)
        self.power = torch.zeros(env.B, env.N, device=env.device)

    def _plan(self, b):
        env = self.env
        p = env.physics
        n, sc = int(env.n[b]), int(env.scene[b])
        demand = env.demand[b, :n].double().cpu().numpy()
        # 原版用 f0 处的路径增益；bank.path 是各槽中心频率处的，按 1/f² 折回 f0
        path0 = env.bank.path[sc, :n, 0].double().cpu().numpy() * ((p.f0_Hz + .5 * p.slot_Hz) / p.f0_Hz) ** 2
        coefficient = env.bank.sig[sc, :n].double().cpu().numpy() * path0
        power = proportional_power(demand, env.budget, env.cap_w)
        estimated = np.minimum(env.budget * demand / demand.sum(), env.cap_w)
        K = p.max_slots_per_beam
        slots = np.full(n, K, int)
        for k in range(1, K + 1):
            snr = coefficient * estimated / k / p.noise_w
            capacity = k * p.slot_Hz * np.minimum(np.log2(1 + snr * p.impl_linear), p.spectral_efficiency_cap)
            slots[(slots == K) & (capacity >= demand)] = k
        order = np.argsort(-demand, kind='stable')
        groups = np.empty(n, int)
        groups[order] = np.arange(n) % env.G
        T = env.T
        for g in range(env.G):
            members = np.flatnonzero(groups == g)
            if slots[members].sum() > T:
                quota = (T - len(members)) * demand[members] / demand[members].sum()
                count = np.minimum(slots[members], 1 + np.floor(quota).astype(int))
                while count.sum() < T and np.any(count < slots[members]):
                    score = np.where(count < slots[members], quota - (count - 1), -np.inf)
                    count[np.argmax(score)] += 1
                slots[members] = count
        self.group[b, :n] = torch.from_numpy(groups).to(self.group)
        self.slots[b, :n] = torch.from_numpy(slots).to(self.slots)
        self.power[b, :n] = torch.from_numpy(power).to(self.power)

    @torch.no_grad()
    def act(self, obs=None):
        env = self.env
        B, T = env.B, env.T
        for b in torch.nonzero((env.t == 0) & env.alive).flatten().tolist():
            self._plan(b)
        cur = env.cur
        g = _own_group(env, self.group[env.arange_b, cur])
        want = self.slots[env.arange_b, cur]
        free_run = env._free_run()
        fr = free_run[env.arange_b, g]                                # [B,T]
        longest = fr.amax(1)
        c = torch.minimum(want, env.count_cap(_one_hot(env, g), free_run)).clamp(min=1)
        # 已放的同极化异组波束在各槽上的耦合权重之和；窗口代价 = 窗口内求和 = Σ_j 重叠槽数 × 权重
        weight = torch.maximum(env.bank.coupling[env.scene, cur], env.bank.coupling[env.scene, :, cur])     # [B,N]
        eligible = (env.grp >= 0) & (env.grp % 2 == (g % 2)[:, None]) & (env.grp != g[:, None])
        t = env.arange_t[None, None, :]
        occupied = (t >= env.start[:, :, None]) & (t < (env.start + env.cnt)[:, :, None])                  # [B,N,T]
        profile = torch.einsum('bn,bnt->bt', (weight * eligible).double(), occupied.double())
        cs = torch.cat([torch.zeros(B, 1, dtype=profile.dtype, device=env.device), profile.cumsum(1)], 1)
        end = (env.arange_t[None, :] + c[:, None]).clamp(max=T)
        cost = cs.gather(1, end) - cs[:, :T]
        s = torch.where(fr >= c[:, None], cost, torch.full_like(cost, float('inf'))).argmin(1)
        first_longest = torch.where(fr >= longest[:, None], env.arange_t[None, :], torch.full_like(fr, T)).amin(1)
        s = torch.where(want > longest, first_longest, s).clamp(max=T - 1)
        pw = _levels_index(env, self.power[env.arange_b, cur])
        return torch.stack([g * T + s, c - 1, pw], 1)
