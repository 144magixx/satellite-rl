"""规则基线：容量感知贪心、固定分配、随机（合法）分配。直接读环境张量，批量给出动作。"""
import math

import torch


def _levels_index(env, watts):
    """把目标功率映射到不超过它的最大档（至少最低档），再按剩余预算裁剪。"""
    levels = env.levels
    remaining = (env.budget - env.used_w - env.reserve * (env.n - env.t - 1).clamp(min=0).float()).clamp(min=0)
    target = torch.minimum(watts, remaining)
    ok = levels[None, :] <= target[:, None] + 1e-6
    idx = torch.where(ok, torch.arange(len(levels), device=env.device)[None, :], torch.zeros(1, dtype=torch.long, device=env.device)).amax(1)
    return idx


def _window_cost(intf_par, c, free_run, allow):
    """intf_par[B,G,T] 每组极化上的干扰，c[B] 槽数；返回 [B,G,T] 窗口干扰和，不可行处为 inf。"""
    B, G, T = intf_par.shape
    cs = torch.cat([torch.zeros(B, G, 1, device=intf_par.device), intf_par.cumsum(-1)], -1)   # [B,G,T+1]
    s = torch.arange(T, device=intf_par.device)
    end = (s[None, :] + c[:, None]).clamp(max=T)                                              # [B,T]
    cost = cs.gather(2, end[:, None, :].expand(B, G, T)) - cs[:, :, :T]
    feasible = allow[:, :, None] & (free_run >= c[:, None, None])
    return torch.where(feasible, cost, torch.full_like(cost, float('inf')))


class GreedyRule:
    """按需求从大到小（环境顺序）：估计槽数 → 功率按需求比例且不超过所需 → 允许组内干扰最小的空闲段。"""

    name = 'greedy'

    def __init__(self, env, margin=1.1):
        self.env, self.margin = env, margin

    @torch.no_grad()
    def act(self, obs=None):
        env = self.env
        p = env.physics
        B, T, G = env.B, env.T, env.G
        cur = env.cur
        scene = env.scene
        d = env.demand[env.arange_b, cur]
        sig = env.bank.sig[scene, cur]
        path = env.bank.path[scene, cur].mean(-1)
        remaining_beams = (env.n - env.t).clamp(min=1)
        ordered = env.demand.gather(1, env.order)
        pos = env.arange_n[None, :]
        remaining_demand = (ordered * ((pos >= env.t[:, None]) & (pos < env.n[:, None]))).sum(1).clamp(min=1.)
        budget_left = (env.budget - env.used_w - env.reserve * (remaining_beams - 1).float()).clamp(min=0)
        share = torch.minimum(budget_left * d / remaining_demand, torch.full_like(d, env.cap_w))
        # 槽数：噪声受限估计，取满足需求的最小槽数
        c_opts = torch.arange(1, p.max_slots_per_beam + 1, device=env.device).float()
        snr = sig[:, None] * path[:, None] * (share[:, None] / c_opts[None, :]) / p.noise_w * p.impl_linear
        cap = c_opts[None, :] * p.slot_Hz * torch.log2(1 + snr).clamp(max=p.spectral_efficiency_cap)
        enough = cap >= d[:, None] * self.margin
        c = torch.where(enough.any(1), enough.float().argmax(1) + 1, torch.full_like(cur, p.max_slots_per_beam))
        # 可行性：允许组内最长空闲段
        allow = env.allowed_groups()
        free_run = env._free_run()
        c = torch.minimum(c, env.count_cap(allow, free_run))
        # 功率：满足需求所需（含裕量）与份额取小
        eff_need = (d * self.margin / (c.float() * p.slot_Hz)).clamp(max=p.spectral_efficiency_cap)
        p_need = c.float() * p.noise_w * (2 ** eff_need - 1) / (sig * path * p.impl_linear)
        watts = torch.minimum(share, p_need)
        pw = _levels_index(env, watts)
        # 组×起点：窗口干扰最小
        intf = env._interference(cur)                                 # [B,2,T]
        parity = torch.arange(G, device=env.device) % 2
        cost = _window_cost(intf[:, parity], c, free_run, allow)
        flat = cost.reshape(B, G * T)
        gs = flat.argmin(1)
        none = torch.isinf(flat.min(1).values)
        gs = torch.where(none, torch.zeros_like(gs), gs)
        return torch.stack([gs, c - 1, pw], 1)


class FixedRule:
    """固定槽数、固定功率档、组轮转、组内首个可容纳的空闲段。"""

    name = 'fixed'

    def __init__(self, env, slots=4, watts=None):
        self.env, self.slots = env, slots
        self.watts = watts if watts is not None else env.budget / 150.

    @torch.no_grad()
    def act(self, obs=None):
        env = self.env
        B, T, G = env.B, env.T, env.G
        g = env.t % G
        free_run = env._free_run()[env.arange_b, g]                   # [B,T]
        c = torch.full((B,), self.slots, dtype=torch.long, device=env.device)
        c = torch.minimum(c, free_run.amax(-1).clamp(min=1))
        ok = free_run >= c[:, None]
        s = torch.where(ok, torch.arange(T, device=env.device)[None, :], torch.full_like(free_run, T)).amin(1).clamp(max=T - 1)
        pw = _levels_index(env, torch.full((B,), float(self.watts), device=env.device))
        return torch.stack([g * T + s, c - 1, pw], 1)


class RandomRule:
    """在合法动作（空闲起点、可容纳槽数、预算内功率）中均匀随机。"""

    name = 'random'

    def __init__(self, env, seed=0):
        self.env = env
        self.gen = torch.Generator(device=env.device)
        self.gen.manual_seed(seed)

    @torch.no_grad()
    def act(self, obs=None):
        env = self.env
        B, T, G = env.B, env.T, env.G
        allow = env.allowed_groups()
        free_run = env._free_run()
        feasible = (allow[:, :, None] & (free_run >= 1)).reshape(B, G * T).float()
        feasible[feasible.sum(1) == 0, 0] = 1.
        gs = torch.multinomial(feasible, 1, generator=self.gen).squeeze(1)
        run = free_run.reshape(B, G * T).gather(1, gs[:, None]).squeeze(1).clamp(min=1, max=env.physics.max_slots_per_beam)
        c = (torch.rand(B, device=env.device, generator=self.gen) * run.float()).floor().long() + 1
        remaining = (env.budget - env.used_w - env.reserve * (env.n - env.t - 1).clamp(min=0).float()).clamp(min=0)
        ok = (env.levels[None, :] <= remaining[:, None] + 1e-6).float()
        ok[ok.sum(1) == 0, 0] = 1.
        pw = torch.multinomial(ok, 1, generator=self.gen).squeeze(1)
        return torch.stack([gs, c - 1, pw], 1)


def make_rule(name, env, seed=0):
    if name == 'greedy':
        return GreedyRule(env)
    if name == 'fixed':
        return FixedRule(env)
    if name == 'random':
        return RandomRule(env, seed)
    raise ValueError(name)
