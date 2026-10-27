"""第三轮状态补充特征（只用动作前已知的量）。各项定义见 docs/Claude维护/S2-LAB/11_第三轮方案.md 第 5 节。

返回的张量都已缩放到大致 0—1 的范围；缩放区间由真实数据体检（scripts/s2lab/feature_audit.py）确定。
"""
import math

import torch
import torch.nn.functional as F

FEATURE_NAMES = ('pneed', 'share', 'outint', 'snr', 'sens', 'future', 'gspec', 'hint', 'hintx')
# 每项进入 14 令牌的位置与维度：标量令牌(rate/power) 或 每槽列(slot，每槽几列) 或 全局令牌(glob)
# 第四轮 hint/hintx：在当前状态下贪心会给的槽数与功率档（独热 10＋13，进需求令牌）、起点窗口（每槽 0/1，进频率块）
FEATURE_SLOT = dict(pneed=('rate', 30), share=('power', 1), outint=('slot', 1), snr=('slot', 2), sens=('slot', 1),
                    future=('glob', 4), gspec=('glob', 17), hint=('rate', 23), hintx=('slot', 1))


def _db(x, lo, hi):
    """线性量 → dB → 线性映射到 [0,1]（lo..hi dB）。"""
    return ((10 * torch.log10(x + 1e-30) - lo) / (hi - lo)).clamp(0, 1)


def compute(env, cur, g, intf, budget_left, names):
    """env: BatchEnv；cur[B] 当前束；g[B] 本束允许组；intf[B,2,T] 当前束受到的干扰（W）；budget_left[B] 扣预留后的可用功率。"""
    out = {}
    B, N, T = env.B, env.N, env.T
    p = env.physics
    b = env.arange_b
    scene = env.scene
    pg = g % 2
    demand = env.demand[b, cur]
    sig = env.bank.sig[scene, cur]
    path_cur = env.bank.path[scene, cur]                                    # [B,T]
    if 'pneed' in names:
        a = p.impl_linear * sig * path_cur.mean(-1) / env.noise           # 每瓦信噪比（整束功率均摊前）
        c = torch.arange(1, 11, device=env.device).float()
        eff = demand[:, None] / (c[None, :] * p.slot_Hz)                 # 每槽所需频谱效率
        unreachable = eff > p.spectral_efficiency_cap
        power = c[None, :] / a[:, None] * (torch.pow(2., eff.clamp(max=p.spectral_efficiency_cap)) - 1)
        over = (power > env.cap_w) & ~unreachable
        val = (torch.log2(power.clamp(min=1e-3) / .5) / math.log2(200.)).clamp(0, 1.2)   # 0.5 W→0，100 W→1
        val = torch.where(unreachable, torch.full_like(val, 1.2), val)
        out['f_pneed'] = torch.cat([val, unreachable.float(), over.float()], 1)          # [B,30]
    if 'share' in names:
        ordered = env.demand.gather(1, env.order)
        pos = env.arange_n[None, :]
        remaining_demand = (ordered * ((pos >= env.t[:, None]) & (pos < env.n[:, None]))).sum(1).clamp(min=1.)
        out['f_share'] = (torch.minimum(budget_left * demand / remaining_demand, torch.full_like(demand, env.cap_w)) / env.cap_w)[:, None]
    occ_beam = env.A.sum(1) > 0                                             # [B,N,T] 已放波束占用的槽
    parity = (env.grp % 2).clamp(min=0)                                     # [B,N]
    placed = env.grp >= 0
    path_all = env.bank.path[scene]                                         # [B,N,T]
    k_out = env.bank.coupling[scene][b, :, cur]                             # [B,N] 接收 j ← 发射 cur（不含路径）
    same_pol = (parity == pg[:, None]) & placed
    if 'outint' in names:
        pol = torch.where(same_pol, torch.ones_like(k_out), torch.full_like(k_out, p.xpd_linear)) * placed
        x = torch.einsum('bn,bnt->bt', k_out * pol, path_all * occ_beam) / env.noise     # 每槽 1 W 给已放束新增的 I/N 之和
        out['f_outint'] = _db(x, -40., 20.)
    if 'snr' in names:
        snr = p.impl_linear * sig[:, None] * path_cur / env.noise                          # 每槽 1 W 的信噪比
        sinr = p.impl_linear * sig[:, None] * path_cur / (env.noise + intf[b, pg])
        out['f_snr'] = torch.stack([_db(snr, -10., 20.), _db(sinr, -10., 20.)], 1)        # [B,2,T]；真实数据中位约 4 dB
    if 'sens' in names:
        cap = env.capacities()
        closeness = torch.exp(-(cap / env.demand.clamp(min=1.) - 1).abs()) * placed           # [B,N]
        other_grp = env.grp != g[:, None]
        w = closeness * (same_pol & other_grp)
        i_all = env._all_interference()                                                     # [B,N,T]
        kappa = k_out[:, :, None] * path_all                                                # [B,N,T] 每瓦到 j 的增益
        x = torch.einsum('bn,bnt->bt', w, kappa * occ_beam / (env.noise + i_all))
        out['f_sens'] = _db(x, -30., 5.)                                                     # 真实数据中位约 −15 dB，最大约 +4 dB
    if 'future' in names:
        pos = env.arange_n[None, :]
        later = (pos > env.t[:, None]) & (pos < env.n[:, None])                             # 按处理顺序在本束之后
        later_beams = torch.zeros(B, N, dtype=torch.bool, device=env.device)
        later_beams.scatter_(1, env.order, later)
        near = env.bank.adjacent[scene][b, cur] & later_beams                              # [B,N]
        k_in = env.bank.coupling[scene][b, cur]                                             # 接收 cur ← 发射 j
        in_sum = (k_in * near).sum(1) * path_cur.mean(-1) / env.noise
        out_sum = (k_out * near * path_all.mean(-1)).sum(1) / env.noise
        # 真实数据：1° 内后续邻居平均约 20 个、需求合计约 10 Gbps，用对数压缩
        out['f_future'] = torch.stack([torch.log1p(near.sum(1).float()) / math.log1p(150.),
                                       torch.log1p((env.demand * near).sum(1) / 1e9) / math.log1p(150.),
                                       torch.log10(1 + in_sum) / 3, torch.log10(1 + out_sum) / 3], 1)
    if 'gspec' in names:
        free = (~env.occ).sum(-1).float()                                                   # [B,G]
        longest = env._free_run().amax(-1).float()
        remaining = (env.n - env.t).clamp(min=1).float()
        out['f_gspec'] = torch.cat([free / T, longest / T, (torch.log1p(free.sum(1) / remaining) / math.log1p(800.))[:, None]], 1)
    if 'hint' in names or 'hintx' in names:
        from .rules import GreedyRule
        a = GreedyRule(env).act()                                   # 只读环境当前状态，不改动
        c = a[:, 1]
        if 'hint' in names:
            out['f_hint'] = torch.cat([F.one_hot(c, 10), F.one_hot(a[:, 2], env.K)], 1).float()
        if 'hintx' in names:
            s = a[:, 0] % T
            t = env.arange_t[None, :]
            out['f_hintx'] = ((t >= s[:, None]) & (t < (s + c + 1)[:, None])).float()
    return out
