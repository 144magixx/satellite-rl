"""文献对照：模拟退火（Cocco et al. 2018）与遗传算法（Paris et al. 2019）的 GPU 批量版。

整套分配 = 每束 (组, 起点, 槽数, 功率档)。候选解通过环境按需求顺序回放（full 掩码下自动截断/修复），
得到真实直出指标；目标 = SGM + W（第一轮的选模标准；第二轮学习方法改按 SGM 选模，两者不同）。起点为容量感知贪心的真实分配。
预算以"真实评价次数"计（32 或 320）：退火含 1 次贪心初解；遗传另算 1 次贪心，实际 321 次。初始化、扰动、交叉是本项目的组合，
不是对原文的逐项复现，报告中称"按文献思路实现"。
"""
from dataclasses import replace
import math

import numpy as np
import torch

from .env import BatchEnv, ItemSampler
from .rules import GreedyRule


class ReplayPolicy:
    def __init__(self, env, cand):
        self.env, self.cand = env, cand

    def act(self, obs=None):
        env = self.env
        cur = env.cur
        b = env.arange_b
        return torch.stack([self.cand['g'][b, cur] * env.T + self.cand['s'][b, cur], self.cand['c'][b, cur] - 1, self.cand['p'][b, cur]], 1)


def _rollout(bank, config, items, options, device, policy_factory):
    env = BatchEnv(bank, config, replace(options, batch_size=len(items)), device, ItemSampler(bank, config, items, False, 0), auto_reset=False)
    policy = policy_factory(env)
    while env.active.any():
        env.step(policy.act(env.observe()))
    metrics = sorted(env.finished, key=lambda m: m['slot'])
    cand = dict(g=env.grp.clamp(min=0).clone(), s=env.start.clone(), c=env.cnt.clamp(min=1).clone(), p=_power_index(env))
    score = torch.tensor([m['SGM'] + m['W'] for m in metrics], device=device)
    return metrics, cand, score


def _power_index(env):
    diff = (env.pw[:, :, None] - env.levels[None, None, :]).abs()
    return diff.argmin(-1)


def _mutate(cand, valid, rng, prob, T, K, G):
    """对每个个体，以 prob 概率改动每个波束的某一项。"""
    B, N = cand['g'].shape
    device = cand['g'].device
    hit = (torch.rand(B, N, device=device, generator=rng) < prob) & valid
    which = torch.randint(0, 4, (B, N), device=device, generator=rng)
    out = {k: v.clone() for k, v in cand.items()}
    new_g = torch.randint(0, G, (B, N), device=device, generator=rng)
    new_s = torch.randint(0, T, (B, N), device=device, generator=rng)
    new_c = (out['c'] + torch.randint(-3, 4, (B, N), device=device, generator=rng)).clamp(1, 10)
    new_p = (out['p'] + torch.randint(-3, 4, (B, N), device=device, generator=rng)).clamp(0, K - 1)
    out['g'] = torch.where(hit & (which == 0), new_g, out['g'])
    out['s'] = torch.where(hit & (which == 1), new_s, out['s'])
    out['c'] = torch.where(hit & (which == 2), new_c, out['c'])
    out['p'] = torch.where(hit & (which == 3), new_p, out['p'])
    return out


def _occupancy(cand, valid, G, T):
    """[B,G,T] 候选解的占用（按 c 槽展开）。"""
    B, N = cand['g'].shape
    device = cand['g'].device
    t = torch.arange(T, device=device)
    span = (t[None, None, :] >= cand['s'][:, :, None]) & (t[None, None, :] < (cand['s'] + cand['c'])[:, :, None]) & valid[:, :, None]   # [B,N,T]
    occ = torch.zeros(B, G, T, dtype=torch.bool, device=device)
    for g in range(G):
        occ[:, g] = (span & (cand['g'] == g)[:, :, None]).any(1)
    return occ


def _free_run(occ):
    T = occ.shape[-1]
    t = torch.arange(T, device=occ.device)
    idx = torch.where(occ, t[None, None, :], torch.full_like(occ, T, dtype=torch.long))
    return torch.flip(torch.cummin(torch.flip(idx, [2]), 2)[0], [2]) - t[None, None, :]


def _smart_mutate(cand, valid, rng, T, K, G, moves=1):
    """每个个体做 moves 次单束改动：搬到某组一个放得下的空闲位置（50%）、槽数±1（25%）、功率档±1（25%）。"""
    out = {k: v.clone() for k, v in cand.items()}
    B, N = out['g'].shape
    device = out['g'].device
    rows = torch.arange(B, device=device)
    for _ in range(moves):
        j = torch.multinomial(valid.float(), 1, generator=rng).squeeze(1)
        op = torch.rand(B, device=device, generator=rng)
        # 搬家
        occ = _occupancy(out, valid, G, T)
        own = torch.zeros(B, T, dtype=torch.bool, device=device)
        t = torch.arange(T, device=device)
        s_j, c_j, g_j = out['s'][rows, j], out['c'][rows, j], out['g'][rows, j]
        own = (t[None, :] >= s_j[:, None]) & (t[None, :] < (s_j + c_j)[:, None])
        occ[rows, g_j] &= ~own
        new_g = torch.randint(0, G, (B,), device=device, generator=rng)
        run = _free_run(occ)[rows, new_g]                                   # [B,T]
        feasible = run >= c_j[:, None]
        has = feasible.any(1)
        probs = feasible.float() + (~has)[:, None].float()
        new_s = torch.multinomial(probs, 1, generator=rng).squeeze(1)
        move = (op < .5) & has
        out['g'][rows, j] = torch.where(move, new_g, g_j)
        out['s'][rows, j] = torch.where(move, new_s, s_j)
        # 槽数 ±1
        dc = torch.where(torch.rand(B, device=device, generator=rng) < .5, -1, 1)
        chg_c = (op >= .5) & (op < .75)
        out['c'][rows, j] = torch.where(chg_c, (c_j + dc).clamp(1, 10), out['c'][rows, j])
        # 功率 ±1
        dp = torch.where(torch.rand(B, device=device, generator=rng) < .5, -1, 1)
        chg_p = op >= .75
        out['p'][rows, j] = torch.where(chg_p, (out['p'][rows, j] + dp).clamp(0, K - 1), out['p'][rows, j])
    return out


def simulated_annealing(bank, config, items, options, device, budget=32, seed=0, t0=.05, t1=.001, per_beam_prob=None):
    """每景一条链；每次迭代对当前解做一次扰动（约 1 个波束），Metropolis 接受；预算 = 评价次数。"""
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    greedy_metrics, cur, cur_score = _rollout(bank, config, items, options, device, lambda env: GreedyRule(env))
    n = torch.tensor([m['beams'] for m in greedy_metrics], device=device)
    valid = torch.arange(cur['g'].shape[1], device=device)[None, :] < n[:, None]
    best, best_score, best_metrics = {k: v.clone() for k, v in cur.items()}, cur_score.clone(), list(greedy_metrics)
    K, G = len(options.power_levels), 8
    for i in range(budget - 1):
        temp = t0 * (t1 / t0) ** (i / max(1, budget - 2))
        proposal = _smart_mutate(cur, valid, rng, 100, K, G, moves=1)
        metrics, realized, score = _rollout(bank, config, items, options, device, lambda env, c=proposal: ReplayPolicy(env, c))
        accept = (score >= cur_score) | (torch.rand(len(items), device=device, generator=rng) < torch.exp((score - cur_score) / temp))
        for k in cur:
            cur[k] = torch.where(accept[:, None], realized[k], cur[k])
        cur_score = torch.where(accept, score, cur_score)
        improved = score > best_score
        for k in best:
            best[k] = torch.where(improved[:, None], realized[k], best[k])
        best_score = torch.where(improved, score, best_score)
        for j in torch.nonzero(improved).flatten().tolist():
            best_metrics[j] = metrics[j]
    for m in best_metrics:
        m['evaluations'] = budget
    return best_metrics


def genetic_algorithm(bank, config, items, options, device, budget=320, population=16, seed=0, mutation_prob=.03, elite=2):
    """每景一个种群（贪心解 + 其变异体）；锦标赛选择、逐束均匀交叉、变异、精英保留；代数 = 预算 ÷ 种群。"""
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    S, P = len(items), population
    generations = max(1, budget // P)
    rep_items = [it for it in items for _ in range(P)]
    greedy_metrics, g_cand, g_score = _rollout(bank, config, items, options, device, lambda env: GreedyRule(env))
    n = torch.tensor([m['beams'] for m in greedy_metrics], device=device)
    N = g_cand['g'].shape[1]
    pop = {k: v.repeat_interleave(P, 0) for k, v in g_cand.items()}
    valid = (torch.arange(N, device=device)[None, :] < n.repeat_interleave(P)[:, None])
    K = len(options.power_levels)
    mutated = _smart_mutate(pop, valid, rng, 100, K, 8, moves=2)
    for k in pop:                                   # 每个种群第 0 个个体保留贪心原解
        mutated[k][::P] = pop[k][::P]
    metrics, pop, score = _rollout(bank, config, rep_items, options, device, lambda env, c=mutated: ReplayPolicy(env, c))
    best_metrics = [None] * S
    best_score = torch.full((S,), -1e9, device=device)

    def update_best(metrics, score):
        sc = score.reshape(S, P)
        idx = sc.argmax(1)
        for i in range(S):
            if sc[i, idx[i]] > best_score[i]:
                best_score[i] = sc[i, idx[i]]
                best_metrics[i] = metrics[i * P + int(idx[i])]
    update_best(metrics, score)
    for _ in range(generations - 1):
        sc = score.reshape(S, P)
        order = sc.argsort(1, descending=True)
        # 锦标赛选择父母（2 选 1）
        a = torch.randint(0, P, (S, P), device=device, generator=rng)
        b = torch.randint(0, P, (S, P), device=device, generator=rng)
        parent1 = torch.where(sc.gather(1, a) >= sc.gather(1, b), a, b)
        a = torch.randint(0, P, (S, P), device=device, generator=rng)
        b = torch.randint(0, P, (S, P), device=device, generator=rng)
        parent2 = torch.where(sc.gather(1, a) >= sc.gather(1, b), a, b)
        base = (torch.arange(S, device=device) * P)[:, None]
        i1, i2 = (base + parent1).flatten(), (base + parent2).flatten()
        mask = torch.rand(S * P, N, device=device, generator=rng) < .5
        child = {k: torch.where(mask, v[i1], v[i2]) for k, v in pop.items()}
        child = _smart_mutate(child, valid, rng, 100, K, 8, moves=1)
        # 精英直接保留
        elite_idx = (base + order[:, :elite]).flatten()
        slots = (base + torch.arange(elite, device=device)[None, :]).flatten()
        for k in child:
            child[k][slots] = pop[k][elite_idx]
        metrics, pop, score = _rollout(bank, config, rep_items, options, device, lambda env, c=child: ReplayPolicy(env, c))
        update_best(metrics, score)
    for m in best_metrics:
        m['evaluations'] = generations * P
    return best_metrics


def run_meta(name, bank, config, items, options, device, seed=0, chunk=32):
    """name: sa32 | sa320 | ga32 | ga320。按 chunk 个场景一批跑，返回与 run_policy 同格式的指标列表。"""
    kind, budget = name[:2], int(name[2:])
    out = []
    for i in range(0, len(items), chunk):
        part = items[i:i + chunk]
        if kind == 'sa':
            res = simulated_annealing(bank, config, part, options, device, budget, seed)
        else:
            res = genetic_algorithm(bank, config, part, options, device, budget, 16 if budget >= 64 else 8, seed)
        out.extend(res)
    stratum = {it['key']: it.get('stratum', '') for it in items}
    for r in out:
        r['stratum'] = stratum.get(r['key'], '')
        r['seconds_per_scene'] = float('nan')
    return out
