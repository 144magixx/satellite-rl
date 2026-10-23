"""确定性评价：一个策略（学习模型或规则）在一组 (场景, 单机需求) 上的直出指标。"""
from dataclasses import replace
from time import perf_counter
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .env import BatchEnv, EnvOptions, ItemSampler, SceneBank
from .rules import make_rule

METRIC_COLUMNS = ('key', 'per_aircraft_bps', 'load_ratio', 'stratum', 'beams', 'aircraft', 'W', 'SI', 'SGM', 'worst10', 'power_w',
                  'slots', 'copol_overlap', 'unserved', 'terminated', 'W_noint', 'interference_loss', 'mean_in_db',
                  'adj_overlap', 'adj_expected', 'adj_ratio', 'objective', 'seconds_per_scene')


def run_policy(policy_factory, bank, config, items, options, device, deterministic=True):
    """policy_factory(env) → 对象，含 act(obs)→action。返回每景指标列表（顺序与 items 一致）。"""
    batch = min(options.batch_size, len(items))
    env = BatchEnv(bank, config, replace(options, batch_size=batch), device, ItemSampler(bank, config, items, False, 0))
    policy = policy_factory(env)
    started = perf_counter()
    steps = 0
    while env.alive.any():
        obs = env.observe()
        action = policy.act(obs) if not hasattr(policy, 'off_policy') else policy.act(obs, explore=not deterministic)
        env.step(action)
        steps += 1
    elapsed = perf_counter() - started
    results = env.finished
    stratum = {it['key']: it.get('stratum', '') for it in items}
    for r in results:
        r['stratum'] = stratum.get(r['key'], '')
        r['seconds_per_scene'] = elapsed / max(1, len(results))
    return results


def evaluate_agent(agent, bank, config, items, options, device):
    return run_policy(lambda env: agent, bank, config, items, options, device)


def evaluate_rule(name, bank, config, items, options, device, seed=0):
    return run_policy(lambda env: make_rule(name, env, seed), bank, config, items, options, device)


def single_scene_timing(policy_factory, bank, config, items, options, device, count=10):
    """批量为 1 的逐景推理用时（秒），同机可比。"""
    times = []
    for it in items[:count]:
        env = BatchEnv(bank, config, replace(options, batch_size=1), device, ItemSampler(bank, config, [it], False, 0))
        policy = policy_factory(env)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        started = perf_counter()
        while env.alive.any():
            obs = env.observe()
            action = policy.act(obs) if not hasattr(policy, 'off_policy') else policy.act(obs, explore=False)
            env.step(action)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(perf_counter() - started)
    return float(np.mean(times)), float(np.std(times))


def summarize(results):
    """按分层和总体给出均值；返回 dict。"""
    out = {}
    groups = {'all': results}
    for r in results:
        groups.setdefault(r.get('stratum') or 'unknown', []).append(r)
    for name, rows in groups.items():
        if not rows:
            continue
        agg = dict(n=len(rows))
        for col in ('W', 'SI', 'SGM', 'worst10', 'power_w', 'slots', 'copol_overlap', 'unserved', 'interference_loss', 'mean_in_db', 'load_ratio'):
            vals = np.asarray([r[col] for r in rows], float)
            agg[col] = float(np.nanmean(vals))
        ratios = np.asarray([r['adj_ratio'] for r in rows], float)
        agg['adj_ratio'] = float(np.nanmean(ratios)) if np.isfinite(ratios).any() else float('nan')
        agg['adj_ratio_pooled'] = float(sum(r['adj_overlap'] for r in rows) / max(1e-9, sum(r['adj_expected'] for r in rows)))
        out[name] = agg
    return out


def write_results(path, results, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(METRIC_COLUMNS) + list((extra or {}).keys())
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow({**r, **(extra or {})})
