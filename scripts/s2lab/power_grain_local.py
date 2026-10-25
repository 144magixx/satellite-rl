"""本机粗查：功率档粒度对容量感知贪心的影响（冒烟数据 20 景 × 4 档需求，口径预算 2000 W）。远端不可用时的替代，只看趋势。"""
from dataclasses import replace

import numpy as np
import torch

from _common import ROOT  # noqa: F401
from satellite_rl.s2lab.env import EnvOptions, SceneBank, DEFAULT_POWER_LEVELS
from satellite_rl.s2lab.evaluate import evaluate_rule, summarize
from satellite_rl.s2lab.scenes import read_config, read_index, load_ratio, stratum_of

d = 'D:/paper/outputs/experiments/s2lab_20261024/smoke_data'
config = replace(read_config(d), power_budget_w=2000., load_thresholds=(.35, .6))
keys = [r['key'] for r in read_index(d)]
device = torch.device('cuda')
bank = SceneBank(d, keys, config.physics_obj(), device)
totals = {k: int(bank.members_total[i]) for i, k in enumerate(keys)}
items = []
for k in keys:
    for v in (30e6, 45e6, 60e6, 80e6):
        items.append(dict(key=k, per_aircraft_bps=v, stratum=stratum_of(load_ratio(totals[k], v, config), config)))
menus = {'默认13档': DEFAULT_POWER_LEVELS}
for step in (5., 1., .5, .1):
    menus[f'{step:g}W步长'] = tuple(float(x) for x in np.arange(step, 50 + 1e-9, step))
for name, levels in menus.items():
    res = evaluate_rule('greedy', bank, config, items, EnvOptions(batch_size=80, power_levels=levels), device)
    s = summarize(res)
    line = ' | '.join(f"{g}: SGM {s[g]['SGM']:.3f} W {s[g]['W']:.3f} P {s[g]['power_w']:.0f}" for g in ('all', 'ample', 'marginal', 'short') if g in s)
    print(f'{name:8s} ({len(levels):3d}档) {line}', flush=True)
