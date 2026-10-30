"""生成并冻结第二阶段方案：随机（种子 1—3）、固定、容量感知贪心、块指针头 T-SAC（训练种子 1—3）。

每份记录：场景 key、单机需求、分档与负载、波束处理顺序、逐束需求、实际组号/起点/槽数/功率，以及环境给出的直出指标。
"""
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from satellite_rl.s2lab.agents import load_policy
from satellite_rl.s2lab.env import BatchEnv, EnvOptions, ItemSampler, SceneBank
from satellite_rl.s2lab.evaluate import act_and_step
from satellite_rl.s2lab.rules import make_rule
from satellite_rl.s2lab.scenes import read_config, read_splits
from satellite_rl.s2lab.train import env_options

PLANS = {
    'random-s1': ('random', 1), 'random-s2': ('random', 2), 'random-s3': ('random', 3),
    'fixed': ('fixed', 0), 'greedy': ('greedy', 0),
    'ptr-s1': ('model', 1), 'ptr-s2': ('model', 2), 'ptr-s3': ('model', 3),
}
TIER = {'random': '随机', 'fixed': '固定', 'greedy': '贪心', 'ptr': '块指针头'}


def tier_of(plan_id):
    return plan_id.split('-')[0]


@torch.no_grad()
def generate(dataset_dir, split, plan_id, model_dir, device):
    config = read_config(dataset_dir)
    physics = config.physics_obj()
    items = read_splits(dataset_dir)[split]
    bank = SceneBank(dataset_dir, sorted({it['key'] for it in items}), physics, device)
    kind, seed = PLANS[plan_id]
    if kind == 'model':
        run = json.loads((Path(model_dir) / f'r4-ptr-s{seed}.json').read_text(encoding='utf-8'))
        agent = load_policy(Path(model_dir) / f'r4-ptr-s{seed}.pt', device)
        opts = env_options(dict(run, batch_envs=len(items)))
        factory = lambda env: agent
        source = dict(model=f'r4-ptr-s{seed}/policy_best_sgm.pt', config=run)
    else:
        opts = EnvOptions(batch_size=len(items))
        factory = lambda env: make_rule(kind, env, seed)
        source = dict(rule=kind, seed=seed)
    env = BatchEnv(bank, config, opts, device, ItemSampler(bank, config, items, False, 0), auto_reset=False)
    policy = factory(env)
    while env.alive.any():
        act_and_step(policy, env, env.observe())
    metrics = {m['slot']: m for m in env.finished}
    records = []
    for b, it in enumerate(items):
        n = int(env.n[b])
        g, s, c, p = env.allocation(b)
        m = metrics[b]
        assert m['key'] == it['key']
        records.append(dict(key=it['key'], split=split, plan=plan_id, per_aircraft_bps=float(env.per_aircraft[b]),
                            stratum=it['stratum'], load_ratio=it['load_ratio'],
                            order=env.order[b, :n].cpu().numpy(), demand=env.demand[b, :n].cpu().numpy().astype(np.float64),
                            groups=g.astype(np.int64), starts=s.astype(np.int64), counts=c.astype(np.int64), power=p.astype(np.float64),
                            requested=env.req[b, :n].cpu().numpy(),
                            direct={k: m[k] for k in ('W', 'SI', 'SGM', 'worst10', 'power_w', 'slots', 'interference_loss', 'unserved')}))
    return dict(plan=plan_id, split=split, dataset=str(dataset_dir), physics=physics.to_dict(),
                budget=config.power_budget_w, cap=config.beam_power_cap_w, source=source, records=records)


def save(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
