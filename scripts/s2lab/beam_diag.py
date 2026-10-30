"""逐束资源诊断（第四轮 D1）：在旧确认集上事先选定的 12 景，记录每个策略给每个波束的槽数、功率、最终容量与满足率，
看资源去了哪里、最差飞机在哪些波束。只用旧数据，不接触新确认集。

  python scripts/s2lab/beam_diag.py --dataset-dir D --runs-dir R --run-ids r4-base-s1 ... [--rules greedy] --out O
选景规则（事先固定）：宽裕、勉强两档各按负载排序取 12.5%/37.5%/62.5%/87.5% 分位处的 4 景；
不够档在负载 <1 的里取 25%/75% 分位 2 景，在负载 ≥1 的里取 25%/75% 分位 2 景。
输出 O/beams.csv（每行一个策略×场景×波束）与 O/scenes.json（所选场景）。
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from _common import default_device
from satellite_rl.s2lab.agents import load_policy
from satellite_rl.s2lab.env import BatchEnv, EnvOptions, ItemSampler, SceneBank
from satellite_rl.s2lab.evaluate import act_and_step
from satellite_rl.s2lab.matrix import read_matrix
from satellite_rl.s2lab.rules import make_rule
from satellite_rl.s2lab.scenes import read_config, read_splits
from satellite_rl.s2lab.train import env_options


def pick_scenes(items):
    out = []
    for s in ('ample', 'marginal'):
        rows = sorted([it for it in items if it['stratum'] == s], key=lambda it: it['load_ratio'])
        out += [rows[int(q * (len(rows) - 1))] for q in (.125, .375, .625, .875)]
    short = sorted([it for it in items if it['stratum'] == 'short'], key=lambda it: it['load_ratio'])
    lo, hi = [it for it in short if it['load_ratio'] < 1], [it for it in short if it['load_ratio'] >= 1]
    out += [lo[int(q * (len(lo) - 1))] for q in (.25, .75)] + [hi[int(q * (len(hi) - 1))] for q in (.25, .75)]
    return out


@torch.no_grad()
def run(factory, bank, config, items, opts, device):
    env = BatchEnv(bank, config, opts, device, ItemSampler(bank, config, items, False, 0), auto_reset=False)
    policy = factory(env)
    while env.active.any():
        act_and_step(policy, env, env.observe())
    cap = env.capacities()
    cap0 = env.capacities(interference=False)
    sgm = env._sgm(cap)
    rows = []
    for b, it in enumerate(items):
        n = int(env.n[b])
        rank = torch.empty(n, dtype=torch.long)
        rank[env.order[b, :n].cpu()] = torch.arange(n)
        d = env.demand[b, :n]
        for i in range(n):
            rows.append(dict(key=it['key'], stratum=it['stratum'], load=it['load_ratio'], beam=i, rank=int(rank[i]), n=n,
                             members=int(env.bank.members[env.scene[b], i]), width=float(env.bank.width[env.scene[b], i]),
                             demand_mbps=float(d[i]) / 1e6, group=int(env.grp[b, i]), start=int(env.start[b, i]),
                             req=int(env.req[b, i]), slots=int(env.cnt[b, i]), power=float(env.pw[b, i]),
                             cap_mbps=float(cap[b, i]) / 1e6, cap_noint_mbps=float(cap0[b, i]) / 1e6,
                             si=float(min(1., cap[b, i] / d[i])), sgm=float(sgm[b, i])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--runs-dir')
    ap.add_argument('--run-ids', nargs='*', default=[])
    ap.add_argument('--rules', nargs='*', default=[])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    device = torch.device(default_device())
    config = read_config(a.dataset_dir)
    items = pick_scenes(read_splits(a.dataset_dir)['confirm'])
    bank = SceneBank(a.dataset_dir, sorted({it['key'] for it in items}), config.physics_obj(), device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'scenes.json').write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding='utf-8')
    rows = []
    for rule in a.rules:
        for r in run(lambda env, n=rule: make_rule(n, env, 0), bank, config, items, EnvOptions(batch_size=len(items)), device):
            rows.append(dict(policy=f'rule-{rule}', **r))
    matrix = {r['run_id']: r for r in read_matrix(a.runs_dir)} if a.run_ids else {}
    for run_id in a.run_ids:
        cfg = matrix[run_id]
        agent = load_policy(Path(a.runs_dir) / run_id / 'policy_best_sgm.pt', device)
        for r in run(lambda env, ag=agent: ag, bank, config, items, env_options(dict(cfg, batch_envs=len(items))), device):
            rows.append(dict(policy=run_id, **r))
        print(run_id, 'done', flush=True)
    with (out / 'beams.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), 'rows')


if __name__ == '__main__':
    main()
