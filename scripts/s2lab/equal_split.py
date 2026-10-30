"""均分固定方案：每束功率 = 总预算 ÷ 波束数（连续值），频率槽 = 800 槽按波束数均分。

组按处理顺序轮转（第 t 个波束进第 t%8 组），每组 100 槽在本组波束间均分（多出的槽给本组排前面的波束），
组内从低到高紧挨着放，不看干扰。每束不超过 10 槽、50 W。
同时在本机重跑原固定规则（4 槽、10 W 档）和容量感知贪心作对照。

  python scripts/s2lab/equal_split.py --dataset-dir D --split confirm --out E
"""
import argparse
import json
from pathlib import Path

import torch

from _common import default_device
from satellite_rl.s2lab.env import EnvOptions, SceneBank
from satellite_rl.s2lab.evaluate import run_policy, summarize, write_results
from satellite_rl.s2lab.rules import make_rule
from satellite_rl.s2lab.scenes import read_config, read_splits


class EqualSplitRule:
    name = 'equal'

    def __init__(self, env):
        self.env = env

    @torch.no_grad()
    def act(self, obs=None):
        env = self.env
        T, G = env.T, env.G
        n = env.n.clamp(min=1)
        g = env.t % G
        k = env.t // G                                                # 本束在本组里的序号
        n_g = ((n - g + G - 1) // G).clamp(min=1)                     # 本组波束数
        c = (T // n_g + (k < T % n_g).long()).clamp(min=1, max=env.physics.max_slots_per_beam)
        free_run = env._free_run()[env.arange_b, g]                   # [B,T]
        s = torch.where(free_run >= c[:, None], env.arange_t[None, :], torch.full_like(free_run, T)).amin(1).clamp(max=T - 1)
        watts = torch.clamp(env.budget / n.float(), max=env.cap_w)
        action = torch.stack([g * T + s, c - 1, torch.zeros_like(c)], 1)
        return action, watts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--split', default='confirm')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=64)
    args = ap.parse_args()
    device = torch.device(default_device())
    dataset_dir = Path(args.dataset_dir)
    config = read_config(dataset_dir)
    items = read_splits(dataset_dir)[args.split]
    bank = SceneBank(dataset_dir, sorted({it['key'] for it in items}), config.physics_obj(), device)
    out_dir = Path(args.out) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = EnvOptions(batch_size=args.batch)
    policies = {'equal': lambda env: EqualSplitRule(env),
                'fixed': lambda env: make_rule('fixed', env),
                'greedy': lambda env: make_rule('greedy', env)}
    summary = {}
    for name, factory in policies.items():
        results = run_policy(factory, bank, config, items, opts, device)
        write_results(out_dir / f'rule-{name}.csv', results)
        summary[name] = summarize(results)
        a = summary[name]['all']
        print(f"{name:8s} n={a['n']} SGM={a['SGM']:.3f} W={a['W']:.3f} worst10={a['worst10']:.3f} P={a['power_w']:.0f} "
              f"slots={a['slots']:.0f} intf_loss={a['interference_loss']:.3f} I/N={a['mean_in_db']:.1f}dB adj={a['adj_ratio_pooled']:.2f}",
              flush=True)
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
