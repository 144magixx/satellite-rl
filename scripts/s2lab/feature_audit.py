"""第三轮特征体检：在真实开发集场景上用贪心走完整景，记录每一步的补充特征，打印分位数、落在 0/上限的比例、常数列比例。
  python scripts/s2lab/feature_audit.py --dataset-dir D:/paper/outputs/experiments/s2lab_round3_20261027/data/v1 [--scenes 20]
"""
import argparse

import numpy as np
import torch

from _common import ROOT  # noqa: F401
from satellite_rl.s2lab.env import BatchEnv, EnvOptions, ItemSampler, SceneBank
from satellite_rl.s2lab.features import FEATURE_NAMES
from satellite_rl.s2lab.rules import make_rule
from satellite_rl.s2lab.scenes import read_config, read_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--scenes', type=int, default=20)
    args = ap.parse_args()
    device = torch.device('cuda')
    config = read_config(args.dataset_dir)
    items = read_splits(args.dataset_dir)['dev'][:args.scenes]
    bank = SceneBank(args.dataset_dir, sorted({it['key'] for it in items}), config.physics_obj(), device)
    env = BatchEnv(bank, config, EnvOptions(batch_size=len(items), group_mode='greedy', reward_mode='delta', legacy_obs=True,
                                            features=FEATURE_NAMES, intf_rel=True),
                   device, ItemSampler(bank, config, items, False, 0), auto_reset=False)
    rule = make_rule('greedy', env)
    rows = {k: [] for k in FEATURE_NAMES}
    while env.active.any():
        obs = env.observe()
        act = env.active
        g = obs['allow'].float().argmax(1)
        for k in FEATURE_NAMES:
            v = obs['f_' + k][act]
            if v.dim() == 2 and v.shape[1] == env.T:             # 每槽量：只看允许组极化上的 100 槽
                rows[k].append(v.reshape(-1).cpu())
            elif v.dim() == 3:
                rows[k].append(v.reshape(v.shape[0], -1).reshape(-1).cpu())
            else:
                rows[k].append(v.cpu())
        env.step(rule.act(obs))
    for k in FEATURE_NAMES:
        x = torch.cat([r.reshape(len(r), -1) if r.dim() > 1 else r[:, None] for r in rows[k]]).numpy()
        flat = x.reshape(-1)
        q = np.percentile(flat, [0, 5, 25, 50, 75, 95, 100])
        print(f"{k:7s} 维度 {x.shape[1]:3d} 分位数 0/5/25/50/75/95/100 = {' '.join(f'{v:.3f}' for v in q)} "
              f"等于0 {np.mean(flat <= 1e-6):.2f} 到上限(≥1) {np.mean(flat >= 1 - 1e-6):.2f} 常数列 {np.mean(x.std(0) < 1e-6):.2f}")
        if x.shape[1] <= 30:
            print('        各列均值', ' '.join(f'{v:.2f}' for v in x.mean(0)))


if __name__ == '__main__':
    main()
