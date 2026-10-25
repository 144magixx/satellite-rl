"""核对"干扰打乱"诊断不是假象：同一状态下，原观测与打乱/置零干扰后的观测各出一次确定性动作，统计三个动作头相同的比例。
  python scripts/s2lab/diag_check.py --dataset-dir D --runs-dir R --run-ids a b ...
"""
import argparse

import torch

from _common import default_device
from satellite_rl.s2lab.agents import load_policy
from satellite_rl.s2lab.diagnostics import DiagnosticPolicy
from satellite_rl.s2lab.env import BatchEnv, ItemSampler, SceneBank
from satellite_rl.s2lab.matrix import read_matrix
from satellite_rl.s2lab.scenes import read_config, read_splits
from satellite_rl.s2lab.train import env_options


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--run-ids', nargs='+')
    args = ap.parse_args()
    device = torch.device(default_device())
    config = read_config(args.dataset_dir)
    items = read_splits(args.dataset_dir)['dev'][:32]
    bank = SceneBank(args.dataset_dir, sorted({it['key'] for it in items}), config.physics_obj(), device)
    matrix = {r['run_id']: r for r in read_matrix(args.runs_dir)}
    for run_id in args.run_ids:
        run = matrix[run_id]
        agent = load_policy(f'{args.runs_dir}/{run_id}/policy_best_sgm.pt', device)
        opts = env_options(dict(run, batch_envs=32))
        env = BatchEnv(bank, config, opts, device, ItemSampler(bank, config, items, False, 0), auto_reset=False)
        diags = {m: DiagnosticPolicy(agent, env, m) for m in ('intf_shuffle', 'intf_zero')}
        same = {m: torch.zeros(3, device=device) for m in diags}
        total = 0
        while env.active.any():
            obs = env.observe()
            a = agent.act(obs, explore=False)
            act = env.active
            for m, d in diags.items():
                b = agent.act(d._edit_obs(obs), explore=False)
                same[m] += (a[act] == b[act]).float().sum(0)
            total += int(act.sum())
            env.step(a)
        print(run_id, total, {m: [round(float(x) / total, 3) for x in v] for m, v in same.items()}, flush=True)


if __name__ == '__main__':
    main()
