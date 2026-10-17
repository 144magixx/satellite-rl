"""W32 complete-terminal policy or W26 rule evaluation, without learning."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from satellite_rl.power30.agent import Power30Agent, build_config
from satellite_rl.power30.environment import Power30Env
from satellite_rl.power30.lean_runner import EVAL_FIELDS, save_beams
from satellite_rl.power30.runner import isolated_rng, seed_all
from satellite_rl.power30.simple_data import load_input
from satellite_rl.t0_width.runner import BackgroundPolicy


def evaluate(policy, records, menu, group, directory, *, repeats=3, node=0):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    episodes = []
    with (directory / 'eval.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=EVAL_FIELDS + ('traffic_weighted',))
        writer.writeheader()
        with isolated_rng(274000):
            for mother, record in enumerate(records):
                for repeat in range(repeats):
                    action_seed = 274000 + mother + repeat * 1000000
                    seed_all(action_seed)
                    started = time.perf_counter()
                    env = Power30Env(load_input(record), menu, group)
                    if isinstance(policy, BackgroundPolicy):
                        policy.bind_environment(env)
                    state, done = env.reset(), False
                    while not done:
                        action = policy.act(state, deterministic=False, action_mask=env.action_mask())
                        state, _, done, _ = env.step(action)
                    env.export_allocation()
                    metrics, beams = env.metrics(), env.beam_results()
                    demand = sum(b['demand_bps'] for b in beams)
                    served = sum(min(b['capacity_bps'], b['demand_bps']) for b in beams)
                    writer.writerow(dict(mother=mother, case_id=record['case_id'], repeat=repeat,
                        action_seed=action_seed, env_step=node, SI=metrics['SI_mean'], SGM=metrics['SGM'],
                        power_w=metrics['total_power_w'], N=env.N, constraints_ok=True,
                        seconds=time.perf_counter() - started, traffic_weighted=served/demand))
                    stream.flush()
                    episodes.append(dict(episode=mother*repeats+repeat, env_step=node,
                        case_id=record['case_id'], complete=True, beams=beams))
    save_beams(directory / 'beams.npz', episodes)
    (directory / 'DONE').write_text('complete\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True, help='checkpoint or random|fixed|greedy')
    parser.add_argument('--menu', choices=('P00', 'P01'), required=True)
    parser.add_argument('--group', choices=('M0', 'M1'), required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    seed_all(274000)
    node = 0
    if args.policy in ('random', 'fixed', 'greedy'):
        if args.menu != 'P00':
            parser.error('W26 background policies require P00')
        policy = BackgroundPolicy(args.policy, args.device)
    else:
        checkpoint = torch.load(args.policy, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        if (config['menu'], config['group']) != (args.menu, args.group):
            parser.error('checkpoint menu/group differs')
        config['training']['device'] = args.device
        policy = Power30Agent(config)
        for name in ('actor', 'critic1', 'critic2'):
            getattr(policy, name).load_state_dict(checkpoint[name])
        node = checkpoint['env_step']
    records = json.loads((Path(args.data_dir) / 'development_runtime.json').read_text(encoding='utf-8'))['rows']
    evaluate(policy, records, args.menu, args.group, args.output_dir, repeats=args.repeats, node=node)


if __name__ == '__main__':
    main()
