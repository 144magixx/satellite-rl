"""Local code check only: eight arms, resume, real updates and one development mother."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from satellite_rl.power30.agent import build_config
from satellite_rl.power30 import lean_runner
from satellite_rl.power30.environment import Power30Env
from satellite_rl.power30.lean_runner import train, evaluate, read_cases, atomic_torch
from satellite_rl.power30.menus import validate_actual_power
from satellite_rl.power30.runner import evaluate_cases, _default_loader, _tree_equal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    lean_runner.NODES = (317,)  # Exercise final policy/export at the short code-check cap.
    records, dev = read_cases(args.data_dir)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='w31_smoke_') as temp:
        for menu in ('P00', 'P10', 'P01', 'P11'):
            for group in ('M0', 'M1'):
                config = build_config(menu, group)
                config['training']['total_steps'] = 317
                directory = Path(temp) / f'{menu}-{group}'
                agent, result = train(config, records, directory, until_step=300)
                assert result['env_step'] == 300 and result['update_step'] == 0
                # Explicit update calls only for this short code check; keep production warmup=2000.
                before = agent.actor.heads[0].weight.detach().clone()
                for _ in range(3):
                    agent.update()
                assert not torch.equal(before, agent.actor.heads[0].weight)
                saved = torch.load(directory / 'resume.pt', weights_only=False)
                saved['agent'] = agent.state_dict()
                atomic_torch(directory / 'resume.pt', saved)
                restored, result = train(config, records, directory, until_step=300)
                assert _tree_equal(agent.state_dict(), restored.state_dict())
                agent, result = train(config, records, directory, until_step=317)
                assert result['env_step'] == 317 and result['update_step'] == 3
                policy = torch.load(directory / 'policy_317.pt', weights_only=False)
                assert policy['config'] == config and policy['env_step'] == 317
                for name in ('actor', 'critic1', 'critic2'):
                    getattr(agent, name).load_state_dict(policy[name])
                saved = torch.load(directory / 'resume.pt', weights_only=False)
                if saved['environment'] is not None:
                    env = saved['environment']
                    validate_actual_power(menu, env['powers'], active=env['counts'] > 0)
                evaluate(agent, dev, menu, group, directory, 317, repeats=1)
                expected, complete = evaluate_cases(agent, dev, group, repeats=1, node=317,
                    env_factory=lambda value, g: Power30Env(value, menu, g))
                actual = list(csv.DictReader((directory / 'eval_317.csv').open()))[0]
                assert complete
                assert float(actual['SI']) == expected[0]['metrics']['SI_mean']
                assert float(actual['SGM']) == expected[0]['metrics']['SGM']
                assert float(actual['power_w']) == expected[0]['metrics']['total_power_w']
                with np.load(directory / 'eval_317_beams.npz') as arrays:
                    validate_actual_power(menu, arrays['power_w'])
                    np.testing.assert_array_equal(arrays['served_bps'],
                        np.minimum(arrays['capacity_bps'], arrays['demand_bps']))
                print(json.dumps(dict(arm=f'{menu}-{group}', steps=317, updates=3,
                    exact_resume=True, evaluation_matches_W30=True, power_menu_ok=True)), flush=True)
    print('elapsed_seconds', time.perf_counter() - started)


if __name__ == '__main__':
    main()
