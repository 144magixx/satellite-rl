"""Requested 300-step P01/M1 local check, one resume and one development mother."""
import argparse
import json
from pathlib import Path
import tempfile

import torch

from satellite_rl.power30 import lean_runner
from satellite_rl.power30.agent import build_config
from satellite_rl.power30.runner import _tree_equal
from satellite_rl.power30.simple_data import load_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    records, dev = lean_runner.read_cases(args.data_dir)
    config = build_config('P01', 'M1', seed=51002)
    assert config['training']['seed'] == config['training']['scene_order_seed'] == 51002
    config['training']['total_steps'] = 300
    lean_runner.NODES = (300,)
    with tempfile.TemporaryDirectory(prefix='w32_smoke_') as temp:
        agent, result = lean_runner.train(config, records, temp, loader=load_input, until_step=150)
        assert result['env_step'] == 150
        restored, result = lean_runner.train(config, records, temp, loader=load_input, until_step=150)
        assert _tree_equal(agent.state_dict(), restored.state_dict())
        agent, result = lean_runner.train(config, records, temp, loader=load_input)
        assert result['env_step'] == 300 and result['update_step'] == 0
        assert (Path(temp) / 'policy_300.pt').exists()
        lean_runner.evaluate(agent, dev[:1], 'P01', 'M1', Path(temp), 300, loader=load_input, repeats=1)
        print(json.dumps(dict(steps=300, resume_exact=True, development_mothers=1,
                             updates=0, note='production warmup unchanged at 2000')))


if __name__ == '__main__':
    main()
