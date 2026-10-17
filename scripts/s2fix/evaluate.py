"""Evaluate G1/R/S1/S2/L1/L2/L3, with the same 50 W constrained SCA."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path

import torch

from satellite_rl.s2fix.common import run_manifest
from satellite_rl.s2fix.data import read_records
from satellite_rl.s2fix.experiments import evaluate_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--set', default='dev_d432_n150')
    parser.add_argument('--methods', nargs='+', default=['G1','R','S1','S2'])
    parser.add_argument('--models', nargs='*', default=[], help='L1=/abs/model.pt L2=/abs/model.pt ...')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    with run_manifest(args.output_dir, args):
        directory = Path(args.output_dir)
        (directory/'beams').mkdir(exist_ok=True)
        models = {method: [] for method in ('L1','L2','L3')}
        for item in args.models:
            method, path = item.split('=',1)
            config = torch.load(path, map_location='cpu', weights_only=False)['config']
            seed = config['seed'] if method == 'L1' else config['training']['seed']
            models[method].append((seed, path))
        records = read_records(args.data_dir, args.set)
        jobs = [(i,r,args.methods,models,str(directory),args.device,args.threads)
                for i,r in enumerate(records)]
        def write(results):
            with (directory/'results.csv').open('w',newline='',encoding='utf-8') as stream:
                writer = None
                for rows in results:
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                    writer.writerows(rows)
                    stream.flush()
                    print(rows[0]['scene_id'], len(rows), flush=True)
        if args.workers == 1:
            write(map(evaluate_scene,jobs))
        else:
            with ProcessPoolExecutor(args.workers) as pool:
                write(pool.map(evaluate_scene,jobs))


if __name__ == '__main__':
    main()
