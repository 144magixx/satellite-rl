"""Evaluate one directory of layouts, with scene-based sharding."""
import argparse
import os
from pathlib import Path
from satellite_rl.s2fix.common import run_manifest
from satellite_rl.s2fix.experiments import evaluate_scene as reference_scene
from satellite_rl.litcmp.pipeline import METHODS, evaluate_scene
from satellite_rl.litcmp.objective import POINTS
from satellite_rl.litcmp.data import records, shard_indices, suffix, parallel, write_csv


def run(job):
    if job['method'] in METHODS:
        return [evaluate_scene(job)]
    record = records(job['set_dir'])[job['index']]
    rows = reference_scene((job['index'], record, [job['method']], {}, job['output_dir'], 'cpu', 1))
    for row in rows:
        row.update(t_init=row['initial_seconds'], t_search=row['search_seconds'],
                   t_refine=row['sca_seconds'], t_total=row['total_seconds'], best_at='')
        row.update({f'w_at_{point}': '' for point in POINTS})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--set-dir', required=True)
    p.add_argument('--methods', nargs='+', choices=(*METHODS, 'G1', 'R', 'S1', 'S2'), required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[51001, 51002, 51003])
    p.add_argument('--workers', type=int, default=os.cpu_count())
    p.add_argument('--shard')
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    with run_manifest(args.output_dir, args):
        (Path(args.output_dir) / 'beams').mkdir(exist_ok=True)
        jobs = [dict(set_dir=args.set_dir, index=i, method=method, seed=seed, output_dir=args.output_dir)
                for i, _ in shard_indices(records(args.set_dir), args.shard) for method in args.methods
                for seed in (args.seeds if method.startswith(('B1', 'B2', 'E1')) or method == 'C2' else [0])]
        rows = [row for batch in parallel(run, jobs, args.workers) for row in batch]
        write_csv(Path(args.output_dir) / f'results{suffix(args.shard)}.csv', rows)
        print(len(rows), 'rows', flush=True)


if __name__ == '__main__':
    main()
