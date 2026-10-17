"""Static literature layouts on the S1-FIX scene sets."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
from satellite_rl.coverage.s1fix.data import static_sets
from satellite_rl.coverage.s1fix.geom import metrics
from satellite_rl.s2fix.common import run_manifest
from satellite_rl.litcmp.layouts import solve_layout
from satellite_rl.litcmp.data import parallel, shard_indices, suffix, write_csv


def run(job):
    name, frame, budget, method, seed = job
    result = solve_layout(frame, budget, method, seed)
    return dict(scene_set=name, date=frame.date,
        frame_time=datetime.fromtimestamp(frame.time, timezone.utc).isoformat(), n=len(frame.icao),
        N_budget=budget, arm=method, effort=seed, spacing=False, feasible=result.feasible,
        t_init=result.t_init, t_opt=result.t_opt, t_check=result.t_check, t_total=result.t_total,
        **metrics(result.layout), spacing_violations=result.spacing_violations)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--sets', nargs='+', default=['dev24', 'conf48'])
    p.add_argument('--budgets', nargs='+', type=int, default=[60, 100, 150, 200])
    p.add_argument('--methods', nargs='+', choices=['A1', 'A2', 'A3'], default=['A1', 'A2', 'A3'])
    p.add_argument('--seeds', nargs='+', type=int, default=[51001, 51002, 51003])
    p.add_argument('--workers', type=int, default=os.cpu_count())
    p.add_argument('--shard')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output)
    output = output.with_stem(output.stem + suffix(args.shard))
    with run_manifest(output.parent, args):
        sets = static_sets(args.data_dir)
        jobs = [(name.upper(), frame, budget, method, seed) for name in args.sets
                for _, frame in shard_indices(sets[name.upper()], args.shard) for budget in args.budgets
                for method in args.methods for seed in (args.seeds if method == 'A2' else [0])]
        write_csv(output, parallel(run, jobs, args.workers))


if __name__ == '__main__':
    main()
