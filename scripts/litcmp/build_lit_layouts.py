"""Build each confirmation geometry once and reuse it at all request levels."""
import argparse
import os
import numpy as np
from pathlib import Path
from satellite_rl.s2fix.common import run_manifest
from satellite_rl.s2fix.data import save_record
from satellite_rl.litcmp.layouts import frame_from_case, to_stage2, solve_layout
from satellite_rl.litcmp.data import source_cases, parallel, shard_indices, write_set


def run(job):
    index, case, method, requests, budget, output = job
    result = solve_layout(frame_from_case(case), budget, method, 51001)
    rows = []
    counts = np.bincount(result.layout.owner)
    for request in requests:
        name = f'conf_d{request // 100000}_n{budget}_{method}'
        converted, layout = to_stage2(result, case, request)
        directory = Path(output) / name / 'objects'
        directory.mkdir(parents=True, exist_ok=True)
        save_record(directory / f'{index:04d}_case.json', converted)
        save_record(directory / f'{index:04d}_layout.json', layout)
        record = dict(case_id=converted['case_id'], date=converted['UTC'][:10], set_name=name,
            case_path=f'objects/{index:04d}_case.json', layout_path=f'objects/{index:04d}_layout.json')
        cap = 1125000000 // request
        check = dict(case_id=converted['case_id'], n=len(case['aircraft_ids']), N_used=len(counts), K=cap,
            max_beam_aircraft=int(counts.max()), over_cap_beams=int((counts > cap).sum()),
            mean_width_deg=float(result.layout.widths.mean()), max_width_deg=float(result.layout.widths.max()),
            spacing_violations=result.spacing_violations, seconds=result.t_total, feasible=result.feasible)
        rows.append((name, record, check))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True, help='S1-FIX frames directory')
    p.add_argument('--methods', nargs='+', choices=['A1', 'A2', 'A3'], default=['A1', 'A2', 'A3'])
    p.add_argument('--requests', nargs='+', type=int, default=[86400000, 43200000, 21600000])
    p.add_argument('--N', type=int, default=150)
    p.add_argument('--workers', type=int, default=os.cpu_count())
    p.add_argument('--shard')
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    with run_manifest(args.output_dir, args):
        cases = source_cases('conf', frames_dir=args.data_dir)
        jobs = [(i, case, method, args.requests, args.N, args.output_dir)
                for i, case in shard_indices(cases, args.shard) for method in args.methods]
        outputs = {}
        for batch in parallel(run, jobs, args.workers):
            for name, record, check in batch:
                outputs.setdefault(name, []).append((record, check))
        for name, rows in outputs.items():
            write_set(args.output_dir, name, rows, args.shard)


if __name__ == '__main__':
    main()
