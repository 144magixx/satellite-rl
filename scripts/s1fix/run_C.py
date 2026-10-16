"""块C：同实际束数的布局对照，固定贪心分配与SCA。"""
import os
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from satellite_rl.coverage.s1fix.data import OUT, static_sets
from satellite_rl.coverage.s1fix.downstream import evaluate_layout
from satellite_rl.coverage.s1fix.geom import metrics
from satellite_rl.coverage.s1fix.io import load_layout, manifest, save_layout, write_rows
from satellite_rl.coverage.s1fix.skeleton import init_baseline, solve


def comparison_job(job):
    scene_set, frame, budget, output, sub, max_iters, time_limit, sca_options = job
    output = Path(output)
    stem = f"{scene_set}_{frame.date}_{frame.index:03d}_N{budget}"
    if sub == "effort":
        # Reuse the exact A3 endpoint. Its actual N may differ from the three
        # common-N variants and is explicitly reported as a separate layout.
        layout = load_layout(output / "A/layouts" / f"{stem}_FLOW-AO_e10.npz")
        row = dict(scene_set=scene_set, date=frame.date,
                   frame_time=datetime.fromtimestamp(frame.time, timezone.utc).isoformat(),
                   n=len(frame.icao), N_budget=budget, comparison_N="",
                   variant="FLOW-AO-e10", layout_source="A3", feasible=True)
        row.update(metrics(layout))
        row.update(evaluate_layout(layout, sca_options,
                                   output / "C/service_arrays" / f"{stem}_FLOW-AO-e10.npz"))
        return [row]
    initial_path = output / "A/layouts" / f"{stem}_INIT_e1.npz"
    if initial_path.exists():
        initial, initial_source = load_layout(initial_path), "A-INIT"
    else:
        result = init_baseline(frame, budget, time_limit=time_limit)
        if not result.feasible:
            return [dict(scene_set=scene_set, date=frame.date, frame_time=frame.time,
                         n=len(frame.icao), N_budget=budget, variant="INIT",
                         feasible=False, N0=result.N0)]
        initial, initial_source = result.layout, "C-INIT"
    # The same feasible INIT is the starting point for every comparison.
    # preserve_N rejects merges/empty-beam proposals that change its actual K.
    common_n = len(initial.widths)
    variants = [("INIT", False, 1), ("FLOW-AO-peak", False, 1), ("FLOW-AO", True, 1)]
    rows = []
    for variant, balance, effort in variants:
        source = initial_source if variant == "INIT" else "C-common-N"
        layout = initial if variant == "INIT" else None
        if layout is None:
            result = solve(frame, common_n, spacing=True, balance=balance, effort=effort,
                           init_layout=initial, preserve_N=True, max_iters=max_iters,
                           time_limit=time_limit)
            layout = result.layout
        row = dict(scene_set=scene_set, date=frame.date,
                   frame_time=datetime.fromtimestamp(frame.time, timezone.utc).isoformat(),
                   n=len(frame.icao), N_budget=budget, comparison_N=common_n,
                   variant=variant, layout_source=source, feasible=layout is not None)
        if layout is not None:
            row.update(metrics(layout))
            row.update(evaluate_layout(layout, sca_options,
                                       output / "C/service_arrays" / f"{stem}_{variant}.npz"))
            save_layout(output / "C/layouts" / f"{stem}_{variant}.npz", layout, frame,
                        scene_set=scene_set, frame_time=frame.time,
                        N_budget=budget, comparison_N=common_n, variant=variant)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(OUT / "data"))
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--subexperiment", choices=("dev", "confirm", "effort", "all"), default="dev")
    parser.add_argument("--budgets", type=int, nargs="+", default=[60, 100, 150, 200])
    parser.add_argument("--limit-scenes", type=int, default=None, help="小型冒烟场景数")
    parser.add_argument("--max-iters", type=int, default=None, help="小型冒烟优化轮数")
    parser.add_argument("--layout-seconds", type=float, default=None)
    parser.add_argument("--sca-seconds", type=float, default=30.)
    parser.add_argument("--sca-maxiter", type=int, default=200)
    args = parser.parse_args()
    sets = static_sets(args.data_dir)
    requested = ["dev", "confirm", "effort"] if args.subexperiment == "all" else [args.subexperiment]
    sca_options = dict(maxiter=args.sca_maxiter, time_limit_seconds=args.sca_seconds)
    with manifest(args.output_dir, "C", vars(args)):
        for sub in requested:
            name = "CONF48" if sub == "confirm" else "DEV24"
            frames = [frame for frame in sets[name] if sub != "effort" or len(frame.icao) >= 1200]
            frames = frames[:args.limit_scenes]
            budgets = [200] if sub == "effort" else args.budgets
            jobs = [(name, frame, budget, args.output_dir, sub, args.max_iters,
                     args.layout_seconds, sca_options) for frame in frames for budget in budgets]
            with ProcessPoolExecutor(args.workers) as pool:
                rows = [row for batch in pool.map(comparison_job, jobs) for row in batch]
            write_rows(Path(args.output_dir) / "C/service.csv", rows, append=True)
            print(f"C/{sub}: {len(rows)}个布局完成", flush=True)


if __name__ == "__main__":
    main()
