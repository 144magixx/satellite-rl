"""块A：静态对照、历史对照、下界和搜索强度。"""
import os
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import csv
import numpy as np

from satellite_rl.coverage.s1fix.data import OUT, HISTORY, static_sets
from satellite_rl.coverage.s1fix.geom import metrics
from satellite_rl.coverage.s1fix.skeleton import solve, init_baseline
from satellite_rl.coverage.s1fix.io import manifest, save_layout, write_rows


def csv_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def history_rows(history):
    return (csv_rows(history / "round10/science_independent_statistics_01/endpoint_metrics.csv"),
            csv_rows(history / "round8/analysis/actual_v4_01/analysis20x3.csv"),
            csv_rows(history / "round8/analysis/mechanism_01/mechanism_rows.csv"))


TIME_LIMIT = 120.  # 每景软限时（秒），effort>1时按倍数放宽


def static_job(job):
    scene_set, frame, budget, arm, effort, output, max_iters = job
    spacing = arm not in ("FLOW-AO-no-spacing", "HIST-cap", "HIST-actual")
    result = (init_baseline if arm == "INIT" else solve)(
        frame, budget, spacing=spacing, effort=effort,
        fixed_width=.5 if arm == "FLOW-AO-fixed" else None,
        max_iters=max_iters, time_limit=TIME_LIMIT * effort)
    row = dict(scene_set=scene_set, date=frame.date,
               frame_time=datetime.fromtimestamp(frame.time, timezone.utc).isoformat(),
               n=len(frame.icao), N_budget=budget, arm=arm, effort=effort,
               spacing=spacing, feasible=result.feasible, N0=result.N0,
               t_init=result.t_init, t_opt=result.t_opt, t_check=result.t_check, t_total=result.t_total)
    if result.feasible:
        row.update(metrics(result.layout))
        name = f"{scene_set}_{frame.date}_{frame.index:03d}_N{budget}_{arm}_e{effort}.npz"
        save_layout(Path(output) / "A/layouts" / name, result.layout, frame,
                    scene_set=scene_set, date=frame.date, frame_time=frame.time,
                    frame_index=frame.index, N_budget=budget, arm=arm, effort=effort)
    return row


def compare_history(results, sets, history):
    hist12, hist20, _ = history_rows(history)
    rows = []
    for name in ("HIST12", "HIST20"):
        for case, frame in enumerate(sets[name]):
            row = dict(scene_set=name, case_index=case, date=frame.date, frame_time=frame.time, n=len(frame.icao))
            historical = ([x for x in hist12 if int(x["original_index"]) == frame.index and x["method"] in ("SH-C", "S2", "B")]
                          if name == "HIST12" else [x for x in hist20 if int(x["case_index"]) == case])
            for old in historical:
                prefix = old["method"] if name == "HIST12" else old["arm"]
                values = (dict(peak=float(old["maximum_count"]), top10=float(old["top10_Mbps"]) / 21.6,
                               CV=float(old["CV"]), N=int(old["N"]), Omega_sr=float(old["Omega_sr"]))
                          if name == "HIST12" else dict(peak=float(old["maxload"]) / 21600000,
                               top10=float(old["top10_mean"]) / 21600000, CV=float(old["CV"]),
                               N=int(old["N"]), Omega_sr=float(old["actual_Omega"])))
                row.update({f"{prefix}_{key}": value for key, value in values.items()})
            for new in results:
                stamp = datetime.fromisoformat(new["frame_time"]).timestamp()
                if new["scene_set"] == name and stamp == frame.time:
                    prefix = "FLOW_cap" if new["arm"] == "HIST-cap" else "FLOW_actual"
                    row.update({f"{prefix}_{key}": new.get(key) for key in ("N_budget", "peak", "top10", "CV", "N_used", "Omega_sr", "feasible")})
            rows.append(row)
    return rows


def bound_job(job):
    from satellite_rl.coverage.s1fix.bound import compute_bound
    name, frame, budget, upper, g, milp_seconds = job
    result = compute_bound(frame, budget, upper, g=g, milp_seconds=milp_seconds)
    return dict(scene_set=name, date=frame.date, frame_time=frame.time, n=len(frame.icao), N_budget=budget, **result)


def parallel(function, jobs, workers):
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(function, jobs))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=str(OUT / "data"))
    p.add_argument("--output-dir", default=str(OUT))
    p.add_argument("--history-dir", default=str(HISTORY))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--subexperiment", choices=["dev", "history", "bounds", "effort", "confirm", "all"], default="dev")
    p.add_argument("--budgets", type=int, nargs="+", default=[60, 100, 150, 200])
    p.add_argument("--limit-scenes", type=int, default=None, help="只用于小型冒烟")
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--milp-seconds", type=float, default=120.)
    args = p.parse_args()
    sets = static_sets(args.data_dir)
    sets = {key: value[:args.limit_scenes] for key, value in sets.items()}
    output = Path(args.output_dir)
    requested = ["dev", "history", "effort", "confirm", "bounds"] if args.subexperiment == "all" else [args.subexperiment]
    with manifest(output, "A", vars(args)):
        for sub in requested:
            jobs = []
            if sub in ("dev", "confirm"):
                name = "DEV24" if sub == "dev" else "CONF48"
                arms = ["FLOW-AO", "INIT"] if sub == "confirm" else ["FLOW-AO", "FLOW-AO-no-spacing", "FLOW-AO-fixed", "INIT"]
                jobs = [(name, frame, budget, arm, 1, output, args.max_iters)
                        for frame in sets[name] for budget in args.budgets for arm in arms]
            elif sub == "effort":
                jobs = [("DEV24", frame, 100, "FLOW-AO", effort, output, args.max_iters)
                        for frame in sets["DEV24"] if len(frame.icao) >= 1200 for effort in (1, 3, 10)]
            elif sub == "history":
                h12, h20, mechanism = history_rows(Path(args.history_dir))
                for frame in sets["HIST12"]:
                    budget = int(next(x["N"] for x in h12 if int(x["original_index"]) == frame.index and x["method"] == "SH-C"))
                    jobs.append(("HIST12", frame, budget, "HIST-actual", 1, output, args.max_iters))
                for case, frame in enumerate(sets["HIST20"]):
                    cap = int(next(x["Ncap"] for x in mechanism if int(x["case_index"]) == case))
                    actual = int(next(x["N"] for x in h20 if int(x["case_index"]) == case and x["arm"] == "G"))
                    jobs.extend(("HIST20", frame, b, arm, 1, output, args.max_iters)
                                for b, arm in ((cap, "HIST-cap"), (actual, "HIST-actual")))
            if sub != "bounds":
                results = parallel(static_job, jobs, args.workers)
                write_rows(output / "A/static_results.csv", results, append=True)
                if sub == "history":
                    write_rows(output / "A/hist_compare.csv", compare_history(results, sets, Path(args.history_dir)))
                print(f"A/{sub}: {len(results)}个场景臂完成", flush=True)
                continue
            previous = csv_rows(output / "A/static_results.csv")
            for name in ("DEV24", "CONF48"):
                for frame in sets[name]:
                    for budget in (100, 200):
                        peaks = [int(float(row["peak"])) for row in previous
                                 if row["scene_set"] == name and row["date"] == frame.date
                                 and datetime.fromisoformat(row["frame_time"]).timestamp() == frame.time
                                 and int(row["N_budget"]) == budget and row.get("peak")
                                 and row["spacing"] == "True"]
                        jobs.append((name, frame, budget, min(peaks, default=len(frame.icao)), .2, args.milp_seconds))
            # 仅三景n<600的DEV24场景，N=200复算g=.1敏感性。
            selected = [frame for frame in sets["DEV24"] if len(frame.icao) < 600][:3]
            jobs += [(name, frame, budget, upper, .1, seconds)
                     for name, frame, budget, upper, g, seconds in jobs.copy()
                     if name == "DEV24" and budget == 200 and any(frame is x for x in selected)]
            results = parallel(bound_job, jobs, args.workers)
            write_rows(output / "A/bounds.csv", results, append=True)
            print(f"A/bounds: {len(results)}项完成", flush=True)


if __name__ == "__main__":
    main()
