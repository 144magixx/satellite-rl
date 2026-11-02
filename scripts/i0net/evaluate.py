"""Evaluate I0 network variants with the S2-FIX protocol: every scene, 3 sampled runs per model,
action seed 274000 + scene index + repeat * 1e6; metrics before (and optionally after) the same 50 W SCA.

    python scripts/i0net/evaluate.py --data-dir D --set dev_d432_n150 --models tag=/path/final.pt ... \
        --output out.csv --workers 14 [--sca] [--start spread]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path
import time

import numpy as np
import torch

from satellite_rl.s2fix.data import read_records, load_input
from satellite_rl.s2fix.evaluate import metrics, run_sca
from satellite_rl.s2fix.i0net import load_actor, run_policy

_ACTORS = {}


def scene_rows(job):
    index, record, models, sca, start, device = job
    torch.set_num_threads(1)
    stage_input = load_input(record)
    rows = []
    for tag, path in models:
        if path not in _ACTORS:
            _ACTORS[path] = load_actor(path, device)
        actor, config = _ACTORS[path]
        for repeat in range(3):
            started = time.perf_counter()
            allocation, probes = run_policy(stage_input, actor, config, 274000 + index + repeat * 1000000, start)
            seconds = time.perf_counter() - started
            before = metrics(stage_input, *allocation)
            row = dict(tag=tag, seed=config["seed"], scene_index=index, scene_id=record["case_id"], repeat=repeat,
                       start=start or config["start"], probes=probes, search_seconds=seconds,
                       W=before["W"], SGM=before["SGM"], aircraft_worst10=before["aircraft_worst10"])
            if sca:
                groups, starts, counts, _ = allocation
                after = metrics(stage_input, groups, starts, counts,
                                np.asarray(run_sca(stage_input, *allocation)["power_w"], float))
                row.update(after_W=after["W"], after_SGM=after["SGM"], after_aircraft_worst10=after["aircraft_worst10"])
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--set", default="dev_d432_n150")
    parser.add_argument("--models", nargs="+", required=True, help="tag=/abs/final.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--sca", action="store_true")
    parser.add_argument("--start", choices=("greedy", "spread"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True   # 与训练一致
    torch.backends.cudnn.allow_tf32 = True
    models = [tuple(item.split("=", 1)) for item in args.models]
    jobs = [(i, r, models, args.sca, args.start, args.device)
            for i, r in enumerate(read_records(args.data_dir, args.set))]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool, open(args.output, "w", newline="", encoding="utf8") as stream:
        writer = None
        for rows in pool.map(scene_rows, jobs):
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            print(rows[0]["scene_id"], len(rows), flush=True)


if __name__ == "__main__":
    main()
