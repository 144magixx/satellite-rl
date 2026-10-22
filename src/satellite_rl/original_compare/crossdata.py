"""交叉检验：把 S2-FIX 确认集的欧洲场景（FLOW-AO v2 布局、每机 43.2 Mbps）放进原版环境。

只换数据、不换环境：波束中心和逐束需求来自 S2-FIX，物理全用原版（束宽固定 0.5°、功率不按槽分摊）。
原版把束数和卫星经度写成模块常量，这里按场景改成 N 和 13°E（S2-FIX 的卫星位置）。

    python -m satellite_rl.original_compare.crossdata DATA_DIR OUT_DIR I0_DIR --workers 6
"""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

import satellite_rl.envs.satellite as orig
from . import physics
from .physics import greedy_allocation, metrics, random_allocation, replay
from .search import SearchEnv, tabu_search

EUROPE_SAT_LON = 13.0


def set_scene_size(n, sat_lon):
    for module in (orig, physics):
        module.beam_num = n
        module.satellite_lon = sat_lon


def europe_scene(layout_path):
    layout = json.loads(Path(layout_path).read_text(encoding="utf8"))
    demand = np.asarray(layout["request_bps_by_beam_int64"], float)
    order = np.argsort(-demand, kind="stable")
    location = np.asarray(layout["centres_lat_lon_deg"], float)[order]
    n = len(order)
    set_scene_size(n, EUROPE_SAT_LON)
    sat = np.deg2rad(EUROPE_SAT_LON)
    ground = np.stack([orig.geo_to_cartesian(lat, lon) for lat, lon in location])
    distance = np.linalg.norm(ground - [orig.R * np.cos(sat), orig.R * np.sin(sat), 0.], axis=1)
    info = dict(beam_location=location, beam_G_Tx_max=np.full((n, 1), 50.), UE_G_Rx_max=np.full((n, 1), 40.),
                distances=distance, rate=demand[order][:, None], noise_temperature=np.full((n, 1), 290.))
    return SimpleNamespace(beam_info=info)


def run_scene(args):
    index, layout_path = args
    from . import run
    from satellite_rl.evaluation.compare_metrics import greedy_policy
    beams = europe_scene(layout_path)
    rows = {}
    for name in ("T-SAC", "MLP-SAC"):
        runs = []
        for repeat in range(run.SAC_REPEATS):
            torch.manual_seed(index * 10 + repeat)
            started = time.perf_counter()
            env = run.run_episode(beams, lambda env, state, a=run._MODELS[name]: a.take_action(state))
            runs.append(run.record(beams, env, time.perf_counter() - started))
        rows[name] = runs
    env = run.run_episode(beams, lambda env, state: greedy_policy(env))
    rows["原版旧贪心"] = [run.record(beams, env, 0.)]
    greedy = greedy_allocation(beams)
    rows["容量感知贪心"] = [metrics(replay(beams, *greedy), beams)]
    free8 = greedy_allocation(beams, fixed_groups=True, free_starts=True)
    rows["贪心＋8组＋任意空位"] = [metrics(replay(beams, *free8), beams)]
    env = SearchEnv(beams, greedy, 32)
    rows["贪心+禁忌32"] = [metrics(replay(beams, *tabu_search(env, 32)), beams)]
    env = SearchEnv(beams, random_allocation(beams, index), 32)
    rows["纯禁忌32"] = [metrics(replay(beams, *tabu_search(env, 32)), beams)]
    demand = physics.demand_of(beams)
    return dict(index=index, N=len(demand), total_demand=float(demand.sum()), methods=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("i0_dir", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    import multiprocessing as mp
    from . import run
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf8"))["rows"]
    jobs = [(i, str(args.data_dir / row["layout_path"])) for i, row in enumerate(manifest)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(args.workers, initializer=run.load_models, initargs=(args.i0_dir,)) as pool, \
            (args.out_dir / "europe_in_original.jsonl").open("w", encoding="utf8") as stream:
        for count, row in enumerate(pool.imap_unordered(run_scene, jobs), 1):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{count}/{len(jobs)}] scene {row['index']} N={row['N']}", flush=True)


if __name__ == "__main__":
    main()
