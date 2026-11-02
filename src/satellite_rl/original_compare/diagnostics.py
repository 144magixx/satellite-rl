"""诊断：原版里直出T-SAC为什么能赢。

A 组（不需模型）：容量感知贪心换槽数、分组、功率规则，看能追回多少；
B 组：重跑直出T-SAC，取它下达的整套动作，只把功率换成贪心的按需求比例分配，反过来再看。

    python -m satellite_rl.original_compare.diagnostics RUN_DIR I0_DIR
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .physics import greedy_allocation, make_scene, metrics, proportional_power, demand_of, replay

VARIANTS = {
    "容量感知贪心": dict(),
    "贪心＋补满本组槽": dict(fill_groups=True),
    "贪心＋固定分组": dict(fixed_groups=True),
    "贪心＋固定分组＋补满槽": dict(fixed_groups=True, fill_groups=True),
    "贪心＋固定分组＋任意空位": dict(fixed_groups=True, free_starts=True),
}


def tsac_actions(beams, seed):
    from . import run
    from satellite_rl.evaluation.compare_metrics import to_env_action
    torch.manual_seed(seed * 10)
    issued = []

    def policy(env, state):
        action = to_env_action(env, run._MODELS["T-SAC"].take_action(state))
        issued.append(list(action))
        return action
    env = run.run_episode(beams, policy)
    groups, starts, slots, power_db = map(np.array, zip(*issued))
    return env, groups, starts, slots, 10 ** (power_db / 10)


def main(run_dir, i0_dir):
    from . import run
    run.load_models(i0_dir)
    run_dir = Path(run_dir)
    seeds = sorted(json.loads(line)["seed"] for line in (run_dir / "results.jsonl").open(encoding="utf8"))
    rows = []
    for seed in seeds:
        beams = make_scene(seed)
        demand = demand_of(beams)
        row = dict(seed=seed)
        for name, options in VARIANTS.items():
            allocation = greedy_allocation(beams, **options)
            row[name] = dict(metrics(replay(beams, *allocation), beams), groups_used=int(len(set(allocation[0]))))
        g = greedy_allocation(beams)
        row["贪心＋每束20W"] = metrics(replay(beams, *g[:3], np.full(len(demand), 20.)), beams)
        env, groups, starts, slots, power = tsac_actions(beams, seed)
        row["直出T-SAC"] = dict(metrics(env, beams), power_levels=np.unique(np.round(power)).tolist())
        row["T-SAC频率方案＋按需求比例功率"] = metrics(replay(beams, groups, starts, slots, proportional_power(demand)), beams)
        rows.append(row)
    with (run_dir / "diagnostics.jsonl").open("w", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    for name in [k for k in rows[0] if k != "seed"]:
        values = {k: np.mean([r[name][k] for r in rows]) for k in ("W", "worst10", "SGM", "slots", "power_w")}
        print(f"{name:20s} " + " ".join(f"{k}={v:.3f}" for k, v in values.items()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
