"""在原版环境的固定种子场景上跑全部方法，每景一行写入 results.jsonl。

用法：
    python -m satellite_rl.original_compare.run extract-i0 --out DIR CKPT...
    python -m satellite_rl.original_compare.run run --out DIR --seeds 1000:1100 --workers 24
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from satellite_rl.paths import project_path
from .physics import greedy_allocation, make_scene, metrics, new_env, random_allocation, replay
from .search import SearchEnv, i0_search, load_i0_actor, tabu_search

SAC_REPEATS = 3          # 直出模型按策略概率抽样，每景跑 3 次取平均（与 S2-FIX 相同）
# 主实验用默认值；环境/几何变体的快速检验用环境变量缩减（ORIG_TABU="32"、ORIG_I0_SAMPLES="1"）
I0_SAMPLE_SEEDS = tuple(range(1, 1 + int(os.environ.get("ORIG_I0_SAMPLES", "3"))))
TABU_BUDGETS = tuple(int(x) for x in os.environ.get("ORIG_TABU", "32,96,320").split(","))
WITH_FREE8 = os.environ.get("ORIG_FREE8", "0") == "1"   # 主实验之后加入的"8组＋组内按干扰挑空位"规则
_MODELS = {}


def extract_i0(checkpoints, out):
    """S2-FIX 的 L1 断点含 70 万条回放，只抽出策略网络权重。"""
    out.mkdir(parents=True, exist_ok=True)
    for path in map(Path, checkpoints):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        torch.save(saved["agent"]["actor"], out / f"{path.parent.name}_actor.pt")
        print(path.parent.name, saved["step"], saved["episodes"])


def load_models(i0_dir):
    torch.set_num_threads(1)
    from satellite_rl.agents.sac_transformer import SAC as SACTransformer
    from satellite_rl.agents.sac_mlp import SAC as SACMLP
    from satellite_rl.evaluation.checkpoints import load_actor_only
    kw = dict(n_states=204, n_hiddens=128, freq_dim=100, slots_dim=10, power_dim=10,
              actor_lr=1e-5, critic_lr=1e-4, alpha_lr=1e-4, target_entropy=-6.9,
              tau=0.005, gamma=0.99, device="cpu")
    _MODELS["T-SAC"] = SACTransformer(**kw)
    _MODELS["MLP-SAC"] = SACMLP(**kw)
    load_actor_only(_MODELS["T-SAC"], project_path("saved_models/sac-t"))
    load_actor_only(_MODELS["MLP-SAC"], project_path("saved_models/sac-fc"))
    _MODELS["I0"] = {p.stem.split("_")[0]: load_i0_actor(p) for p in sorted(Path(i0_dir).glob("*_actor.pt"))}


def run_episode(beams, policy):
    from satellite_rl.evaluation.compare_metrics import to_env_action
    env, state = new_env(beams)
    done = False
    while not done:
        action = policy(env, state)
        if len(action) == 3:
            action = to_env_action(env, action)
        state, _, done, _ = env.step(action)
    return env


def record(beams, env, seconds, **extra):
    return dict(metrics(env, beams), seconds=seconds, **extra)


def search_record(beams, env, allocation, init_seconds, started):
    total = time.perf_counter() - started
    return record(beams, replay(beams, *allocation), total, init_seconds=init_seconds,
                  search_seconds=total - init_seconds, probes=env.probe_count,
                  probe_seconds=env.probe_seconds, base_W=env.base_eval.W)


def run_scene(seed):
    from satellite_rl.evaluation.compare_metrics import greedy_policy
    beams = make_scene(seed)
    rows = {}

    for name in ("T-SAC", "MLP-SAC"):
        agent, runs = _MODELS[name], []
        for repeat in range(SAC_REPEATS):
            torch.manual_seed(seed * 10 + repeat)
            started = time.perf_counter()
            env = run_episode(beams, lambda env, state: agent.take_action(state))
            runs.append(record(beams, env, time.perf_counter() - started, repeat=repeat))
        rows[name] = runs

    started = time.perf_counter()
    env = run_episode(beams, lambda env, state: greedy_policy(env))
    rows["原版旧贪心"] = [record(beams, env, time.perf_counter() - started)]

    started = time.perf_counter()
    greedy = greedy_allocation(beams)
    greedy_seconds = time.perf_counter() - started
    rows["容量感知贪心"] = [record(beams, replay(beams, *greedy), greedy_seconds, init_seconds=greedy_seconds)]

    started = time.perf_counter()
    initial = random_allocation(beams, seed)
    init_seconds = time.perf_counter() - started
    env = SearchEnv(beams, initial, 32)
    rows["纯禁忌32"] = [search_record(beams, env, tabu_search(env, 32), init_seconds, started)]

    for budget in TABU_BUDGETS:
        started = time.perf_counter() - greedy_seconds   # 计入同一次贪心构造耗时
        env = SearchEnv(beams, greedy, budget)
        rows[f"贪心+禁忌{budget}"] = [search_record(beams, env, tabu_search(env, budget), greedy_seconds, started)]

    runs = []
    for model_seed, actor in _MODELS["I0"].items():
        for sample_seed in I0_SAMPLE_SEEDS:
            started = time.perf_counter() - greedy_seconds
            env = SearchEnv(beams, greedy, 32)
            runs.append(dict(search_record(beams, env, i0_search(env, actor, sample_seed), greedy_seconds, started),
                             model=model_seed, sample_seed=sample_seed, decisions=env.decision_count))
    rows["改进式SAC(I0)"] = runs

    if WITH_FREE8:
        started = time.perf_counter()
        free8 = greedy_allocation(beams, fixed_groups=True, free_starts=True)
        free8_seconds = time.perf_counter() - started
        rows["8组任意空位"] = [record(beams, replay(beams, *free8), free8_seconds, init_seconds=free8_seconds)]
        runs = []
        for model_seed, actor in _MODELS["I0"].items():
            for sample_seed in I0_SAMPLE_SEEDS:
                started = time.perf_counter() - free8_seconds
                env = SearchEnv(beams, free8, 32)
                runs.append(dict(search_record(beams, env, i0_search(env, actor, sample_seed), free8_seconds, started),
                                 model=model_seed, sample_seed=sample_seed, decisions=env.decision_count))
        rows["8组任意空位+I0"] = runs
    return dict(seed=seed, methods=rows)


def run(out, seeds, workers, i0_dir):
    import multiprocessing as mp
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.jsonl"
    done = {json.loads(line)["seed"] for line in path.open(encoding="utf8")} if path.exists() else set()
    todo = [s for s in seeds if s not in done]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    started = time.perf_counter()
    with mp.get_context("spawn").Pool(workers, initializer=load_models, initargs=(i0_dir,)) as pool, \
            path.open("a", encoding="utf8") as stream:
        for count, row in enumerate(pool.imap_unordered(run_scene, todo), 1):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{count}/{len(todo)}] seed {row['seed']} "
                  f"{time.perf_counter() - started:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract-i0")
    extract.add_argument("--out", type=Path, required=True)
    extract.add_argument("checkpoints", nargs="+")
    go = sub.add_parser("run")
    go.add_argument("--out", type=Path, required=True)
    go.add_argument("--seeds", required=True, help="start:stop")
    go.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    go.add_argument("--i0-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract-i0":
        extract_i0(args.checkpoints, args.out)
    else:
        start, stop = map(int, args.seeds.split(":"))
        run(args.out, range(start, stop), args.workers, args.i0_dir)


if __name__ == "__main__":
    main()
