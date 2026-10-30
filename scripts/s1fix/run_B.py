"""块B缓存、非学习基线、PPO训练及最终检查点评价。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from satellite_rl.coverage.s1fix.data import sequence_sets
from satellite_rl.coverage.s1fix.dynamic import BASELINES, FullCache, evaluate, summarize
from satellite_rl.coverage.s1fix.io import manifest, write_rows
from satellite_rl.coverage.s1fix.ppo_dyn import load_policy, merge_curves, train


def build_one(arguments):
    directory, frame = arguments
    return str(FullCache(directory).build(frame))


def baseline_one(arguments):
    sequences, directory, role, arm, param = arguments
    rows = evaluate(sequences, FullCache(directory), arm, param)
    for row in rows:
        row["role"] = role
    return rows, summarize(rows, role)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    # 相同公共选项可放在子命令前，也可放在后。
    def common(target, suppress=False):
        default = argparse.SUPPRESS if suppress else None
        target.add_argument("--data-dir", type=Path, default=default)
        target.add_argument("--output-dir", type=Path, default=default)
        target.add_argument("--workers", type=int, default=argparse.SUPPRESS if suppress else 1)
    common(parser)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("cache", "baselines", "train", "eval"):
        sub = commands.add_parser(name)
        common(sub, True)
        if name in ("cache", "baselines", "eval"):
            sub.add_argument("--roles", nargs="+", choices=("train", "dev", "conf"),
                             default=("train", "dev", "conf") if name == "cache" else ("dev", "conf"))
            sub.add_argument("--sequence-limit", type=int)
        if name == "train":
            sub.add_argument("--lam", type=float, default=.2)
            sub.add_argument("--seed", type=int, default=51001)
            sub.add_argument("--no-forecast", action="store_true")
            sub.add_argument("--episodes", type=int, default=1200)
            sub.add_argument("--episodes-per-update", type=int, default=16)
            sub.add_argument("--epochs", type=int, default=4)
            sub.add_argument("--sequence-limit", type=int)
        if name == "eval":
            sub.add_argument("--models", nargs="*", type=Path)
    args = parser.parse_args()
    if args.data_dir is None or args.output_dir is None:
        parser.error("--data-dir和--output-dir为必填绝对路径")
    return args


def main():
    args = arguments()
    out = args.output_dir
    directory = out / "B" / "full_cache"
    with manifest(out, f"B_{args.command}", vars(args)):
        if args.command == "train":
            sequences = sequence_sets(args.data_dir, "train")[:args.sequence_limit]
            path = train(sequences, FullCache(directory), out, args.lam, args.seed,
                         not args.no_forecast, args.episodes, args.episodes_per_update, args.epochs)
            print(path, flush=True)
            return
        groups = {role: sequence_sets(args.data_dir, role)[:args.sequence_limit] for role in args.roles}
        if args.command == "cache":
            unique = {(frame.date, frame.index): frame for sequences in groups.values()
                      for sequence in sequences for frame in sequence}
            jobs = [(directory, frame) for frame in unique.values()]
            with ProcessPoolExecutor(args.workers) as pool:
                for i, _ in enumerate(pool.map(build_one, jobs), 1):
                    if i % 100 == 0:
                        print(f"full cache {i}/{len(jobs)}", flush=True)
            print(f"full cache: {len(jobs)} unique frames", flush=True)
        elif args.command == "baselines":
            jobs = [(seq, directory, role, arm, param) for role, seq in groups.items()
                    for arm, param in BASELINES]
            with ProcessPoolExecutor(args.workers) as pool:
                for rows, summary in pool.map(baseline_one, jobs):
                    write_rows(out / "B" / "eval_frames.csv", rows, append=True)
                    write_rows(out / "B" / "eval_summary.csv", [summary], append=True)
        else:
            merge_curves(out)
            models = args.models or sorted((out / "B" / "models").glob("*_final.pt"))
            for path in models:
                policy, config = load_policy(path)
                arm = "B4" if config["forecast"] else "B4-forecast"
                for role, sequences in groups.items():
                    rows = evaluate(sequences, FullCache(directory), arm, config["lam"], config["seed"],
                                    policy, config["forecast"], config["lam"])
                    for row in rows:
                        row["role"] = role
                    write_rows(out / "B" / "eval_frames.csv", rows, append=True)
                    write_rows(out / "B" / "eval_summary.csv", [summarize(rows, role)], append=True)


if __name__ == "__main__":
    main()
