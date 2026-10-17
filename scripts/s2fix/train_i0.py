"""Train L1 on a generated S2-FIX training set."""
import argparse
import torch
from satellite_rl.s2fix.common import run_manifest
from satellite_rl.s2fix.data import read_records
from satellite_rl.s2fix.i0 import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=51001)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    with run_manifest(args.output_dir, args):
        train(read_records(args.data_dir, "train_d864_n150"), args.output_dir,
              seed=args.seed, steps=args.steps, device=args.device)


if __name__ == "__main__":
    main()
