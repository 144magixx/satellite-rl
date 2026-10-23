"""Train one I0 network variant on the S2-FIX training set (light checkpoints)."""
import argparse

import torch

from satellite_rl.s2fix.common import run_manifest
from satellite_rl.s2fix.data import read_records
from satellite_rl.s2fix.i0net import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=51001)
    parser.add_argument("--steps", type=int, default=80000)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--network", default="mlp", choices=("mlp", "deepsets", "transformer"))
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--pointer", action="store_true")
    parser.add_argument("--start", default="greedy", choices=("greedy", "spread"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    with run_manifest(args.output_dir, args):
        train(read_records(args.data_dir, "train_d432_n150"), args.output_dir, args.seed, args.steps, args.device,
              network=args.network, hidden_width=args.hidden_width, embed_dim=args.embed_dim,
              num_layers=args.num_layers, num_heads=args.num_heads, ffn_mult=args.ffn_mult,
              pointer=args.pointer, start=args.start)


if __name__ == "__main__":
    main()
