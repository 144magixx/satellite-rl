"""Train L2 (Transformer) or L3 (MLP)."""
import argparse
import torch
from satellite_rl.s2fix.common import run_manifest
from satellite_rl.s2fix.data import read_records
from satellite_rl.s2fix.tsac import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=51001)
    parser.add_argument('--encoder', choices=('transformer', 'mlp'), default='transformer')
    parser.add_argument('--steps', type=int, default=150000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    with run_manifest(args.output_dir, args):
        train(read_records(args.data_dir, 'train_d432_n150'), args.output_dir,
              seed=args.seed, encoder=args.encoder, steps=args.steps, device=args.device)


if __name__ == '__main__':
    main()
