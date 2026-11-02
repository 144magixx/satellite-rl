"""Train L2 (Transformer) or L3 (MLP); network size options default to L2.

--grouping：greedy（L2，贪心定组）、order8（报告21的 X2，组号=处理顺序%8）、choose（X4，策略自己选组）。"""
import argparse
import torch
from satellite_rl.s2fix.common import run_manifest
from satellite_rl.s2fix.data import read_records
from satellite_rl.s2fix.tsac import train, GROUPINGS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=51001)
    parser.add_argument('--encoder', choices=('transformer', 'mlp'), default='transformer')
    parser.add_argument('--steps', type=int, default=150000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--embed-dim', type=int)
    parser.add_argument('--num-heads', type=int)
    parser.add_argument('--num-layers', type=int)
    parser.add_argument('--ff-dim', type=int)
    parser.add_argument('--save-every', type=int, default=10000)
    parser.add_argument('--grouping', choices=GROUPINGS, default='greedy')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    network = {key: value for key, value in (('embedding_dim', args.embed_dim), ('num_heads', args.num_heads),
                                             ('num_layers', args.num_layers), ('feedforward_dim', args.ff_dim))
               if value is not None}
    with run_manifest(args.output_dir, args):
        train(read_records(args.data_dir, 'train_d432_n150'), args.output_dir,
              seed=args.seed, encoder=args.encoder, steps=args.steps, device=args.device,
              network=network, save_every=args.save_every, grouping=args.grouping)


if __name__ == '__main__':
    main()
