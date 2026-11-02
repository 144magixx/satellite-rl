"""Build FLOW-AO datasets using the corrected source-object lookup."""
import argparse
import os
from satellite_rl.s2fix.common import run_manifest
from satellite_rl.litcmp.data import SETS, build_set, worker_init


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sets', nargs='+', choices=SETS, required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--workers', type=int, default=os.cpu_count())
    p.add_argument('--shard')
    p.add_argument('--limit', type=int, help='Only for local smoke checks')
    args = p.parse_args()
    worker_init()
    with run_manifest(args.output_dir, args):
        for name in args.sets:
            checks = build_set(name, args.output_dir, args.workers, args.limit, args.shard)
            print(name, len(checks), 'scenes', flush=True)


if __name__ == '__main__':
    main()
