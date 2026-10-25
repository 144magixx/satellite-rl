"""重写 runs 目录下的 matrix.json 与 queue.json（追加新运行后在远端执行；已有运行的状态不受影响）。
  python scripts/s2lab/write_matrix.py --runs-dir <runs>                         # 第一轮
  python scripts/s2lab/write_matrix.py --runs-dir <runs2> --round 2 --account A  # 第二轮，按账号分队列
"""
import argparse

from _common import ROOT  # noqa: F401
from satellite_rl.s2lab.matrix import write_matrix, write_matrix_round2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--round', type=int, default=1)
    ap.add_argument('--account', default='A')
    args = ap.parse_args()
    runs = write_matrix(args.runs_dir) if args.round == 1 else write_matrix_round2(args.runs_dir, args.account)
    print(len(runs))


if __name__ == '__main__':
    main()
