"""重写 runs 目录下的 matrix.json 与 queue.json（追加新运行后在远端执行；已有运行的状态不受影响）。
  python scripts/s2lab/write_matrix.py --runs-dir <runs>
"""
import argparse

from _common import ROOT  # noqa: F401
from satellite_rl.s2lab.matrix import write_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True)
    args = ap.parse_args()
    runs = write_matrix(args.runs_dir)
    print(len(runs))


if __name__ == '__main__':
    main()
