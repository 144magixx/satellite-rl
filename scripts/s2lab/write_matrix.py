"""重写 runs 目录下的 matrix.json 与 queue.json（追加新运行后在远端执行；已有运行的状态不受影响）。
  python scripts/s2lab/write_matrix.py --runs-dir <runs>                         # 第一轮
  python scripts/s2lab/write_matrix.py --runs-dir <runs2> --round 2 --account A  # 第二轮，按账号分队列
  python scripts/s2lab/write_matrix.py --runs-dir <runs3> --round 3 --account A [--extra extra_runs.json]   # 第三轮（extra：组合版、续训）
"""
import argparse
import json
from pathlib import Path

from _common import ROOT  # noqa: F401
from satellite_rl.s2lab.matrix import write_matrix, write_matrix_round2, write_matrix_round3, write_matrix_round4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--round', type=int, default=1)
    ap.add_argument('--account', default='A')
    ap.add_argument('--extra', help='第三轮追加运行（JSON 列表，每项含 account 字段）')
    args = ap.parse_args()
    if args.round == 1:
        runs = write_matrix(args.runs_dir)
    elif args.round == 2:
        runs = write_matrix_round2(args.runs_dir, args.account)
    else:
        extra = json.loads(Path(args.extra).read_text(encoding='utf-8')) if args.extra else []
        runs = write_matrix_round3(args.runs_dir, args.account, extra) if args.round == 3 else write_matrix_round4(args.runs_dir, extra)
    print(len(runs))


if __name__ == '__main__':
    main()
