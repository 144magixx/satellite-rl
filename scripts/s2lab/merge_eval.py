"""把另一个评价输出目录（如 eval_meta）的 CSV 与 summary.json 条目并入主评价目录。
  python scripts/s2lab/merge_eval.py --src <root>/eval_meta --dst <root>/eval --split confirm
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--split', default='confirm')
    args = ap.parse_args()
    src, dst = Path(args.src) / args.split, Path(args.dst) / args.split
    dst.mkdir(parents=True, exist_ok=True)
    moved = 0
    for csv in src.glob('*.csv'):
        shutil.copy2(csv, dst / csv.name)
        moved += 1
    summary_dst = dst / 'summary.json'
    summary = json.loads(summary_dst.read_text(encoding='utf-8')) if summary_dst.exists() else {}
    extra = json.loads((src / 'summary.json').read_text(encoding='utf-8')) if (src / 'summary.json').exists() else {}
    summary.update(extra)
    summary_dst.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'merged {moved} csv, {len(extra)} summary entries -> {dst}')


if __name__ == '__main__':
    main()
