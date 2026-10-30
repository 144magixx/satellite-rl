"""抽取五个日包；不展开中间文件。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from satellite_rl.coverage.s1fix.data import DATES, OUT, extract_day, raw_path, verify_history
from satellite_rl.coverage.s1fix.io import manifest, write_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", default="D:/paper/outputs")
    p.add_argument("--output-dir", default=str(OUT))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--dates", nargs="+", default=list(DATES))
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    from pathlib import Path
    data = Path(args.output_dir) / "data"
    with manifest(args.output_dir, "extract_frames", vars(args)):
        if not args.verify_only:
            with ProcessPoolExecutor(args.workers) as pool:
                jobs = [pool.submit(extract_day, date, raw_path(date, args.raw_root), data) for date in args.dates]
                for job in jobs:
                    job.result()
        rows = verify_history(data)
        write_rows(data / "history_counts.csv", rows)
        print(f"历史场景匹配：{sum(row['match'] for row in rows)}/{len(rows)}", flush=True)
        if not all(row["match"] for row in rows):
            raise ValueError("历史飞机数不一致，详见data/history_counts.csv")


if __name__ == "__main__":
    main()
