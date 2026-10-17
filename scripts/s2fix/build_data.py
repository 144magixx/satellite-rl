"""Build the six S2-FIX FLOW-AO datasets."""
import argparse
from satellite_rl.s2fix.data import SETS, INPUTS, FRAMES, build_set
from satellite_rl.s2fix.common import run_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sets", nargs="+", choices=SETS, default=list(SETS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--inputs-dir", default=str(INPUTS))
    parser.add_argument("--frames-dir", default=str(FRAMES))
    args = parser.parse_args()
    with run_manifest(args.output_dir, vars(args)):
        for name in args.sets:
            checks = build_set(name, args.output_dir, args.workers, args.limit, args.inputs_dir, args.frames_dir)
            for row in checks: print(name, row, flush=True)


if __name__ == "__main__": main()
