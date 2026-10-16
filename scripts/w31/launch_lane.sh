#!/usr/bin/env bash
set -euo pipefail
source_dir=$1
python_bin=$2
lane=$3
root=/home/u2025140002/second_stage_satellite_rl/s2_w31
data=/home/u2025140002/second_stage_satellite_rl/s2_w30/shared_data/input596
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
case "$lane" in
    20627) menus="P00 P11" ;;
    20653) menus="P10 P01" ;;
    *) exit 2 ;;
esac
for menu in $menus; do
    for group in M0 M1; do
        run_dir="$root/runs/$menu-$group"
        mkdir -p "$run_dir"
        nohup setsid "$python_bin" -m satellite_rl.power30.lean_runner \
            --menu "$menu" --group "$group" --data-dir "$data" \
            --run-dir "$run_dir" --threads 3 >"$run_dir/worker.log" 2>&1 </dev/null &
        echo "$menu-$group $! $lane $run_dir"
    done
done
