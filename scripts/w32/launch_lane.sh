#!/usr/bin/env bash
set -euo pipefail
source_dir=$1
python_bin=$2
port=$3
mode=${4:-three}
root=/home/u2025140002/second_stage_satellite_rl/s2_w32
export PYTHONPATH="$source_dir/src"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1
threads=3
arms=()
if [[ $mode == two ]]; then
    threads=2
    if [[ $port == 20627 ]]; then menu=P00; else menu=P01; fi
    for seed in 51001 51002 51003; do
        for group in M0 M1; do arms+=("$menu-$group-s$seed"); done
    done
elif [[ $port == 20655 ]]; then
    for menu in P00 P01; do
        for group in M0 M1; do arms+=("$menu-$group-s51003"); done
    done
else
    if [[ $port == 20627 ]]; then menu=P00; else menu=P01; fi
    for seed in 51001 51002; do
        for group in M0 M1; do arms+=("$menu-$group-s$seed"); done
    done
fi
for arm in "${arms[@]}"; do
    IFS=- read -r menu group seed <<< "$arm"
    directory="$root/runs/$arm"
    mkdir -p "$directory"
    nohup setsid "$python_bin" -m satellite_rl.power30.lean_runner --menu "$menu" --group "$group" \
        --seed "${seed#s}" --data-dir "$root/data/train_n100" --loader simple \
        --run-dir "$directory" --threads "$threads" > "$directory/stdout.log" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$directory/pid"
    printf '%s %s %s %s\n' "$arm" "$pid" "$port" "$directory"
done
