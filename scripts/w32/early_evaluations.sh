#!/usr/bin/env bash
set -euo pipefail
source_dir=$1
python_bin=$2
port=$3
root=/home/u2025140002/second_stage_satellite_rl/s2_w32
export PYTHONPATH="$source_dir/src"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
evaluate() {
    local policy=$1 menu=$2 group=$3 data=$4 output=$5 repeats=$6
    "$python_bin" "$source_dir/scripts/w32/eval_policy.py" --policy "$policy" \
        --menu "$menu" --group "$group" --data-dir "$root/data/$data" \
        --output-dir "$output" --repeats "$repeats" --threads 2 --device cpu
}
if [[ $port == 20655 ]]; then
    old=/home/u2025140002/second_stage_satellite_rl/s2_w31/runs
    evaluate "$old/P00-M0/policy_150000.pt" P00 M0 dev_n100 "$root/evals/transfer_w31/P00-M0" 3
    evaluate "$old/P01-M1/policy_150000.pt" P01 M1 dev_n100 "$root/evals/transfer_w31/P01-M1" 3
else
    if [[ $port == 20627 ]]; then data=dev_n100; output=baselines_dev; else data=conf_n100; output=baselines_conf; fi
    evaluate random P00 M0 "$data" "$root/evals/$output/random" 3
    evaluate fixed P00 M0 "$data" "$root/evals/$output/fixed" 1
    evaluate greedy P00 M0 "$data" "$root/evals/$output/greedy" 1
fi
