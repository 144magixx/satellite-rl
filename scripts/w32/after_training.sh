#!/usr/bin/env bash
set -euo pipefail
source_dir=$1
python_bin=$2
root=/home/u2025140002/second_stage_satellite_rl/s2_w32
export PYTHONPATH="$source_dir/src"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1

evaluate() {
    local policy=$1 menu=$2 group=$3 data=$4 output=$5
    if [[ ! -f "$output/DONE" ]]; then
        "$python_bin" "$source_dir/scripts/w32/eval_policy.py" --policy "$policy" \
            --menu "$menu" --group "$group" --data-dir "$root/data/$data" \
            --output-dir "$output" --repeats 3 --threads 2
    fi
}

if [[ ${3:-} == watch ]]; then
    port=$4
    threads=$5
    shift 5
    arms=("$@")
    declare -A retries
    for arm in "${arms[@]}"; do retries[$arm]=0; done
    while true; do
        remaining=0
        for arm in "${arms[@]}"; do
            directory="$root/runs/$arm"
            [[ -f "$directory/DONE" ]] && continue
            remaining=$((remaining+1))
            pid=$(cat "$directory/pid")
            # Match the command as well: a recycled PID is not this training worker.
            if kill -0 "$pid" 2>/dev/null && [[ $(ps -p "$pid" -o args=) == *"--run-dir $directory "* ]]; then
                continue
            fi
            if (( retries[$arm] >= 2 )); then
                printf '%s port=%s arm=%s retries exhausted\n' "$(date -Is)" "$port" "$arm"
                exit 1
            fi
            retries[$arm]=$((retries[$arm]+1))
            IFS=- read -r menu group seed <<< "$arm"
            nohup setsid "$python_bin" -m satellite_rl.power30.lean_runner \
                --menu "$menu" --group "$group" --seed "${seed#s}" --data-dir "$root/data/train_n100" \
                --loader simple --run-dir "$directory" --threads "$threads" >> "$directory/stdout.log" 2>&1 < /dev/null &
            pid=$!
            printf '%s\n' "$pid" > "$directory/pid"
            printf '%s port=%s arm=%s restart=%s pid=%s\n' "$(date -Is)" "$port" "$arm" "${retries[$arm]}" "$pid"
        done
        (( remaining == 0 )) && exit 0
        sleep 300
    done
fi

arms=()
for menu in P00 P01; do
    for group in M0 M1; do
        for seed in 51001 51002 51003; do arms+=("$menu-$group-s$seed"); done
    done
done
while true; do
    count=0
    for arm in "${arms[@]}"; do [[ -f "$root/runs/$arm/DONE" ]] && count=$((count+1)); done
    (( count == 12 )) && break
    sleep 300
done
for menu in P00 P01; do
    arm="$menu-M0-s51001"
    for n in 60 150 200; do
        evaluate "$root/runs/$arm/policy_150000.pt" "$menu" M0 "dev_n$n" "$root/evals/n_robustness/$arm/n$n"
    done
done
for arm in "${arms[@]}"; do
    IFS=- read -r menu group seed <<< "$arm"
    evaluate "$root/runs/$arm/policy_150000.pt" "$menu" "$group" conf_n100 "$root/evals/confirmation/$arm"
done
date -Is > "$root/DONE_ALL"
