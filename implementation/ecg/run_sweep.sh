#!/bin/bash
# FULL SWEEP: 3 METHODS x 3 LEARNING RATES x 3 SEEDS, 1 BIT, 20 EPOCHS, 3 RUNS AT A TIME
cd "$(dirname "$0")"
mkdir -p logs
for s in 0 1 2; do for lr in 1e-3 3e-3 1e-2; do for m in float ste g; do
  echo "$m $lr $s"
done; done; done | xargs -P 3 -L 1 bash -c 'python3 train_ecg.py --method $0 --lr $1 --seed $2 > logs/${0}_lr${1}_s${2}.log 2>&1; echo "done $0 lr=$1 seed=$2: $(tail -1 logs/${0}_lr${1}_s${2}.log)"'
echo SWEEP FINISHED
