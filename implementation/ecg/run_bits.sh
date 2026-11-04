#!/bin/bash
# STE VS. g AT 2 AND 4 BITS, 3 SEEDS, AT THE LEARNING RATE THAT IS BEST FOR FLOAT (RUN AFTER run_sweep.sh)
cd "$(dirname "$0")"
LR=$(python3 -c "
import glob, json, numpy as np
from collections import defaultdict
f1 = defaultdict(list)
for f in glob.glob('runs/float_*.json'):
    r = json.load(open(f)); f1[r['lr']].append(r['epoch'][-1]['macro_f1'])
print(max(f1, key=lambda lr: np.mean(f1[lr])))" 2>/dev/null)
echo "learning rate best for float: $LR"
for s in 0 1 2; do for b in 2 4; do for m in ste g; do
  echo "$m $b $s"
done; done; done | xargs -P 3 -L 1 bash -c 'python3 train_ecg.py --method $0 --bits $1 --seed $2 --lr '"$LR"' > logs/${0}_k${1}_s${2}.log 2>&1; echo "done $0 k=$1 seed=$2: $(tail -1 logs/${0}_k${1}_s${2}.log)"'
echo BITS FINISHED
