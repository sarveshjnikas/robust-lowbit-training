#!/bin/bash
# EXACT PAPER SETTING FOR FIG. 1a / FIG. 9: nanoGPT train_shakespeare_char, A1W1, CHANNEL-WISE,
# BATCH 64, CONTEXT 256, 5000-STEP SCHEDULE BUT STOPPED AT 1500 (FIG. 1a: STE BLOWS UP BY STEP 500-1250), LR 1e-3, CLIP 1.0, SEED 1337 (THE SCRIPT DEFAULTS).
# ONE RUN AT A TIME: THE GPU IS SHARED, SO PARALLEL RUNS WOULD NOT BE FASTER.
cd "$(dirname "$0")"
for cfg in "ste affine" "ridge affine" "ste linear" "ridge linear"; do
  set -- $cfg
  python3 train_1bit.py --method $1 --scheme $2 --iters 1500 --name full_$1_$2 > logs/full_$1_$2.log 2>&1
  echo "done $1 $2: $(grep step logs/full_$1_$2.log | tail -1)"
done
echo REPRO FINISHED
