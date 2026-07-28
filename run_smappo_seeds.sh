#!/usr/bin/env bash
set -euo pipefail

FILE="docs/source/examples/isaaclab/torch_machine_tending_smappo.py"
SEEDS=(1 10 42) #(1 10 42 60 100 147 555 678 888 963)
TMP_DIR="seeds_exps_ablationNoVel/tmp"
OUTPUT_DIR="seeds_exps_ablationNoVel/logs"
MODEL_NAME="noVelSMAPPO_500K_DTnLID120_dec6"

mkdir -p "$TMP_DIR"
mkdir -p "$OUTPUT_DIR"

for seed in "${SEEDS[@]}"; do
  tmp_file="$TMP_DIR/torch_machine_tending_smappo_seed_${seed}.py"
  cp "$FILE" "$tmp_file"
  sed -i -E "s/^seed = .*/seed = ${seed}/" "$tmp_file"
  python "$tmp_file" --num_envs 256 --headless > "$OUTPUT_DIR/MT2_C1_25_${MODEL_NAME}_s${seed}.log" 2>&1 &
done
