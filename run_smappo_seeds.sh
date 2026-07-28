#!/usr/bin/env bash
set -euo pipefail

REPEAT_NUM=12

FILE="docs/source/examples/isaaclab/torch_machine_tending_smappo.py"
SEEDS=(1 2 3 42 5 6 7 8 9 10)
TMP_DIR="seeds_exps_S1_P05_O1L01A0${REPEAT_NUM}"
OUTPUT_DIR="Eval_logs_P05_O1L01A0"

mkdir -p "$TMP_DIR"
mkdir -p "$OUTPUT_DIR"

for seed in "${SEEDS[@]}"; do
  tmp_file="$TMP_DIR/torch_machine_tending_smappo_seed_${seed}.py"
  cp "$FILE" "$tmp_file"
  sed -i -E "s/^seed = .*/seed = ${seed}/" "$tmp_file"
  python "$tmp_file" --num_envs 1 --headless > "$OUTPUT_DIR/SMAPPO_dec6_1_s1_${REPEAT_NUM}_P05_O1L01A0_Eval_${seed}.log" 2>&1 &
done
