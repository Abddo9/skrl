#!/usr/bin/env bash
set -euo pipefail

REPEAT_NUM=12
GPU_INDEX=0                        
MIN_FREE_VRAM_MIB=3072             
VRAM_CHECK_INTERVAL_SECONDS=240 
BETWEEN_JOB_DELAY_SECONDS=30

FILE="docs/source/examples/isaaclab/torch_machine_tending_smappo.py"
SEEDS=(1 2 3 42 5 6 7 8 9 10)
TMP_DIR="seeds_exps_S2_P2_O1V2L01A01${REPEAT_NUM}"
OUTPUT_DIR="Eval_logs_P2_O1V2L01A01"

mkdir -p "$TMP_DIR"
mkdir -p "$OUTPUT_DIR"

wait_for_free_vram() {             
  while true; do                  
    if free_vram_mib=$(nvidia-smi --id="$GPU_INDEX" \
      --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null); then # Read free memory (in MiB) from the chosen GPU.
      free_vram_mib=${free_vram_mib//[[:space:]]/} # Remove spaces/newlines from the output.

      if [[ "$free_vram_mib" =~ ^[0-9]+$ ]] && \
         (( free_vram_mib > MIN_FREE_VRAM_MIB )); then # Require a valid number strictly above 3072 MiB.

        echo "GPU ${GPU_INDEX} has ${free_vram_mib} MiB free; starting job."
        return                    
      fi

      echo "GPU ${GPU_INDEX} has ${free_vram_mib:-unknown} MiB free; need more than ${MIN_FREE_VRAM_MIB} MiB. Retrying in 4 minutes."
    else                           
      echo "Unable to read GPU ${GPU_INDEX} VRAM with nvidia-smi. Retrying in 4 minutes."
    fi

    sleep "$VRAM_CHECK_INTERVAL_SECONDS"
                                   
  done
}


for seed in "${SEEDS[@]}"; do
  wait_for_free_vram 

  tmp_file="$TMP_DIR/torch_machine_tending_smappo_seed_${seed}.py"
  cp "$FILE" "$tmp_file"
  sed -i -E "s/^seed = .*/seed = ${seed}/" "$tmp_file"
  python "$tmp_file" --num_envs 1 --headless > "$OUTPUT_DIR/SMAPPO_dec6_1_s2_${REPEAT_NUM}_P2_O1V2L01A01_Eval_${seed}.log" 2>&1 &
  disown
  echo "Waiting for $BETWEEN_JOB_DELAY_SECONDS seconds before trying the next job..."
  sleep "$BETWEEN_JOB_DELAY_SECONDS"
done
