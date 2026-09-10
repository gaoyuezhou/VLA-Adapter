#!/bin/bash
# Launch the full baseline sweep: 4 patch_policy suites (+ optional RLDS sanity run), given seeds.
# Usage: slurm/launch_all.sh [seed ...]      (default seed 42)
cd "$(dirname "$0")/.."
SEEDS=("${@:-42}")
for seed in "${SEEDS[@]}"; do
  for ds in pusht block_push cube libero_goal; do
    sbatch slurm/finetune.sbatch dataset="$ds" seed="$seed"
  done
done
