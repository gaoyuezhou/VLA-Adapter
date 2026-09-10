#!/bin/bash
# Submit one eval job per saved checkpoint of a training run.
# Usage: slurm/eval_sweep.sh <eval_env> <run_dir (without --<step>_chkpt suffix)> [step_stride (default 1)] [hydra overrides...]
# Example: slurm/eval_sweep.sh pusht "<run_root_dir>/vla-adapter+pusht.PushTDataset+b64+lr-0.0002+lora-r64+dropout-0.0+seed42--image_aug" 4
set -euo pipefail
EVAL_ENV="$1"; RUN_DIR="$2"; STRIDE="${3:-1}"; shift 3 2>/dev/null || shift $#
cd "$(dirname "$0")/.."
i=0
for d in $(ls -d "${RUN_DIR}"--*_chkpt | sort -t- -k2 -V); do
  [[ -f "$d/config.json" && -n "$(ls "$d"/action_head--*_checkpoint.pt 2>/dev/null)" ]] || { echo "skip incomplete $d"; continue; }
  if (( i % STRIDE == 0 )); then sbatch slurm/eval.sbatch "$EVAL_ENV" "$d" "$@"; fi
  i=$((i+1))
done
