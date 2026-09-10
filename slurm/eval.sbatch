#!/bin/bash
#SBATCH --job-name=vla-adapter-eval
#SBATCH --partition=open_research
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/eval-%j.out
#SBATCH --error=logs/slurm/eval-%j.out

# Evaluate one checkpoint in one patch_policy env.
# Usage: sbatch slurm/eval.sbatch <eval_env: pusht|blockpush|cube|libero_goal> <checkpoint_dir> [hydra overrides...]
set -euo pipefail
EVAL_ENV="$1"; CKPT="$2"; shift 2
source "$SLURM_SUBMIT_DIR/slurm/cluster.env"
cd "${REPO_DIR}"; source "${CONDA_SH}"; conda activate "${CONDA_ENV}"
STEP=$(basename "$CKPT" | sed -E 's/.*--([0-9]+)_chkpt$/\1/')
SEED=$(basename "$CKPT" | sed -E 's/.*\+seed([0-9]+).*/\1/')
python vla-scripts/run_eval.py eval_env="$EVAL_ENV" pretrained_checkpoint="$CKPT" run_id_note="seed${SEED}-step${STEP}" "$@"
echo EVAL_DONE
