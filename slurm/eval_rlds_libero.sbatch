#!/bin/bash
#SBATCH --job-name=vla-adapter-rlds-eval
#SBATCH --partition=open_research
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/rlds-eval-%j.out
#SBATCH --error=logs/slurm/rlds-eval-%j.out

# Original VLA-Adapter LIBERO eval (upstream LIBERO protocol, run_libero_eval.py) for the RLDS sanity-check run.
# Usage: sbatch slurm/eval_rlds_libero.sbatch <checkpoint_dir> [num_trials_per_task (default 50)] [extra draccus args...]
set -euo pipefail
CKPT="$1"; TRIALS="${2:-50}"; shift; shift || true
source "$SLURM_SUBMIT_DIR/slurm/cluster.env"
cd "${REPO_DIR}"; source "${CONDA_SH}"; conda activate "${CONDA_ENV}"
export LIBERO_CONFIG_PATH="${UPSTREAM_LIBERO_CONFIG}"   # upstream LIBERO (pip), not envs/libero
mkdir -p "${EXP_ROOT}/eval_logs_rlds"
python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True --num_images_in_input 2 --use_film False --use_pro_version True --use_minivlm True \
  --pretrained_checkpoint "$CKPT" --task_suite_name libero_goal --num_trials_per_task "$TRIALS" \
  --local_log_dir "${EXP_ROOT}/eval_logs_rlds" "$@"
echo RLDS_EVAL_DONE
