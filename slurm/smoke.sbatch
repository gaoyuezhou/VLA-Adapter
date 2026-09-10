#!/bin/bash
#SBATCH --job-name=vla-adapter-smoke
#SBATCH --partition=open_research
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/smoke-%j.out
#SBATCH --error=logs/slurm/smoke-%j.out

# Smoke test: tiny training run (a few steps, 1 GPU) followed by a short eval on the saved checkpoint.
# Usage: sbatch slurm/smoke.sbatch <dataset> <eval_env>      e.g. sbatch slurm/smoke.sbatch pusht pusht
#        sbatch slurm/smoke.sbatch rlds none                  (RLDS libero path, training only)
set -euo pipefail
DATASET="$1"; EVAL_ENV="${2:-none}"
source "$SLURM_SUBMIT_DIR/slurm/cluster.env"
cd "${REPO_DIR}"; source "${CONDA_SH}"; conda activate "${CONDA_ENV}"
export WANDB_MODE=offline
ROOT="${EXP_ROOT}/smoke"
mkdir -p "$ROOT"
EXTRA=()
if [[ "$DATASET" == "rlds" ]]; then EXTRA=(use_proprio=true); fi
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  dataset="$DATASET" batch_size=2 max_steps=20 save_freq=10 wandb_mode=offline \
  env_vars.run_root_dir="$ROOT" run_id_note=smoke "${EXTRA[@]}"
if [[ "$EVAL_ENV" != "none" ]]; then
  case "$DATASET" in
    pusht) LABEL="pusht.PushTDataset";; block_push) LABEL="block_pushing.PushMultiviewTrajectoryDataset";;
    cube) LABEL="cube.CubeDataset";; libero_goal) LABEL="libero.LiberoGoalDataset";; *) LABEL="$DATASET";;
  esac
  CKPT=$(ls -d "$ROOT"/vla-adapter+"$LABEL"+*smoke--20_chkpt | head -1)
  echo "Evaluating checkpoint: $CKPT"
  python vla-scripts/run_eval.py eval_env="$EVAL_ENV" pretrained_checkpoint="$CKPT" \
    num_trials=2 max_steps=40 use_wandb=false env_vars.run_root_dir="$ROOT" run_id_note=smoke
fi
echo SMOKE_DONE
