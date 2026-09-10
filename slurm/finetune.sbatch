#!/bin/bash
#SBATCH --job-name=vla-adapter-ft
#SBATCH --partition=open_research
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=1600G
#SBATCH --time=96:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm/slurm-%j.out
#SBATCH --error=logs/slurm/slurm-%j.out

# Submit vla-scripts/finetune.py to SLURM as a single-node, 8-GPU job with
# automatic resume across preemption/requeue.
#
# Resource settings mirror patch_policy/eval_configs/train_blockpush.yaml
# (submitit_slurm launcher): open_research, gpu:8, 1600G mem, 96h. submitit used
# 8 tasks x 24 cpus_per_task = 192 CPUs/node; here we launch one task and let
# torchrun spawn the 8 procs, so cpus-per-task holds all 192.
#
# AUTO-RESUME: the cluster is PreemptMode=REQUEUE with JobRequeue=1, so on
# preemption this script re-runs from the top with the same job ID. Before each
# torchrun, resolve_resume.py looks for the latest COMPLETE checkpoint for these
# exact overrides and, if found, injects `resume=true resume_step=<N> vla_path=<dir>`.
# The run_dir is deterministic from the overrides, so re-submitting the same
# command also resumes. Resume relies on periodic save_freq checkpoints (grace
# time is 0 -> no last-second save on preemption).
#
# Usage:
#   sbatch slurm/finetune.sbatch dataset=block_push seed=44
#   sbatch --partition=scaling --gres=gpu:4 slurm/finetune.sbatch dataset=cube seed=1
#
# Everything after the script name is forwarded verbatim to finetune.py as Hydra overrides.

set -euo pipefail

source "$SLURM_SUBMIT_DIR/slurm/cluster.env"
cd "${REPO_DIR}"

# Activate the conda env non-interactively.
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

# Use however many GPUs SLURM allocated on this node.
NPROC="${SLURM_GPUS_ON_NODE:-8}"

# Guard against nodes whose GPUs fail CUDA init (seen on gb-west-a1-gpu-003: "CUDA unknown error"): exclude this
# node from the job and requeue instead of burning the allocation on a crash.
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.device_count() > 0 else 1)" 2>/dev/null; then
  echo "NO CUDA devices visible on ${SLURMD_NODENAME}; excluding node and requeueing job ${SLURM_JOB_ID}"
  scontrol update JobId="${SLURM_JOB_ID}" ExcNodeList="${SLURMD_NODENAME}" || true
  scontrol requeue "${SLURM_JOB_ID}" || true
  exit 0
fi

# Resolve the latest complete checkpoint for these overrides (empty on a fresh run).
# `|| true` so a resolver hiccup degrades to "train from scratch" rather than
# killing the job under `set -e`.
RESUME_INFO="$(python slurm/resolve_resume.py "$@" || true)"

RESUME_ARGS=()
if [[ -n "${RESUME_INFO}" ]]; then
  RESUME_STEP="$(printf '%s' "${RESUME_INFO}" | cut -f1)"
  RESUME_CKPT="$(printf '%s' "${RESUME_INFO}" | cut -f2)"
  RESUME_ARGS=(resume=true "resume_step=${RESUME_STEP}" "resum_vla_path=${RESUME_CKPT}")
  echo "AUTO-RESUME: continuing from step ${RESUME_STEP} at ${RESUME_CKPT}"
  # Pre-warm the HF dynamic-module cache single-process: all 8 ranks otherwise
  # race to populate transformers_modules/<chkpt>/ for the custom prismatic code
  # and a loser crashes with "no attribute 'PrismaticProcessor'". `|| true` so a
  # pre-warm hiccup never kills the job.
  echo "AUTO-RESUME: pre-warming HF dynamic-module cache for ${RESUME_CKPT}"
  python slurm/prewarm_hf_modules.py "${RESUME_CKPT}" || true
else
  echo "AUTO-RESUME: no complete checkpoint found -> training from scratch"
fi

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME} | ${NPROC} GPUs | overrides: $* ${RESUME_ARGS[*]:-}"

torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node "${NPROC}" \
  vla-scripts/finetune.py "$@" "${RESUME_ARGS[@]}"
