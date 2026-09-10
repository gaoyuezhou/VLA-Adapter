# VLA-Adapter as a baseline on the patch_policy sim suites

This branch trains and evaluates VLA-Adapter on the four simulated suites used by
[patch_policy](https://github.com/gaoyuezhou/patch_policy): **PushT**, **BlockPush**, **Cube**, **LIBERO-Goal**.
The integration mirrors the one done for OpenVLA-OFT (`adazing/openvla-oft`, branch `kathy`), ported onto VLA-Adapter.
The original RLDS/LIBERO training path is untouched and selectable with `dataset=rlds`.

Contents
1. [What changed](#1-what-changed)
2. [Setting up on a new cluster](#2-setting-up-on-a-new-cluster)
3. [Launching the experiments](#3-launching-the-experiments)
4. [Evaluating](#4-evaluating)
5. [Per-env settings and decisions](#5-per-env-settings-and-decisions)
6. [Gotchas we hit](#6-gotchas-we-hit)

---

## 1. What changed

| Piece | Where | Notes |
|---|---|---|
| Configurable robot constants | `prismatic/vla/constants.py` (`set_constants`) | All consumers read `C.ACTION_DIM` etc. at call time, so 2-D / 5-D actions work. |
| Hydra configs | `vla-scripts/configs/{finetune,eval}.yaml`, `dataset/*.yaml`, `eval_env/*.yaml`, `env_vars/env_vars.yaml` | `finetune.py` and `run_eval.py` are Hydra entry points (draccus removed from `finetune.py`). |
| patch_policy data path | `prismatic/vla/datasets/trajectory/`, `patch_policy_adapter.py`, `repeating_loader.py` | Reads the patch_policy `.pth/.npy` datasets directly (no RLDS conversion). q01/q99 normalization stats are computed on the fly and saved to `dataset_statistics.json` in every checkpoint. |
| Sim envs | `envs/` | Vendored from patch_policy (pusht, block_pushing, cube, libero incl. assets, ~450 MB). |
| Unified eval | `vla-scripts/run_eval.py` | Success metrics + wandb metrics match patch_policy's eval. |
| Adapter head fixes | `prismatic/models/action_heads.py` | Task/action token split inferred from the tensor (was hard-coded to 2 cameras = 512 tokens); proprio optional. |
| Resume | `vla-scripts/finetune.py` | Resume reloads the merged VLA from the checkpoint dir (upstream never restored LoRA weights). |
| Startup race fixes | `experiments/robot/openvla_utils.py` | `config.json` / modeling-file sync is a no-op when unchanged and atomic otherwise (concurrent jobs share `pretrained_models/configs`). |
| Slurm | `slurm/` | `finetune.sbatch` (8 GPU, auto-resume, bad-GPU-node guard), `smoke.sbatch`, `eval.sbatch`, `eval_sweep.sh`, `eval_rlds_libero.sbatch`, `resolve_resume.py`, `cluster.env`. |
| Env freeze | `requirements/vla-adapter-freeze.txt` | `pip freeze` of the working env (minus torch / flash-attn / editable packages). |

## 2. Setting up on a new cluster

Verified on H200 nodes (compute capability 9.0) with the CUDA 12.1 torch wheels. Any Ampere+ GPU with a 12.x driver should work.

### 2.1 Conda environment

```bash
conda create -n vla-adapter python=3.10 -y && conda activate vla-adapter
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
cd VLA-Adapter && pip install -e .
pip install packaging ninja
# flash-attn 2.5.5 prebuilt wheel (cu12 / torch 2.2 / cp310). Build from source if your combo differs.
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# everything else (sim envs, hydra, wandb, pinned TF stack) at the exact versions that worked:
pip install -r requirements/vla-adapter-freeze.txt
```

If you prefer not to replay the whole freeze, the packages that actually matter beyond `pip install -e .` are:

```bash
pip install "hydra-core>=1.2" omegaconf hydra-submitit-launcher msgpack msgpack-numpy imageio imageio-ffmpeg
pip install pymunk==6.8.0 pybullet==3.2.7 mujoco==3.2.7 gymnasium==1.1.1 dm_control==1.0.27 shapely==2.0.7 \
            scikit-image pygame easydict termcolor lxml gym==0.23.1 bddl==1.0.1 numba future gin-config
pip install robosuite==1.4.1 robomimic==0.2.0 tf-agents==0.19.0 --no-deps
pip install tensorflow-probability==0.23.0 "tensorflow-metadata==1.17.3" "array_record==0.7.1" "numpy<2"
```

Pins that bite if you drift: `tensorflow-metadata==1.17.3` (newer versions need protobuf 5, TF 2.15 needs protobuf 4;
symptom `ImportError: cannot import name 'runtime_version' from 'google.protobuf'`), `numpy<2`, `transformers==4.40.1`.

Sanity check (CPU is fine):

```bash
python -c "import flash_attn, pymunk, pybullet, mujoco, gymnasium, robosuite, bddl, tf_agents, dlimp, hydra, msgpack; print('ok')"
```

### 2.2 Data

```bash
# patch_policy datasets (HF: gaoyuezhou/patch-policy-datasets) -> one root with pusht_dataset/ cube_dataset/ libero_dataset/ block_push_dataset/
huggingface-cli download --repo-type dataset gaoyuezhou/patch-policy-datasets --local-dir <dataset_root>   # then unzip the 4 zips
# (optional, RLDS sanity-check run) LIBERO-Goal RLDS from OpenVLA
huggingface-cli download --repo-type dataset openvla/modified_libero_rlds --include "libero_goal_no_noops/*" --local-dir <rlds_root>
```

### 2.3 Model weights

```bash
M=<ai_models_dir>
for repo in timm/vit_large_patch14_reg4_dinov2.lvd142m timm/ViT-SO400M-14-SigLIP Qwen/Qwen2.5-0.5B \
            Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b; do
  huggingface-cli download --resume-download "$repo" --local-dir "$M/$repo"
done
```
`vlm_path` (below) points at the `Stanford-ILIAD/...` dir. The timm backbones are pulled from the HF cache at load time
(they are downloaded on first use if you skip them here; see gotcha on concurrent downloads).

### 2.4 Paths

Edit two files:

* `vla-scripts/configs/env_vars/env_vars.yaml` — `dataset_root`, `rlds_root`, `vlm_path`, `run_root_dir`, `wandb_entity`, `wandb_project`.
* `slurm/cluster.env` — `CONDA_SH`, `CONDA_ENV`, `EXP_ROOT`, and (only for the RLDS eval) `UPSTREAM_LIBERO_CONFIG`.

Then adjust the `#SBATCH` headers (partition, `--gres`, `--mem`, `--cpus-per-task`, `--time`) in `slurm/*.sbatch`, or override at submit time
(`sbatch --partition=X --gres=gpu:8 ...`). Slurm logs go to `logs/slurm/` relative to the repo; create it once: `mkdir -p logs/slurm`.

### 2.5 (Only for the RLDS sanity-check eval) upstream LIBERO

`run_libero_eval.py` (the original VLA-Adapter eval, upstream LIBERO protocol with pruned init states) needs the pip LIBERO:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git <somewhere>/LIBERO
echo "<somewhere>/LIBERO" > $CONDA_PREFIX/lib/python3.10/site-packages/libero_repo.pth   # editable install of LIBERO is broken; a .pth works
mkdir -p <somewhere>/LIBERO/.libero_config && cat > <somewhere>/LIBERO/.libero_config/config.yaml <<EOF2
assets: <somewhere>/LIBERO/libero/libero/./assets
bddl_files: <somewhere>/LIBERO/libero/libero/./bddl_files
benchmark_root: <somewhere>/LIBERO/libero/libero
datasets: <somewhere>/LIBERO/libero/datasets
init_states: <somewhere>/LIBERO/libero/libero/./init_files
EOF2
```
and set `UPSTREAM_LIBERO_CONFIG` in `slurm/cluster.env` to that `.libero_config` dir. The vendored `envs/libero` used by
`run_eval.py` writes its own config under `envs/libero/.libero_config/` (gitignored) and never touches `~/.libero`.

## 3. Launching the experiments

### 3.1 Smoke test first (1 GPU, ~10 min each)

```bash
sbatch slurm/smoke.sbatch pusht pusht
sbatch slurm/smoke.sbatch block_push blockpush
sbatch slurm/smoke.sbatch cube cube
sbatch slurm/smoke.sbatch libero_goal libero_goal
sbatch slurm/smoke.sbatch rlds none            # RLDS training path only
```
Each trains 20 steps, saves/merges a checkpoint at steps 10 and 20, reloads it, and runs 2 eval episodes with video under
`$EXP_ROOT/smoke/eval_logs/`. **Run the first one alone** before the rest so the timm weights land in the HF cache once (see gotchas).

### 3.2 Full runs (8 GPUs)

Recipe = VLA-Adapter LIBERO recipe: batch 8 per GPU (64 effective), lr 2e-4, LoRA rank 64, Pro head, image aug, Qwen2.5-0.5B backbone.

```bash
# the four patch_policy suites
for ds in pusht block_push cube libero_goal; do
  sbatch slurm/finetune.sbatch dataset=$ds seed=42 max_steps=100000 num_steps_before_decay=100000 save_freq=5000
done
# port sanity check: original RLDS LIBERO-Goal recipe (proprio on, as published). Expect ~97-98% with the upstream eval.
sbatch slurm/finetune.sbatch dataset=rlds use_proprio=true seed=42 max_steps=100000 num_steps_before_decay=100000 save_freq=5000
```
Everything after the script name is a Hydra override. More seeds: `seed=43`, etc. Runs land in
`<run_root_dir>/vla-adapter+<dataset_label>+b64+lr-0.0002+lora-r64+dropout-0.0+seed42--image_aug`, checkpoints as `<run_dir>--<step>_chkpt`.

`finetune.sbatch` is `--requeue`-safe: on restart it finds the latest *complete* checkpoint for the same overrides and resumes
(`slurm/resolve_resume.py`). It also detects nodes whose GPUs fail CUDA init and requeues itself excluding that node.

Without slurm: `torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py dataset=pusht seed=42 ...`

## 4. Evaluating

```bash
# one checkpoint
sbatch slurm/eval.sbatch pusht <run_dir>--50000_chkpt
# every checkpoint of a run (3rd arg = stride over saved checkpoints)
slurm/eval_sweep.sh pusht        "<run_root_dir>/vla-adapter+pusht.PushTDataset+b64+lr-0.0002+lora-r64+dropout-0.0+seed42--image_aug" 2
slurm/eval_sweep.sh blockpush    "<run_root_dir>/vla-adapter+block_pushing.PushMultiviewTrajectoryDataset+...--image_aug" 2
slurm/eval_sweep.sh cube         "<run_root_dir>/vla-adapter+cube.CubeDataset+...--image_aug" 2
slurm/eval_sweep.sh libero_goal  "<run_root_dir>/vla-adapter+libero.LiberoGoalDataset+...--image_aug" 2
# RLDS sanity-check run, original upstream LIBERO protocol (50 trials x 10 tasks)
sbatch slurm/eval_rlds_libero.sbatch "<run_root_dir>/vla-adapter+libero_goal_no_noops+...--image_aug--100000_chkpt" 50
```
Defaults (`configs/eval.yaml`): 100 episodes per task (10 tasks for LIBERO-Goal), 300 max steps, full action chunk executed open-loop,
videos under `$EXP_ROOT/eval_logs/<run_id>/videos/`, results to wandb and a `.txt` log.
Success: pusht max coverage >= 0.8; blockpush both blocks entered; cube all cubes entered; libero goal predicate fired within the episode.
Continuous metrics (coverage / entered / moved, mean/max/min) are logged too, matching patch_policy's `eval_on_env`.

## 5. Per-env settings and decisions

| env | action dim | chunk (= open-loop steps) | views | proprio | instruction | dataset config | eval config |
|---|---|---|---|---|---|---|---|
| pusht | 2 (dataset /500, env x500) | 5 | 1 | none | fixed string | `dataset/pusht.yaml` | `eval_env/pusht.yaml` |
| block_push | 2 (dataset /0.03, env x0.03) | 5 | 2 | none | fixed string | `dataset/block_push.yaml` | `eval_env/blockpush.yaml` |
| cube | 5 | 8 | 1 | none | fixed string | `dataset/cube.yaml` | `eval_env/cube.yaml` |
| libero_goal | 7 | 8 | 2 | none | from task name | `dataset/libero_goal.yaml` | `eval_env/libero_goal.yaml` |

Decisions: no proprio anywhere (patch_policy has none); chunk lengths as in the OpenVLA-OFT fork so those numbers stay comparable;
LIBERO eval uses patch_policy's `LiberoEnv` protocol (sorted task order, seeded resets, no pruned init states, 224x224 upright images),
so it is comparable to patch_policy but not to VLA-Adapter's published LIBERO-Goal number — that is what the RLDS run + `eval_rlds_libero.sbatch` are for.
`dataset/*.yaml` and `eval_env/*.yaml` must agree on action_dim / num_actions_chunk / num_views.

## 6. Gotchas we hit

* **Concurrent first-time weight downloads.** Several jobs starting together each download the timm SigLIP weights into the shared HF cache;
  one mmaps a half-written safetensors file and dies with SIGBUS (`exitcode -7`) while loading `ViT-SO400M-14-SigLIP`. Populate the cache
  with one job first.
* **Shared `pretrained_models/configs/config.json`.** Fixed in code (no-op / atomic write), but this is why several jobs launched in the same
  second used to crash with `JSONDecodeError`.
* **Bad GPU nodes.** `CUDA unknown error` / `ProcessGroupNCCL ... no GPUs found` at startup = broken node, not code. The launcher now
  excludes the node and requeues.
* **EGL noise at exit.** LIBERO eval prints `EGLError ... eglMakeCurrent` tracebacks during interpreter teardown. Harmless.
* **Login nodes without GPU/EGL** can still import everything and run PushT/BlockPush/Cube envs; LIBERO rendering needs a GPU node.
