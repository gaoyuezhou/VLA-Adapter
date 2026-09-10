#!/usr/bin/env python
"""
Resolve the latest COMPLETE training checkpoint for a set of Hydra overrides so
slurm/finetune.sbatch can auto-resume after a SLURM preemption/requeue.

Given the SAME overrides you pass to finetune.py, this prints one line:

    <step>\t<checkpoint_dir>

for the highest-step checkpoint that is fully written, or prints nothing if
there is no resumable checkpoint (i.e. a fresh run). It ALWAYS exits 0 so a
resolver hiccup degrades to "train from scratch" instead of killing the job.

Notes / invariants this relies on (see vla-scripts/finetune.py):
  * run_dir is deterministic: <run_root_dir>/<run_id>  (finetune.py:740)
  * checkpoints live at   <run_dir>--<step>_chkpt       (finetune.py:557)
  * the merged base model (model*.safetensors + index) is written LAST, after
    the component saves and a dist.barrier (finetune.py:599-613), so a complete
    shard set is our "this checkpoint finished writing" signal. Preemption grace
    on open_research is 0, so half-written checkpoints are possible -> we verify
    completeness and fall back to the previous step if the newest is truncated.

compute_run_id() MUST stay in sync with finetune.py:get_run_id (lines ~113-127,
the fresh-run branch). Resume itself is the repo's built-in mechanism: passing
resume=true resume_step=<step> resum_vla_path=<chkpt_dir> (finetune.py:104-127,206-208).
"""
import glob
import json
import os
import re
import sys

CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "vla-scripts", "configs")
)


def compute_run_id(cfg):
    # Mirror of finetune.py:get_run_id, fresh-run branch (resume/override off).
    dataset_label = cfg.dataset.get("dataset_name", cfg.dataset.get("dataset_class", "unknown"))
    run_id = (
        f"{cfg.get('run_id_prefix', 'vla-adapter')}+{dataset_label}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_fz:
        run_id += f"+frozen+dropout-{cfg.lora_dropout}"
    if cfg.use_lora:
        run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    run_id += f"+seed{cfg.seed}"
    if cfg.image_aug:
        run_id += "--image_aug"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    return run_id


def checkpoint_is_complete(cfg, ckpt_dir, step):
    """True only if the merged model AND every component resume will load exist."""
    if not os.path.isfile(os.path.join(ckpt_dir, "config.json")):
        return False

    # Merged base model: all shards named in the index must be present.
    index = os.path.join(ckpt_dir, "model.safetensors.index.json")
    shards = {os.path.basename(p) for p in glob.glob(os.path.join(ckpt_dir, "*.safetensors"))}
    if os.path.isfile(index):
        try:
            with open(index) as f:
                needed = set(json.load(f).get("weight_map", {}).values())
        except Exception:
            return False
        if not needed or not needed.issubset(shards):
            return False
    elif not shards:
        # No merged model saved at all -> not resumable by finetune.py.
        return False

    # Auxiliary components, gated on the same cfg flags as save_training_checkpoint.
    if cfg.use_l1_regression or cfg.use_diffusion:
        if not os.path.isfile(os.path.join(ckpt_dir, f"action_head--{step}_checkpoint.pt")):
            return False
    if cfg.use_proprio:
        if not os.path.isfile(os.path.join(ckpt_dir, f"proprio_projector--{step}_checkpoint.pt")):
            return False
    if cfg.use_diffusion:
        if not os.path.isfile(os.path.join(ckpt_dir, f"noisy_action_projector--{step}_checkpoint.pt")):
            return False
    return True


def main():
    overrides = sys.argv[1:]
    try:
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(version_base="1.2", config_dir=CONFIG_DIR):
            cfg = compose(config_name="finetune", overrides=overrides)
        run_dir = os.path.join(cfg.run_root_dir, compute_run_id(cfg))
    except Exception as e:
        sys.stderr.write(f"[resolve_resume] could not resolve config, starting fresh: {e}\n")
        return

    candidates = []
    for d in glob.glob(f"{run_dir}--*_chkpt"):
        m = re.search(r"--(\d+)_chkpt$", d)
        if m and os.path.isdir(d):
            candidates.append((int(m.group(1)), d))
    candidates.sort(reverse=True)

    for step, d in candidates:
        try:
            if checkpoint_is_complete(cfg, d, step):
                sys.stdout.write(f"{step}\t{d}\n")
                return
            sys.stderr.write(f"[resolve_resume] skipping incomplete checkpoint: {d}\n")
        except Exception as e:
            sys.stderr.write(f"[resolve_resume] error checking {d}: {e}\n")
    # Nothing resumable -> print nothing (fresh run).


if __name__ == "__main__":
    main()
