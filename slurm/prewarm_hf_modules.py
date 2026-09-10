#!/usr/bin/env python
"""
Pre-warm the HuggingFace dynamic-module cache for a local checkpoint dir.

When finetune.py resumes, every rank calls AutoProcessor/AutoConfig/AutoModel
.from_pretrained(vla_path, trust_remote_code=True) on the SAME fresh checkpoint
dir. transformers copies the custom *.py (configuration/processing/modeling_
prismatic) into ~/.cache/huggingface/modules/transformers_modules/<dir>/ on first
use, then imports them. With 8 torchrun ranks hitting a never-seen dir at once,
they race to populate/import that module and a loser sees a half-written file:
    AttributeError: module '...processing_prismatic' has no attribute 'PrismaticProcessor'

Running this ONCE, single-process, before torchrun resolves every auto_map class
so the cache is fully written before the ranks read it. No model weights are
loaded -- only the custom module files are copied and imported.

Usage: python slurm/prewarm_hf_modules.py <checkpoint_dir>
Always exits 0; a pre-warm failure just leaves the original (racy) path in place.
"""
import json
import os
import sys

from transformers.dynamic_module_utils import get_class_from_dynamic_module

AUTO_MAP_FILES = ("config.json", "preprocessor_config.json", "processor_config.json")


def main():
    if len(sys.argv) < 2:
        return
    path = sys.argv[1]
    refs = set()
    for fn in AUTO_MAP_FILES:
        fp = os.path.join(path, fn)
        if not os.path.isfile(fp):
            continue
        try:
            with open(fp) as f:
                refs.update(json.load(f).get("auto_map", {}).values())
        except Exception as e:
            sys.stderr.write(f"[prewarm] could not read {fn}: {e}\n")

    for ref in sorted(refs):
        try:
            get_class_from_dynamic_module(ref, path)
            sys.stderr.write(f"[prewarm] cached {ref}\n")
        except Exception as e:
            sys.stderr.write(f"[prewarm] skip {ref}: {e}\n")


if __name__ == "__main__":
    main()
