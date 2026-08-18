#!/usr/bin/env bash
set -euo pipefail

# 1. GCDP-Obj-Attn-AttnPool
uv run python policy/main.py experiment=GCDP-Obj-Attn-AttnPool__SCLR__default__train seed=3 trainer/logger=wandb
CKPT_ATTN_POOL="$(ls -td logs/GCDP-Obj-Attn-AttnPool__SCLR__default__train/runs/*/*/checkpoints/last.ckpt | head -n 1)"
uv run python policy/eval.py experiment=GCDP-Obj-Attn-AttnPool__SCLR__default__test ckpt_path="${CKPT_ATTN_POOL}" seed=3 trainer/logger=wandb

# 2. GCDP-Obj-Attn-MLPPool
uv run python policy/main.py experiment=GCDP-Obj-Attn-MLPPool__SCLR__default__train seed=3 trainer/logger=wandb
CKPT_MLP_POOL="$(ls -td logs/GCDP-Obj-Attn-MLPPool__SCLR__default__train/runs/*/*/checkpoints/last.ckpt | head -n 1)"
uv run python policy/eval.py experiment=GCDP-Obj-Attn-MLPPool__SCLR__default__test ckpt_path="${CKPT_MLP_POOL}" seed=3 trainer/logger=wandb
