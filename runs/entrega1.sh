#!/bin/bash
# Entrega 1 — PT-Latn small language model: single-GPU pretraining pipeline
#
# Hardware target: 1× NVIDIA RTX A6000 (48 GB VRAM, Ampere SM 86)
# Expected wall-clock: ~12–18 h for depth=12 (~85 M params, ~2 B tokens)
#
# Usage:
#   bash runs/entrega1.sh                         # no wandb logging
#   WANDB_RUN=entrega1_d12 bash runs/entrega1.sh  # with wandb
#
# Prerequisites:
#   conda activate DL   (or whichever env has nanochat deps + torch cu128)
#   Optional: huggingface-cli login  (not required — dataset is public)
#   Optional: wandb login

set -euo pipefail

# -----------------------------------------------------------------------------
# Environment
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/mnt/E-SSD/barcelos/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"
mkdir -p logs

# wandb: pass WANDB_RUN=name to enable logging; "dummy" disables it
WANDB_RUN="${WANDB_RUN:-dummy}"

echo "========================================================================"
echo "  entrega1.sh  |  PT-Latn SLM  |  depth=12  |  single A6000"
echo "  NANOCHAT_BASE_DIR : $NANOCHAT_BASE_DIR"
echo "  WANDB_RUN         : $WANDB_RUN"
echo "========================================================================"

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# [1/5] Download HF fineweb-2/por_Latn data
#
# 55 shards × 50 000 docs × ~3 KB avg ≈ 8.25 GB raw text ≈ 2.2 B tokens
# The first ~14 shards (~2 B chars) are consumed by the tokenizer trainer;
# all 55 are consumed by model pretraining.
# Download is idempotent: already-present shards are skipped.

echo ""
echo "[1/5] Downloading HF fineweb-2/por_Latn shards ..."
python -m nanochat.dataset_hf -n 55

# -----------------------------------------------------------------------------
# [2/5] Train Portuguese BPE tokenizer
#
# Byte-level BPE (GPT-4 style), vocab size 32 768 (= 2^15, nanochat default).
# Streams docs from the downloaded parquets up to --max-chars=2 B.
# Byte-level handles accented chars (á é ç ã ó) without any normalization pass.
# Expected compression: ~3.5–4.0 chars/token on PT-Latn text.

echo ""
echo "[2/5] Training Portuguese BPE tokenizer (vocab=32 768, up to 2 B chars) ..."
python -m scripts.tok_train --dataset=hf

# -----------------------------------------------------------------------------
# [3/5] Evaluate tokenizer compression ratio
#
# Report saved to logs/tokenizer_eval.txt.
# Key metric: fwe-train / fwe-val rows — target ≈ 4.5 bytes/token ≈ 3.9 chars/token.

echo ""
echo "[3/5] Evaluating tokenizer compression ratio ..."
python -m scripts.tok_eval --dataset=hf 2>&1 | tee logs/tokenizer_eval.txt

# -----------------------------------------------------------------------------
# [4/5] Pretrain the base model
#
# Flag rationale (all decisions documented in README):
#
#   --depth=12
#       ~85 M total parameters; compute-optimal for a single-GPU run.
#
#   --target-param-data-ratio=23
#       ≈ 2 B training tokens (2e9 / 85e6 ≈ 23.5 × params).
#
#   --device-batch-size=16
#       16 seq × 2048 tok = 32 768 tok/step; fits A6000 48 GB in bf16.
#       grad_accum_steps auto-computed to reach total_batch_size ≈ 524 288.
#
#   --optimizer=adamw
#       AdamW for all parameter groups; Muon disabled.
#       Rationale: well-understood baseline; Newton-Schulz orthogonalization
#       adds conceptual complexity without clear benefit at this scale.
#
#   --dataset=hf
#       Reads from HF fineweb-2/por_Latn local parquet shards.
#
#   --window-pattern=L
#       REQUIRED on A6000 (Ampere SM 86): FA3 is not available; the SDPA
#       fallback does not support sliding-window efficiently (falls through
#       to an explicit bool-mask path).  Default pattern "SSSL" would double
#       VRAM usage and halve throughput.  "L" uses the fast is_causal=True
#       path for all layers.
#
#   --core-metric-every=-1
#       CORE benchmark is English ICL tasks; results are near-random for a
#       Portuguese model and waste ~15 min per 2 000 steps.  Disabled here;
#       the README notes this limitation explicitly.
#
#   No --fp8     : A6000 SM 86 has no FP8 tensor cores; bf16 auto-detected.
#   No torchrun  : single process, DDP auto-disabled.

echo ""
echo "[4/5] Pretraining depth=12 PT-Latn model (~85 M params, ~2 B tokens) ..."
echo "      Expected wall-clock: 12–18 h on A6000."
python -m scripts.base_train           \
    --save-every=250                  \
    --depth=12                         \
    --target-param-data-ratio=23       \
    --device-batch-size=16             \
    --optimizer=adamw                  \
    --dataset=hf                       \
    --window-pattern=L                 \
    --core-metric-every=-1             \
    --run="$WANDB_RUN"                 \
    2>&1 | tee logs/d12_train.log

# -----------------------------------------------------------------------------
# [5/5] Evaluate the base model
#
#   --eval bpb,sample   : BPB on PT-Latn val + unconditioned samples.
#                         CORE omitted (English benchmark; near-random for PT).
#   --dataset=hf        : BPB measured on fineweb-2/por_Latn validation shard
#                         (the last shard); this is the primary quality metric.
#   --device-batch-size=8 : conservative; keeps VRAM safe during eval.

echo ""
echo "[5/5] Evaluating base model (BPB on PT-Latn val + samples) ..."
python -m scripts.base_eval  \
    --device-batch-size=8    \
    --eval bpb,sample        \
    --dataset=hf             \
    2>&1 | tee logs/d12_eval.log

# -----------------------------------------------------------------------------
python -m nanochat.report generate

echo ""
echo "========================================================================"
echo "  entrega1.sh complete."
echo ""
echo "  Artifacts:"
echo "    Tokenizer eval  :  logs/tokenizer_eval.txt"
echo "    Training log    :  logs/d12_train.log"
echo "    Eval log        :  logs/d12_eval.log"
echo "    Checkpoint      :  \$NANOCHAT_BASE_DIR/base_checkpoints/d12/"
echo "    Report          :  \$NANOCHAT_BASE_DIR/report/report.md"
echo "========================================================================"
