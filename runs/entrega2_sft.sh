#!/bin/bash
# Entrega 3 — Supervised Fine-Tuning (SFT) on ju-resplande/qa-pt
#
# Continues from the d12_midtrain checkpoint (Entrega 2) with the loss mask
# enabled: only assistant-response tokens contribute to the cross-entropy loss.
# This teaches the model to answer and terminate, not predict the question.
#
# Key differences from entrega2_midtrain.sh:
#   - loss mask ENABLED (--loss-mask=assistant_only)
#   - init checkpoint: d12_midtrain (not d12 base)
#   - LRs 10x smaller than mid-training (already 10x smaller than pre-training)
#   - Output checkpoint: d12_sft  (d12 and d12_midtrain are NEVER touched)
#
# Hardware  : 1x NVIDIA RTX A6000 (48 GB VRAM, Ampere SM 86), single process.
# Token budget  : ~630 M tokens (2 epochs x ~315 M tokens/epoch with packing).
# Expected wall-clock: ~2-4 h on A6000.
#
# Prerequisites:
#   conda activate DL   (env with torch + CUDA, datasets, pyarrow, rustbpe)
#   Entrega 2 must be complete:
#       $NANOCHAT_BASE_DIR/base_checkpoints/d12_midtrain/ must exist
#       $NANOCHAT_BASE_DIR/data_qa_pt/ must have 16 parquet shards
#
# Usage:
#   bash runs/entrega3_sft.sh
#   WANDB_RUN=entrega3_d12_sft bash runs/entrega3_sft.sh
#
#   # Verify mask only (prints 3 sequences colour-coded, then exits):
#   bash runs/entrega3_sft.sh --verify-mask

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/mnt/E-SSD/barcelos/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"
mkdir -p logs

WANDB_RUN="${WANDB_RUN:-dummy}"
VERIFY_MASK="${1:-}"

echo "========================================================================"
echo "  entrega3_sft.sh  |  PT-Latn SFT  |  qa-pt + loss mask  |  1x A6000"
echo "  NANOCHAT_BASE_DIR : $NANOCHAT_BASE_DIR"
echo "  WANDB_RUN         : $WANDB_RUN"
echo "  CUDA device       : $CUDA_VISIBLE_DEVICES"
echo "========================================================================"

# ---------------------------------------------------------------------------
# [0/3] Sanity: qa-pt shards already on disk from Entrega 2 (idempotent)
#
# The parquet shards in $NANOCHAT_BASE_DIR/data_qa_pt/ are reused directly.
# Re-run this only if the data directory was deleted.

if [ ! -d "$NANOCHAT_BASE_DIR/data_qa_pt" ]; then
    echo ""
    echo "[0/3] qa-pt shards not found, re-tokenizing (15 train + 1 val) ..."
    python -m nanochat.dataset_qa_pt -n 16 --verify 0
else
    echo ""
    echo "[0/3] qa-pt shards found in $NANOCHAT_BASE_DIR/data_qa_pt/  (skipping re-tokenization)"
fi

# ---------------------------------------------------------------------------
# [1/3] Verify loss mask (optional — pass --verify-mask as first arg)
#
# Prints 3 packed sequences with colour-coded mask (green=keep, red=mask).
# Expected fraction of kept tokens: 40-70%.  0% or 100% = bug.
# Use this to confirm the mask before committing to a full run.

if [ "${VERIFY_MASK}" = "--verify-mask" ]; then
    echo ""
    echo "[1/3] Verifying loss mask (3 sequences, then exit) ..."
    python -m scripts.sft                           \
        --init-checkpoint=d12_midtrain              \
        --depth=12                                  \
        --device-batch-size=16                      \
        --loss-mask=assistant_only                  \
        --verify-mask

    echo "Mask verified. Re-run without --verify-mask to start training."
    exit 0
fi

# ---------------------------------------------------------------------------
# [2/3] SFT
#
# Flag rationale:
#
#   --init-checkpoint=d12_midtrain
#       Load Entrega-2 weights from base_checkpoints/d12_midtrain/ (latest step).
#       Optimizer is NOT resumed — fresh AdamW.
#
#   --depth=12  --window-pattern=L
#       Must match the Entrega-2 model config exactly.
#
#   --loss-mask=assistant_only
#       THE defining SFT difference.  targets=-1 for user/structure tokens;
#       only assistant response tokens (+ <|assistant_end|>) contribute to loss.
#
#   --epochs=2
#       2 full passes over the 15 training shards (~1172 steps at 524k batch).
#       Enough to fine-tune without overfitting on this dataset size.
#
#   --embedding-lr=0.003  --unembedding-lr=0.00008
#   --matrix-lr=0.0002    --scalar-lr=0.005
#       All 10x smaller than mid-training LRs (which were already 10x smaller
#       than pre-training).  Prevents destabilizing the mid-trained representations.
#
#   --warmup-steps=50
#       Very short warmup; model is already trained twice over.
#
#   --model-tag=d12_sft
#       Output checkpoint tag.  d12 and d12_midtrain are NEVER touched.
#
#   No --fp8   : A6000 SM 86 has no FP8 tensor cores.
#   No torchrun: single process; DDP auto-disabled.

echo ""
echo "[2/3] SFT (loss mask ENABLED, 2 epochs, ~1172 steps, ~2-4 h) ..."
python -m scripts.sft                               \
    --init-checkpoint=d12_midtrain                  \
    --depth=12                                      \
    --window-pattern=L                              \
    --device-batch-size=16                          \
    --total-batch-size=524288                       \
    --optimizer=adamw                               \
    --epochs=2                                      \
    --embedding-lr=0.003                            \
    --unembedding-lr=0.00008                        \
    --matrix-lr=0.0002                              \
    --scalar-lr=0.005                               \
    --warmup-steps=50                               \
    --warmdown-ratio=0.5                            \
    --loss-mask=assistant_only                      \
    --eval-every=50                                 \
    --save-every=200                                \
    --dataset=qa_pt                                 \
    --model-tag=d12_sft                             \
    --run="$WANDB_RUN"                              \
    2>&1 | tee logs/d12_sft.log

# ---------------------------------------------------------------------------
# [3/3] Comparative samples: base (E1) vs. mid-train (E2) vs. SFT (E3)
#
# Generates responses for 5 PT-BR questions from all three checkpoints.
# SFT should produce more focused answers that terminate cleanly at
# <|assistant_end|>.  Output saved to samples_entrega3.md.

echo ""
echo "[3/3] Generating comparative samples (base vs. mid-train vs. SFT) ..."
python -m scripts.sample_entrega3                   \
    --mid-tag=d12_midtrain                          \
    --sft-tag=d12_sft                               \
    --max-tokens=200                                \
    --temperature=0.7                               \
    --output=samples_entrega3.md

echo ""
echo "========================================================================"
echo "  entrega3_sft.sh complete."
echo ""
echo "  Artifacts:"
echo "    Training log        :  logs/d12_sft.log"
echo "    Checkpoint          :  \$NANOCHAT_BASE_DIR/base_checkpoints/d12_sft/"
echo "    Comparison samples  :  samples_entrega3.md"
echo "========================================================================"
