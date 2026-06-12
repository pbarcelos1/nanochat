#!/bin/bash
# Entrega 2 — Mid-training on ju-resplande/qa-pt
#
# Continues the PT-Latn base model on a Portuguese question-answering dataset
# to teach dialog structure (chat template) before SFT.  Loss is computed over
# all tokens (no prompt mask); masking is deferred to SFT (Etapa 3).
#
# Hardware  : 1× NVIDIA RTX A6000 (48 GB VRAM, Ampere SM 86), single process.
# Token budget  : 300 M tokens  (~572 optimizer steps at batch 524 288).
# Expected wall-clock: ~2–4 h on A6000.
#
# Prerequisites:
#   conda activate DL   (or whichever env has nanochat deps + torch cu128)
#   Entrega 1 must be complete: base checkpoint at
#       $NANOCHAT_BASE_DIR/base_checkpoints/d12/
#
# Usage:
#   bash runs/entrega2_midtrain.sh
#   WANDB_RUN=entrega2_d12_midtrain bash runs/entrega2_midtrain.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/mnt/E-SSD/barcelos/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"
mkdir -p logs

WANDB_RUN="${WANDB_RUN:-dummy}"

echo "========================================================================"
echo "  entrega2_midtrain.sh  |  PT-Latn mid-training  |  qa-pt  |  single A6000"
echo "  NANOCHAT_BASE_DIR : $NANOCHAT_BASE_DIR"
echo "  WANDB_RUN         : $WANDB_RUN"
echo "  CUDA device       : $CUDA_VISIBLE_DEVICES"
echo "========================================================================"

# ---------------------------------------------------------------------------
# [1/3] Pre-tokenize qa-pt shards
#
# Downloads 16 shards from ju-resplande/qa-pt (streaming, CC0 licence).
# Filters accepted answers, dedupes, applies chat template, packs into
# fixed-length token blocks and writes parquet to $NANOCHAT_BASE_DIR/data_qa_pt/.
# 16 shards = 15 train + 1 val ≈ 307 M training tokens.
# Idempotent: skips shards already on disk.
# --verify 10: prints 10 decoded examples so we can inspect the template.

echo ""
echo "[1/3] Pre-tokenizing qa-pt (16 shards, 15 train + 1 val ≈ 307 M tokens) …"
python -m nanochat.dataset_qa_pt -n 16 --verify 10

# ---------------------------------------------------------------------------
# [2/3] Mid-train
#
# Flag rationale:
#
#   --base-checkpoint=d12
#       Load Entrega-1 weights from base_checkpoints/d12/ (latest step).
#
#   --depth=12  --window-pattern=L
#       Must match the Entrega-1 model config exactly.
#
#   --device-batch-size=16  --total-batch-size=524288
#       Same as Entrega 1.  grad_accum = 524288 / (16×2048) = 16.
#
#   --optimizer=adamw
#       AdamW throughout (no Muon), consistent with Entrega 1.
#
#   --target-tokens=3e8
#       300 M token budget → ~572 optimizer steps.
#
#   --embedding-lr=0.03  --unembedding-lr=0.0008
#   --matrix-lr=0.002    --scalar-lr=0.05
#       All 10× smaller than the Entrega-1 base LRs (muP-scaled).
#       Justified: fine-tuning on a much smaller dataset with a pre-trained
#       model; large LRs would destabilise the existing PT-BR representation.
#
#   --warmup-steps=100
#       Short warmup because we are already in a trained model; avoids
#       overshooting the optimal LR immediately.
#
#   --eval-every=50  --save-every=100
#       Frequent enough for a ~572-step run to get a useful loss curve.
#
#   --dataset=qa_pt
#       Reads pre-tokenized blocks from $NANOCHAT_BASE_DIR/data_qa_pt/.
#
#   No --fp8   : A6000 SM 86 has no FP8 tensor cores.
#   No torchrun: single process; DDP auto-disabled.

echo ""
echo "[2/3] Mid-training (300 M tokens, ~572 steps, ~2–4 h) …"
python -m scripts.mid_train                   \
    --base-checkpoint=d12                     \
    --depth=12                                \
    --window-pattern=L                        \
    --device-batch-size=16                    \
    --total-batch-size=524288                 \
    --optimizer=adamw                         \
    --target-tokens=3e8                       \
    --embedding-lr=0.03                       \
    --unembedding-lr=0.0008                   \
    --matrix-lr=0.002                         \
    --scalar-lr=0.05                          \
    --warmup-steps=100                        \
    --warmdown-ratio=0.65                     \
    --eval-every=50                           \
    --save-every=100                          \
    --dataset=qa_pt                           \
    --model-tag=d12_midtrain                  \
    --run="$WANDB_RUN"                        \
    2>&1 | tee logs/d12_midtrain.log

# ---------------------------------------------------------------------------
# [3/3] Sample comparison: base (Entrega 1) vs. mid-trained (Entrega 2)
#
# Outputs samples_entrega2.md with side-by-side answers to 5 PT-BR questions.
# The last prompt (dog care) is in-distribution for qa-pt and expected to show
# the clearest improvement in answer structure.

echo ""
echo "[3/3] Generating comparison samples …"
python -m scripts.sample_entrega2            \
    --mid-tag=d12_midtrain                   \
    --max-tokens=200                         \
    --temperature=0.7                        \
    --output=samples_entrega2.md

echo ""
echo "========================================================================"
echo "  entrega2_midtrain.sh complete."
echo ""
echo "  Artifacts:"
echo "    Pre-tokenized data  :  \$NANOCHAT_BASE_DIR/data_qa_pt/"
echo "    Training log        :  logs/d12_midtrain.log"
echo "    Checkpoint          :  \$NANOCHAT_BASE_DIR/base_checkpoints/d12_midtrain/"
echo "    Comparison samples  :  samples_entrega2.md"
echo "========================================================================"
