#!/bin/bash
# Entrega 4 — BLUEX multiple-choice evaluation for d12, d12_midtrain, d12_sft.
# Scores each checkpoint via log-likelihood (sum and mean normalisation).
# d12 uses plain prompt format; d12_midtrain and d12_sft use the chat template.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/E-SSD/barcelos/.cache/nanochat}"
export OMP_NUM_THREADS=1

LOG=logs/eval_bluex.log
mkdir -p logs
: > "$LOG"  # truncate / create

RESULTS_FILE=results_bluex.md

echo "# BLUEX Evaluation Results" > "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "Benchmark: BLUEX (portuguese-benchmark-datasets/BLUEX)" >> "$RESULTS_FILE"
echo "Method: log-likelihood scoring (no autoregressive generation)" >> "$RESULTS_FILE"
echo "Date: $(date -u '+%Y-%m-%d %H:%M UTC')" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "| checkpoint | prompt_format | acc_sum | acc_mean | n_scored | baseline |" >> "$RESULTS_FILE"
echo "|-----------|--------------|---------|----------|----------|----------|" >> "$RESULTS_FILE"

for ckpt in d12 d12_midtrain d12_sft; do
    echo "======================================================" | tee -a "$LOG"
    echo "Evaluating checkpoint: $ckpt" | tee -a "$LOG"
    echo "======================================================" | tee -a "$LOG"

    output=$(conda run -n DL python -m scripts.eval \
        --checkpoint "$ckpt" \
        --task mc \
        --benchmark bluex \
        2>&1 | tee -a "$LOG")

    # Extract result line
    acc_sum=$(echo "$output" | grep "Accuracy (sum)" | awk '{print $NF}')
    acc_mean=$(echo "$output" | grep "Accuracy (mean)" | awk '{print $NF}')
    n_scored=$(echo "$output" | grep "Questions scored" | awk '{print $NF}')
    baseline=$(echo "$output" | grep "Baseline random" | awk '{print $NF}')

    if [ "$ckpt" = "d12" ]; then
        fmt="plain"
    else
        fmt="chat"
    fi

    echo "| $ckpt | $fmt | $acc_sum | $acc_mean | $n_scored | $baseline |" >> "$RESULTS_FILE"

    echo "" | tee -a "$LOG"
done

echo "" >> "$RESULTS_FILE"
echo "_Accuracy is reported over the text-only subset (questions without associated images excluded)._" >> "$RESULTS_FILE"

echo "======================================================" | tee -a "$LOG"
echo "Done. Results written to $RESULTS_FILE" | tee -a "$LOG"
echo "======================================================" | tee -a "$LOG"

cat "$RESULTS_FILE"
