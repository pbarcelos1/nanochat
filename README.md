# nanochat — Entrega 1: PT-Latn small language model

This is a fork of [karpathy/nanochat](https://github.com/karpathy/nanochat)
(upstream commit `0aaca56`) adapted for **Entrega 1** of a graduate deep-learning
course. The goal is to pretrain a small language model on Brazilian/European
Portuguese text, trained to convergence on a single NVIDIA RTX A6000 GPU.

The upstream README is preserved at [README_upstream.md](README_upstream.md).

---

## What changed vs upstream

| Area | Change | Justification |
|------|--------|---------------|
| **Dataset** | Added `nanochat/dataset_hf.py` — streams `HuggingFaceFW/fineweb-2 / por_Latn` from the HF Hub and writes local parquet shards with the same schema as the upstream shards. | High-quality multilingual Common Crawl data for Portuguese. No license issues; the subset is public. |
| **Tokenizer** | `--dataset=hf` flag on `scripts/tok_train.py` and `scripts/tok_eval.py` — trains a byte-level BPE tokenizer (GPT-4 style, vocab=32 768) on Portuguese text. | Portuguese is ~1.15 bytes/char (accented chars); a PT-trained tokenizer achieves ~4.5 bytes/token vs GPT-2's ~2.8 bytes/token on PT text, nearly doubling training efficiency. |
| **Dataloader** | Added `data_dir` parameter to `nanochat/dataloader.py` — threads down to `list_parquet_files`. | Allows BPB evaluation and training to read HF shards without touching the upstream data path. |
| **Optimizer** | Added `build_adamw_only_optimizer()` in `nanochat/optim.py` and `--optimizer={muon,adamw}` flag in `scripts/base_train.py`. | Muon (Newton-Schulz orthogonalization) adds conceptual complexity without a clear benefit at this scale; AdamW is the standard baseline. Muon code is retained but not used. |
| **Window pattern** | Launcher sets `--window-pattern=L`. | A6000 is Ampere SM 86; Flash Attention 3 is unavailable. The SDPA fallback builds an explicit bool mask for sliding-window layers (SSSL pattern), doubling VRAM and halving throughput. `L` uses the fast `is_causal=True` SDPA path for all layers. |
| **CORE eval** | Launcher sets `--core-metric-every=-1`. | CORE is an English ICL benchmark; results are near-random for a Portuguese-only model. Disabling it saves ~15 min per 2 000 steps. |
| **BPB eval** | `--dataset=hf` flag on `scripts/base_eval.py` — measures BPB on the PT-Latn validation shard. | Upstream eval would measure BPB on English data; meaningless for a PT model. |
| **Dynamo cache** | `torch._dynamo.config.cache_size_limit = 64` before `torch.compile` in `scripts/base_train.py`. | `evaluate_bpb` calls `model(…, loss_reduction='none')` while training calls `model(…)`, and `model.eval()` / `model.train()` switches each add cache entries. The default limit of 8 is exceeded before the first training step. |
| **Sampling script** | New `scripts/sample_entrega1.py` — loads latest checkpoint, generates 200 tokens per prompt for 5 hardcoded Portuguese prompts, writes `samples.md`. | Qualitative check for a base model (no instruction tuning). |
| **Launcher** | New `runs/entrega1.sh` — end-to-end pipeline: download → tokenize → eval tokenizer → pretrain → eval model. | Single entry point for the full Entrega 1 run. |
| **Architecture doc** | README (this file) corrects upstream description: FFN uses **relu²** (`F.relu(x).square()`), not SwiGLU. | The code in `nanochat/gpt.py` was always relu²; the upstream README description was ambiguous. We do not change the architecture. |

---

## Architecture

Model instantiated with `--depth=12` (all other hyperparameters auto-derived by nanochat's scaling-law logic):

| Hyperparameter | Value |
|----------------|-------|
| Layers (`n_layer`) | 12 |
| d_model (`n_embd`) | 768 |
| Attention heads (`n_head`) | 6 |
| KV heads (`n_kv_head`) | 6 |
| Sequence length | 2 048 |
| Vocabulary size | 32 768 |
| Window pattern | `L` (full causal, all layers) |

**Parameter counts** (from actual training run):

| Component | Parameters |
|-----------|-----------|
| `wte` (token embeddings) | 25 165 824 |
| `value_embeds` (per-layer value embedding tables) | 150 994 944 |
| `lm_head` (unembedding) | 25 165 824 |
| `transformer_matrices` (attn + FFN weight matrices) | 84 935 088 |
| scalars (λ, gate, smear params) | 50 |
| **total** | **286 261 730** |

The **scaling parameters** used for the Chinchilla token-budget formula are
`transformer_matrices + lm_head` ≈ **110 M**. At a Chinchilla ratio of 23
tokens/param, this gives ~2.53 B training tokens.

Key architecture choices (unchanged from upstream):
- **Attention**: RoPE positional encoding, GQA, sliding-window configurable via `--window-pattern`
- **FFN**: **relu²** (i.e. `F.relu(x).square()`) — *not* SwiGLU
- **Normalization**: RMSNorm (`F.rms_norm`)
- **Precision**: bf16 (auto-detected on CUDA SM 86+)

---

## Hardware

| | |
|-|-|
| GPU | 1× NVIDIA RTX A6000 (48 GB VRAM, Ampere SM 86) |
| Precision | bf16 (FP8 unavailable on SM 86) |
| Attention backend | PyTorch SDPA (`is_causal=True`; FA3 unavailable on SM 86) |
| Throughput | ~59 000 tok/s (step time ~8.9 s, batch = 16 × 2 048 × 16 grad-accum = 524 288 tok) |
| Training time | ~12–14 h for the full 4 830-step run |

---

## How to reproduce

### Prerequisites

```bash
conda activate DL          # env with torch 2.9.1+cu128, datasets, pyarrow, rustbpe
# optional:
huggingface-cli login      # not required — dataset is public
wandb login                # optional; omit WANDB_RUN to skip logging
```

### Full pipeline (one command)

```bash
# plain run (no W&B logging)
bash runs/entrega1.sh

# with W&B logging
WANDB_RUN=entrega1_d12 bash runs/entrega1.sh
```

The script runs five stages in order:

| Stage | Script | Notes |
|-------|--------|-------|
| [1/5] Download data | `nanochat/dataset_hf.py -n 55` | 55 shards × 50 000 docs ≈ 2.1 B tokens; idempotent |
| [2/5] Train tokenizer | `scripts/tok_train.py --dataset=hf` | Byte-level BPE, vocab=32 768, up to 2 B chars; ~90 s |
| [3/5] Eval tokenizer | `scripts/tok_eval.py --dataset=hf` | Compression table vs GPT-2 / GPT-4; logged to `logs/tokenizer_eval.txt` |
| [4/5] Pretrain | `scripts/base_train.py` | ~12–14 h; checkpoint saved to `$NANOCHAT_BASE_DIR/base_checkpoints/d12/` |
| [5/5] Eval model | `scripts/base_eval.py` | BPB on PT-Latn val + unconditioned samples |

### Individual stages

```bash
# Download only 2 shards for a quick smoke test
python -m nanochat.dataset_hf -n 2

# Train tokenizer on HF data
python -m scripts.tok_train --dataset=hf

# Eval tokenizer
python -m scripts.tok_eval --dataset=hf

# Pretrain (single GPU, no torchrun)
python -m scripts.base_train \
    --depth=12 \
    --target-param-data-ratio=23 \
    --device-batch-size=16 \
    --optimizer=adamw \
    --dataset=hf \
    --window-pattern=L \
    --core-metric-every=-1

# Eval model (BPB + samples, no CORE)
python -m scripts.base_eval \
    --device-batch-size=8 \
    --eval bpb,sample \
    --dataset=hf

# Generate Portuguese samples from the trained model
python -m scripts.sample_entrega1
```

---

## Results

*To be filled in after the full training run completes.*

**Tokenizer** (evaluated on `HuggingFaceFW/fineweb-2 / por_Latn`):

| Metric | Value |
|--------|-------|
| fwe-train bytes/token | 4.58 |
| fwe-val bytes/token | 4.52 |
| vs GPT-2 (PT text) | +39% compression |
| vs GPT-4 (PT text) | +21% compression |

**Base model** (d12, 4 830 steps, ~2.53 B tokens):

| Metric | Value |
|--------|-------|
| val bpb @ step 0 (random) | 3.184 |
| val bpb @ step 250 | 1.787 |
| val bpb @ step 500 | 1.304 |
| val bpb @ step 750 | 1.179 |
| val bpb @ step 1000 | 1.143 |
| val bpb @ step 1250 | 1.104 |
| val bpb @ final (step 4830) | *pending* |
| CORE metric | N/A — English benchmark, near-random for PT model |

---

## Limitations

- **Base model only** — no instruction tuning, no RLHF. Outputs are raw continuations, not responses.
- **CORE benchmark not applicable** — CORE measures English ICL accuracy; results are near-random for a Portuguese model and are not reported.
- **Single GPU** — the launcher is designed for one A6000. For multi-GPU runs, use `torchrun --nproc_per_node=N` and remove `--window-pattern=L` if running on SM 90 hardware with FA3.
- **Portuguese only** — the tokenizer and training data are exclusively PT-Latn. Cross-lingual transfer is not evaluated.

---

## Credits

- **Upstream**: [karpathy/nanochat](https://github.com/karpathy/nanochat), forked at commit `0aaca56`. All architecture, training loop, tokenizer, evaluation, and inference code is from upstream unless noted above.
- **Dataset**: [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), subset `por_Latn`. License: ODC-By 1.0.
- **Course**: Entrega 1 — PUCRS graduate deep learning course, 2026.

## License

MIT (same as upstream)
