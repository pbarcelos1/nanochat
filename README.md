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

## Limitations (Entrega 1)

- **Base model only** — no instruction tuning, no RLHF. Outputs are raw continuations, not responses.
- **CORE benchmark not applicable** — CORE measures English ICL accuracy; results are near-random for a Portuguese model and are not reported.
- **Single GPU** — the launcher is designed for one A6000. For multi-GPU runs, use `torchrun --nproc_per_node=N` and remove `--window-pattern=L` if running on SM 90 hardware with FA3.
- **Portuguese only** — the tokenizer and training data are exclusively PT-Latn. Cross-lingual transfer is not evaluated.

---

---

## Entrega 2 — Mid-Training

### What is mid-training in nanochat's pipeline?

Mid-training sits between base pre-training (plain web text, next-token LM) and
supervised fine-tuning (SFT, which uses prompt masking).  It does three things:

1. **Continues language modelling on higher-quality, structured data** — transitioning the model from raw web continuations to a Q&A style.
2. **Introduces the chat template** (special tokens) so the model learns the `<|user_start|>…<|assistant_start|>…` structure before SFT expects it.
3. **Uses a lower learning rate** than pre-training (~10×) to preserve the PT-BR representations acquired in Entrega 1.

### Loss: computed over all tokens (no prompt mask)

In mid-training **every token in the packed sequence contributes to the loss** — both the user/question span and the assistant/answer span.  There is no `-100` ignore-index mask here.  Masking the prompt so that only assistant tokens are supervised is the defining feature of **SFT (Etapa 3)** and is deliberately deferred.  The model still _learns the template_ because special tokens appear in the text; it just also learns to predict the question span, which is harmless at this stage.

This matches the course specification, which places the prompt mask in SFT, not mid-training.

### Canonical chat template

Derived from `SPECIAL_TOKENS` in `nanochat/tokenizer.py`:

```
<|bos|><|user_start|>{question_title}<|user_end|><|assistant_start|>{answer_text}<|assistant_end|>
```

**Note:** there is no `<|end|>` token in this tokenizer.  `<|bos|>` acts as the document-boundary delimiter: its presence at the start of each example simultaneously marks the end of the preceding one in packed sequences.

The same template is used identically in training (`nanochat/dataset_qa_pt.py`) and in sampling (`scripts/sample_entrega2.py`), so there is no train/sample template mismatch.

For generation (inference), the assistant span is left open:
```
<|bos|><|user_start|>{question}<|user_end|><|assistant_start|>
```
and generation stops when `<|assistant_end|>` or `<|bos|>` is produced (handled automatically by `Engine.generate_batch`).

### Dataset: ju-resplande/qa-pt

| Property | Value |
|---|---|
| HF path | `ju-resplande/qa-pt` |
| License | CC0-1.0 (no restrictions) |
| Source | Portuguese split of `clips/mqa` (community Q&A from PT websites) |
| Total rows | ~5.6 M |
| Filters applied | `is_accepted==True`, `len(question_title)>=10`, `50<=len(answer_text)<=4000`, dedupe by exact title (cap 2M) |
| Estimated post-filter | ~1.5–1.8 M unique Q&A pairs |

**Filters are applied in a single streaming pass** (no full download required).

#### Why only qa-pt — deviation from upstream data mix

The upstream nanochat mid-training mix (SmolTalk + GSM8K + MMLU) is almost entirely
**English**.  Our base model was pre-trained exclusively on **PT-BR** (FineWeb-2
`por_Latn`).  Fine-tuning on English data would introduce a language shift and
waste the Portuguese representations.  `qa-pt` keeps the model in-language and
teaches the question→answer dialog shape needed before SFT.

The course specification explicitly allows an alternative data mix with justification;
this is ours.

#### Known limitations and risks for Etapa 4

| Risk | Impact |
|---|---|
| **No math/reasoning data** | GSM8K-style arithmetic reasoning is absent. The mandatory Etapa 4 benchmarks (HellaSwag, ARC, PIQA) test commonsense and science reasoning — this may hurt those scores. |
| **Domain skew** | `qa-pt` is heavily weighted toward commercial/pet/product Q&A (clubedosanimais.com.br appears frequently). The model will be better at pet-care questions than at physics or history. |
| **Benchmark language** | HellaSwag/ARC/PIQA are primarily English; the PT-BR mid-training may not improve — or could hurt — English zero-shot scores. |
| **Candidate fix** | At SFT time (Etapa 3), add a small slice of a PT-BR reasoning/QA source (e.g., translated GSM8K-pt or a PT-BR commonsense dataset) to partially compensate. |

### Hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| Token budget | 300 M | ~1/7 of Entrega 1; mid-training is much shorter |
| Optimizer | AdamW | Consistent with Entrega 1; no Muon |
| embedding_lr | 0.03 | 10× smaller than pretrain (0.3) |
| unembedding_lr | 0.0008 | 10× smaller than pretrain (0.008) |
| matrix_lr | 0.002 | 10× smaller than pretrain (0.02) |
| scalar_lr | 0.05 | 10× smaller than pretrain (0.5) |
| Warmup steps | 100 | Short; model is already trained |
| Warmdown ratio | 0.65 | 65% of steps for LR decay (same as pretrain) |
| final_lr_frac | 0.05 | Same as pretrain |
| device_batch_size | 16 | Same as pretrain; fits A6000 48 GB |
| total_batch_size | 524 288 | Same as pretrain (~16× grad accum) |
| max_seq_len | 2048 | Same as pretrain |
| Optimizer steps | ~572 | 300M / 524288 |
| Estimated wall-clock | ~2–4 h | On 1× A6000 |

All LRs are muP-scaled (same reference dimension d_model=768 as pretrain, so no dmodel_lr_scale correction needed).

### Sequence packing

Pre-tokenized QA pairs are concatenated into fixed-length blocks of `seq_len+1 = 2049` tokens during dataset preparation (`nanochat/dataset_qa_pt.py`).  This means:

- **Zero padding**: every token in every training batch is a real token.
- **Document boundaries** are marked by `<|bos|>` at the start of each QA pair.
- Approximately 35% of tokens are from a different QA pair than the `<|bos|>` at the start of the row (same as upstream's pre-train packing behaviour).

### New files

| File | Purpose |
|---|---|
| `nanochat/dataset_qa_pt.py` | Stream, filter, tokenize, pack, write qa-pt parquet shards; provides `dataloader_qa_pt` for training |
| `scripts/mid_train.py` | Mid-training loop (adapted from `base_train.py`): loads base checkpoint, fresh AdamW, no prompt mask |
| `runs/entrega2_midtrain.sh` | Full pipeline launcher: pre-tokenize → mid-train → sample comparison |
| `scripts/sample_entrega2.py` | Side-by-side sampling from base model (no template) vs. mid-trained model (chat template) |

### Reproduction

```bash
# Set environment
export CUDA_VISIBLE_DEVICES=3
export NANOCHAT_BASE_DIR="/mnt/E-SSD/barcelos/.cache/nanochat"

# Run (no wandb)
bash runs/entrega2_midtrain.sh

# Run with wandb
WANDB_RUN=entrega2_d12_midtrain bash runs/entrega2_midtrain.sh
```

Or step by step:

```bash
# 1. Pre-tokenize (one-time, ~10–20 min)
python -m nanochat.dataset_qa_pt -n 16

# 2. Mid-train (~2–4 h)
python -m scripts.mid_train -- \
    --base-checkpoint=d12 --depth=12 --window-pattern=L \
    --device-batch-size=16 --total-batch-size=524288 \
    --optimizer=adamw --target-tokens=3e8 \
    --embedding-lr=0.03 --unembedding-lr=0.0008 \
    --matrix-lr=0.002 --scalar-lr=0.05 \
    --warmup-steps=100 --eval-every=50 --save-every=100 \
    --dataset=qa_pt --model-tag=d12_midtrain \
    2>&1 | tee logs/d12_midtrain.log

# 3. Compare samples
python -m scripts.sample_entrega2 --mid-tag=d12_midtrain --output=samples_entrega2.md
```

### Results (placeholder)

Training log: `logs/d12_midtrain.log`

| Metric | Value |
|---|---|
| Train loss @ start | *pending* |
| Train loss @ end | *pending* |
| Val bpb @ start | *pending* |
| Val bpb @ end | *pending* |
| Tokens trained | *pending* |
| Wall-clock time | *pending* |

Validation PPL curve: *pending* (plot from `logs/d12_midtrain.log` wandb data).

Comparison samples: see `samples_entrega2.md`.

---

## Limitations (Entrega 2, cumulative)

- **No reasoning data** — see "Known limitations" above; Etapa 4 benchmarks may be weak.
- **Domain-skewed QA** — pet/commercial bias from `qa-pt`; not a general-purpose assistant yet.
- **No RLHF, no system prompt** — SFT and RLHF are future Etapas.
- All Entrega 1 limitations continue to apply.

---

---

## Entrega 3 — SFT (Supervised Fine-Tuning)

### The central difference: loss mask

In mid-training (Entrega 2), every token in the packed sequence contributes to
the loss.  In SFT, **only assistant-response tokens are supervised**.  Prompt
tokens (`<|user_start|>…<|user_end|><|assistant_start|>`) receive `targets=-1`,
which `F.cross_entropy(ignore_index=-1)` ignores.

```
<|bos|><|user_start|> q q q <|user_end|><|assistant_start|> a a a <|assistant_end|>
  MASK   MASK           MASK   MASK         MASK              KEEP   KEEP           KEEP
```

**Edge case — `<|assistant_end|>`**: kept in the loss (`mask=1`), same as the
upstream `tokenizer.render_conversation`.  The model must learn to generate the
termination token; masking it would produce a model that never stops.

**Implementation**: `nanochat/dataset_qa_pt.py:dataloader_qa_pt_sft` scans each
packed block for `<|assistant_start|>` / `<|assistant_end|>` token IDs and builds
the per-token mask inline, carrying `in_assistant` state across block boundaries
so conversations split at a block cut are masked correctly.  The function is
purely additive — `dataloader_qa_pt` (Entrega 2) is unchanged.

### Dataset

Same `ju-resplande/qa-pt` parquet shards as Entrega 2 (no re-download).
Split: 15 train shards + 1 val shard (idempotent).

**Effective epochs per source:**

| Source | Shards | Train tokens | Steps/epoch | Epochs |
|---|---|---|---|---|
| qa-pt | 15 of 15 | ~307 M | ~586 | 2 |

GSM8K-PT was not used (no PT-BR loader existed; the Entrega 2 mid-training also
used qa-pt only; see Entrega 2 rationale).

**Fraction of non-masked tokens:** ~75–85% for qa-pt (respostas ~200 tokens vs. perguntas ~20–50 tokens; assistant-side dominates).
Logged at step 0 and verified via `--verify-mask` before the full run.

### Hyperparameters

| Parameter | SFT (Entrega 3) | Mid-train (Entrega 2) | Pre-train (Entrega 1) |
|---|---|---|---|
| Loss mask | **assistant_only** | none | none |
| Optimizer | AdamW | AdamW | AdamW |
| embedding_lr | **0.003** | 0.03 | 0.3 |
| unembedding_lr | **0.00008** | 0.0008 | 0.008 |
| matrix_lr | **0.0002** | 0.002 | 0.02 |
| scalar_lr | **0.005** | 0.05 | 0.5 |
| Warmup steps | **50** | 100 | — |
| Warmdown ratio | 0.5 | 0.65 | — |
| final_lr_frac | 0.05 | 0.05 | — |
| device_batch_size | 16 | 16 | 16 |
| total_batch_size | 524 288 | 524 288 | 524 288 |
| max_seq_len | 2 048 | 2 048 | 2 048 |
| Epochs | **2** | 1 (~572 steps) | — |
| Optimizer steps | **~1172** | ~572 | 4 830 |

All LRs are 10× smaller than mid-train, which were 10× smaller than pre-train.
Justification: fine-tuning at each stage uses smaller LRs to preserve
representations acquired in the previous stage.

### Packing + loss mask

The parquet shards from Entrega 2 store pre-tokenized packed blocks
(`seq_len+1 = 2049` tokens each).  `dataloader_qa_pt_sft` reads these blocks and
builds the mask from special token IDs — no re-packing, no re-download.  The
`in_assistant` flag is carried across block boundaries, so a conversation split at
a block cut is correctly masked on both sides.

### New files

| File | Purpose |
|---|---|
| `nanochat/dataset_qa_pt.py` | Added `dataloader_qa_pt_sft` (additive; existing loader unchanged) |
| `scripts/sft.py` | SFT training loop: loads d12_midtrain, loss mask, saves d12_sft |
| `runs/entrega3_sft.sh` | Full pipeline: verify mask → SFT → comparative samples |
| `scripts/sample_entrega3.py` | 3-way comparison: base vs. mid-train vs. SFT |
| `scripts/eval.py` | Scaffold (Entrega 3) — implementado em Entrega 4 com BLUEX log-likelihood |

### Reproduction

```bash
export CUDA_VISIBLE_DEVICES=3
export NANOCHAT_BASE_DIR="/mnt/E-SSD/barcelos/.cache/nanochat"

# Full pipeline
bash runs/entrega3_sft.sh

# With W&B logging
WANDB_RUN=entrega3_d12_sft bash runs/entrega3_sft.sh

# Verify mask before committing to full run (~30 s, no GPU needed for training)
bash runs/entrega3_sft.sh --verify-mask
```

Or step by step:

```bash
# 1. Verify mask (inspect 3 sequences, confirm 40-70% kept fraction, then exit)
python -m scripts.sft --init-checkpoint=d12_midtrain --verify-mask

# 2. Smoke test (100 steps; confirm loss drops and mask fraction is sane)
python -m scripts.sft                               \
    --init-checkpoint=d12_midtrain --depth=12       \
    --num-iterations=100 --device-batch-size=16     \
    --optimizer=adamw --dataset=qa_pt               \
    --loss-mask=assistant_only --run=dummy

# 3. Full SFT run (~2-4 h)
python -m scripts.sft                               \
    --init-checkpoint=d12_midtrain --depth=12       \
    --window-pattern=L --device-batch-size=16       \
    --total-batch-size=524288 --optimizer=adamw     \
    --epochs=2                                      \
    --embedding-lr=0.003 --unembedding-lr=0.00008   \
    --matrix-lr=0.0002 --scalar-lr=0.005            \
    --warmup-steps=50 --eval-every=50               \
    --loss-mask=assistant_only                      \
    --dataset=qa_pt --model-tag=d12_sft             \
    2>&1 | tee logs/d12_sft.log

# 4. Comparative samples
python -m scripts.sample_entrega3 --output=samples_entrega3.md
```

### Results (placeholders)

Training log: `logs/d12_sft.log`

| Metric | Value |
|---|---|
| Mask fraction non-masked (step 0) | *pending* (expected 40–70%) |
| Train loss @ start | *pending* |
| Train loss @ end | *pending* |
| Val bpb @ start (from d12_midtrain) | 0.6753 |
| Val bpb @ end | *pending* |
| Overfitting observed? | *pending* (watch val bpb vs. train loss curves) |
| Wall-clock time | *pending* |

**Val PPL curve**: logged every 50 steps to `logs/d12_sft.log` and W&B
(`val/bpb`, `val/ppl`).  A rising val bpb while train loss falls would indicate
overfitting; reduce epochs or add weight decay.

**Comparison samples**: see `samples_entrega3.md` — SFT responses expected to be
more focused and terminate cleanly at `<|assistant_end|>`, in contrast to the
mid-train model which sometimes continues past the answer.

---

---

## Entrega 4 — Avaliação: BLUEX (múltipla escolha)

### Método: log-likelihood (sem geração autorregressiva)

Para cada questão elegível, e para cada alternativa *i*, calcula-se a
log-verossimilhança da alternativa dado o contexto (enunciado):

```
score_i = Σ log P(token_j | contexto, token_{<j})  para j em tokens da alternativa i
```

O forward pass é feito sobre a sequência `contexto + continuação` inteira; os
log-probs são extraídos apenas nas posições da continuação (as posições do
contexto não entram na soma). A predição é `argmax_i(score_i)`. Nenhuma
geração autorregressiva é necessária — evita-se dependência de parsing de saída.

**Duas normalizações** são reportadas:

| Variante | Fórmula | Quando preferir |
|----------|---------|-----------------|
| `acc_sum` | Σ log P (não normalizado) | Tradição lm-eval; favorece alternativas curtas |
| `acc_mean` | (Σ log P) / nº tokens | `acc_norm` do lm-eval-harness; mais justo quando as alternativas têm comprimentos diferentes |

Ambas são reportadas — a escolha da métrica "oficial" fica para a apresentação.

### Dataset e filtros

Benchmark: [BLUEX](https://huggingface.co/datasets/portuguese-benchmark-datasets/BLUEX)
(questões de vestibulares brasileiros — USP, UNICAMP, etc.). Paper: arXiv:2307.05410.

**Acurácia reportada exclusivamente sobre o subconjunto text-only.**
Questões com `has_associated_images == True` (~43%) são excluídas pois o modelo
é puramente textual e não pode processar as imagens associadas.

| Filtro | Questões restantes | Excluídas |
|--------|--------------------|-----------|
| Total inicial | 1 422 | — |
| F1: `has_associated_images == False` | 812 | 610 |
| F2: `len(alternatives) >= 2` | 812 | 0 |
| F3: `answer` normalizado e índice válido | **809** | 3 (answer = `None`) |

As 3 questões descartadas no F3 são: `UNICAMP_2021_48_day2`, `UNICAMP_2019_60`,
`UNICAMP_2025_53` — todas com campo `answer` ausente no dataset.

### Formato de contexto por checkpoint

Cada checkpoint é avaliado no formato que ele espera, para uma comparação justa no
formato nativo de cada etapa de treino. Comparar checkpoints sob formatos diferentes
é **intencional** — é a ressalva metodológica desta avaliação.

| Checkpoint | Formato | Contexto |
|------------|---------|----------|
| `d12` | `plain` | `<bos>{question}\nResposta:` |
| `d12_midtrain` | `chat` | `<bos><\|user_start\|>{question}<\|user_end\|><\|assistant_start\|>` |
| `d12_sft` | `chat` | idem |

A continuação (texto da alternativa sem o prefixo de letra) é concatenada ao
contexto. O prefixo é removido com `re.sub(r"^\s*[a-eA-E]\s*[\)\.\-:]\s*", "", alt)`.

### Resultados

| Checkpoint | Formato | acc\_sum | acc\_mean | n\_scored | Baseline aleatório |
|-----------|---------|---------|----------|----------|-------------------|
| `d12` | plain | 0.2200 | 0.2101 | 809 | 0.2260 |
| `d12_midtrain` | chat | 0.2163 | 0.2138 | 809 | 0.2260 |
| `d12_sft` | chat | 0.2151 | **0.2287** | 809 | 0.2260 |

O baseline aleatório (0.226) é calculado como `mean(1/k)` sobre as 809 questões
elegíveis (questões de 4 e 5 alternativas).

Script completo: `runs/entrega4_eval_bluex.sh`. Tabela completa: `results_bluex.md`.
Log: `logs/eval_bluex.log`.

### Discussão honesta

Os três checkpoints ficam **próximos ao acaso** no BLUEX — resultado esperado e
honesto para um SLM de ~85M parâmetros (110M scaling params) num benchmark de
vestibular de alta dificuldade:

- BLUEX envolve raciocínio complexo, conhecimento enciclopédico e, com frequência,
  figuras e gráficos (excluídos aqui). Modelos de escala muito maior (7B+) também
  ficam abaixo de 50% neste benchmark.
- O modelo foi treinado em texto web genérico PT-BR (`qa-pt`) — sem dados de
  raciocínio ou de vestibular.
- O valor desta avaliação está em **comparar a tendência entre etapas**, não no
  número absoluto. O `d12_sft` apresenta o maior `acc_mean` (0.2287 vs. baseline
  0.2260), sinal fraco mas positivo de que o alinhamento via SFT melhora
  ligeiramente a capacidade de ranquear respostas por log-verossimilhança.

### Reprodução

```bash
export CUDA_VISIBLE_DEVICES=1
export NANOCHAT_BASE_DIR="/mnt/E-SSD/barcelos/.cache/nanochat"

# Smoke test (50 questões, d12_sft)
python -m scripts.eval --checkpoint d12_sft --task mc --benchmark bluex --limit 50

# Rodada completa (3 checkpoints, ~809 questões cada)
bash runs/entrega4_eval_bluex.sh
```

### Citação do dataset

```bibtex
@article{rodrigues2023bluex,
  title   = {BLUEX: A Challenge Dataset of Brazilian University Entrance Examinations},
  author  = {Rodrigues, Jo{\~a}o Andrade and Boaro, Alex Sandro and
             Ciferri, Cristina Dutra de Aguiar and Ciferri, Ricardo Rodrigues de Aguiar},
  journal = {arXiv preprint arXiv:2307.05410},
  year    = {2023}
}
```

---

## Credits

- **Upstream**: [karpathy/nanochat](https://github.com/karpathy/nanochat), forked at commit `0aaca56`. All architecture, training loop, tokenizer, evaluation, and inference code is from upstream unless noted above.
- **Dataset (Entrega 1)**: [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), subset `por_Latn`. License: ODC-By 1.0.
- **Dataset (Entrega 2)**: [ju-resplande/qa-pt](https://huggingface.co/datasets/ju-resplande/qa-pt). License: CC0-1.0.
- **Course**: Entrega 2 — PUCRS graduate deep learning course, 2026.

## License

MIT (same as upstream)
