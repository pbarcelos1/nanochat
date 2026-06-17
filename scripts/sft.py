"""
Supervised fine-tuning (SFT) — Etapa 3.

Key differences from scripts/mid_train.py (Etapa 2):
  - Loads d12_midtrain (mid-train checkpoint); AdamW optimizer is always fresh.
  - Loss mask ENABLED: only assistant-response tokens contribute to loss.
    targets=-1 for prompt / structural tokens (cross_entropy ignores -1).
  - LRs 10x smaller than mid-training (already 10x smaller than pre-training).
  - Same qa-pt parquet shards, read with per-token mask via dataloader_qa_pt_sft.
  - Saves checkpoint to base_checkpoints/d12_sft.

Usage:
    python -m scripts.sft \\
        --init-checkpoint=d12_midtrain --depth=12 \\
        --device-batch-size=16 --optimizer=adamw \\
        --epochs=2 --dataset=qa_pt --loss-mask=assistant_only \\
        --run=entrega3_d12_sft 2>&1 | tee logs/d12_sft.log

    # Verify loss mask (3 sequences with colour-coded mask, then exit):
    python -m scripts.sft --init-checkpoint=d12_midtrain --verify-mask
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict

import wandb
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import (
    compute_init, compute_cleanup, print0, DummyWandb, print_banner,
    get_base_dir, autodetect_device_type, get_peak_flops,
    COMPUTE_DTYPE, COMPUTE_DTYPE_REASON,
)
from nanochat.optim import build_adamw_only_optimizer
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint, find_last_step
from nanochat.loss_eval import evaluate_bpb, evaluate_perplexity
from nanochat.flash_attention import HAS_FA3
from nanochat.dataset_qa_pt import (
    DATA_DIR_QA_PT, BLOCKS_PER_SHARD,
    list_parquet_files_qa_pt, dataloader_qa_pt_sft,
)
print_banner()

# ---------------------------------------------------------------------------
# CLI

parser = argparse.ArgumentParser(description="SFT — Etapa 3")

# Logging
parser.add_argument("--run",          type=str, default="dummy",
                    help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type",  type=str, default="",
                    help="cuda|cpu|mps (empty = autodetect)")
# Init checkpoint (source weights)
parser.add_argument("--init-checkpoint",      type=str, default="d12_midtrain",
                    help="Tag of the mid-train checkpoint to load weights from")
parser.add_argument("--init-checkpoint-step", type=int, default=-1,
                    help="Step of the init checkpoint (-1 = latest)")
# Output tag (never overwrites d12 or d12_midtrain)
parser.add_argument("--model-tag",    type=str, default="d12_sft",
                    help="Tag (directory name) for the SFT checkpoint output")
# Architecture — must match the init checkpoint exactly
parser.add_argument("--depth",         type=int, default=12)
parser.add_argument("--aspect-ratio",  type=int, default=64)
parser.add_argument("--head-dim",      type=int, default=128)
parser.add_argument("--window-pattern",type=str, default="L")
# Training horizon
parser.add_argument("--epochs",         type=int, default=2,
                    help="Epochs over qa-pt train set (default 2)")
parser.add_argument("--num-iterations", type=int, default=-1,
                    help="Override epoch-derived step count (-1 = derive from --epochs)")
# Batch
parser.add_argument("--max-seq-len",        type=int, default=2048)
parser.add_argument("--device-batch-size",  type=int, default=16)
parser.add_argument("--total-batch-size",   type=int, default=524288)
# Optimizer
parser.add_argument("--optimizer",      type=str, default="adamw", choices=["adamw"])
# Learning rates — defaults are 10x smaller than mid-training
parser.add_argument("--embedding-lr",   type=float, default=0.003,
                    help="LR for embedding params (mid-train was 0.03)")
parser.add_argument("--unembedding-lr", type=float, default=0.00008,
                    help="LR for unembedding params (mid-train was 0.0008)")
parser.add_argument("--matrix-lr",      type=float, default=0.0002,
                    help="LR for transformer matrix params (mid-train was 0.002)")
parser.add_argument("--scalar-lr",      type=float, default=0.005,
                    help="LR for scalar params (mid-train was 0.05)")
# LR schedule (linear warmup → flat → linear warmdown)
parser.add_argument("--warmup-steps",   type=int,   default=50,
                    help="Linear warmup steps (short; model already trained)")
parser.add_argument("--warmdown-ratio", type=float, default=0.5,
                    help="Fraction of iterations for LR warmdown")
parser.add_argument("--final-lr-frac",  type=float, default=0.05,
                    help="Final LR as fraction of peak LR")
# Loss
parser.add_argument("--loss-mask",      type=str, default="assistant_only",
                    choices=["assistant_only"],
                    help="assistant_only: only assistant response tokens in loss")
# Evaluation
parser.add_argument("--eval-every",     type=int, default=50,
                    help="Evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens",    type=int, default=10_485_760,
                    help="Tokens for val bpb evaluation (~10M)")
# Checkpointing
parser.add_argument("--save-every",     type=int, default=200,
                    help="Save checkpoint every N steps (-1 = only at end)")
# Dataset
parser.add_argument("--dataset",        type=str, default="qa_pt", choices=["qa_pt"])
# Diagnostic
parser.add_argument("--verify-mask",    action="store_true",
                    help="Print 3 packed sequences with mask annotations then exit")

args = parser.parse_args()
user_config = vars(args).copy()

# ---------------------------------------------------------------------------
# Compute init

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
synchronize    = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops  = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float("inf")

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = (
    DummyWandb() if use_dummy_wandb
    else wandb.init(project="nanochat-sft", name=args.run, config=user_config)
)

if not HAS_FA3:
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback.")

# ---------------------------------------------------------------------------
# Tokenizer (frozen from Entrega 1)

tokenizer   = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size  = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# ---------------------------------------------------------------------------
# Verify mask mode: show 3 packed sequences and exit before model setup

if args.verify_mask:
    asst_start_id = tokenizer.encode_special("<|assistant_start|>")
    asst_end_id   = tokenizer.encode_special("<|assistant_end|>")
    print0("\n" + "=" * 72)
    print0("VERIFY MASK: 3 packed sequences from the training set")
    print0("  GREEN = KEEP (assistant response tokens, contribute to loss)")
    print0("  RED   = MASK (prompt / structure tokens, ignored)")
    print0("=" * 72)
    loader = dataloader_qa_pt_sft(
        DATA_DIR_QA_PT, "train",
        args.device_batch_size, args.max_seq_len, device, tokenizer,
    )
    total_kept, total_toks = 0, 0
    for seq_i in range(3):
        x, y, state = next(loader)
        row_ids = x[0].cpu().tolist()    # input tokens, positions 0..T-1
        row_y   = y[0].cpu().tolist()    # targets, -1 = masked
        kept    = sum(1 for v in row_y if v >= 0)
        total_kept += kept
        total_toks += len(row_y)
        frac = kept / len(row_y)
        print0(
            f"\n--- seq {seq_i + 1} | ep:{state['epoch']} "
            f"pq:{state['pq_idx']} rg:{state['rg_idx']} ---"
        )
        print0(f"  tokens: {len(row_ids)} | kept: {kept} ({frac:.1%})")
        # Build visualisation mask from the actual y tensor (correct cross-block state).
        # y[i] >= 0  →  token x[i+1] is kept; shift back by 1 to align with x[i].
        # Position 0 is approximated as the same mask as position 1 (one token off,
        # but accurate for all other positions regardless of cross-block state).
        viz_mask = [1 if v >= 0 else 0 for v in row_y]       # mask for positions 1..T
        viz_mask = [viz_mask[0]] + viz_mask[:-1]              # shift: align with x[0..T-1]
        viz = tokenizer.visualize_tokenization(row_ids, viz_mask)
        print0(viz[:700] + (" ..." if len(viz) > 700 else ""))
    print0(
        f"\nOverall fraction kept (3 seqs × row 0): "
        f"{total_kept}/{total_toks} = {total_kept / total_toks:.1%}"
    )
    print0("Expected range: 40–90% (qa-pt: long answers → typically ~80%)  |  0% or 100% → BUG")
    print0("=" * 72)
    compute_cleanup()
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Build model skeleton (same architecture as init checkpoint)

base_dim     = args.depth * args.aspect_ratio
model_dim    = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads    = model_dim // args.head_dim
model_config = GPTConfig(
    sequence_len   = args.max_seq_len,
    vocab_size     = vocab_size,
    n_layer        = args.depth,
    n_head         = num_heads,
    n_kv_head      = num_heads,
    n_embd         = model_dim,
    window_pattern = args.window_pattern,
)
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")

with torch.device("meta"):
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()

# ---------------------------------------------------------------------------
# Load init checkpoint (weights only — optimizer is NOT resumed)

base_dir        = get_base_dir()
init_ckpt_dir   = os.path.join(base_dir, "base_checkpoints", args.init_checkpoint)
output_ckpt_dir = os.path.join(base_dir, "base_checkpoints", args.model_tag)

assert args.model_tag not in {"d12", "d12_midtrain"}, (
    f"--model-tag={args.model_tag} would overwrite an existing checkpoint. Use d12_sft."
)

try:
    init_step = (
        find_last_step(init_ckpt_dir)
        if args.init_checkpoint_step == -1
        else args.init_checkpoint_step
    )
except FileNotFoundError:
    raise RuntimeError(
        f"Init checkpoint not found: {init_ckpt_dir}\n"
        f"Run Entrega 2 first (runs/entrega2_midtrain.sh)."
    )

print0(f"Loading init checkpoint: {args.init_checkpoint} @ step {init_step}")
model_data, _, init_meta = load_checkpoint(
    init_ckpt_dir, init_step, device, load_optimizer=False
)
# Unwrap keys added by torch.compile (prefix "_orig_mod.")
model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
model.load_state_dict(model_data, strict=True, assign=True)
del model_data
print0(f"  init val_bpb = {init_meta.get('val_bpb', 'n/a'):.4f}  (from Entrega 2)")

# ---------------------------------------------------------------------------
# Compile

orig_model = model
torch._dynamo.config.cache_size_limit = 64
model = torch.compile(model, dynamic=False)

param_counts        = orig_model.num_scaling_params()
num_flops_per_token = orig_model.estimate_flops()
print0(f"Parameters: {param_counts['total']:,}")
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# ---------------------------------------------------------------------------
# Optimizer — fresh AdamW, 10x smaller LRs than mid-training
# Weight decay = 0 (standard for fine-tuning)

optimizer = build_adamw_only_optimizer(
    model,
    embedding_lr   = args.embedding_lr,
    unembedding_lr = args.unembedding_lr,
    matrix_lr      = args.matrix_lr,
    scalar_lr      = args.scalar_lr,
    weight_decay   = 0.0,
)
print0(
    f"Optimizer: fresh AdamW | "
    f"emb={args.embedding_lr} unemb={args.unembedding_lr} "
    f"mat={args.matrix_lr} scl={args.scalar_lr}"
)

# ---------------------------------------------------------------------------
# Batch sizes / gradient accumulation

tokens_per_fwdbwd       = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert args.total_batch_size % world_tokens_per_fwdbwd == 0, (
    f"total_batch_size ({args.total_batch_size}) must be divisible by "
    f"device_batch_size*max_seq_len*world_size = {world_tokens_per_fwdbwd}"
)
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens/micro-batch/rank : {args.device_batch_size} × {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Total batch size        : {args.total_batch_size:,}  →  grad_accum = {grad_accum_steps}")

# ---------------------------------------------------------------------------
# Training horizon (epoch-derived or explicit)
#
# 1 epoch = 15 train shards × BLOCKS_PER_SHARD blocks / (total_batch_size / max_seq_len) steps
# Default: 586 steps/epoch  →  2 epochs ≈ 1172 steps

all_paths        = list_parquet_files_qa_pt(DATA_DIR_QA_PT)
num_train_shards = len(all_paths) - 1
blocks_per_epoch = num_train_shards * BLOCKS_PER_SHARD
blocks_per_step  = args.total_batch_size // args.max_seq_len   # = 256
steps_per_epoch  = blocks_per_epoch // blocks_per_step

if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print0(f"Using user-provided num_iterations: {num_iterations:,}")
else:
    num_iterations = args.epochs * steps_per_epoch
    print0(
        f"Derived num_iterations = {args.epochs} epochs × {steps_per_epoch} steps/epoch "
        f"= {num_iterations:,} steps"
    )
    print0(
        f"  basis: {num_train_shards} train shards × {BLOCKS_PER_SHARD:,} blocks/shard "
        f"= {blocks_per_epoch:,} blocks  |  {blocks_per_step} blocks/optimizer step"
    )
total_tokens_planned = args.total_batch_size * num_iterations
print0(f"Planned training tokens : {total_tokens_planned:,}  ({total_tokens_planned / 1e6:.1f} M)")

# ---------------------------------------------------------------------------
# Data loaders

def build_train_loader(resume_state_dict=None):
    return dataloader_qa_pt_sft(
        DATA_DIR_QA_PT, "train",
        args.device_batch_size, args.max_seq_len, device, tokenizer,
        resume_state_dict=resume_state_dict,
    )

def build_val_loader():
    for x, y, _ in dataloader_qa_pt_sft(
        DATA_DIR_QA_PT, "val",
        args.device_batch_size, args.max_seq_len, device, tokenizer,
    ):
        yield x, y

# ---------------------------------------------------------------------------
# LR schedule: linear warmup → flat → linear warmdown

def get_lr_multiplier(it):
    warmup_iters   = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress + (1 - progress) * args.final_lr_frac

# ---------------------------------------------------------------------------
# Training loop state

step                = 0
val_bpb             = None
val_ppl             = None
min_val_bpb         = float("inf")
min_val_ppl         = float("inf")
smooth_train_loss   = 0.0
total_training_time = 0.0
ema_beta            = 0.9

eval_steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size))

train_loader = build_train_loader()
x, y, dataloader_state_dict = next(train_loader)

# ---------------------------------------------------------------------------
# Main training loop

while True:
    last_step    = (step == num_iterations)
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # ---- validation bpb + perplexity (watch val rise while train falls = overfitting) ----
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_bpb = evaluate_bpb(model, build_val_loader(), eval_steps, token_bytes)
        val_ppl = evaluate_perplexity(model, build_val_loader(), eval_steps, token_bytes)
        print0(f"step {step:05d} | val bpb: {val_bpb:.6f} | val ppl: {val_ppl:.2f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        if val_ppl < min_val_ppl:
            min_val_ppl = val_ppl
        wandb_run.log({
            "step":                 step,
            "total_training_flops": flops_so_far,
            "total_training_time":  total_training_time,
            "val/bpb":              val_bpb,
            "val/ppl":              val_ppl,
        })
        model.train()

    # ---- checkpoint ----
    if last_step or (step > 0 and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            output_ckpt_dir,
            step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step":                  step,
                "val_bpb":               val_bpb,
                "model_config":          model_config_kwargs,
                "user_config":           user_config,
                "device_batch_size":     args.device_batch_size,
                "max_seq_len":           args.max_seq_len,
                "total_batch_size":      args.total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "val_ppl":               val_ppl,
                "loop_state": {
                    "min_val_bpb":         min_val_bpb,
                    "min_val_ppl":         min_val_ppl,
                    "smooth_train_loss":   smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # ---- forward / backward ----
    synchronize()
    t0 = time.time()

    for micro_step in range(grad_accum_steps):
        # targets has -1 for prompt tokens; GPT.forward uses ignore_index=-1
        loss       = model(x, y)
        train_loss = loss.detach()
        (loss / grad_accum_steps).backward()
        x, y, dataloader_state_dict = next(train_loader)

    # ---- optimizer step ----
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # ---- logging ----
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_loss     = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done          = 100 * step / num_iterations
    tok_per_sec       = int(args.total_batch_size / dt)
    flops_per_sec     = num_flops_per_token * args.total_batch_size / dt
    mfu               = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    epoch_approx      = dataloader_state_dict["epoch"]

    if step > 10:
        total_training_time += dt
    steps_done = step - 10
    eta_str = (
        f" | eta: {(total_training_time / steps_done) * (num_iterations - step) / 60:.1f}m"
        if steps_done > 0 else ""
    )

    # On step 1: verify mask fraction (should be 40–70%; 0% or 100% = bug)
    if step == 0:
        mask_frac = (y >= 0).float().mean().item()
        print0(
            f"  MASK CHECK step 0: non-masked fraction = {mask_frac:.1%}  "
            f"({'OK' if 0.3 < mask_frac < 0.95 else 'WARNING: out of expected range (0%=no mask, 100%=all masked)'})"
        )

    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | "
        f"loss: {debiased_loss:.6f} | lrm: {lrm:.4f} | "
        f"dt: {dt * 1000:.1f}ms | tok/s: {tok_per_sec:,} | "
        f"mfu: {mfu:.2f}% | ep: {epoch_approx} | "
        f"time: {total_training_time / 60:.1f}m{eta_str}"
    )

    if step % 50 == 0:
        wandb_run.log({
            "step":                 step,
            "total_training_flops": flops_so_far,
            "total_training_time":  total_training_time,
            "train/loss":           debiased_loss,
            "train/lrm":            lrm,
            "train/tok_per_sec":    tok_per_sec,
            "train/mfu":            mfu,
            "train/epoch":          epoch_approx,
        })

    # GC management (same as mid_train.py)
    step += 1
    if step == 1:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

# ---------------------------------------------------------------------------
# Post-training stats

print0(f"Peak memory     : {get_max_memory() / 1024 / 1024:.2f} MiB")
print0(f"Total train time: {total_training_time / 60:.2f} m")
if val_bpb is not None:
    print0(f"Min val bpb     : {min_val_bpb:.6f}")
if val_ppl is not None:
    print0(f"Min val ppl     : {min_val_ppl:.2f}")

wandb_run.finish()
compute_cleanup()
