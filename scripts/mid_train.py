"""
Mid-training script for nanochat (Entrega 2).

Continues language-modelling on ju-resplande/qa-pt with the nanochat chat
template, bridging the gap between base pre-training (plain web text) and SFT
(masked prompt fine-tuning).

Key differences from base_train.py:
  - Loads the Entrega-1 base checkpoint; does NOT resume optimizer state.
  - Fresh AdamW with LRs 10× smaller than pre-training defaults.
  - No prompt mask: loss is computed over every token in the packed sequence.
    The -100 / ignore_index mask is deferred to SFT (Etapa 3).
  - Dataset: pre-tokenized qa-pt blocks (nanochat.dataset_qa_pt).
  - Token budget controlled by --target-tokens instead of scaling laws.
  - Checkpoint saved to base_checkpoints/{model_tag} (default: d12_midtrain).

Usage:
    python -m scripts.mid_train -- \\
        --base-checkpoint=d12 --depth=12 --device-batch-size=16 \\
        --optimizer=adamw --target-tokens=3e8 --dataset=qa_pt

    # Resume a mid-train run from a checkpoint:
    python -m scripts.mid_train -- \\
        --resume-from-step=250 --model-tag=d12_midtrain ...
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.common import (
    compute_init, compute_cleanup, print0, DummyWandb, print_banner,
    get_base_dir, autodetect_device_type, get_peak_flops,
    COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized,
)
from nanochat.optim import build_adamw_only_optimizer
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import (
    save_checkpoint, load_checkpoint, find_last_step,
)
from nanochat.loss_eval import evaluate_bpb, evaluate_perplexity
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
print_banner()

# ---------------------------------------------------------------------------
# CLI

parser = argparse.ArgumentParser(description="Mid-training (Entrega 2)")

# Logging
parser.add_argument("--run",       type=str, default="dummy",
                    help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="",
                    help="cuda|cpu|mps (empty = autodetect)")
# Base checkpoint (what we start from)
parser.add_argument("--base-checkpoint", type=str, default="d12",
                    help="Tag of the base pre-train checkpoint to load weights from")
parser.add_argument("--base-checkpoint-step", type=int, default=-1,
                    help="Step of the base checkpoint (-1 = latest)")
# Model architecture (must match the base checkpoint)
parser.add_argument("--depth",       type=int, default=12)
parser.add_argument("--aspect-ratio", type=int, default=64)
parser.add_argument("--head-dim",    type=int, default=128)
parser.add_argument("--window-pattern", type=str, default="L",
                    help="Attention window pattern (L = full; must match base ckpt)")
# Resume a mid-train run (not the base checkpoint)
parser.add_argument("--resume-from-step", type=int, default=-1,
                    help="Resume mid-train from this step (-1 = start fresh)")
# Output tag for mid-train checkpoint
parser.add_argument("--model-tag",   type=str, default="d12_midtrain",
                    help="Tag (directory name) for the mid-train checkpoint")
# Training horizon
parser.add_argument("--target-tokens",  type=float, default=3e8,
                    help="Token budget for mid-training (default 3e8 = 300M)")
parser.add_argument("--num-iterations", type=int, default=-1,
                    help="Override step count directly (-1 = derive from target-tokens)")
# Batch
parser.add_argument("--max-seq-len",      type=int, default=2048)
parser.add_argument("--device-batch-size", type=int, default=16)
parser.add_argument("--total-batch-size",  type=int, default=524288,
                    help="Total tokens per optimizer step (default: same as pretrain)")
# Optimizer
parser.add_argument("--optimizer", type=str, default="adamw",
                    choices=["adamw"], help="Optimizer (only adamw supported)")
# Learning rates (defaults are 10× smaller than pretrain)
parser.add_argument("--embedding-lr",   type=float, default=0.03,
                    help="LR for embedding params (pretrain was 0.3)")
parser.add_argument("--unembedding-lr", type=float, default=0.0008,
                    help="LR for unembedding / lm_head params (pretrain was 0.008)")
parser.add_argument("--matrix-lr",      type=float, default=0.002,
                    help="LR for transformer matrix params (pretrain was 0.02)")
parser.add_argument("--scalar-lr",      type=float, default=0.05,
                    help="LR for scalar params (pretrain was 0.5)")
# LR schedule
parser.add_argument("--warmup-steps",    type=int,   default=100,
                    help="Linear warmup steps (short because we resume a trained model)")
parser.add_argument("--warmdown-ratio",  type=float, default=0.65,
                    help="Fraction of iterations used for LR warmdown (cosine)")
parser.add_argument("--final-lr-frac",   type=float, default=0.05,
                    help="LR at end of warmdown as fraction of peak LR")
# Evaluation
parser.add_argument("--eval-every",   type=int, default=50,
                    help="Evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens",  type=int, default=10_485_760,
                    help="Tokens to evaluate val loss on (~10M default)")
# Checkpointing
parser.add_argument("--save-every",   type=int, default=100,
                    help="Save checkpoint every N steps (-1 = only at end)")
# Dataset
parser.add_argument("--dataset",      type=str, default="qa_pt",
                    choices=["qa_pt"],
                    help="Dataset for mid-training (only qa_pt supported)")

args   = parser.parse_args()
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

# wandb
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = (
    DummyWandb() if use_dummy_wandb
    else wandb.init(project="nanochat-midtrain", name=args.run, config=user_config)
)

if not HAS_FA3:
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback.")

# ---------------------------------------------------------------------------
# Tokenizer

tokenizer  = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# ---------------------------------------------------------------------------
# Build model skeleton on meta device (shapes/dtypes only, no data)

base_dim  = args.depth * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = model_dim // args.head_dim
model_config = GPTConfig(
    sequence_len=args.max_seq_len,
    vocab_size=vocab_size,
    n_layer=args.depth,
    n_head=num_heads,
    n_kv_head=num_heads,
    n_embd=model_dim,
    window_pattern=args.window_pattern,
)
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")

with torch.device("meta"):
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()

# ---------------------------------------------------------------------------
# Load checkpoint weights
# Two modes:
#   (a) fresh mid-train: load BASE checkpoint weights, fresh optimizer
#   (b) resume mid-train: load MID-TRAIN checkpoint weights + optimizer

base_dir          = get_base_dir()
output_dirname    = args.model_tag
mid_train_ckpt_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming           = args.resume_from_step != -1

if resuming:
    print0(f"Resuming mid-training from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(
        mid_train_ckpt_dir, args.resume_from_step, device,
        load_optimizer=True, rank=ddp_rank,
    )
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data
    dataloader_resume_state_dict = meta_data.get("dataloader_state_dict")
else:
    # Load BASE checkpoint (weights only; optimizer is reset)
    base_ckpt_dir  = os.path.join(base_dir, "base_checkpoints", args.base_checkpoint)
    base_step      = (
        find_last_step(base_ckpt_dir)
        if args.base_checkpoint_step == -1
        else args.base_checkpoint_step
    )
    assert base_step is not None, (
        f"No checkpoint found in {base_ckpt_dir}.  "
        f"Run Entrega 1 first (runs/entrega1.sh)."
    )
    print0(f"Loading base checkpoint: {args.base_checkpoint} @ step {base_step}")
    model_data, _, base_meta = load_checkpoint(
        base_ckpt_dir, base_step, device, load_optimizer=False
    )
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data
    print0(f"  base val_bpb={base_meta.get('val_bpb', 'n/a'):.4f}")
    dataloader_resume_state_dict = None

# ---------------------------------------------------------------------------
# Compile

orig_model = model
torch._dynamo.config.cache_size_limit = 64
model = torch.compile(model, dynamic=False)

param_counts       = model.num_scaling_params()
num_flops_per_token = model.estimate_flops()
print0(f"Parameters: {param_counts['total']:,}")
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# ---------------------------------------------------------------------------
# Optimizer — fresh AdamW with 10× smaller LRs; no warm-start from base ckpt.
# Weight decay = 0 (standard for fine-tuning; matches nanochat chat_sft.py).

optimizer = build_adamw_only_optimizer(
    model,
    embedding_lr   = args.embedding_lr,
    unembedding_lr = args.unembedding_lr,
    matrix_lr      = args.matrix_lr,
    scalar_lr      = args.scalar_lr,
    weight_decay   = 0.0,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# ---------------------------------------------------------------------------
# Batch sizes and gradient accumulation

tokens_per_fwdbwd       = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert args.total_batch_size % world_tokens_per_fwdbwd == 0, (
    f"total_batch_size ({args.total_batch_size}) must be divisible by "
    f"device_batch_size*max_seq_len*world_size={world_tokens_per_fwdbwd}"
)
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens/micro-batch/rank : {args.device_batch_size} × {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Total batch size        : {args.total_batch_size:,}  →  grad_accum = {grad_accum_steps}")

# ---------------------------------------------------------------------------
# Training horizon

if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print0(f"Using user-provided num_iterations: {num_iterations:,}")
else:
    num_iterations = int(args.target_tokens / args.total_batch_size)
    print0(f"Derived num_iterations = {int(args.target_tokens):,} / {args.total_batch_size:,} = {num_iterations:,}")
total_tokens_planned = args.total_batch_size * num_iterations
print0(f"Planned training tokens : {total_tokens_planned:,}  ({total_tokens_planned / 1e6:.1f} M)")

# ---------------------------------------------------------------------------
# Dataset loaders

from nanochat.dataset_qa_pt import DATA_DIR_QA_PT, dataloader_qa_pt

def _val_batches():
    """Fresh val generator; yields (x, y) without state_dict."""
    for x, y, _ in dataloader_qa_pt(
        DATA_DIR_QA_PT, "val",
        args.device_batch_size, args.max_seq_len, device,
    ):
        yield x, y

build_val_loader = _val_batches

train_loader = dataloader_qa_pt(
    DATA_DIR_QA_PT, "train",
    args.device_batch_size, args.max_seq_len, device,
    resume_state_dict=dataloader_resume_state_dict,
)
x, y, dataloader_state_dict = next(train_loader)

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

if not resuming:
    step               = 0
    val_bpb            = None
    val_ppl            = None
    min_val_bpb        = float("inf")
    min_val_ppl        = float("inf")
    smooth_train_loss  = 0.0
    total_training_time = 0.0
else:
    step               = meta_data["step"]
    loop_state         = meta_data["loop_state"]
    val_bpb            = meta_data.get("val_bpb")
    val_ppl            = meta_data.get("val_ppl")
    min_val_bpb        = loop_state["min_val_bpb"]
    min_val_ppl        = loop_state.get("min_val_ppl", float("inf"))
    smooth_train_loss  = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

eval_steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size))

# ---------------------------------------------------------------------------
# Main training loop

while True:
    last_step = (step == num_iterations)
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # ---- validation bpb + perplexity ----
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
    if last_step or (
        step > 0
        and step != args.resume_from_step
        and args.save_every > 0
        and step % args.save_every == 0
    ):
        save_checkpoint(
            mid_train_ckpt_dir,
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
        loss       = model(x, y)   # no -1 targets → full-sequence LM loss
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
    t1  = time.time()
    dt  = t1 - t0

    # ---- logging ----
    ema_beta          = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_loss     = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done          = 100 * step / num_iterations
    tok_per_sec       = int(args.total_batch_size / dt)
    flops_per_sec     = num_flops_per_token * args.total_batch_size / dt
    mfu               = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)

    if step > 10:
        total_training_time += dt
    steps_done = step - 10
    if steps_done > 0:
        avg_step_time   = total_training_time / steps_done
        eta_seconds     = avg_step_time * (num_iterations - step)
        eta_str         = f" | eta: {eta_seconds / 60:.1f}m"
    else:
        eta_str = ""

    dl_state = (
        f"ep:{dataloader_state_dict['epoch']} "
        f"pq:{dataloader_state_dict['pq_idx']} "
        f"rg:{dataloader_state_dict['rg_idx']}"
    )
    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | "
        f"loss: {debiased_loss:.6f} | lrm: {lrm:.4f} | "
        f"dt: {dt * 1000:.1f}ms | tok/s: {tok_per_sec:,} | "
        f"mfu: {mfu:.2f}% | {dl_state} | "
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
        })

    # GC management (mirrors base_train.py)
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1
    if first_step_of_run:
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
