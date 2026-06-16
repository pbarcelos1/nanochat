"""
Upload nanochat PT-Latn checkpoints to the Hugging Face Hub.

Supported phases:
  base      — base_checkpoints/d12           → nanochat-pt-latn-d12
  midtrain  — base_checkpoints/d12_midtrain  → nanochat-pt-latn-d12-midtrain
  sft       — base_checkpoints/d12_sft       → nanochat-pt-latn-d12-sft

Uploads per phase:
  - model_<step>.pt
  - meta_<step>.json
  - tokenizer/tokenizer.pkl
  - tokenizer/token_bytes.pt
  - README.md (model card)

Prerequisites:
  huggingface-cli login   (or set HF_TOKEN env var)

Usage:
  python -m scripts.upload_hf --phase base
  python -m scripts.upload_hf --phase midtrain
  python -m scripts.upload_hf --phase sft
  python -m scripts.upload_hf --phase sft --step 500 --repo myuser/my-model
"""
import os
import json
import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import find_last_step

# ---------------------------------------------------------------------------
# Phase configuration
# ---------------------------------------------------------------------------

PHASES = {
    "base": {
        "checkpoint_subdir": Path("base_checkpoints") / "d12",
        "repo_suffix": "nanochat-pt-latn-d12",
        "description": "base pre-trained",
        "trained_on": "HuggingFaceFW/fineweb-2 (por_Latn)",
        "loss_mask": False,
    },
    "midtrain": {
        "checkpoint_subdir": Path("base_checkpoints") / "d12_midtrain",
        "repo_suffix": "nanochat-pt-latn-d12-midtrain",
        "description": "mid-trained on QA-PT (no loss mask)",
        "trained_on": "ju-resplande/qa-pt (full sequence, no mask)",
        "loss_mask": False,
    },
    "sft": {
        "checkpoint_subdir": Path("base_checkpoints") / "d12_sft",
        "repo_suffix": "nanochat-pt-latn-d12-sft",
        "description": "SFT fine-tuned on QA-PT (assistant-only loss mask)",
        "trained_on": "ju-resplande/qa-pt (assistant tokens only)",
        "loss_mask": True,
    },
}

# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------

def make_model_card(meta: dict, repo_id: str, phase: str) -> str:
    cfg = meta["model_config"]
    step = meta["step"]
    val_bpb = meta.get("val_bpb", float("nan"))
    total_time_h = meta.get("loop_state", {}).get("total_training_time", 0) / 3600
    phase_cfg = PHASES[phase]

    phase_note = {
        "base": """\
This is a **base model** — raw continuations, no chat/instruction following.
""",
        "midtrain": """\
This is a **mid-trained** model: the base pre-trained checkpoint continued on
Portuguese QA data (no loss mask). It bridges raw web-text pre-training and
supervised fine-tuning.
""",
        "sft": """\
This is a **supervised fine-tuned (SFT)** chat model. It was trained with an
assistant-only loss mask on Portuguese QA data. It follows simple instruction /
answer prompts using the nanochat chat template.
""",
    }[phase]

    return f"""\
---
language:
- pt
license: mit
tags:
- nanochat
- portuguese
- causal-lm
- {phase}
base_model: pbarcelos1/nanochat-pt-latn-d12
---

# nanochat PT-Latn d12 — {phase_cfg["description"]}

{phase_note}
Forked from [karpathy/nanochat](https://github.com/karpathy/nanochat) and
trained on {phase_cfg["trained_on"]}.

## Model details

| | |
|-|-|
| Architecture | Decoder-only transformer (relu² FFN, RoPE, GQA, RMSNorm) |
| Layers | {cfg["n_layer"]} |
| d_model | {cfg["n_embd"]} |
| Attention heads | {cfg["n_head"]} (KV heads: {cfg["n_kv_head"]}) |
| Sequence length | {cfg["sequence_len"]} |
| Vocabulary | {cfg["vocab_size"]:,} (byte-level BPE, PT-trained) |
| Trained steps | {step:,} |
| Final val bpb | {val_bpb:.4f} |
| Training time | {total_time_h:.1f} h on 1× NVIDIA RTX A6000 |

## Usage

Load with nanochat:

```python
import os
import torch
from huggingface_hub import snapshot_download
from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.engine import Engine

local_dir = snapshot_download(repo_id="{repo_id}")
os.environ["NANOCHAT_BASE_DIR"] = local_dir

device = torch.device("cuda:0")
step = find_last_step(local_dir)
model, tokenizer, meta = build_model(local_dir, step, device, phase="eval")

engine = Engine(model, tokenizer)
tokens = tokenizer("Era uma vez,", prepend="<|bos|>")
samples, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=200, temperature=0.8)
print(tokenizer.decode(samples[0]))
```
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Upload nanochat checkpoint to HF Hub")
    parser.add_argument("--phase", choices=list(PHASES), default="base",
                        help="Which checkpoint to upload: base | midtrain | sft")
    parser.add_argument("--step", type=int, default=None,
                        help="Checkpoint step to upload (default: latest)")
    parser.add_argument("--repo", type=str, default=None,
                        help="HF repo id, e.g. username/model-name (default: auto)")
    parser.add_argument("--private", action="store_true",
                        help="Create a private repository")
    parser.add_argument("--base-dir", type=str,
                        default=os.environ.get("NANOCHAT_BASE_DIR"),
                        help="Override NANOCHAT_BASE_DIR")
    args = parser.parse_args()

    if args.base_dir:
        os.environ["NANOCHAT_BASE_DIR"] = args.base_dir

    base_dir = Path(get_base_dir())
    phase_cfg = PHASES[args.phase]
    checkpoint_dir = base_dir / phase_cfg["checkpoint_subdir"]
    tokenizer_dir  = base_dir / "tokenizer"

    step = args.step if args.step is not None else find_last_step(str(checkpoint_dir))
    print(f"Phase: {args.phase} | step: {step} | dir: {checkpoint_dir}")

    meta_path  = checkpoint_dir / f"meta_{step:06d}.json"
    model_path = checkpoint_dir / f"model_{step:06d}.pt"
    for p in [meta_path, model_path]:
        assert p.exists(), f"Not found: {p}"

    with open(meta_path) as f:
        meta = json.load(f)
    meta["step"] = step

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = args.repo or f"{user}/{phase_cfg['repo_suffix']}"
    print(f"Target repo: {repo_id}")

    create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)

    files_to_upload = [
        (model_path,                              f"model_{step:06d}.pt"),
        (meta_path,                               f"meta_{step:06d}.json"),
        (tokenizer_dir / "tokenizer.pkl",         "tokenizer/tokenizer.pkl"),
        (tokenizer_dir / "token_bytes.pt",        "tokenizer/token_bytes.pt"),
    ]

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as tmp:
        tmp.write(make_model_card(meta, repo_id, args.phase))
        card_path = tmp.name

    files_to_upload.append((Path(card_path), "README.md"))

    for local_path, repo_filename in files_to_upload:
        print(f"  uploading {local_path.name} → {repo_filename} …", flush=True)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_filename,
            repo_id=repo_id,
            repo_type="model",
        )

    os.unlink(card_path)
    print(f"\nDone. Model available at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
