"""
Upload the trained nanochat PT-Latn model to the Hugging Face Hub.

Uploads:
  - model_<step>.pt          (model weights, ~757 MB)
  - meta_<step>.json         (model config + training metadata)
  - tokenizer/tokenizer.pkl  (trained BPE tokenizer)
  - tokenizer/token_bytes.pt (token → byte-length cache for BPB eval)
  - README.md                (model card)

Prerequisites:
  huggingface-cli login      (or set HF_TOKEN env var)

Usage:
  python -m scripts.upload_hf                           # uploads step 4830 (latest)
  python -m scripts.upload_hf --step 1000               # uploads a specific step
  python -m scripts.upload_hf --repo myuser/my-model    # custom repo name
"""
import os
import json
import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import find_last_step

def make_model_card(meta: dict, repo_id: str) -> str:
    cfg = meta["model_config"]
    step = meta["step"]
    val_bpb = meta["val_bpb"]
    total_time_h = meta["loop_state"]["total_training_time"] / 3600
    return f"""\
---
language:
- pt
license: mit
tags:
- nanochat
- portuguese
- causal-lm
- pretraining
base_model: karpathy/nanochat
---

# nanochat PT-Latn d12 — base language model

Pretrained Portuguese (PT-Latn) language model, forked from
[karpathy/nanochat](https://github.com/karpathy/nanochat) and trained on
[HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
subset `por_Latn`.

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
| Training tokens | ~2.53 B |
| Final val bpb | {val_bpb:.4f} (PT-Latn validation shard) |
| Training time | {total_time_h:.1f} h on 1× NVIDIA RTX A6000 |

## Usage

This is a **base model** (not instruction-tuned). Load with nanochat:

```python
import os
import torch
from huggingface_hub import snapshot_download
from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.engine import Engine

# Download repo to local cache (~760 MB, cached after first run)
local_dir = snapshot_download(repo_id="{repo_id}")
os.environ["NANOCHAT_BASE_DIR"] = local_dir  # needed for tokenizer lookup

device = torch.device("cuda:0")
step = find_last_step(local_dir)
model, tokenizer, meta = build_model(local_dir, step, device, phase="eval")

engine = Engine(model, tokenizer)
tokens = tokenizer("Era uma vez,", prepend="<|bos|>")
samples, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=200, temperature=0.8)
print(tokenizer.decode(samples[0]))
```

## Limitations

- Base model only — raw continuations, no chat/instruction following
- Portuguese text only — no cross-lingual transfer evaluated
- CORE benchmark (English ICL) is not applicable; results are near-random
"""

def main():
    parser = argparse.ArgumentParser(description="Upload nanochat PT-Latn checkpoint to HF Hub")
    parser.add_argument("--step", type=int, default=None,
                        help="Checkpoint step to upload (default: latest)")
    parser.add_argument("--repo", type=str, default=None,
                        help="HF repo id, e.g. username/model-name (default: auto from whoami)")
    parser.add_argument("--private", action="store_true",
                        help="Create a private repository")
    parser.add_argument("--base-dir", type=str,
                        default=os.environ.get("NANOCHAT_BASE_DIR"),
                        help="Override NANOCHAT_BASE_DIR")
    args = parser.parse_args()

    if args.base_dir:
        os.environ["NANOCHAT_BASE_DIR"] = args.base_dir

    base_dir = get_base_dir()
    checkpoint_dir = Path(base_dir) / "base_checkpoints" / "d12"
    tokenizer_dir  = Path(base_dir) / "tokenizer"

    step = args.step if args.step is not None else find_last_step(str(checkpoint_dir))
    print(f"Uploading step {step} from {checkpoint_dir}")

    meta_path  = checkpoint_dir / f"meta_{step:06d}.json"
    model_path = checkpoint_dir / f"model_{step:06d}.pt"
    for p in [meta_path, model_path]:
        assert p.exists(), f"Not found: {p}"

    with open(meta_path) as f:
        meta = json.load(f)
    meta["step"] = step  # ensure step is set even if missing

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = args.repo or f"{user}/nanochat-pt-latn-d12"
    print(f"Target repo: {repo_id}")

    create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)

    files_to_upload = [
        (model_path,                              f"model_{step:06d}.pt"),
        (meta_path,                               f"meta_{step:06d}.json"),
        (tokenizer_dir / "tokenizer.pkl",         "tokenizer/tokenizer.pkl"),
        (tokenizer_dir / "token_bytes.pt",        "tokenizer/token_bytes.pt"),
    ]

    # Write model card to a temp file
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as tmp:
        tmp.write(make_model_card(meta, repo_id))
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
