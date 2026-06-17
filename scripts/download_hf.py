"""
Download nanochat PT-Latn checkpoints from the Hugging Face Hub.

Supported phases:
  base      — pbarcelos1/nanochat-pt-latn-d12        → base_checkpoints/d12/
  midtrain  — pbarcelos1/nanochat-pt-latn-d12-midtrain → base_checkpoints/d12_midtrain/
  sft       — pbarcelos1/nanochat-pt-latn-d12-sft    → base_checkpoints/d12_sft/

The shared tokenizer is downloaded once from the base repo into tokenizer/.

Usage:
  python -m scripts.download_hf                     # all phases
  python -m scripts.download_hf --phase base
  python -m scripts.download_hf --phase midtrain sft
  python -m scripts.download_hf --base-dir /data/nanochat
"""
import os
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download, hf_hub_download
from nanochat.common import get_base_dir

REPOS = {
    "base":     "pbarcelos1/nanochat-pt-latn-d12",
    "midtrain": "pbarcelos1/nanochat-pt-latn-d12-midtrain",
    "sft":      "pbarcelos1/nanochat-pt-latn-d12-sft",
}

CHECKPOINT_SUBDIRS = {
    "base":     Path("base_checkpoints") / "d12",
    "midtrain": Path("base_checkpoints") / "d12_midtrain",
    "sft":      Path("base_checkpoints") / "d12_sft",
}

TOKENIZER_FILES = [
    "tokenizer/tokenizer.pkl",
    "tokenizer/token_bytes.pt",
]


def main():
    parser = argparse.ArgumentParser(description="Download nanochat checkpoints from HF Hub")
    parser.add_argument(
        "--phase", nargs="+", choices=list(REPOS), default=list(REPOS),
        metavar="PHASE",
        help="Phases to download: base | midtrain | sft (default: all)",
    )
    parser.add_argument(
        "--base-dir", type=str, default=os.environ.get("NANOCHAT_BASE_DIR"),
        help="Override NANOCHAT_BASE_DIR",
    )
    parser.add_argument(
        "--skip-tokenizer", action="store_true",
        help="Skip downloading the tokenizer",
    )
    args = parser.parse_args()

    if args.base_dir:
        os.environ["NANOCHAT_BASE_DIR"] = args.base_dir

    base_dir = Path(get_base_dir())
    print(f"NANOCHAT_BASE_DIR: {base_dir}\n")

    for phase in args.phase:
        repo_id = REPOS[phase]
        checkpoint_dir = base_dir / CHECKPOINT_SUBDIRS[phase]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{phase}] Downloading {repo_id}")
        print(f"       → {checkpoint_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(checkpoint_dir),
            ignore_patterns=["tokenizer/*", "README.md"],
        )
        print(f"[{phase}] Done\n")

    if not args.skip_tokenizer:
        tokenizer_dir = base_dir / "tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        source_repo = REPOS["base"]
        print(f"[tokenizer] Downloading from {source_repo}")
        print(f"            → {tokenizer_dir}")
        for filename in TOKENIZER_FILES:
            hf_hub_download(
                repo_id=source_repo,
                filename=filename,
                local_dir=str(base_dir),
            )
        print("[tokenizer] Done\n")

    print("All downloads complete.")


if __name__ == "__main__":
    main()
