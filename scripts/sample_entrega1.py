"""
Generate text samples from the trained PT-Latn base model.

This is a base model (not instruction-tuned), so prompts are completed
as continuations — no chat formatting, no system prompt.

Usage:
    python -m scripts.sample_entrega1.py
    python -m scripts.sample_entrega1.py --model-tag d12 --max-tokens 200 --temperature 0.8
    python -m scripts.sample_entrega1.py --step 5000

Output:
    stdout + samples.md (in the current working directory)
"""
import os
import argparse
import torch

from nanochat.common import autodetect_device_type, compute_init, compute_cleanup, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# ---------------------------------------------------------------------------
PROMPTS = [
    "Era uma vez, numa pequena cidade do interior,",
    "A capital do Brasil é",
    "O nome do filho do Lucas é",
    "A fotossíntese é o processo pelo qual as plantas",
    "def fibonacci(n):\n    ",
]

# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PT-Latn base model sampling")
    parser.add_argument("--model-tag", type=str, default=None,
                        help="Checkpoint tag, e.g. d12 (default: auto-detect largest)")
    parser.add_argument("--step", type=int, default=None,
                        help="Checkpoint step (default: latest)")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Tokens to generate per prompt (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k sampling (default: off)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default="samples.md",
                        help="Output file (default: samples.md)")
    parser.add_argument("--device-type", type=str, default="",
                        help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--base-dir", type=str,
                        default=os.environ.get("NANOCHAT_BASE_DIR",
                                               "/mnt/E-SSD/barcelos/.cache/nanochat"),
                        help="Base directory for checkpoints "
                             "(default: $NANOCHAT_BASE_DIR or entrega1 cache path)")
    args = parser.parse_args()

    if args.base_dir:
        os.environ["NANOCHAT_BASE_DIR"] = args.base_dir

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)

    model, tokenizer, meta = load_model(
        "base", device, phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )
    step = meta["step"]
    model_cfg = meta["model_config"]
    print0(f"Loaded: step={step}  depth={model_cfg['n_layer']}  "
           f"d_model={model_cfg['n_embd']}  vocab={model_cfg['vocab_size']}")
    print0(f"Sampling: max_tokens={args.max_tokens}  temperature={args.temperature}  "
           f"top_k={args.top_k}  seed={args.seed}")

    engine = Engine(model, tokenizer)

    header = (
        f"# PT-Latn base model — text samples\n\n"
        f"- checkpoint step: {step}\n"
        f"- depth: {model_cfg['n_layer']}  d_model: {model_cfg['n_embd']}\n"
        f"- max_tokens: {args.max_tokens}  temperature: {args.temperature}"
        f"  top_k: {args.top_k}  seed: {args.seed}\n"
    )
    sections = [header]

    print0("\n" + "=" * 72)
    print0("PT-Latn base model samples")
    print0("=" * 72)

    for i, prompt in enumerate(PROMPTS):
        tokens = tokenizer(prompt, prepend="<|bos|>")
        samples, _ = engine.generate_batch(
            tokens,
            num_samples=1,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed + i,
        )
        text = tokenizer.decode(samples[0])

        print0(f"\n--- prompt {i + 1}/{len(PROMPTS)} ---")
        print0(text)

        sections.append(f"## Prompt {i + 1}\n\n```\n{text}\n```\n")

    output_md = "\n".join(sections)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(output_md)
    print0(f"\nSamples written to: {args.output}")

    compute_cleanup()


if __name__ == "__main__":
    main()
