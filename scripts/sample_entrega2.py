"""
Generate and compare samples from base, mid-trained, and SFT models.

Formats each prompt as:
    <|bos|><|user_start|>{question}<|user_end|><|assistant_start|>

Generation stops at <|assistant_end|> or <|bos|> (handled by Engine automatically).
Temperature 0.7, up to 200 tokens per response.

Output: samples_entrega3.md (side-by-side: base | mid-train | SFT)

Usage:
    python -m scripts.sample_entrega3
    python -m scripts.sample_entrega3 --sft-tag d12_sft --max-tokens 200
"""

import os
import argparse
import torch

from nanochat.common import autodetect_device_type, compute_init, compute_cleanup, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# ---------------------------------------------------------------------------
# Same 5 prompts as Entrega 2 for direct cross-stage comparability.
# Prompt 4 is in-distribution for qa-pt; prompt 5 tests arithmetic reasoning.

PROMPTS = [
    "Shih tzu é a melhor raça?",
    "Cachorro ou gato?",
    "Qual o nome do filho do Lucas?",
    "Quais cuidados básicos um cachorro precisa?",     # in-distribution qa-pt
    "Se um trem percorre 60 km em 1,5 horas, qual é a sua velocidade média?",  # arithmetic
]

# ---------------------------------------------------------------------------


def _sample_base(engine, tokenizer, prompt: str, max_tokens: int,
                 temperature: float, top_k, seed: int) -> str:
    """Sample from the BASE model (Entrega 1): plain text, no special tokens."""
    tokens = tokenizer(prompt, prepend="<|bos|>")
    samples, _ = engine.generate_batch(
        tokens, num_samples=1, max_tokens=max_tokens,
        temperature=temperature, top_k=top_k, seed=seed,
    )
    return tokenizer.decode(samples[0])


def _sample_chat(engine, tokenizer, prompt: str, max_tokens: int,
                 temperature: float, top_k, seed: int) -> str:
    """
    Sample from mid-trained or SFT model using the canonical chat template:
        <|bos|><|user_start|>{question}<|user_end|><|assistant_start|>
    """
    bos        = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end   = tokenizer.encode_special("<|user_end|>")
    asst_start = tokenizer.encode_special("<|assistant_start|>")

    prefix_ids = [bos, user_start]
    prefix_ids.extend(tokenizer.encode(prompt))
    prefix_ids.extend([user_end, asst_start])

    samples, _ = engine.generate_batch(
        prefix_ids, num_samples=1, max_tokens=max_tokens,
        temperature=temperature, top_k=top_k, seed=seed,
    )
    return tokenizer.decode(samples[0])


def main():
    parser = argparse.ArgumentParser(
        description="Compare base / mid-train / SFT models on PT-BR questions"
    )
    parser.add_argument("--base-tag",    type=str, default=None,
                        help="Base checkpoint tag (default: auto-detect)")
    parser.add_argument("--base-step",   type=int, default=None)
    parser.add_argument("--mid-tag",     type=str, default="d12_midtrain")
    parser.add_argument("--mid-step",    type=int, default=None)
    parser.add_argument("--sft-tag",     type=str, default="d12_sft")
    parser.add_argument("--sft-step",    type=int, default=None)
    parser.add_argument("--max-tokens",  type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k",       type=int, default=None)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output",      type=str, default="samples_entrega3.md")
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--base-dir",    type=str,
                        default=os.environ.get("NANOCHAT_BASE_DIR",
                                               "/mnt/E-SSD/barcelos/.cache/nanochat"))
    args = parser.parse_args()

    if args.base_dir:
        os.environ["NANOCHAT_BASE_DIR"] = args.base_dir

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)

    # ---- load base model ----
    print0("\nLoading BASE model (Entrega 1) …")
    base_model, tokenizer, base_meta = load_model(
        "base", device, phase="eval",
        model_tag=args.base_tag, step=args.base_step,
    )
    base_step = base_meta["step"]
    print0(f"  step={base_step}")
    base_engine = Engine(base_model, tokenizer)

    # ---- load mid-train model ----
    print0("\nLoading MID-TRAIN model (Entrega 2) …")
    mid_model, _, mid_meta = load_model(
        "base", device, phase="eval",
        model_tag=args.mid_tag, step=args.mid_step,
    )
    mid_step = mid_meta["step"]
    print0(f"  step={mid_step}  tag={args.mid_tag}")
    mid_engine = Engine(mid_model, tokenizer)

    # ---- load SFT model ----
    print0("\nLoading SFT model (Entrega 3) …")
    sft_model, _, sft_meta = load_model(
        "base", device, phase="eval",
        model_tag=args.sft_tag, step=args.sft_step,
    )
    sft_step = sft_meta["step"]
    print0(f"  step={sft_step}  tag={args.sft_tag}")
    sft_engine = Engine(sft_model, tokenizer)

    # ---- generate ----
    print0(
        f"\nSampling: max_tokens={args.max_tokens}  temperature={args.temperature}  "
        f"top_k={args.top_k}  seed={args.seed}"
    )

    sections = [
        f"# Entrega 3 — Base vs. Mid-trained vs. SFT model comparison\n\n"
        f"**Base checkpoint**: `{args.base_tag or 'd12'}` @ step {base_step}  \n"
        f"**Mid-train checkpoint**: `{args.mid_tag}` @ step {mid_step}  \n"
        f"**SFT checkpoint**: `{args.sft_tag}` @ step {sft_step}  \n"
        f"**max_tokens**: {args.max_tokens}  "
        f"**temperature**: {args.temperature}  "
        f"**top_k**: {args.top_k}  "
        f"**seed**: {args.seed}\n\n"
        f"*Base model: plain text prompt (no special tokens — never trained on chat template).*  \n"
        f"*Mid-train + SFT: `<|bos|><|user_start|>…<|user_end|><|assistant_start|>` prefix.*  \n"
        f"*SFT is expected to produce more focused, better-terminated responses*  \n"
        f"*(loss mask teaches the model to answer and stop, not continue raw text).*\n"
    ]

    print0("\n" + "=" * 72)
    print0("Entrega 3 — Base vs. Mid-train vs. SFT samples")
    print0("=" * 72)

    for i, prompt in enumerate(PROMPTS):
        print0(f"\n--- prompt {i + 1}/{len(PROMPTS)}: {prompt[:60]} ---")

        base_text = _sample_base(
            base_engine, tokenizer, prompt,
            args.max_tokens, args.temperature, args.top_k, args.seed + i,
        )
        mid_text = _sample_chat(
            mid_engine, tokenizer, prompt,
            args.max_tokens, args.temperature, args.top_k, args.seed + i,
        )
        sft_text = _sample_chat(
            sft_engine, tokenizer, prompt,
            args.max_tokens, args.temperature, args.top_k, args.seed + i,
        )

        print0(f"\nBASE:\n{base_text}\n")
        print0(f"\nMID-TRAIN:\n{mid_text}\n")
        print0(f"\nSFT:\n{sft_text}\n")

        sections.append(
            f"## Pergunta {i + 1}: {prompt}\n\n"
            f"### Base model (Entrega 1 — sem chat template)\n\n"
            f"```\n{base_text}\n```\n\n"
            f"### Mid-trained model (Entrega 2 — all-token loss)\n\n"
            f"```\n{mid_text}\n```\n\n"
            f"### SFT model (Entrega 3 — loss mask assistente)\n\n"
            f"```\n{sft_text}\n```\n"
        )

    output_md = "\n".join(sections)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(output_md)
    print0(f"\nSamples written to: {args.output}")

    compute_cleanup()


if __name__ == "__main__":
    main()
