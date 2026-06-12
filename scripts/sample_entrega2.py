"""
Generate and compare samples from the base model and the mid-trained model.

The base model (Entrega 1) is prompted with plain text (no chat template)
because it was never trained on special tokens.

The mid-trained model (Entrega 2) is prompted with the canonical chat
template and generation stops at <|assistant_end|> or <|bos|>:
    <|bos|><|user_start|>{question}<|user_end|><|assistant_start|>

The Engine already stops at <|assistant_end|> and <|bos|> automatically
(see nanochat/engine.py generate_batch).

Temperature 0.7 (lower than base model's 0.8 — QA benefits from less
randomness).

Output: samples_entrega2.md

Usage:
    python -m scripts.sample_entrega2
    python -m scripts.sample_entrega2 --mid-tag d12_midtrain --max-tokens 200
"""

import os
import argparse
import torch

from nanochat.common import autodetect_device_type, compute_init, compute_cleanup, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# ---------------------------------------------------------------------------
# Prompts: 5 PT-BR questions covering different domains.
# Prompt 5 is deliberately in-distribution for qa-pt (pet care is a heavy
# domain in that dataset).  It should produce the most coherent mid-train
# answer and serves as evidence that mid-training worked.

PROMPTS = [
    "Qual é a capital do Brasil?",
    "Como funciona a fotossíntese?",
    "O que é uma equação de segundo grau?",
    "Qual a diferença entre Python e JavaScript?",
    "Quais cuidados básicos um cachorro precisa?",   # in-distribution for qa-pt
]

# ---------------------------------------------------------------------------


def _sample_base(engine, tokenizer, prompt: str, max_tokens: int,
                 temperature: float, top_k, seed: int) -> str:
    """Sample from the BASE model: plain text, no special tokens."""
    tokens = tokenizer(prompt, prepend="<|bos|>")
    samples, _ = engine.generate_batch(
        tokens,
        num_samples=1,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )
    return tokenizer.decode(samples[0])


def _sample_midtrain(engine, tokenizer, prompt: str, max_tokens: int,
                     temperature: float, top_k, seed: int) -> str:
    """
    Sample from the MID-TRAINED model using the canonical chat template.
    The prompt is formatted as:
        <|bos|><|user_start|>{question}<|user_end|><|assistant_start|>
    Generation stops when the engine produces <|assistant_end|> or <|bos|>.
    """
    bos        = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end   = tokenizer.encode_special("<|user_end|>")
    asst_start = tokenizer.encode_special("<|assistant_start|>")

    prefix_ids = [bos, user_start]
    prefix_ids.extend(tokenizer.encode(prompt))
    prefix_ids.extend([user_end, asst_start])

    samples, _ = engine.generate_batch(
        prefix_ids,
        num_samples=1,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )
    return tokenizer.decode(samples[0])


def main():
    parser = argparse.ArgumentParser(
        description="Compare base vs. mid-trained model on PT-BR questions"
    )
    parser.add_argument("--base-tag",   type=str, default=None,
                        help="Base checkpoint tag (default: auto-detect)")
    parser.add_argument("--base-step",  type=int, default=None,
                        help="Base checkpoint step (default: latest)")
    parser.add_argument("--mid-tag",    type=str, default="d12_midtrain",
                        help="Mid-train checkpoint tag (default: d12_midtrain)")
    parser.add_argument("--mid-step",   type=int, default=None,
                        help="Mid-train checkpoint step (default: latest)")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max generation tokens (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-k",      type=int, default=None,
                        help="Top-k (default: off)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output",     type=str, default="samples_entrega2.md")
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--base-dir",   type=str,
                        default=os.environ.get("NANOCHAT_BASE_DIR",
                                               "/mnt/E-SSD/barcelos/.cache/nanochat"))
    args = parser.parse_args()

    if args.base_dir:
        os.environ["NANOCHAT_BASE_DIR"] = args.base_dir

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)

    # ---- load base model ----
    print0("\nLoading BASE model …")
    base_model, tokenizer, base_meta = load_model(
        "base", device, phase="eval",
        model_tag=args.base_tag,
        step=args.base_step,
    )
    base_step = base_meta["step"]
    base_cfg  = base_meta["model_config"]
    print0(f"  step={base_step}  depth={base_cfg['n_layer']}  d_model={base_cfg['n_embd']}")
    base_engine = Engine(base_model, tokenizer)

    # ---- load mid-train model ----
    print0("\nLoading MID-TRAIN model …")
    mid_model, _, mid_meta = load_model(
        "base", device, phase="eval",
        model_tag=args.mid_tag,
        step=args.mid_step,
    )
    mid_step = mid_meta["step"]
    mid_cfg  = mid_meta["model_config"]
    print0(f"  step={mid_step}  tag={args.mid_tag}")
    mid_engine = Engine(mid_model, tokenizer)

    # ---- generate ----
    print0(f"\nSampling: max_tokens={args.max_tokens}  temperature={args.temperature}  "
           f"top_k={args.top_k}  seed={args.seed}")

    sections = [
        f"# Entrega 2 — Base vs. Mid-trained model comparison\n\n"
        f"**Base checkpoint**: `{args.base_tag or 'd12'}` @ step {base_step}  \n"
        f"**Mid-train checkpoint**: `{args.mid_tag}` @ step {mid_step}  \n"
        f"**max_tokens**: {args.max_tokens}  "
        f"**temperature**: {args.temperature}  "
        f"**top_k**: {args.top_k}  "
        f"**seed**: {args.seed}\n\n"
        f"*Note: the base model (Entrega 1) was not trained on the chat template,*  \n"
        f"*so its prompts are formatted as plain text (no special tokens).*  \n"
        f"*The mid-trained model uses `<|bos|><|user_start|>…<|assistant_start|>` prefix.*\n"
    ]

    print0("\n" + "=" * 72)
    print0("Entrega 2 — Base vs. Mid-trained model samples")
    print0("=" * 72)

    for i, prompt in enumerate(PROMPTS):
        print0(f"\n--- prompt {i + 1}/{len(PROMPTS)}: {prompt[:60]} ---")

        base_text = _sample_base(
            base_engine, tokenizer, prompt,
            args.max_tokens, args.temperature, args.top_k, args.seed + i,
        )
        mid_text  = _sample_midtrain(
            mid_engine, tokenizer, prompt,
            args.max_tokens, args.temperature, args.top_k, args.seed + i,
        )

        print0(f"\nBASE:\n{base_text}\n")
        print0(f"\nMID-TRAIN:\n{mid_text}\n")

        sections.append(
            f"## Pergunta {i + 1}: {prompt}\n\n"
            f"### Base model (Entrega 1)\n\n"
            f"```\n{base_text}\n```\n\n"
            f"### Mid-trained model (Entrega 2)\n\n"
            f"```\n{mid_text}\n```\n"
        )

    output_md = "\n".join(sections)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(output_md)
    print0(f"\nSamples written to: {args.output}")

    compute_cleanup()


if __name__ == "__main__":
    main()
