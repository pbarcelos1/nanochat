"""
Calcula a perplexidade do modelo base PT-Latn no conjunto de validação.

Perplexidade = exp(média da cross-entropy por token), excluindo tokens especiais.

Uso:
    python -m scripts.perplexity_entrega1
    python -m scripts.perplexity_entrega1 --model-tag d12 --split val
    python -m scripts.perplexity_entrega1 --split-tokens 524288 --device-batch-size 8
    python -m scripts.perplexity_entrega1 --dataset hf   # fineweb-2/por_Latn
"""
import os
import argparse
import math

from nanochat.common import autodetect_device_type, compute_init, compute_cleanup, print0
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_perplexity


def main():
    parser = argparse.ArgumentParser(description="Perplexidade do modelo PT-Latn")
    parser.add_argument("--model-tag", type=str, default=None,
                        help="Tag do checkpoint (ex: d12; padrão: maior disponível)")
    parser.add_argument("--step", type=int, default=None,
                        help="Step do checkpoint (padrão: último)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                        help="Split a avaliar (padrão: val)")
    parser.add_argument("--split-tokens", type=int, default=524288,
                        help="Número de tokens a avaliar (padrão: 524288 ≈ 0.5M)")
    parser.add_argument("--device-batch-size", type=int, default=8,
                        help="Batch size por dispositivo (padrão: 8)")
    parser.add_argument("--dataset", type=str, default="hf", choices=["upstream", "hf"],
                        help="Dataset: hf (fineweb-2/por_Latn, padrão) ou upstream (ClimbMix)")
    parser.add_argument("--device-type", type=str, default="",
                        help="cuda|cpu|mps (vazio = autodetect)")
    parser.add_argument("--base-dir", type=str,
                        default=os.environ.get("NANOCHAT_BASE_DIR",
                                               "/mnt/E-SSD/barcelos/.cache/nanochat"),
                        help="Diretório base dos checkpoints")
    args = parser.parse_args()

    if args.base_dir:
        os.environ["NANOCHAT_BASE_DIR"] = args.base_dir

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    model, tokenizer, meta = load_model(
        "base", device, phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )
    step = meta["step"]
    model_cfg = meta["model_config"]
    sequence_len = model_cfg["sequence_len"]

    print0(f"Modelo: step={step}  depth={model_cfg['n_layer']}  "
           f"d_model={model_cfg['n_embd']}  vocab={model_cfg['vocab_size']}")

    token_bytes = get_token_bytes(device=device)

    tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
    split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step
    if split_tokens == 0:
        split_tokens = tokens_per_step
    steps = split_tokens // tokens_per_step

    if args.dataset == "hf":
        from nanochat.dataset_hf import DATA_DIR_HF
        data_dir = DATA_DIR_HF
        print0(f"Dataset: HF fineweb-2/por_Latn  ({data_dir})")
    else:
        data_dir = None
        print0("Dataset: upstream (ClimbMix)")

    print0(f"Split: {args.split}  tokens avaliados: {split_tokens:,}  steps: {steps}")

    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, sequence_len, args.split, device=device,
        data_dir=data_dir,
    )

    ppl = evaluate_perplexity(model, loader, steps, token_bytes)

    print0(f"\nPerplexidade ({args.split}): {ppl:.4f}")
    print0(f"  (log-ppl = {math.log(ppl):.4f} nats/token)")

    compute_cleanup()


if __name__ == "__main__":
    main()
