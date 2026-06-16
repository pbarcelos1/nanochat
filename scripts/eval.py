"""
Evaluation scaffold — Etapa 4.

Perplexity (compute_perplexity) is a placeholder for a future step.
Multiple-choice evaluation (eval_multiple_choice) is implemented for BLUEX
via per-token log-likelihood — no autoregressive generation required.

Usage:
    python -m scripts.eval --checkpoint d12_sft --task ppl
    python -m scripts.eval --checkpoint d12_sft --task mc --benchmark bluex
    python -m scripts.eval --checkpoint d12 --task mc --benchmark bluex --limit 50
"""

import argparse
import os
import re

import torch
import torch.nn.functional as F


def compute_perplexity(checkpoint, val_split="por_Latn"):
    """
    Compute perplexity of `checkpoint` on `val_split`.

    Args:
        checkpoint: str — checkpoint tag (e.g. "d12_sft")
        val_split:  str — validation split identifier

    Returns:
        float — perplexity
    """
    raise NotImplementedError("Etapa 4")


# ---------------------------------------------------------------------------
# BLUEX helpers
# ---------------------------------------------------------------------------

_STRIP_PREFIX = re.compile(r"^\s*[a-eA-E]\s*[\)\.\-:]\s*")


def _strip_alt_prefix(text):
    """Remove letter label ('a) ', 'b. ', etc.) from the start of an alternative."""
    return _STRIP_PREFIX.sub("", text)


def _load_bluex():
    from datasets import load_dataset
    return load_dataset("portuguese-benchmark-datasets/BLUEX", split="questions")


def _apply_filters(ds):
    """
    Apply the three eligibility filters in order.
    Returns (eligible_rows, counts_dict).

    Filters:
      1. has_associated_images == False  (model is text-only)
      2. len(alternatives) >= 2
      3. answer normalises to a–e AND maps to a valid index in alternatives
    """
    rows = list(ds)
    n_total = len(rows)

    f1 = [r for r in rows if not r["has_associated_images"]]

    f2 = [r for r in f1 if len(r["alternatives"]) >= 2]

    f3, discarded_ids = [], []
    for r in f2:
        if r["answer"] is None:
            discarded_ids.append(r["id"])
            continue
        ans = r["answer"].strip().lower()
        if ans not in {"a", "b", "c", "d", "e"}:
            discarded_ids.append(r["id"])
            continue
        if (ord(ans) - ord("a")) >= len(r["alternatives"]):
            discarded_ids.append(r["id"])
            continue
        f3.append(r)

    counts = {
        "n_total": n_total,
        "n_f1_no_image": len(f1),
        "n_f2_min2_alts": len(f2),
        "n_eligible": len(f3),
        "discarded_f3": discarded_ids,
    }
    return f3, counts


def _build_context_ids(question, prompt_format, tokenizer):
    """
    Encode the question as a context sequence.

    plain  (d12 base):  <bos> {question}\nResposta:
    chat   (mid/sft):   <bos> <|user_start|> {question} <|user_end|> <|assistant_start|>

    The continuation (stripped alternative text) is appended by the caller.
    """
    bos = tokenizer.get_bos_token_id()
    if prompt_format == "plain":
        return [bos] + tokenizer.encode(f"{question}\nResposta:")
    else:
        usr_s = tokenizer.encode_special("<|user_start|>")
        usr_e = tokenizer.encode_special("<|user_end|>")
        ast_s = tokenizer.encode_special("<|assistant_start|>")
        return [bos, usr_s] + tokenizer.encode(question) + [usr_e, ast_s]


def _score_question(model, tokenizer, ctx_ids, cont_ids_list, device):
    """
    Batched forward over all (ctx + cont_k) sequences for one question.
    Returns (scores_sum, scores_mean), each a list of length K.

    Right-padding is used; padding positions are excluded from the score by
    gathering only the continuation slice for each alternative.
    """
    K = len(cont_ids_list)
    ctx_len = len(ctx_ids)

    full_seqs = [ctx_ids + cont for cont in cont_ids_list]
    max_len = max(len(s) for s in full_seqs)

    padded = torch.zeros(K, max_len, dtype=torch.long, device=device)
    for i, seq in enumerate(full_seqs):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    logits = model(padded)                     # (K, max_len, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1)  # (K, max_len, vocab_size)

    scores_sum, scores_mean = [], []
    for i, cont_ids in enumerate(cont_ids_list):
        cont_len = len(cont_ids)
        # At logits position p, the model predicts token p+1.
        # Continuation token j is at full_seq position ctx_len+j,
        # so it is predicted by logits at position ctx_len+j-1.
        pos_start = ctx_len - 1
        pos_end = ctx_len + cont_len - 1  # exclusive
        lp = log_probs[i, pos_start:pos_end, :]           # (cont_len, vocab_size)
        targets = torch.tensor(cont_ids, dtype=torch.long, device=device)
        gathered = lp.gather(1, targets.unsqueeze(1)).squeeze(1)  # (cont_len,)
        scores_sum.append(gathered.sum().item())
        scores_mean.append(gathered.mean().item())

    return scores_sum, scores_mean


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def eval_multiple_choice(checkpoint, benchmark="bluex",
                         prompt_format="auto", length_norm="both",
                         limit=None, batch_size=8):
    """
    Evaluate `checkpoint` on a multiple-choice benchmark via log-likelihood.

    For each question, scores P(alternative | context) for every alternative
    by summing per-token log-probabilities over the continuation.  No
    autoregressive generation is needed.

    Args:
        checkpoint:    str  — checkpoint tag, e.g. "d12", "d12_midtrain", "d12_sft"
        benchmark:     str  — "bluex" (only BLUEX implemented)
        prompt_format: str  — "plain" | "chat" | "auto"
                              auto → plain for d12, chat for midtrain/sft
        length_norm:   str  — "sum" | "mean" | "both"
        limit:         int  — evaluate only the first N eligible questions (debug)
        batch_size:    int  — reserved; alternatives per question are always batched together

    Returns:
        dict with acc_sum, acc_mean, n_scored, filter counts, baseline.
    """
    assert benchmark == "bluex", f"Only 'bluex' is implemented; got '{benchmark}'"

    # ------------------------------------------------------------------ setup
    from nanochat.checkpoint_manager import load_model_from_dir
    from nanochat.common import get_base_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if prompt_format == "auto":
        prompt_format = "plain" if checkpoint == "d12" else "chat"
    assert prompt_format in ("plain", "chat"), f"Invalid prompt_format: {prompt_format}"

    print(
        f"[eval] checkpoint={checkpoint}  prompt_format={prompt_format}  "
        f"device={device}  limit={limit}",
        flush=True,
    )

    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, "base_checkpoints")
    model, tokenizer, _ = load_model_from_dir(
        checkpoints_dir, device, phase="eval", model_tag=checkpoint
    )
    model.eval()

    # ----------------------------------------------------------------- dataset
    print("[eval] loading BLUEX …", flush=True)
    ds = _load_bluex()
    eligible, counts = _apply_filters(ds)
    print(
        f"[filters] total={counts['n_total']}"
        f"  → no-image={counts['n_f1_no_image']}"
        f"  → ≥2-alts={counts['n_f2_min2_alts']}"
        f"  → valid-answer={counts['n_eligible']}",
        flush=True,
    )
    if counts["discarded_f3"]:
        print(f"  discarded (invalid answer): {counts['discarded_f3']}", flush=True)

    baseline = sum(1.0 / len(r["alternatives"]) for r in eligible) / len(eligible)
    print(f"[eval] random baseline = {baseline:.4f}", flush=True)

    to_score = eligible[:limit] if limit is not None else eligible

    # --------------------------------------------------------------- scoring
    correct_sum = correct_mean = 0
    debug_printed = False

    with torch.inference_mode():
        for n, row in enumerate(to_score):
            question = row["question"]
            alts_raw = row["alternatives"]
            ans_idx = ord(row["answer"].strip().lower()) - ord("a")

            ctx_ids = _build_context_ids(question, prompt_format, tokenizer)
            cont_ids_list = [tokenizer.encode(_strip_alt_prefix(a)) for a in alts_raw]

            scores_sum, scores_mean = _score_question(
                model, tokenizer, ctx_ids, cont_ids_list, device
            )

            pred_sum = max(range(len(scores_sum)), key=lambda i: scores_sum[i])
            pred_mean = max(range(len(scores_mean)), key=lambda i: scores_mean[i])

            if pred_sum == ans_idx:
                correct_sum += 1
            if pred_mean == ans_idx:
                correct_mean += 1

            # One example for manual sanity check
            if not debug_printed:
                print(f"\n[debug] id={row['id']}  correct={row['answer'].strip().lower()}")
                for i, (alt, cs, cm) in enumerate(
                    zip(alts_raw, scores_sum, scores_mean)
                ):
                    tags = ""
                    if i == ans_idx:
                        tags += " ← correct"
                    if i == pred_sum:
                        tags += " ← pred(sum)"
                    if i == pred_mean and pred_mean != pred_sum:
                        tags += " ← pred(mean)"
                    print(
                        f"  [{i}] sum={cs:8.3f}  mean={cm:6.3f}  "
                        f"{repr(_strip_alt_prefix(alt)[:60])}{tags}"
                    )
                print(flush=True)
                debug_printed = True

            if (n + 1) % 100 == 0:
                print(
                    f"  {n+1}/{len(to_score)}"
                    f"  acc_sum={correct_sum/(n+1):.4f}"
                    f"  acc_mean={correct_mean/(n+1):.4f}",
                    flush=True,
                )

    n_scored = len(to_score)
    acc_sum = correct_sum / n_scored
    acc_mean = correct_mean / n_scored

    result = {
        "checkpoint": checkpoint,
        "prompt_format": prompt_format,
        "benchmark": benchmark,
        **counts,
        "n_scored": n_scored,
        "acc_sum": acc_sum,
        "acc_mean": acc_mean,
        "baseline": baseline,
    }

    print(
        f"\n[result] checkpoint={checkpoint}"
        f"  n={n_scored}"
        f"  acc_sum={acc_sum:.4f}"
        f"  acc_mean={acc_mean:.4f}"
        f"  baseline={baseline:.4f}",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation — Etapa 4")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint tag (e.g. d12, d12_midtrain, d12_sft)")
    parser.add_argument("--task", type=str, choices=["ppl", "mc"], required=True,
                        help="ppl: perplexity | mc: multiple-choice accuracy")
    parser.add_argument("--val-split", type=str, default="por_Latn",
                        help="Validation split for perplexity (default: por_Latn)")
    parser.add_argument("--benchmark", type=str, default="bluex",
                        help="Benchmark for --task mc (default: bluex)")
    parser.add_argument("--prompt-format", type=str, default="auto",
                        choices=["auto", "plain", "chat"],
                        help="Context format: auto infers from checkpoint name")
    parser.add_argument("--length-norm", type=str, default="both",
                        choices=["sum", "mean", "both"],
                        help="Log-prob normalisation (default: both)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N eligible questions")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Alternatives per forward call (default: 8)")
    args = parser.parse_args()

    if args.task == "ppl":
        result = compute_perplexity(args.checkpoint, args.val_split)
        print(f"Perplexity: {result:.2f}")
    elif args.task == "mc":
        result = eval_multiple_choice(
            args.checkpoint,
            benchmark=args.benchmark,
            prompt_format=args.prompt_format,
            length_norm=args.length_norm,
            limit=args.limit,
            batch_size=args.batch_size,
        )
        print(f"\nAccuracy (sum)  = {result['acc_sum']:.4f}")
        print(f"Accuracy (mean) = {result['acc_mean']:.4f}")
        print(f"Baseline random = {result['baseline']:.4f}")
        print(f"Questions scored: {result['n_scored']}")
