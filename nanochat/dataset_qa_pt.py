"""
ju-resplande/qa-pt → mid-training dataset for nanochat.

Pre-tokenizes QA pairs with the nanochat chat template, packs them into
fixed-length token blocks, and writes parquet shards for the mid-training
dataloader.  The tokenization happens once at download time; the training
loop reads raw integer arrays without re-tokenizing.

Chat template (canonical – derived from SPECIAL_TOKENS in nanochat/tokenizer.py):
    <|bos|><|user_start|>{question_title}<|user_end|><|assistant_start|>{answer_text}<|assistant_end|>

<|bos|> acts as the document-boundary delimiter: its presence at the start
of each example simultaneously marks the end of the preceding one when blocks
are consumed sequentially.  There is no <|end|> token in this tokenizer.

Filters (applied in a single streaming pass):
    is_accepted == True
    len(question_title) >= 10
    len(answer_text)   >= 50  and  <= 4000
    dedupe by exact question_title  (cap: MAX_DEDUP_SET to bound RAM)

Shard layout:
    One shard = one parquet file, BLOCKS_PER_SHARD rows, ROW_GROUP_SIZE rows/group.
    Each row = one packed block of SEQ_LEN+1 int32 token IDs.
    Training tokens per shard ≈ BLOCKS_PER_SHARD × SEQ_LEN = 20.48 M.

Token budget:
    For 300 M training tokens use -n 16  (15 train + 1 val ≈ 307 M train tokens).

Usage:
    python -m nanochat.dataset_qa_pt -n 16   # 15 train + 1 val ≈ 307 M tokens
    python -m nanochat.dataset_qa_pt -n 2    # quick smoke test
"""

import os
import argparse

import torch
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer

# ---------------------------------------------------------------------------
HF_DATASET_NAME  = "ju-resplande/qa-pt"
SEQ_LEN          = 2048          # must match --max-seq-len in mid_train.py
BLOCKS_PER_SHARD = 10_000        # ~20.48 M training tokens per shard
ROW_GROUP_SIZE   = 200           # parquet row-groups per shard
MAX_DEDUP_SET    = 2_000_000     # cap on seen-question set to bound RAM

base_dir       = get_base_dir()
DATA_DIR_QA_PT = os.path.join(base_dir, "data_qa_pt")

# ---------------------------------------------------------------------------


def _tokenize_qa_pair(tokenizer, question: str, answer: str) -> list:
    """
    Produce a flat list of int token IDs for one QA pair.
    Layout: <|bos|><|user_start|>{q}<|user_end|><|assistant_start|>{a}<|assistant_end|>
    Text content is encoded with encode_ordinary (no special tokens in text);
    special tokens are inserted via encode_special (correct IDs, not bytes).
    """
    bos        = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end   = tokenizer.encode_special("<|user_end|>")
    asst_start = tokenizer.encode_special("<|assistant_start|>")
    asst_end   = tokenizer.encode_special("<|assistant_end|>")

    ids = [bos, user_start]
    ids.extend(tokenizer.encode(question))  # encode_ordinary for plain text
    ids.append(user_end)
    ids.append(asst_start)
    ids.extend(tokenizer.encode(answer))    # encode_ordinary for plain text
    ids.append(asst_end)
    return ids


def _write_shard(shard_idx: int, blocks: list, data_dir: str) -> None:
    """Write blocks to a parquet shard (atomic: writes to .tmp then renames)."""
    filepath  = os.path.join(data_dir, f"shard_{shard_idx:05d}.parquet")
    temp_path = filepath + ".tmp"
    schema    = pa.schema([pa.field("ids", pa.list_(pa.int32()))])
    writer    = pq.ParquetWriter(temp_path, schema)
    for start in range(0, len(blocks), ROW_GROUP_SIZE):
        chunk = blocks[start : start + ROW_GROUP_SIZE]
        arr   = pa.array([b for b in chunk], type=pa.list_(pa.int32()))
        batch = pa.record_batch({"ids": arr}, schema=schema)
        writer.write_batch(batch)
    writer.close()
    os.rename(temp_path, filepath)


def list_parquet_files_qa_pt(data_dir=None) -> list:
    """Return sorted list of qa-pt shard paths on disk."""
    data_dir = DATA_DIR_QA_PT if data_dir is None else data_dir
    assert os.path.exists(data_dir), (
        f"qa-pt data directory not found: {data_dir}\n"
        f"Run:  python -m nanochat.dataset_qa_pt -n 16"
    )
    files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    )
    assert files, f"No parquet shards found in {data_dir}"
    return [os.path.join(data_dir, f) for f in files]


def download_shards(num_shards: int, data_dir=None, verify: int = 0) -> None:
    """
    Stream qa-pt from HF Hub, filter, tokenize, pack, write parquet shards.
    Idempotent: skips shards already present on disk.

    verify > 0: after writing, print that many decoded examples for manual
                inspection of the chat template.
    """
    data_dir = DATA_DIR_QA_PT if data_dir is None else data_dir
    os.makedirs(data_dir, exist_ok=True)

    existing = {
        f for f in os.listdir(data_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    }
    shards_needed = set(
        i for i in range(num_shards)
        if f"shard_{i:05d}.parquet" not in existing
    )
    if not shards_needed:
        print(f"All {num_shards} shards already present in {data_dir}, nothing to do.")
        if verify:
            _verify_examples(data_dir, verify)
        return

    train_shards = max(0, num_shards - 1)
    print(f"Streaming {HF_DATASET_NAME} …")
    print(f"Target : {num_shards} shards  "
          f"({train_shards} train + 1 val  =  "
          f"~{train_shards * BLOCKS_PER_SHARD * SEQ_LEN / 1e6:.0f} M train tokens)")
    print(f"Output : {data_dir}")
    print(f"Needed : {len(shards_needed)} shards  "
          f"(skipping {num_shards - len(shards_needed)} existing)")

    tokenizer = get_tokenizer()
    ds        = load_dataset(HF_DATASET_NAME, split="train", streaming=True)

    seen_questions: set   = set()
    token_buf:      list  = []        # rolling buffer of token IDs
    blocks_buf:     list  = []        # completed blocks for current shard
    shard_idx             = 0
    total_examples        = 0
    total_tokens          = 0
    verify_examples       = []        # collected for --verify

    for row in ds:
        # ---- filters ----
        if not row.get("is_accepted", False):
            continue
        q = (row.get("question_title") or "").strip()
        a = (row.get("answer_text")    or "").strip()
        if len(q) < 10 or len(a) < 50 or len(a) > 4000:
            continue
        # dedupe
        if len(seen_questions) < MAX_DEDUP_SET:
            if q in seen_questions:
                continue
            seen_questions.add(q)

        # ---- tokenize ----
        ids = _tokenize_qa_pair(tokenizer, q, a)
        token_buf.extend(ids)
        total_examples += 1
        total_tokens   += len(ids)

        if verify and len(verify_examples) < verify:
            verify_examples.append((q, a, ids))

        # ---- pack into SEQ_LEN+1 blocks ----
        while len(token_buf) >= SEQ_LEN + 1:
            block     = token_buf[: SEQ_LEN + 1]
            token_buf = token_buf[SEQ_LEN + 1 :]

            if shard_idx < num_shards:
                blocks_buf.append(block)

            if len(blocks_buf) >= BLOCKS_PER_SHARD:
                if shard_idx in shards_needed:
                    _write_shard(shard_idx, blocks_buf, data_dir)
                    print(
                        f"  shard {shard_idx:05d} written  "
                        f"({total_examples:,} examples | "
                        f"{total_tokens / 1e6:.1f} M tokens)"
                    )
                else:
                    print(f"  shard {shard_idx:05d} skipped (already exists)")
                blocks_buf = []
                shard_idx += 1
                if shard_idx >= num_shards:
                    break

        if shard_idx >= num_shards:
            break

    # flush final partial shard
    if blocks_buf and shard_idx < num_shards and shard_idx in shards_needed:
        _write_shard(shard_idx, blocks_buf, data_dir)
        print(f"  shard {shard_idx:05d} written "
              f"(final/partial, {len(blocks_buf):,} blocks)")

    written = len([f for f in os.listdir(data_dir) if f.endswith(".parquet")])
    print(f"\nDone.  {written} shards on disk in {data_dir}")
    print(f"Examples filtered : {total_examples:,}")
    print(f"Tokens (approx)   : {total_tokens:,}  ({total_tokens / 1e9:.3f} B)")
    print(f"Unique questions  : {len(seen_questions):,}")

    if verify and verify_examples:
        _verify_examples_inline(tokenizer, verify_examples)


def _verify_examples_inline(tokenizer, examples):
    """Print decoded formatted examples to verify the chat template."""
    print("\n" + "=" * 72)
    print("VERIFY: first formatted examples (decode of tokenized IDs)")
    print("=" * 72)
    for i, (q, a, ids) in enumerate(examples):
        decoded = tokenizer.decode(ids)
        print(f"\n--- example {i + 1} ---")
        print(f"question : {q[:120]}")
        print(f"answer   : {a[:120]}")
        print(f"decoded  : {decoded[:300]}")
        print(f"n_tokens : {len(ids)}")
    print("=" * 72)


def _verify_examples(data_dir, n):
    """Load and decode n examples from the first shard for manual inspection."""
    tokenizer = get_tokenizer()
    paths = list_parquet_files_qa_pt(data_dir)
    pf    = pq.ParquetFile(paths[0])
    rg    = pf.read_row_group(0)
    blocks = rg.column("ids").to_pylist()
    print("\n" + "=" * 72)
    print("VERIFY: first packed blocks from shard_00000")
    print("=" * 72)
    for i, block in enumerate(blocks[:n]):
        decoded = tokenizer.decode(block[:200])  # first 200 tokens
        print(f"\n--- block {i + 1} (first 200 of {len(block)} tokens) ---")
        print(decoded)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Dataloader for mid-training
# ---------------------------------------------------------------------------

def dataloader_qa_pt(data_dir, split, B, T, device, resume_state_dict=None):
    """
    Infinite iterator of (inputs[B,T], targets[B,T], state_dict) from
    pre-tokenized qa-pt parquet shards.

    split="train" → all shards except the last (held-out for validation).
    split="val"   → last shard only.

    state_dict keys: pq_idx, rg_idx, epoch  (compatible with base_train format).
    inputs are int32, targets are int64 (matches GPT.forward expectations).
    No loss mask is applied: every token is a learning target.
    """
    assert split in {"train", "val"}, f"Unknown split: {split}"
    all_paths = list_parquet_files_qa_pt(data_dir)
    paths = all_paths[:-1] if split == "train" else all_paths[-1:]
    assert paths, f"No {split} shards found in {data_dir}"

    resume_pq_idx = resume_state_dict.get("pq_idx", 0) if resume_state_dict else 0
    resume_rg_idx = resume_state_dict.get("rg_idx", None) if resume_state_dict else None
    epoch         = resume_state_dict.get("epoch",  1)    if resume_state_dict else 1
    first_pass    = True
    block_batch:  list = []

    while True:
        for pq_i, filepath in enumerate(paths):
            # on first pass: skip already-consumed shards
            if first_pass and pq_i < resume_pq_idx:
                continue
            pf = pq.ParquetFile(filepath)
            # on first pass at the resume shard: skip consumed row-groups
            if first_pass and resume_rg_idx is not None and pq_i == resume_pq_idx:
                rg_start      = resume_rg_idx + 1
                resume_rg_idx = None   # clear; only advance once
            else:
                rg_start = 0
            for rg_i in range(rg_start, pf.num_row_groups):
                rg     = pf.read_row_group(rg_i)
                blocks = rg.column("ids").to_pylist()
                for block in blocks:
                    if len(block) != T + 1:
                        continue  # skip blocks from a different SEQ_LEN
                    block_batch.append(block)
                    if len(block_batch) >= B:
                        arr         = torch.tensor(block_batch[:B], dtype=torch.long)
                        block_batch = block_batch[B:]
                        x = arr[:, :-1].to(device=device, dtype=torch.int32).contiguous()
                        y = arr[:, 1:].to(device=device,  dtype=torch.int64).contiguous()
                        yield x, y, {"pq_idx": pq_i, "rg_idx": rg_i, "epoch": epoch}
        first_pass = False
        epoch      += 1


# ---------------------------------------------------------------------------
# Dataloader for SFT (Etapa 3) — same parquet shards with per-token loss mask
# ---------------------------------------------------------------------------

def dataloader_qa_pt_sft(data_dir, split, B, T, device, tokenizer, resume_state_dict=None):
    """
    Yields (inputs[B,T], targets[B,T], state_dict) for SFT (Etapa 3).

    Identical to dataloader_qa_pt except that targets has -1 (ignore_index)
    for all non-assistant positions.  The mask is derived from special token
    IDs by scanning each packed block; in_assistant state is carried across
    block boundaries so conversations split at a cut are masked correctly.

    Mask rules (aligned with tokenizer.render_conversation):
      <|assistant_start|>  → mask=0  (start token itself is not supervised)
      assistant text       → mask=1  (model learns to generate the response)
      <|assistant_end|>    → mask=1  (model learns to terminate the response)
      everything else      → mask=0  (user turn, BOS, structure tokens)

    split="train" → all shards except the last.
    split="val"   → last shard only (same held-out set as mid-training eval).
    """
    asst_start_id = tokenizer.encode_special("<|assistant_start|>")
    asst_end_id   = tokenizer.encode_special("<|assistant_end|>")

    assert split in {"train", "val"}, f"Unknown split: {split}"
    all_paths = list_parquet_files_qa_pt(data_dir)
    paths = all_paths[:-1] if split == "train" else all_paths[-1:]
    assert paths, f"No {split} shards found in {data_dir}"

    resume_pq_idx = resume_state_dict.get("pq_idx", 0) if resume_state_dict else 0
    resume_rg_idx = resume_state_dict.get("rg_idx", None) if resume_state_dict else None
    epoch         = resume_state_dict.get("epoch",  1)    if resume_state_dict else 1
    first_pass    = True
    block_batch:  list = []
    mask_batch:   list = []
    in_assistant        = False  # cross-block state for mask reconstruction

    while True:
        for pq_i, filepath in enumerate(paths):
            if first_pass and pq_i < resume_pq_idx:
                continue
            pf = pq.ParquetFile(filepath)
            if first_pass and resume_rg_idx is not None and pq_i == resume_pq_idx:
                rg_start      = resume_rg_idx + 1
                resume_rg_idx = None
            else:
                rg_start = 0
            for rg_i in range(rg_start, pf.num_row_groups):
                rg     = pf.read_row_group(rg_i)
                blocks = rg.column("ids").to_pylist()
                for block in blocks:
                    if len(block) != T + 1:
                        continue
                    # Build per-token mask by scanning for special tokens.
                    # in_assistant is carried across block boundaries.
                    mask = []
                    for tok_id in block:
                        if tok_id == asst_start_id:
                            in_assistant = True
                            mask.append(0)   # start token not supervised
                        elif tok_id == asst_end_id:
                            in_assistant = False
                            mask.append(1)   # end token IS supervised
                        else:
                            mask.append(1 if in_assistant else 0)
                    block_batch.append(block)
                    mask_batch.append(mask)
                    if len(block_batch) >= B:
                        arr      = torch.tensor(block_batch[:B], dtype=torch.long)
                        mask_arr = torch.tensor(mask_batch[:B],  dtype=torch.int8)
                        block_batch = block_batch[B:]
                        mask_batch  = mask_batch[B:]
                        x = arr[:, :-1].to(device=device, dtype=torch.int32).contiguous()
                        y = arr[:, 1:].to(device=device,  dtype=torch.int64).contiguous()
                        # Shift mask by 1 to align with targets, then apply ignore_index
                        mask_t = mask_arr[:, 1:].to(device=device)
                        y[mask_t == 0] = -1
                        yield x, y, {"pq_idx": pq_i, "rg_idx": rg_i, "epoch": epoch}
        first_pass   = False
        epoch       += 1
        in_assistant = False  # reset at epoch boundary (shard 0, block 0 = start of user turn)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download & pre-tokenize ju-resplande/qa-pt for mid-training"
    )
    parser.add_argument("-n", "--num-shards", type=int, default=16,
                        help="Total shards (last is val; default 16 ≈ 307M train tokens)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override output dir (default: $NANOCHAT_BASE_DIR/data_qa_pt)")
    parser.add_argument("--verify", type=int, default=10,
                        help="Print N decoded examples after download (default 10)")
    args = parser.parse_args()
    download_shards(args.num_shards, data_dir=args.data_dir, verify=args.verify)
