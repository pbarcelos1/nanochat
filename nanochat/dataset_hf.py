"""
HuggingFaceFW/fineweb-2 → por_Latn dataset downloader for nanochat pretraining.

Streams documents from the HF Hub (no need to download the full ~500 GB subset)
and writes them as local parquet shards with a 'text' column — identical schema
to the upstream nanochat shards, so nanochat.dataloader works unchanged.

Shard layout:
  One shard  = one parquet file, DOCS_PER_SHARD documents.
  One parquet = ROW_GROUP_SIZE docs per row-group (50 groups/shard by default).
  Row-group granularity matches how nanochat.dataloader does DDP sharding
  (each rank reads every world_size-th row-group), so the layout is correct
  for future multi-GPU use even though the target is single-GPU.

Data budget:
  Portuguese BPE is ~3.5-4.0 chars/token (accented chars cost more bytes).
  2B tokens × 3.75 chars/token ≈ 7.5 GB raw text.
  At 50k docs/shard and ~3 KB avg doc length: ~50 shards ≈ 7.5 GB ≈ 2B tokens.
  Use -n 55 for a comfortable margin; -n 2 for a quick smoke test.

Usage:
    python -m nanochat.dataset_hf -n 55   # ~2.1B tokens of training data
    python -m nanochat.dataset_hf -n 2    # quick smoke test
"""

import os
import argparse

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset  # noqa: F401 (used in download_shards)

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
HF_DATASET_NAME  = "HuggingFaceFW/fineweb-2"
HF_SUBSET        = "por_Latn"
DOCS_PER_SHARD   = 50_000   # documents per parquet file
ROW_GROUP_SIZE   = 1_000    # documents per row-group inside each shard

base_dir    = get_base_dir()
DATA_DIR_HF = os.path.join(base_dir, "data_fineweb2_pt")

# The last shard is reserved as the validation split (mirrors upstream convention).
# Training code must pass split="train" (all shards except last) or split="val" (last shard).

# -----------------------------------------------------------------------------

def list_parquet_files_hf(data_dir=None):
    """Return sorted list of full paths to all HF parquet shards on disk."""
    data_dir = DATA_DIR_HF if data_dir is None else data_dir
    assert os.path.exists(data_dir), (
        f"HF data directory not found: {data_dir}\n"
        f"Run:  python -m nanochat.dataset_hf -n 55"
    )
    parquet_files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    )
    assert parquet_files, f"No parquet shards found in {data_dir}"
    return [os.path.join(data_dir, f) for f in parquet_files]


def parquets_iter_batched_hf(split, start=0, step=1, data_dir=None):
    """
    Same interface as nanochat.dataset.parquets_iter_batched but reads from
    the HF fineweb-2/por_Latn shards.  Used by tok_train.py and tok_eval.py
    so they don't need to import from nanochat.dataset at all.

    split="train" → all shards except the last (validation) shard.
    split="val"   → last shard only.
    start/step    → row-group-level DDP sharding (start=rank, step=world_size).
    """
    assert split in ("train", "val"), "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files_hf(data_dir=data_dir)
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts


def _write_shard(shard_idx, docs, data_dir):
    """Write a list of text strings as a single parquet shard with multiple row-groups."""
    filepath = os.path.join(data_dir, f"shard_{shard_idx:05d}.parquet")
    temp_path = filepath + ".tmp"

    schema = pa.schema([pa.field("text", pa.string())])
    writer = pq.ParquetWriter(temp_path, schema)
    for start in range(0, len(docs), ROW_GROUP_SIZE):
        chunk = docs[start:start + ROW_GROUP_SIZE]
        batch = pa.record_batch({"text": chunk}, schema=schema)
        writer.write_batch(batch)
    writer.close()
    os.rename(temp_path, filepath)


def download_shards(num_shards, data_dir=None):
    """
    Stream por_Latn documents from HF and write num_shards parquet files.

    Skips shards that already exist on disk (safe to re-run after interruption).
    Prints running character count so you can gauge token budget.
    """
    data_dir = DATA_DIR_HF if data_dir is None else data_dir
    os.makedirs(data_dir, exist_ok=True)

    # Check which shards are already done
    existing = {
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    }
    shards_needed = [i for i in range(num_shards) if f"shard_{i:05d}.parquet" not in existing]
    if not shards_needed:
        print(f"All {num_shards} shards already present in {data_dir}, nothing to do.")
        return

    print(f"Streaming {HF_SUBSET} from {HF_DATASET_NAME}...")
    print(f"Target: {num_shards} shards ({num_shards * DOCS_PER_SHARD:,} docs)")
    print(f"Output: {data_dir}")
    print(f"Shards to write: {len(shards_needed)} (skipping {num_shards - len(shards_needed)} existing)")

    ds = load_dataset(HF_DATASET_NAME, name=HF_SUBSET, split="train", streaming=True)

    total_chars = 0
    docs_buffer = []
    shard_idx = 0           # index into the full range [0, num_shards)
    stream_idx = 0          # documents consumed from the stream so far
    next_needed = set(shards_needed)

    for doc in ds:
        text = doc["text"]
        total_chars += len(text)

        # Are we still filling shard shard_idx?
        if shard_idx in next_needed:
            docs_buffer.append(text)
        else:
            # This shard already exists; advance the stream without storing
            pass

        stream_idx += 1
        if stream_idx % DOCS_PER_SHARD == 0:
            if shard_idx in next_needed:
                _write_shard(shard_idx, docs_buffer, data_dir)
                print(
                    f"  shard {shard_idx:05d} written  "
                    f"({DOCS_PER_SHARD:,} docs | "
                    f"{total_chars / 1e9:.3f} B chars total)"
                )
            docs_buffer = []
            shard_idx += 1
            if shard_idx >= num_shards:
                break

    # Flush any partial final shard
    if docs_buffer and shard_idx < num_shards and shard_idx in next_needed:
        _write_shard(shard_idx, docs_buffer, data_dir)
        print(f"  shard {shard_idx:05d} written  ({len(docs_buffer):,} docs | final/partial)")

    written = len([f for f in os.listdir(data_dir) if f.endswith('.parquet')])
    print(f"\nDone. {written} shards on disk in {data_dir}")
    print(f"Total chars streamed: {total_chars:,}  (~{total_chars / 1e9:.2f} B)")
    print(f"Estimated tokens:     ~{total_chars / 3.75 / 1e9:.2f} B  (at 3.75 chars/tok)")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HF fineweb-2/por_Latn shards")
    parser.add_argument("-n", "--num-shards", type=int, default=55,
                        help="Number of shards to download (default 55 ≈ 2.1B tokens)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override output directory (default: $NANOCHAT_BASE_DIR/data_fineweb2_pt)")
    args = parser.parse_args()
    download_shards(args.num_shards, data_dir=args.data_dir)
