from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterator

import torch


PAPER_PROTOCOL = "rescomp-iclr26-weight-only-v1"


def iter_concatenated_json_gzip(path: Path, chunk_size: int = 1 << 20) -> Iterator[dict]:
    """Read JSONL and the historical C4 shards that contain concatenated JSON objects."""
    decoder = json.JSONDecoder()
    buffer = ""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        eof = False
        while not eof:
            block = handle.read(chunk_size)
            eof = block == ""
            buffer += block
            offset = 0
            length = len(buffer)
            while True:
                while offset < length and buffer[offset].isspace():
                    offset += 1
                if offset >= length:
                    buffer = ""
                    break
                try:
                    obj, end = decoder.raw_decode(buffer, offset)
                except json.JSONDecodeError:
                    buffer = buffer[offset:]
                    break
                if isinstance(obj, dict):
                    yield obj
                offset = end
            if eof and buffer.strip():
                obj, end = decoder.raw_decode(buffer)
                if buffer[end:].strip():
                    raise ValueError(f"Unparsed trailing C4 content in {path}")
                if isinstance(obj, dict):
                    yield obj


def c4_texts(path: Path, limit: int | None = None) -> list[str]:
    texts: list[str] = []
    for record in iter_concatenated_json_gzip(path):
        text = str(record.get("text", "")).strip()
        if text:
            texts.append(text)
        if limit is not None and len(texts) >= limit:
            break
    return texts


def calibration_texts(path: Path, count: int, seed: int, min_chars: int = 512) -> list[str]:
    """Deterministically sample C4 using the same index policy as author code."""
    rng = random.Random(seed)
    pool: list[str] = []
    for record in iter_concatenated_json_gzip(path):
        text = str(record.get("text", "")).strip()
        if len(text) >= min_chars:
            pool.append(text)
        if len(pool) >= max(10000, count * 40):
            break
    if len(pool) < count:
        raise RuntimeError(f"Only found {len(pool)} usable C4 documents in {path}")
    return [pool[rng.randrange(len(pool))] for _ in range(count)]


def token_blocks(
    tokenizer,
    texts: list[str],
    seq_len: int,
    count: int,
    seed: int,
) -> list[torch.Tensor]:
    rng = random.Random(seed)
    candidates: list[torch.Tensor] = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if ids.numel() < seq_len + 1:
            continue
        start = rng.randint(0, ids.numel() - seq_len - 1)
        candidates.append(ids[start : start + seq_len].clone())
        if len(candidates) >= count:
            break
    if len(candidates) < count:
        joined = tokenizer("\n\n".join(texts), add_special_tokens=False, return_tensors="pt").input_ids[0]
        while len(candidates) < count and joined.numel() >= seq_len + 1:
            start = rng.randint(0, joined.numel() - seq_len - 1)
            candidates.append(joined[start : start + seq_len].clone())
    if len(candidates) < count:
        raise RuntimeError(f"Only created {len(candidates)} token blocks; requested {count}")
    return candidates


def calibration_cache_path(
    data_root: Path,
    model_key: str,
    seed: int,
    count: int = 128,
    seq_len: int = 2048,
) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_key)
    return data_root / "prepared" / "paper_calibration" / f"{safe_model}_c4_s{seed}_n{count}_l{seq_len}.pt"


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_author_c4_calibration_tokens(
    tokenizer,
    data_root: Path,
    model_key: str,
    seed: int,
    count: int = 128,
    seq_len: int = 2048,
) -> Path:
    """Create the exact document-wise C4 sampler used by GPTAQ/ResComp.

    A document index is sampled uniformly from the local C4 train shard. Short
    documents are rejected, and one random contiguous token span is retained.
    The resulting tensor is shared by every compared quantizer and by NAP.
    """
    from datasets import load_dataset

    output = calibration_cache_path(data_root, model_key, seed, count, seq_len)
    metadata = output.with_suffix(".json")
    if output.is_file() and metadata.is_file():
        blocks = torch.load(output, map_location="cpu", weights_only=True)
        if tuple(blocks.shape) == (count, seq_len):
            return output

    parquet = data_root / "raw" / "c4_parquet" / "train.parquet"
    if not parquet.is_file():
        raise FileNotFoundError(f"Paper-aligned C4 parquet is missing: {parquet}")
    dataset = load_dataset(
        "parquet",
        data_files={"train": str(parquet)},
        split="train",
        verification_mode="no_checks",
    )
    rng = random.Random(seed)
    blocks: list[torch.Tensor] = []
    sampled_documents: list[int] = []
    sampled_starts: list[int] = []
    while len(blocks) < count:
        document_index = rng.randint(0, len(dataset) - 1)
        encoded = tokenizer(str(dataset[document_index]["text"]), return_tensors="pt").input_ids[0]
        if encoded.numel() < seq_len + 1:
            continue
        start = rng.randint(0, encoded.numel() - seq_len - 1)
        blocks.append(encoded[start : start + seq_len].cpu().clone())
        sampled_documents.append(document_index)
        sampled_starts.append(start)

    tensor = torch.stack(blocks).contiguous()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(tensor, temporary)
    temporary.replace(output)
    metadata.write_text(
        json.dumps(
            {
                "protocol": PAPER_PROTOCOL,
                "dataset": "allenai/c4",
                "local_source": str(parquet),
                "local_source_sha256": _sha256(parquet),
                "seed": seed,
                "samples": count,
                "seq_len": seq_len,
                "tokenizer_class": type(tokenizer).__name__,
                "use_fast": bool(getattr(tokenizer, "is_fast", False)),
                "sampled_document_indices": sampled_documents,
                "sampled_token_starts": sampled_starts,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output


def nap_subblocks_from_calibration(
    calibration_path: Path,
    seq_len: int,
    count: int,
    seed: int,
) -> list[torch.Tensor]:
    source = torch.load(calibration_path, map_location="cpu", weights_only=True)
    if source.ndim != 2 or source.shape[1] < seq_len:
        raise ValueError(f"Invalid calibration tensor {calibration_path}: shape={tuple(source.shape)}")
    rng = random.Random(seed)
    order = list(range(source.shape[0]))
    rng.shuffle(order)
    blocks: list[torch.Tensor] = []
    for index in range(count):
        row = source[order[index % len(order)]]
        start = rng.randint(0, row.numel() - seq_len) if row.numel() > seq_len else 0
        blocks.append(row[start : start + seq_len].clone())
    return blocks


def load_wikitext_test(parquet_path: Path) -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files={"test": str(parquet_path)}, split="test")
    return [str(value) for value in dataset["text"]]


def build_ppl_ids(dataset: str, tokenizer, wiki_parquet: Path, c4_validation: Path) -> torch.Tensor:
    if dataset == "wikitext2":
        text = "\n\n".join(load_wikitext_test(wiki_parquet))
    elif dataset == "c4":
        # GPTAQ/ResComp concatenate the first 1,100 validation documents with
        # spaces and then retain exactly the first 256 x 2,048 tokens.
        text = " ".join(c4_texts(c4_validation, limit=1100))
    else:
        raise ValueError(dataset)
    # Match the author code's tokenizer call, including the tokenizer's default
    # handling of special tokens.
    return tokenizer(text, return_tensors="pt").input_ids
