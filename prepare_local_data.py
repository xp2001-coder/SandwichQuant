#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nap_llm.data import iter_concatenated_json_gzip


def convert(source: Path, destination: Path, batch_size: int = 4096) -> None:
    if destination.exists():
        print(f"SKIP existing {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".parquet.incomplete")
    writer = None
    batch: list[dict[str, str]] = []
    count = 0
    try:
        for record in iter_concatenated_json_gzip(source):
            batch.append({"text": str(record.get("text", "")), "timestamp": str(record.get("timestamp", "")), "url": str(record.get("url", ""))})
            if len(batch) >= batch_size:
                table = pa.Table.from_pylist(batch)
                writer = writer or pq.ParquetWriter(temp, table.schema, compression="zstd")
                writer.write_table(table)
                count += len(batch)
                batch.clear()
                print(f"{source.name}: {count} records", flush=True)
        if batch:
            table = pa.Table.from_pylist(batch)
            writer = writer or pq.ParquetWriter(temp, table.schema, compression="zstd")
            writer.write_table(table)
            count += len(batch)
        if writer is None:
            raise RuntimeError(f"No records parsed from {source}")
        writer.close()
        writer = None
        temp.replace(destination)
        print(f"DONE {destination}: {count} records", flush=True)
    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.data_root / "raw/c4/en"
    destination = args.data_root / "raw/c4_parquet"
    convert(source / "c4-train.00000-of-01024.json.gz", destination / "train.parquet")
    convert(source / "c4-validation.00000-of-00008.json.gz", destination / "validation.parquet")


if __name__ == "__main__":
    main()
