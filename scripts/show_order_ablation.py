#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = ("piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "boolq")
ORDER = (
    ("ptq", "PTQ"),
    ("ptq_ours", "PTQ -> Post"),
    ("ours_ptq", "Pre -> PTQ"),
    ("ours_ptq_ours", "SandwichQuant"),
)


def value(path: Path, key: str) -> float | None:
    if not path.is_file():
        return None
    return float(json.loads(path.read_text(encoding="utf-8"))[key])


def fmt(item: float | None, scale: float = 1.0) -> str:
    return "--" if item is None else f"{item * scale:.2f}"


def show(run_dir: Path, label: str) -> None:
    if not run_dir.is_dir():
        return
    print(f"\n{label}")
    print(f"{'Order':27s} {'Wiki2':>8s} {'C4':>8s} {'Avg6':>8s}")
    for variant, name in ORDER:
        root = run_dir / "results" / variant
        wiki = value(root / "ppl_wikitext2.json", "ppl")
        c4 = value(root / "ppl_c4.json", "ppl")
        scores = [value(root / f"task_{task}.json", "value") for task in TASKS]
        avg = None if any(item is None for item in scores) else sum(scores) / len(scores)
        print(f"{name:27s} {fmt(wiki):>8s} {fmt(c4):>8s} {fmt(avg, 100):>8s}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs_order_ablation"))
    args = parser.parse_args()
    show(args.runs_root / "llama3_8b_gptq_w3a16_order_s0", "Llama3-8B | GPTQ W3A16 g128")
    show(args.runs_root / "qwen3_8b_gptq_w3a16_order_s0", "Qwen3-8B | GPTQ W3A16 g128")
    show(
        args.runs_root / "llama3_8b_quarot_rescomp_w2a4kv4_order_s0",
        "Llama3-8B | QuaRot+ResComp W2A4KV4 g128",
    )
    show(
        args.runs_root / "qwen3_8b_quarot_rescomp_w2a4kv4_order_s0",
        "Qwen3-8B | QuaRot+ResComp W2A4KV4 g128",
    )


if __name__ == "__main__":
    main()
