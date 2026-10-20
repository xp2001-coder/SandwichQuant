#!/usr/bin/env python3
"""Four-order ablation for Llama3-8B or Qwen3-8B GPTQ W3A16.

The already completed PTQ and PTQ->Ours runs are reused.  For the two new
orders we transfer only the learned normalization-affine state to the dense
checkpoint, quantize it from scratch, and optionally run a second post-PTQ
normalization-affine adaptation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from nap_llm.common import Job, PPL_DATASETS, TASKS, atomic_json, log, seed_everything, stage_done, stage_record
from nap_llm.data import calibration_cache_path, ensure_author_c4_calibration_tokens, nap_subblocks_from_calibration
from nap_llm.evaluate import evaluate_ppl, evaluate_task
from nap_llm.modeling import load_dense_model, load_norm_state, load_tokenizer, unload
from nap_llm.nap import train_nap
from nap_llm.quantize import run_quantization


PROTOCOL = "nap-order-ablation-w3a16-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-key", choices=("llama3_8b", "qwen3_8b"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True,
                        help="Completed GPTQ + affine run containing artifact/, nap/ and results/.")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--minutes", type=float, default=355.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nap-steps", type=int, default=200)
    parser.add_argument("--nap-seq-len", type=int, default=256)
    parser.add_argument("--nap-lr", type=float, default=5e-4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--quant-reserve-minutes", type=float, default=45.0)
    return parser.parse_args()


def copy_existing_results(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Completed source results are missing: {source}")
    shutil.copytree(source, destination, dirs_exist_ok=True)


def save_preconditioned_hf(model_path: str, norm_state: Path, output: Path) -> None:
    if (output / "config.json").is_file():
        return
    temporary = output.with_name(output.name + ".incomplete")
    temporary.mkdir(parents=True, exist_ok=True)
    model = load_dense_model(model_path)
    load_norm_state(model, norm_state)
    model.save_pretrained(
        temporary,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer = load_tokenizer(model_path, use_fast=False)
    tokenizer.save_pretrained(temporary)
    unload(model)
    if output.exists():
        shutil.rmtree(output)
    temporary.replace(output)


def evaluate_variant(
    artifact: Path,
    model_path: str,
    output_dir: Path,
    data_root: Path,
    norm_state: Path | None,
    batch_size: int,
    deadline: float,
) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    wiki = data_root / "raw/wikitext2/wikitext-2-raw-v1/test-00000-of-00001.parquet"
    c4 = data_root / "raw/c4/en/c4-validation.00000-of-00008.json.gz"
    for dataset in PPL_DATASETS:
        output = output_dir / f"ppl_{dataset}.json"
        if output.is_file():
            continue
        if time.time() > deadline - 5 * 60:
            return False
        evaluate_ppl(artifact, model_path, dataset, output, wiki, c4, norm_delta=norm_state)
    for task in TASKS:
        output = output_dir / f"task_{task}.json"
        if output.is_file():
            continue
        if time.time() > deadline - 5 * 60:
            return False
        evaluate_task(artifact, model_path, task, output, norm_delta=norm_state, batch_size=batch_size)
    return True


def main() -> int:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.source_run = args.source_run.resolve()
    args.runs_root = args.runs_root.resolve()
    seed_everything(args.seed)
    deadline = time.time() + args.minutes * 60

    source = args.source_run
    source_artifact = source / "artifact/model.pt"
    source_norm = source / "nap/norm_latest.safetensors"
    for required in (source_artifact, source_norm, Path(args.model_path) / "config.json"):
        if not required.is_file():
            raise FileNotFoundError(f"Required input is missing: {required}")

    run_dir = args.runs_root / f"{args.model_key}_gptq_w3a16_order_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results = run_dir / "results"
    copy_existing_results(source / "results/baseline", results / "ptq")
    copy_existing_results(source / "results/nap", results / "ptq_ours")
    atomic_json(
        run_dir / "protocol.json",
        {
            "protocol": PROTOCOL,
            "model_path": args.model_path,
            "base_quantizer": "GPTQ W3A16 g128",
            "source_run": str(source),
            "seed": args.seed,
            "nap_steps_per_phase": args.nap_steps,
            "definition": {
                "ptq": "original dense -> GPTQ",
                "ptq_ours": "original dense -> GPTQ -> Ours",
                "ours_ptq": "transfer Phi* to dense -> GPTQ from scratch",
                "ours_ptq_ours": "transfer Phi* to dense -> GPTQ from scratch -> Ours",
            },
        },
    )

    pre_hf = run_dir / "preconditioned_hf"
    if not stage_done(run_dir, "prepare_preconditioned_hf"):
        with stage_record(run_dir, "prepare_preconditioned_hf"):
            save_preconditioned_hf(args.model_path, source_norm, pre_hf)

    artifact = run_dir / "ours_ptq/artifact"
    if not stage_done(run_dir, "quantize_preconditioned"):
        remaining = (deadline - time.time()) / 60
        if remaining <= args.quant_reserve_minutes:
            log(f"Not enough time to start GPTQ: remaining={remaining:.1f}m")
            return 75
        job = Job(
            job_id=f"{args.model_key}_ours_gptq_w3a16_s0",
            priority=0,
            lane="order",
            enabled=True,
            # Reuse the exact cached 128 x 2048 token tensor used by the
            # original paper-aligned run, rather than creating an equivalent
            # but differently named calibration cache.
            model_key=args.model_key,
            model_path=str(pre_hf),
            method="gptq",
            bits=3,
            group_size=128,
            seed=args.seed,
            run_nap=True,
        )
        with stage_record(run_dir, "quantize_preconditioned"):
            run_quantization(job, artifact, [], args.data_root)

    if not evaluate_variant(
        artifact,
        args.model_path,
        results / "ours_ptq",
        args.data_root,
        None,
        args.eval_batch_size,
        deadline,
    ):
        return 75

    post_dir = run_dir / "ours_ptq_ours/nap"
    post_norm = post_dir / "norm_latest.safetensors"
    if not stage_done(run_dir, "post_ours"):
        tokenizer = load_tokenizer(args.model_path, use_fast=False)
        calibration = calibration_cache_path(args.data_root, args.model_key, args.seed, 128, 2048)
        if not calibration.is_file():
            calibration = ensure_author_c4_calibration_tokens(
                tokenizer, args.data_root, args.model_key, args.seed, count=128, seq_len=2048
            )
        blocks = nap_subblocks_from_calibration(
            calibration, args.nap_seq_len, args.nap_steps, args.seed + 100003
        )
        with stage_record(run_dir, "post_ours"):
            summary = train_nap(
                args.model_path,
                artifact,
                post_dir,
                blocks,
                args.seed,
                max_steps=args.nap_steps,
                lr=args.nap_lr,
                deadline_end=deadline,
                parameterization="direct",
            )
        if not summary["completed"]:
            (run_dir / "state/post_ours.done.json").unlink(missing_ok=True)
            return 75

    if not evaluate_variant(
        artifact,
        args.model_path,
        results / "ours_ptq_ours",
        args.data_root,
        post_norm,
        args.eval_batch_size,
        deadline,
    ):
        return 75

    atomic_json(run_dir / "COMPLETE.json", {"protocol": PROTOCOL, "finished": time.time()})
    log(f"COMPLETE: {args.model_key} GPTQ W3A16 four-order ablation")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted; completed stages are preserved")
        raise SystemExit(130)
