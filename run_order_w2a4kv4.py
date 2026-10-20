#!/usr/bin/env python3
"""Four-order ablation for Llama3-8B QuaRot+ResComp W2A4KV4.

GPU 0 prepares and evaluates Ours->PTQ.  GPU 1 may be launched at the same
time; it waits for the checkpoint, then performs the second Ours phase and
evaluates Ours->PTQ->Ours.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch

import run_w2a4kv4 as w2
from nap_llm.common import atomic_json, log, seed_everything
from nap_llm.modeling import load_tokenizer, unload


PROTOCOL = "nap-order-ablation-w2a4kv4-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("prepare", "post", "all"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-key", choices=("llama3_8b", "qwen3_8b"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--w2-runs-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--minutes", type=float, default=355.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nap-steps", type=int, default=200)
    parser.add_argument("--nap-seq-len", type=int, default=256)
    parser.add_argument("--nap-lr", type=float, default=5e-4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--quant-reserve-minutes", type=float, default=30.0)
    parser.add_argument("--wait-minutes", type=float, default=300.0)
    return parser.parse_args()


def copy_existing_results(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Completed source results are missing: {source}")
    shutil.copytree(source, destination, dirs_exist_ok=True)


def runner_args(args: argparse.Namespace, run_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model_path=args.model_path,
        model_key=args.model_key,
        method="rescomp_gptaq",
        rotation="quarot",
        optimized_rotation_path=None,
        data_root=args.data_root,
        runs_root=args.runs_root,
        minutes=args.minutes,
        quant_reserve_minutes=args.quant_reserve_minutes,
        seed=args.seed,
        calibration_samples=128,
        calibration_seq_len=2048,
        nap_steps=args.nap_steps,
        nap_seq_len=args.nap_seq_len,
        nap_lr=args.nap_lr,
        eval_batch_size=args.eval_batch_size,
        eval_start_margin=5.0,
        nap_checkpoint_margin=5.0,
        smoke_layers=0,
        baseline_ppl_only=False,
        skip_tasks=False,
    )


def wait_for(path: Path, deadline: float) -> bool:
    while not path.is_file() and time.time() < deadline:
        log(f"Waiting for GPU0 artifact: {path}")
        time.sleep(20)
    return path.is_file()


def prepare_quantized(args, run_dir: Path, source_norm: Path, deadline: float):
    project_root = Path(__file__).resolve().parent
    modules = w2.official_modules(project_root)
    rargs = runner_args(args, run_dir)
    checkpoint = run_dir / "ours_ptq/artifact/model.pt"
    model, qargs = w2.build_quarot_model(rargs, run_dir, modules)
    if checkpoint.is_file():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:8]}")
        log(f"Restored preconditioned W2A4KV4 graph; missing keys={len(missing)}")
        return model, modules, rargs
    if w2.remaining_minutes(deadline) <= args.quant_reserve_minutes:
        raise SystemExit(75)

    # Transfer only Phi* to the dense rotated graph before the new PTQ pass.
    w2.load_affine(model, source_norm)
    loader = modules.data.get_loaders(
        "wikitext2",
        nsamples=128,
        seed=args.seed,
        model=args.model_path,
        seqlen=2048,
        hf_token=None,
        eval_mode=False,
    )
    log(f"Quantizing preconditioned {args.model_key} with QuaRot+ResComp, W2A4KV4")
    modules.rescomp.gptaq_fwrd(model, loader, torch.device("cuda:0"), qargs)
    w2.save_quant_checkpoint(
        model,
        checkpoint,
        {
            "protocol": PROTOCOL,
            "order": "ours_ptq",
            "base_quantizer": "QuaRot+ResComp W2A4KV4 g128",
            "seed": args.seed,
            "preconditioner": str(source_norm),
        },
    )
    atomic_json(run_dir / "state/quantize_preconditioned.done.json", {"finished": time.time()})
    return model, modules, rargs


def load_pre_quantized(args, run_dir: Path):
    project_root = Path(__file__).resolve().parent
    modules = w2.official_modules(project_root)
    rargs = runner_args(args, run_dir)
    checkpoint = run_dir / "ours_ptq/artifact/model.pt"
    model, _ = w2.build_quarot_model(rargs, run_dir, modules)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:8]}")
    log(f"Loaded Ours->PTQ graph; missing keys={len(missing)}")
    return model, modules, rargs


def main() -> int:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.w2_runs_root = args.w2_runs_root.resolve()
    args.runs_root = args.runs_root.resolve()
    seed_everything(args.seed)
    deadline = time.time() + args.minutes * 60

    source = args.w2_runs_root / f"{args.model_key}_quarot_rescomp_w2a4kv4_s{args.seed}"
    source_norm = source / "nap/norm_latest.safetensors"
    # The preconditioning experiment needs only Phi* and the completed JSON
    # metrics from the source run. Its quantized artifact is not
    # consumed: Ours->PTQ deliberately quantizes a fresh rotated dense graph.
    for required in (source_norm, Path(args.model_path) / "config.json"):
        if not required.is_file():
            raise FileNotFoundError(f"Required input is missing: {required}")

    run_dir = args.runs_root / f"{args.model_key}_quarot_rescomp_w2a4kv4_order_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    copy_existing_results(source / "results/baseline", run_dir / "results/ptq")
    copy_existing_results(source / "results/nap", run_dir / "results/ptq_ours")
    atomic_json(
        run_dir / "protocol.json",
        {
            "protocol": PROTOCOL,
            "model_path": args.model_path,
            "base_quantizer": "QuaRot+ResComp W2A4KV4 g128",
            "source_run": str(source),
            "seed": args.seed,
            "nap_steps_per_phase": args.nap_steps,
        },
    )

    checkpoint = run_dir / "ours_ptq/artifact/model.pt"
    if args.role in {"prepare", "all"}:
        model, modules, rargs = prepare_quantized(args, run_dir, source_norm, deadline)
        model.to("cuda").eval()
        tokenizer = load_tokenizer(args.model_path, use_fast=False)
        if not w2.evaluate_variant(rargs, model, tokenizer, run_dir, "ours_ptq", deadline):
            return 75
        atomic_json(run_dir / "state/ours_ptq_eval.done.json", {"finished": time.time()})
        unload(model)
        if args.role == "prepare":
            return 0

    if args.role in {"post", "all"}:
        wait_deadline = min(deadline, time.time() + args.wait_minutes * 60)
        if not wait_for(checkpoint, wait_deadline):
            log("Timed out waiting for the Ours->PTQ artifact")
            return 75
        model, modules, rargs = load_pre_quantized(args, run_dir)
        model.to("cuda").eval()
        tokenizer = load_tokenizer(args.model_path, use_fast=False)
        if not w2.train_nap_full_quant(rargs, model, modules, run_dir / "ours_ptq_ours", deadline):
            return 75
        if not w2.evaluate_variant(rargs, model, tokenizer, run_dir, "ours_ptq_ours", deadline):
            return 75
        atomic_json(run_dir / "state/ours_ptq_ours_eval.done.json", {"finished": time.time()})
        unload(model)

    needed = [
        run_dir / "results" / variant / f"ppl_{dataset}.json"
        for variant in ("ptq", "ptq_ours", "ours_ptq", "ours_ptq_ours")
        for dataset in ("wikitext2", "c4")
    ]
    if all(path.is_file() for path in needed):
        atomic_json(run_dir / "COMPLETE.json", {"protocol": PROTOCOL, "finished": time.time()})
        log(f"PPL-complete: {args.model_key} W2A4KV4 four-order ablation")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted; completed stages are preserved")
        raise SystemExit(130)
