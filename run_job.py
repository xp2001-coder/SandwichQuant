#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from nap_llm.common import Deadline, Job, PPL_DATASETS, TASKS, atomic_json, log, seed_everything, stage_done, stage_record
from nap_llm.data import (
    calibration_cache_path,
    calibration_texts,
    ensure_author_c4_calibration_tokens,
    nap_subblocks_from_calibration,
    token_blocks,
)
from nap_llm.evaluate import evaluate_ppl, evaluate_task
from nap_llm.modeling import load_tokenizer
from nap_llm.nap import train_nap
from nap_llm.preflight import run_preflight
from nap_llm.quantize import run_quantization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one resumable NAP LLM experiment")
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiments.csv"))
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--minutes", type=float, default=300)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--calibration-seq-len", type=int, default=2048)
    parser.add_argument("--nap-documents", type=int, default=128)
    parser.add_argument("--nap-seq-len", type=int, default=256)
    parser.add_argument("--nap-steps", type=int, default=200)
    parser.add_argument("--nap-lr", type=float, default=None)
    parser.add_argument("--nap-parameterization", choices=("auto", "direct", "relative"), default="auto")
    parser.add_argument("--nap-max-log-scale", type=float, default=0.1)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--only-stage", default="", help="preflight|quantize|nap|baseline:ppl:*|baseline:task:*|nap:ppl:*|nap:task:*")
    parser.add_argument("--stop-after-stage", action="store_true")
    return parser.parse_args()


def load_job(matrix: Path, job_id: str) -> Job:
    with matrix.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [Job.from_row(row) for row in csv.DictReader(handle)]
    matches = [job for job in rows if job.job_id == job_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one job_id={job_id!r}; found {len(matches)}")
    return matches[0]


def should_run(args: argparse.Namespace, stage: str) -> bool:
    return not args.only_stage or args.only_stage == stage


def main() -> int:
    args = parse_args()
    args.matrix = args.matrix.resolve()
    args.data_root = args.data_root.resolve()
    args.runs_root = args.runs_root.resolve()
    job = load_job(args.matrix, args.job_id)
    if not job.enabled:
        log(f"Job {job.job_id} is disabled")
        return 0
    seed_everything(job.seed)
    deadline = Deadline.after_minutes(args.minutes)
    run_dir = args.runs_root / job.job_id
    artifact = run_dir / "artifact"
    results_dir = run_dir / "results"
    nap_dir = run_dir / "nap"
    run_dir.mkdir(parents=True, exist_ok=True)
    job_file = run_dir / "job.json"
    fingerprint = job.fingerprint()
    if job_file.exists():
        import json
        previous = json.loads(job_file.read_text(encoding="utf-8"))
        previous_fingerprint = previous.get("fingerprint")
        if previous_fingerprint and previous_fingerprint != fingerprint:
            raise RuntimeError(
                f"Matrix row for {job.job_id} changed after results were created. "
                "Use a new job_id or an empty runs directory; refusing to mix protocols."
            )
    atomic_json(job_file, {**job.__dict__, "fingerprint": fingerprint})

    stage = "preflight"
    if should_run(args, stage) and not stage_done(run_dir, stage):
        with stage_record(run_dir, stage):
            run_preflight(job, args.data_root, run_dir / "preflight.json")
        if args.stop_after_stage:
            return 0

    c4_train = args.data_root / "raw/c4/en/c4-train.00000-of-01024.json.gz"
    calibration = None
    stage = "quantize"
    if should_run(args, stage) and not stage_done(run_dir, stage):
        paper_methods = {"gptq", "gptaq", "awq", "qep_gptq", "qep_awq", "rescomp", "rescomp_gptaq"}
        long_official = job.method in {
            "gptq", "gptaq", "awq", "qep_gptq", "qep_awq", "rescomp", "rescomp_gptaq"
        } and (not job.external_artifact or not Path(job.external_artifact).is_file())
        if long_official:
            method_reserve_name = f"NAP_{job.method.upper()}_RESERVE_MINUTES"
            reserve_text = os.environ.get(
                method_reserve_name,
                os.environ.get("NAP_LONG_QUANT_RESERVE_MINUTES", "90"),
            )
        else:
            method_reserve_name = "NAP_SHORT_QUANT_RESERVE_MINUTES"
            reserve_text = os.environ.get(method_reserve_name, "15")
        try:
            reserve = max(1.0, float(reserve_text))
        except ValueError as exc:
            raise ValueError(
                f"{method_reserve_name} / NAP_LONG_QUANT_RESERVE_MINUTES "
                f"must be a number of minutes, got {reserve_text!r}"
            ) from exc
        log(
            f"Quantization reservation check: method={job.method}, "
            f"required={reserve:.1f} minutes, remaining={deadline.remaining_minutes():.1f} minutes"
        )
        if not deadline.can_start(reserve):
            if long_official:
                log(
                    f"{job.method} quantization needs at least {reserve:.1f} minutes; "
                    f"adjust {method_reserve_name} or NAP_LONG_QUANT_RESERVE_MINUTES "
                    f"if this worker is known to be faster"
                )
            log("Not enough reservation time to start quantization")
            return 75
        external_sampler = (
            job.method in paper_methods
            and os.environ.get("NAP_QUANT_BACKEND", "official").lower() != "gptqmodel"
        )
        calibration = [] if external_sampler else calibration_texts(c4_train, args.calibration_samples, job.seed)
        with stage_record(run_dir, stage):
            run_quantization(job, artifact, calibration, args.data_root)
        if args.stop_after_stage:
            return 0

    wiki = args.data_root / "raw/wikitext2/wikitext-2-raw-v1/test-00000-of-00001.parquet"
    c4_validation = args.data_root / "raw/c4/en/c4-validation.00000-of-00008.json.gz"

    def run_evals(variant: str, norm_delta: Path | None) -> bool:
        for dataset in PPL_DATASETS:
            stage_name = f"{variant}_ppl_{dataset}"
            selector = f"{variant}:ppl:{dataset}"
            if not should_run(args, selector) or stage_done(run_dir, stage_name):
                continue
            if not deadline.can_start(15):
                return False
            with stage_record(run_dir, stage_name):
                evaluate_ppl(artifact, job.model_path, dataset, results_dir / variant / f"ppl_{dataset}.json", wiki, c4_validation, norm_delta=norm_delta)
            if args.stop_after_stage:
                return False
        skip_tasks = os.environ.get("NAP_SKIP_TASKS", "0").lower() in {"1", "true", "yes"}
        for task in ([] if skip_tasks else TASKS):
            stage_name = f"{variant}_task_{task}"
            selector = f"{variant}:task:{task}"
            if not should_run(args, selector) or stage_done(run_dir, stage_name):
                continue
            if not deadline.can_start(15):
                return False
            with stage_record(run_dir, stage_name):
                evaluate_task(artifact, job.model_path, task, results_dir / variant / f"task_{task}.json", norm_delta=norm_delta, batch_size=args.eval_batch_size)
            if args.stop_after_stage:
                return False
        return True

    if not args.only_stage or args.only_stage.startswith("baseline:"):
        if not run_evals("baseline", None):
            return 75

    if job.run_nap:
        stage = "nap"
        if should_run(args, stage) and not stage_done(run_dir, stage):
            if not deadline.can_start(20):
                return 75
            paper_methods = {"gptq", "gptaq", "awq", "qep_gptq", "qep_awq", "rescomp", "rescomp_gptaq"}
            tokenizer = load_tokenizer(job.model_path, use_fast=False)
            if job.method in paper_methods:
                calibration_path = calibration_cache_path(
                    args.data_root, job.model_key, job.seed, args.calibration_samples, args.calibration_seq_len
                )
                if not calibration_path.is_file():
                    calibration_path = ensure_author_c4_calibration_tokens(
                        tokenizer,
                        args.data_root,
                        job.model_key,
                        job.seed,
                        count=args.calibration_samples,
                        seq_len=args.calibration_seq_len,
                    )
                blocks = nap_subblocks_from_calibration(
                    calibration_path, args.nap_seq_len, args.nap_steps, job.seed + 100003
                )
            else:
                if calibration is None:
                    calibration = calibration_texts(c4_train, max(args.nap_documents, args.calibration_samples), job.seed)
                blocks = token_blocks(
                    tokenizer, calibration[: args.nap_documents], args.nap_seq_len, args.nap_steps, job.seed + 100003
                )
            with stage_record(run_dir, stage):
                parameterization = args.nap_parameterization
                if parameterization == "auto":
                    parameterization = "relative" if job.method in {"awq", "qep_awq"} else "direct"
                lr = args.nap_lr
                if lr is None:
                    lr = 1e-4 if parameterization == "relative" else 5e-4
                summary = train_nap(
                    job.model_path,
                    artifact,
                    nap_dir,
                    blocks,
                    job.seed,
                    max_steps=args.nap_steps,
                    lr=lr,
                    deadline_end=deadline.end_time,
                    parameterization=parameterization,
                    max_log_scale=args.nap_max_log_scale,
                )
            if not summary["completed"]:
                (run_dir / "state" / f"{stage}.done.json").unlink(missing_ok=True)
                log("NAP checkpoint saved; resume in the next reservation")
                return 75
            if args.stop_after_stage:
                return 0
        norm_delta = nap_dir / "norm_latest.safetensors"
        if norm_delta.exists() and (not args.only_stage or args.only_stage.startswith("nap:")):
            if not run_evals("nap", norm_delta):
                return 75

    all_expected = []
    skip_tasks = os.environ.get("NAP_SKIP_TASKS", "0").lower() in {"1", "true", "yes"}
    for variant in (["baseline", "nap"] if job.run_nap else ["baseline"]):
        all_expected.extend([results_dir / variant / f"ppl_{name}.json" for name in PPL_DATASETS])
        if not skip_tasks:
            all_expected.extend([results_dir / variant / f"task_{name}.json" for name in TASKS])
    if all(path.exists() for path in all_expected):
        marker = "PPL_ONLY_COMPLETE.json" if skip_tasks else "COMPLETE.json"
        atomic_json(run_dir / marker, {"job_id": job.job_id, "completed": time.time(), "fingerprint": job.fingerprint()})
        log(f"{'PPL-ONLY COMPLETE' if skip_tasks else 'COMPLETE'}: {job.job_id}")
        return 0
    return 75 if not args.only_stage else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted; completed stages and NAP checkpoints are preserved")
        raise SystemExit(130)
