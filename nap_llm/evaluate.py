from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .common import atomic_json, log
from .data import PAPER_PROTOCOL, build_ppl_ids
from .modeling import load_norm_state, load_quant_model, load_tokenizer, unload


METRIC_BY_TASK = {
    "piqa": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "winogrande": "acc,none",
    "boolq": "acc,none",
}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return str(value)


def _load_variant(artifact: Path, model_path: str, norm_delta: Path | None):
    model, wrapper = load_quant_model(artifact, model_path)
    if norm_delta is not None:
        load_norm_state(model, norm_delta)
    model.eval()
    model.config.use_cache = False
    return model, wrapper


@torch.inference_mode()
def evaluate_ppl(
    artifact: Path,
    model_path: str,
    dataset: str,
    output: Path,
    wiki_parquet: Path,
    c4_validation: Path,
    norm_delta: Path | None = None,
    seq_len: int = 2048,
    max_blocks: int | None = None,
) -> dict:
    tokenizer = load_tokenizer(model_path, use_fast=False)
    ids = build_ppl_ids(dataset, tokenizer, wiki_parquet, c4_validation)
    available_blocks = ids.numel() // seq_len
    protocol_blocks = 256 if dataset == "c4" else available_blocks
    requested_blocks = protocol_blocks if max_blocks is None else min(protocol_blocks, max_blocks)
    blocks = min(requested_blocks, available_blocks)
    if blocks == 0:
        raise RuntimeError(f"{dataset} has fewer than {seq_len} tokens")
    model, wrapper = _load_variant(artifact, model_path, norm_delta)
    total_nll = torch.zeros((), device="cuda", dtype=torch.float64)
    total_tokens = 0
    for index in range(blocks):
        batch = ids[:, index * seq_len : (index + 1) * seq_len].to("cuda")
        logits = model(input_ids=batch, use_cache=False).logits[:, :-1].float()
        labels = batch[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum")
        total_nll += nll.double()
        total_tokens += labels.numel()
        if (index + 1) % 16 == 0:
            log(f"{dataset} PPL blocks {index + 1}/{blocks}")
    mean_nll = float((total_nll / total_tokens).cpu())
    result = {
        "dataset": dataset,
        "ppl": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "tokens": total_tokens,
        "blocks": blocks,
        "seq_len": seq_len,
        "protocol": PAPER_PROTOCOL,
        "tokenizer_fast": bool(getattr(tokenizer, "is_fast", False)),
    }
    atomic_json(output, result)
    unload(model, wrapper)
    return result


def evaluate_task(
    artifact: Path,
    model_path: str,
    task: str,
    output: Path,
    norm_delta: Path | None = None,
    batch_size: int = 4,
) -> dict:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tokenizer = load_tokenizer(model_path)
    model, wrapper = _load_variant(artifact, model_path, norm_delta)
    harness_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, max_length=2048)
    raw = lm_eval.simple_evaluate(model=harness_model, tasks=[task], num_fewshot=0, batch_size=batch_size, log_samples=False)
    task_result = raw["results"][task]
    metric_name = METRIC_BY_TASK[task]
    if metric_name not in task_result:
        available = sorted(task_result)
        raise KeyError(f"Expected {metric_name} for {task}; available={available}")
    result = {
        "task": task,
        "metric": metric_name,
        "value": float(task_result[metric_name]),
        "stderr": float(task_result.get(metric_name.replace(",none", "_stderr,none"), float("nan"))),
        "lm_eval_version": getattr(lm_eval, "__version__", "unknown"),
        "num_fewshot": 0,
        "raw_results": _jsonable(raw.get("results", {})),
    }
    atomic_json(output, result)
    unload(harness_model, model, wrapper)
    return result
